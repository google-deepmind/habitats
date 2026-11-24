# coding=utf-8
# Copyright 2026 Google DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrapper to apply evolved reward functions to environments."""

from collections.abc import Callable
# These imports are needed as it is used in the evolved reward functions.
import math  # pylint: disable=unused-import
from typing import Any

from absl import logging
from brax import envs as brax_envs
import dm_env
from dm_env import specs
from habitats.environment import brax_env
from habitats.lib import numerics
import jax
import jax.numpy as jnp
import numpy as np


# Reward given for a transition whose evolved reward or observation is NaN or
# infinite. Zero is neutral: it neither rewards nor punishes the agent, so no
# penalty scale has to be guessed for each evolved reward function.
_NON_FINITE_REWARD = 0.0

# Brax metric that is 1.0 on the step where a NaN or infinite reward or
# observation ends the episode, and 0.0 otherwise. Since that step is the last
# one of its episode, the mean of the metric over episodes is the fraction of
# episodes that ended this way.
NON_FINITE_TERMINATION_METRIC = 'non_finite_termination'

# Minimum number of seconds between two warnings about non-finite transitions
# in dm_env environments, which can occur at every episode.
_NON_FINITE_WARNING_INTERVAL_SECONDS = 60


class RewardError(Exception):
  """Base class for exceptions in this module."""


class RewardCompilationError(RewardError):
  """Raised when the reward function code cannot be compiled or is invalid."""


class BraxEvolvedRewardWrapper(brax_envs.Wrapper):
  """Wraps a Brax environment to compute rewards using an evolved reward function.

  See `EvolvedRewardWrapper` for `ignore_base_termination`.
  """

  def __init__(
      self,
      env: brax_envs.Env,
      reward_fn: Callable[..., Any],
      ignore_base_termination: bool = True,
  ):
    super().__init__(env)
    self.reward_fn = reward_fn
    self.ignore_base_termination = ignore_base_termination

  def __getattr__(self, name: str) -> Any:
    return getattr(self.env, name)

  def reset(self, rng: jax.Array) -> brax_envs.State:
    """Resets the environment, initializing the metrics set by `step`."""
    state = self.env.reset(rng)
    # Brax expects `reset` to return the same metrics as `step`.
    return state.replace(  # pyrefly: ignore[missing-attribute]
        metrics={
            **state.metrics,
            NON_FINITE_TERMINATION_METRIC: jnp.zeros_like(
                state.reward, dtype=jnp.float32
            ),
        }
    )

  def step(self, state: brax_envs.State, action: jax.Array) -> brax_envs.State:
    """Steps the environment, replacing its reward with the evolved one.

    A transition with a NaN or infinite reward or observation ends the episode
    instead of reaching the learner; see `_end_episode_if_non_finite`.

    Args:
      state: The current state.
      action: The action to apply.

    Returns:
      The next state.

    Raises:
      RewardError: If the evolved reward function fails.
    """
    prev_obs = state.obs
    next_state = self.env.step(state, action)
    obs = next_state.obs
    base_reward = next_state.reward
    done = next_state.done

    try:
      res = self.reward_fn(
          self, prev_obs, action, obs, done, next_state.info, base_reward
      )
      if isinstance(res, tuple):
        reward, custom_done = res
      else:
        reward, custom_done = res, False
      if not self.ignore_base_termination:
        custom_done = jnp.logical_or(custom_done, done)
      next_state = next_state.replace(  # pyrefly: ignore[missing-attribute]
          reward=reward,
          done=jnp.where(
              custom_done, jnp.ones_like(done), jnp.zeros_like(done)
          ),
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      raise RewardError(
          f'Failed to execute brax evolved reward function: {e}\n'
      ) from e

    return _end_episode_if_non_finite(next_state)


def _end_episode_if_non_finite(state: brax_envs.State) -> brax_envs.State:
  """Ends the episode if the reward or observation of `state` isn't finite.

  NaN and infinite values in the reward, observation and metrics are replaced
  with zeros, so that they reach neither the learner, nor the observation
  normalizer, nor the episode metrics. Ending the episode makes Brax's
  `AutoResetWrapper` replace the (possibly diverged) physics state.

  Args:
    state: A state returned by the wrapped environment, with the evolved reward
      applied.

  Returns:
    `state` with NaN and infinite values replaced with zeros, done if its reward
    or observation had any.
  """
  is_finite = jnp.isfinite(state.reward)
  for obs_leaf in jax.tree.leaves(state.obs):
    is_finite &= jnp.all(jnp.isfinite(obs_leaf))
  return state.replace(  # pyrefly: ignore[missing-attribute]
      reward=_zero_non_finite_jax(state.reward),
      obs=jax.tree.map(_zero_non_finite_jax, state.obs),
      done=jnp.where(is_finite, state.done, 1.0),
      metrics={
          **jax.tree.map(_zero_non_finite_jax, state.metrics),
          NON_FINITE_TERMINATION_METRIC: jnp.logical_not(is_finite).astype(
              jnp.float32
          ),
      },
  )


def _zero_non_finite_jax(values: jax.typing.ArrayLike) -> jax.Array:
  """Returns `values` with NaN and infinite entries replaced with zeros."""
  return jnp.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _zero_non_finite_numpy(values: Any) -> Any:
  """Same as `_zero_non_finite_jax`, but on the host, with NumPy."""
  return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


class EvolvedRewardWrapper(dm_env.Environment, brax_env.BraxEnv):
  """Wraps an environment to use an evolved reward function.

  The reward function is defined as Python code that must implement
  `compute_reward(env, state, action, next_state, done, info, base_reward)`.

  A transition with a NaN or infinite reward or observation ends the episode
  with a zero reward. The check runs on the host, so it supports pure-Python
  environments (e.g. Tic-Tac-Toe) as well as JAX-based ones.

  With `ignore_base_termination`, only the evolved reward ends an episode: it
  ends when the evolved reward returns `done=True`, or at the caller's step
  limit, and the base environment's termination is ignored. Otherwise (the
  default) the episode also ends when the base environment terminates, which
  suits environments whose episodes can't continue past it (e.g. a finished
  Tic-Tac-Toe game). `to_brax_env` passes the setting on, so a policy trained on
  the Brax environment is evaluated under the same termination rule here.
  """

  reward_fn: Callable[..., Any]
  current_state: Any | None

  def __init__(
      self,
      env: dm_env.Environment,
      reward_fn_code: str,
      ignore_base_termination: bool = False,
  ):
    self.env = env
    self.reward_fn_code = reward_fn_code
    self.ignore_base_termination = ignore_base_termination

    # Compile the code and extract the compute_reward function
    local_vars = {}
    try:
      exec(reward_fn_code, globals(), local_vars)  # pylint: disable=exec-used
    except Exception as e:
      raise RewardCompilationError(
          f'Failed to compile reward code: {e}\nCode:\n{reward_fn_code}'
      ) from e

    if 'compute_reward' not in local_vars:
      raise RewardCompilationError(
          "Reward code must define 'compute_reward' function."
      )

    self.reward_fn = local_vars['compute_reward']
    self.current_state = None

  def __getattr__(self, name: str) -> Any:
    return getattr(self.env, name)

  def reset(self) -> dm_env.TimeStep:
    timestep = self.env.reset()
    self.current_state = timestep.observation
    return timestep

  def step(self, action: Any) -> dm_env.TimeStep:
    prev_state = self.current_state
    timestep = self.env.step(action)
    obs = timestep.observation
    self.current_state = obs

    done = timestep.last()
    base_reward = timestep.reward
    if base_reward is None:
      base_reward = 0.0

    info = {}
    custom_done = False
    try:
      # Pass the env, state transition, done flag, info dict, and base reward.
      res = self.reward_fn(
          self.env, prev_state, action, obs, done, info, base_reward
      )
      if isinstance(res, tuple):
        reward, custom_done = res
      else:
        reward = res
    # Catch all to prevent LLM-generated reward code from crashing the training.
    except Exception:  # pylint: disable=broad-exception-caught
      reward = base_reward
      logging.exception('Error in evolved reward function')

    if not numerics.all_finite((reward, obs)):
      logging.log_every_n_seconds(
          logging.WARNING,
          'Ending the episode: the evolved reward (%s) or the observation is'
          ' not finite.',
          _NON_FINITE_WARNING_INTERVAL_SECONDS,
          reward,
      )
      return dm_env.termination(
          reward=_NON_FINITE_REWARD,
          observation=jax.tree.map(_zero_non_finite_numpy, obs),
      )

    if self.ignore_base_termination:
      if custom_done:
        return dm_env.termination(float(reward), obs)
      # The episode continues, so the base termination's zero discount must not
      # leak through.
      return dm_env.transition(float(reward), obs)

    # Force termination if custom_done is True.
    # If the base env already terminated (done is True), we preserve that.
    if custom_done and not done:
      return dm_env.termination(float(reward), obs)
    else:
      return dm_env.TimeStep(
          step_type=timestep.step_type,
          reward=float(reward),
          discount=timestep.discount,
          observation=timestep.observation,
      )

  def action_spec(self) -> specs.Array:
    return self.env.action_spec()

  def observation_spec(self) -> specs.Array:
    return self.env.observation_spec()

  def close(self) -> None:
    self.env.close()

  def to_brax_env(self) -> brax_envs.Env:
    """Converts to an on-device Brax environment if the underlying env is a BraxEnv."""
    if isinstance(self.env, brax_env.BraxEnv):
      base_brax = self.env.to_brax_env()
      return BraxEvolvedRewardWrapper(
          base_brax, self.reward_fn, self.ignore_base_termination
      )
    raise NotImplementedError(
        f'Underlying environment {type(self.env).__name__} does not implement'
        ' BraxEnv.'
    )
