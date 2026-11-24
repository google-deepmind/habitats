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

"""Unit tests for reward_wrapper.py.

Tests cover:
1. EvolvedRewardWrapper with standard dm_env environments.
2. BraxEvolvedRewardWrapper with Brax environments (invoking reward function,
   modifying rewards, custom_done).
3. JAX JIT compilation (jax.jit, jax.vmap) with BraxEvolvedRewardWrapper.
4. Failure modes and diagnostics when LLM-generated reward functions are not
   JIT-compilable (tracer conversion errors, type errors, mutation errors).
5. Stress testing with diverse dummy reward functions simulating LLM outputs.
6. Ending episodes on NaN or infinite rewards and observations, in both
   wrappers.
"""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from brax import envs as brax_envs
import dm_env
from dm_env import specs
from habitats.environment import brax_env
from habitats.evolution import reward_wrapper
import jax
import jax.numpy as jnp
import numpy as np


class DummyDmEnv(dm_env.Environment):
  """A deterministic dm_env environment for testing EvolvedRewardWrapper."""

  def __init__(self):
    self._step_count = 0

  def reset(self) -> dm_env.TimeStep:
    self._step_count = 0
    return dm_env.restart(np.array([0.0, 0.0], dtype=np.float32))

  def step(self, action: int | float) -> dm_env.TimeStep:
    self._step_count += 1
    reward = 1.0  # base reward
    obs = np.array([float(self._step_count), float(action)], dtype=np.float32)
    if self._step_count >= 5:
      return dm_env.termination(reward, obs)
    return dm_env.transition(reward, obs)

  def observation_spec(self) -> specs.Array:
    return specs.Array(shape=(2,), dtype=np.float32, name="observation")

  def action_spec(self) -> specs.Array:
    return specs.Array(shape=(), dtype=np.float32, name="action")


class DummyBraxEnv(brax_envs.Env):
  """A minimal Brax environment for testing BraxEvolvedRewardWrapper."""

  def __init__(self, obs_dim: int = 4, action_dim: int = 2):
    self._obs_dim = obs_dim
    self._action_dim = action_dim

  def reset(self, rng: jax.Array) -> brax_envs.State:
    del rng
    obs = jnp.zeros(self._obs_dim, dtype=jnp.float32)
    reward = jnp.array(0.0, dtype=jnp.float32)
    done = jnp.array(0.0, dtype=jnp.float32)
    metrics = {"step_count": jnp.array(0, dtype=jnp.int32)}
    info = {"step": jnp.array(0, dtype=jnp.int32)}
    return brax_envs.State(
        pipeline_state=None,
        obs=obs,
        reward=reward,
        done=done,
        metrics=metrics,
        info=info,
    )

  def step(self, state: brax_envs.State, action: jax.Array) -> brax_envs.State:
    obs = state.obs + jnp.sum(action) * 0.1 + 0.5
    reward = jnp.array(1.0, dtype=jnp.float32)  # base reward
    done = jnp.array(0.0, dtype=jnp.float32)
    step = state.info.get("step", jnp.array(0)) + 1
    return state.replace(
        obs=obs,
        reward=reward,
        done=done,
        info={"step": step},
    )

  @property
  def observation_size(self) -> int:
    return self._obs_dim

  @property
  def action_size(self) -> int:
    return self._action_dim

  @property
  def backend(self) -> str:
    return "mjx"


class DummyBraxDmEnv(dm_env.Environment, brax_env.BraxEnv):
  """A dm_env environment that also implements brax_env.BraxEnv."""

  def __init__(self):
    self._brax_env = DummyBraxEnv()
    self._state = None

  def reset(self) -> dm_env.TimeStep:
    rng = jax.random.PRNGKey(0)
    self._state = self._brax_env.reset(rng)
    return dm_env.restart(np.array(self._state.obs))

  def step(self, action: int | float) -> dm_env.TimeStep:
    self._state = self._brax_env.step(self._state, jnp.asarray(action))
    return dm_env.transition(
        reward=float(self._state.reward),
        observation=np.array(self._state.obs),
    )

  def observation_spec(self) -> specs.Array:
    return specs.Array(shape=(4,), dtype=np.float32, name="observation")

  def action_spec(self) -> specs.Array:
    return specs.Array(shape=(2,), dtype=np.float32, name="action")

  def to_brax_env(self) -> brax_envs.Env:
    return self._brax_env


class AlwaysTerminatingBraxEnv(DummyBraxEnv):
  """A DummyBraxEnv whose base task reports termination on every step."""

  def step(self, state: brax_envs.State, action: jax.Array) -> brax_envs.State:
    next_state = super().step(state, action)
    return next_state.replace(done=jnp.array(1.0, dtype=jnp.float32))


class AlwaysTerminatingBraxDmEnv(DummyBraxDmEnv):
  """A dm_env adapter of AlwaysTerminatingBraxEnv, like MJXEnv for Brax."""

  def __init__(self):
    super().__init__()
    self._brax_env = AlwaysTerminatingBraxEnv()

  def step(self, action: int | float) -> dm_env.TimeStep:
    self._state = self._brax_env.step(self._state, jnp.asarray(action))
    reward = float(self._state.reward)
    obs = np.array(self._state.obs)
    if self._state.done:
      return dm_env.termination(reward, obs)
    return dm_env.transition(reward, obs)


class NanObservationDmEnv(DummyDmEnv):
  """A DummyDmEnv whose steps return NaN observations."""

  def step(self, action: int | float) -> dm_env.TimeStep:
    timestep = super().step(action)
    return timestep._replace(
        observation=np.full_like(timestep.observation, np.nan)
    )


class DiscreteDmEnv(dm_env.Environment):
  """A pure-Python environment with integer observations, like Tic-Tac-Toe."""

  def reset(self) -> dm_env.TimeStep:
    return dm_env.restart(0)

  def step(self, action: int) -> dm_env.TimeStep:
    return dm_env.transition(reward=0.0, observation=action)

  def observation_spec(self) -> specs.DiscreteArray:
    return specs.DiscreteArray(num_values=9, name="observation")

  def action_spec(self) -> specs.DiscreteArray:
    return specs.DiscreteArray(num_values=9, name="action")


class NanObservationBraxEnv(DummyBraxEnv):
  """A DummyBraxEnv whose steps return NaN observations."""

  def step(self, state: brax_envs.State, action: jax.Array) -> brax_envs.State:
    next_state = super().step(state, action)
    return next_state.replace(obs=jnp.full_like(next_state.obs, jnp.nan))


def _non_finite_terminations(state: brax_envs.State) -> np.ndarray:
  return np.asarray(
      state.metrics[reward_wrapper.NON_FINITE_TERMINATION_METRIC]
  )


class EvolvedRewardWrapperTest(absltest.TestCase):
  """Tests for standard EvolvedRewardWrapper wrapping dm_env."""

  def test_custom_reward_basic(self):
    """Verifies adding a custom reward function that scales and shifts base reward."""
    base_env = DummyDmEnv()
    reward_code = """
def compute_reward(env, state, action, next_state, done, info, base_reward) -> float:
  return base_reward * 2.0 + 3.0
"""
    wrapped = reward_wrapper.EvolvedRewardWrapper(base_env, reward_code)
    wrapped.reset()
    ts = wrapped.step(0)
    # base_reward is 1.0; custom is 1.0 * 2.0 + 3.0 = 5.0
    self.assertEqual(ts.reward, 5.0)
    self.assertFalse(ts.last())

  def test_custom_reward_state_and_action_dependent(self):
    """Verifies reward function receives state transition, action, and done."""
    base_env = DummyDmEnv()
    reward_code = """
def compute_reward(env, state, action, next_state, done, info, base_reward) -> float:
  # state is previous obs, next_state is current obs
  delta_x = next_state[0] - state[0]
  return float(delta_x + action)
"""
    wrapped = reward_wrapper.EvolvedRewardWrapper(base_env, reward_code)
    wrapped.reset()  # state is [0.0, 0.0]
    ts = wrapped.step(2.5)  # next_state is [1.0, 2.5], delta_x is 1.0
    self.assertAlmostEqual(ts.reward, 1.0 + 2.5)

  def test_custom_reward_tuple_return_triggers_early_termination(self):
    """Verifies returning (reward, True) terminates the episode early."""
    base_env = DummyDmEnv()
    reward_code = """
def compute_reward(env, state, action, next_state, done, info, base_reward):
  # Terminate on first step with reward 10.0
  return 10.0, True
"""
    wrapped = reward_wrapper.EvolvedRewardWrapper(base_env, reward_code)
    wrapped.reset()
    ts = wrapped.step(0)
    self.assertEqual(ts.reward, 10.0)
    self.assertTrue(ts.last())
    self.assertEqual(ts.discount, 0.0)
    self.assertEqual(ts.step_type, dm_env.StepType.LAST)

  def test_custom_reward_tuple_return_preserves_not_done(self):
    """Verifies returning (reward, False) preserves ongoing episode."""
    base_env = DummyDmEnv()
    reward_code = """
def compute_reward(env, state, action, next_state, done, info, base_reward):
  return 42.0, False
"""
    wrapped = reward_wrapper.EvolvedRewardWrapper(base_env, reward_code)
    wrapped.reset()
    ts = wrapped.step(0)
    self.assertEqual(ts.reward, 42.0)
    self.assertFalse(ts.last())
    self.assertEqual(ts.discount, 1.0)
    self.assertEqual(ts.step_type, dm_env.StepType.MID)

  def test_custom_reward_preserves_base_env_termination(self):
    """Verifies if base environment terminates, custom_done=False does not undo it."""
    base_env = DummyDmEnv()
    reward_code = """
def compute_reward(env, state, action, next_state, done, info, base_reward):
  return base_reward, False
"""
    wrapped = reward_wrapper.EvolvedRewardWrapper(base_env, reward_code)
    wrapped.reset()
    # Step 5 times to reach DummyDmEnv termination
    for _ in range(4):
      wrapped.step(0)
    ts = wrapped.step(0)
    self.assertTrue(ts.last())

  def test_ignores_base_termination_when_set(self):
    """Verifies an episode only ends when the evolved reward says so."""
    reward_code = """
def compute_reward(env, state, action, next_state, done, info, base_reward):
  return base_reward
"""
    wrapped = reward_wrapper.EvolvedRewardWrapper(
        AlwaysTerminatingBraxDmEnv(), reward_code, ignore_base_termination=True
    )
    wrapped.reset()
    for _ in range(3):
      ts = wrapped.step(np.zeros(2, dtype=np.float32))
      self.assertFalse(ts.last())
      self.assertEqual(ts.step_type, dm_env.StepType.MID)
      self.assertEqual(ts.discount, 1.0)
      self.assertEqual(ts.reward, 1.0)

  def test_ignoring_base_termination_still_ends_on_custom_done(self):
    """Verifies an episode ends when the evolved reward returns done."""
    reward_code = """
def compute_reward(env, state, action, next_state, done, info, base_reward):
  return 2.0, next_state[0] > 1.2
"""
    wrapped = reward_wrapper.EvolvedRewardWrapper(
        AlwaysTerminatingBraxDmEnv(), reward_code, ignore_base_termination=True
    )
    wrapped.reset()
    action = np.zeros(2, dtype=np.float32)
    # Each step adds 0.5 to the observation: 0.5, 1.0, then 1.5 > 1.2.
    self.assertFalse(wrapped.step(action).last())
    self.assertFalse(wrapped.step(action).last())
    ts = wrapped.step(action)
    self.assertTrue(ts.last())
    self.assertEqual(ts.discount, 0.0)
    self.assertEqual(ts.reward, 2.0)

  def test_ends_episodes_at_same_step_as_brax_wrapper(self):
    """Verifies evaluation and Brax training end episodes at the same step."""
    reward_code = """
def compute_reward(env, state, action, next_state, done, info, base_reward):
  return base_reward, next_state[0] > 1.2
"""
    # AlwaysTerminatingBraxDmEnv's base task terminates on every step, and the
    # evolved reward ends the episode from the third step on.
    for ignore_base_termination, expected_dones in (
        (True, [False, False, True, True]),
        (False, [True, True, True, True]),
    ):
      with self.subTest(ignore_base_termination=ignore_base_termination):
        dm_wrapped = reward_wrapper.EvolvedRewardWrapper(
            AlwaysTerminatingBraxDmEnv(), reward_code, ignore_base_termination
        )
        brax_wrapped = dm_wrapped.to_brax_env()
        action = np.zeros(2, dtype=np.float32)

        dm_wrapped.reset()
        dm_dones = [dm_wrapped.step(action).last() for _ in range(4)]
        brax_state = brax_wrapped.reset(jax.random.PRNGKey(0))
        brax_dones = []
        for _ in range(4):
          brax_state = brax_wrapped.step(brax_state, jnp.asarray(action))
          brax_dones.append(bool(brax_state.done))

        self.assertEqual(dm_dones, expected_dones)
        self.assertEqual(dm_dones, brax_dones)

  def test_invalid_code_syntax_error(self):
    """Verifies RewardCompilationError is raised for invalid Python syntax."""
    base_env = DummyDmEnv()
    broken_code = """
def compute_reward(env, state, action, next_state, done, info, base_reward):
  return base_reward +
"""
    with self.assertRaises(reward_wrapper.RewardCompilationError):
      reward_wrapper.EvolvedRewardWrapper(base_env, broken_code)

  def test_invalid_code_missing_compute_reward(self):
    """Verifies RewardCompilationError when compute_reward is missing."""
    base_env = DummyDmEnv()
    code_without_fn = """
def other_function():
  return 1.0
"""
    with self.assertRaises(reward_wrapper.RewardCompilationError) as ctx:
      reward_wrapper.EvolvedRewardWrapper(base_env, code_without_fn)
    self.assertIn("compute_reward", str(ctx.exception))

  def test_runtime_exception_falls_back_to_base_reward(self):
    """Verifies runtime errors in LLM reward code are caught and fallback to base reward."""
    base_env = DummyDmEnv()
    buggy_code = """
def compute_reward(env, state, action, next_state, done, info, base_reward):
  # Divide by zero to trigger runtime exception
  return 1.0 / 0.0
"""
    wrapped = reward_wrapper.EvolvedRewardWrapper(base_env, buggy_code)
    wrapped.reset()
    # Must not raise an exception; should log and fall back to base_reward (1.0)
    ts = wrapped.step(0)
    self.assertEqual(ts.reward, 1.0)

  def test_spec_and_attribute_delegation(self):
    """Verifies action/observation specs and attributes are properly delegated."""
    base_env = DummyDmEnv()
    reward_code = """
def compute_reward(env, state, action, next_state, done, info, base_reward):
  return base_reward
"""
    wrapped = reward_wrapper.EvolvedRewardWrapper(base_env, reward_code)
    self.assertEqual(wrapped.action_spec().shape, base_env.action_spec().shape)
    self.assertEqual(
        wrapped.observation_spec().shape, base_env.observation_spec().shape
    )

  def test_to_brax_env_success(self):
    """Verifies to_brax_env converts underlying BraxEnv to BraxEvolvedRewardWrapper."""
    brax_dm = DummyBraxDmEnv()
    reward_code = """
def compute_reward(env, state, action, next_state, done, info, base_reward):
  return base_reward + 1.0
"""
    wrapped = reward_wrapper.EvolvedRewardWrapper(brax_dm, reward_code)
    brax_wrapped = wrapped.to_brax_env()
    self.assertIsInstance(brax_wrapped, reward_wrapper.BraxEvolvedRewardWrapper)

  def test_to_brax_env_failure_for_non_brax_env(self):
    """Verifies to_brax_env raises NotImplementedError if env does not implement BraxEnv."""
    non_brax = DummyDmEnv()
    reward_code = """
def compute_reward(env, state, action, next_state, done, info, base_reward):
  return base_reward
"""
    wrapped = reward_wrapper.EvolvedRewardWrapper(non_brax, reward_code)
    with self.assertRaises(NotImplementedError):
      wrapped.to_brax_env()


class BraxEvolvedRewardWrapperTest(absltest.TestCase):
  """Tests for BraxEvolvedRewardWrapper interacting with Brax environment."""

  def setUp(self):
    super().setUp()
    self.base_env = DummyBraxEnv()

  def test_brax_wrapper_step_invokes_reward_fn(self):
    """Verifies Brax actually calls the custom reward function during step."""
    mock_reward_fn = mock.MagicMock(return_value=jnp.array(5.0))
    wrapped = reward_wrapper.BraxEvolvedRewardWrapper(
        self.base_env, mock_reward_fn
    )

    rng = jax.random.PRNGKey(0)
    state = wrapped.reset(rng)
    action = jnp.zeros(2, dtype=jnp.float32)

    next_state = wrapped.step(state, action)

    # Verify reward_fn was invoked exactly once with expected arguments
    mock_reward_fn.assert_called_once()
    args, _ = mock_reward_fn.call_args
    # signature: (env, prev_obs, action, obs, done, info, base_reward)
    self.assertLen(args, 7)
    self.assertIs(args[0], wrapped)
    np.testing.assert_allclose(args[1], state.obs)  # prev_obs
    np.testing.assert_allclose(args[2], action)  # action
    self.assertEqual(float(next_state.reward), 5.0)

  def test_brax_wrapper_scalar_reward_replacement(self):
    """Verifies replacing base reward with a computed JAX reward."""

    def custom_reward_fn(env, prev_obs, action, obs, done, info, base_reward):
      del env, prev_obs, action, done, info
      return base_reward + jnp.sum(obs) * 2.0

    wrapped = reward_wrapper.BraxEvolvedRewardWrapper(
        self.base_env, custom_reward_fn
    )
    rng = jax.random.PRNGKey(0)
    state = wrapped.reset(rng)
    action = jnp.ones(2, dtype=jnp.float32)

    next_state = wrapped.step(state, action)

    # In DummyBraxEnv: obs = state.obs + sum(action)*0.1 + 0.5
    # = 0 + 0.2 + 0.5 = 0.7 for 4 dims.
    # base_reward is 1.0
    # expected = 1.0 + (0.7 * 4) * 2.0 = 1.0 + 5.6 = 6.6
    expected_reward = 1.0 + float(jnp.sum(next_state.obs)) * 2.0
    self.assertAlmostEqual(float(next_state.reward), expected_reward, places=5)

  def test_brax_wrapper_tuple_return_custom_done(self):
    """Verifies (reward, custom_done) updates both reward and done in Brax State."""

    def custom_reward_fn(env, prev_obs, action, obs, done, info, base_reward):
      del env, prev_obs, action, obs, done, info
      return base_reward + 3.0, jnp.array(True)

    wrapped = reward_wrapper.BraxEvolvedRewardWrapper(
        self.base_env, custom_reward_fn
    )
    rng = jax.random.PRNGKey(0)
    state = wrapped.reset(rng)
    action = jnp.zeros(2, dtype=jnp.float32)

    next_state = wrapped.step(state, action)

    self.assertEqual(float(next_state.reward), 4.0)
    self.assertEqual(float(next_state.done), 1.0)

  def test_brax_wrapper_tuple_return_not_done(self):
    """Verifies custom_done=False keeps done at 0.0 when base env is not done."""

    def custom_reward_fn(env, prev_obs, action, obs, done, info, base_reward):
      del env, prev_obs, action, obs, done, info
      return base_reward + 3.0, jnp.array(False)

    wrapped = reward_wrapper.BraxEvolvedRewardWrapper(
        self.base_env, custom_reward_fn
    )
    rng = jax.random.PRNGKey(0)
    state = wrapped.reset(rng)
    action = jnp.zeros(2, dtype=jnp.float32)

    next_state = wrapped.step(state, action)

    self.assertEqual(float(next_state.reward), 4.0)
    self.assertEqual(float(next_state.done), 0.0)

  def test_brax_wrapper_overrides_base_env_termination_when_custom_done_false(
      self,
  ):
    """Verifies custom_done=False suppresses base environment termination."""

    class TerminatingBraxEnv(DummyBraxEnv):

      def step(self, state, action):
        next_s = super().step(state, action)
        return next_s.replace(done=jnp.array(1.0, dtype=jnp.float32))

    base_env = TerminatingBraxEnv()

    def custom_reward_fn(env, prev_obs, action, obs, done, info, base_reward):
      del env, prev_obs, action, obs, done, info, base_reward
      return 10.0, jnp.array(False)

    wrapped = reward_wrapper.BraxEvolvedRewardWrapper(
        base_env, custom_reward_fn
    )
    state = wrapped.reset(jax.random.PRNGKey(0))
    next_state = wrapped.step(state, jnp.zeros(2, dtype=jnp.float32))

    self.assertEqual(float(next_state.reward), 10.0)
    self.assertEqual(float(next_state.done), 0.0)

  def test_brax_wrapper_unjitted_runtime_exception_raises_reward_error(self):
    """Verifies unjitted errors in reward_fn are raised as RewardError."""

    def broken_reward_fn(env, prev_obs, action, obs, done, info, base_reward):
      raise ValueError("Simulated failure in reward calculation")

    wrapped = reward_wrapper.BraxEvolvedRewardWrapper(
        self.base_env, broken_reward_fn
    )
    rng = jax.random.PRNGKey(0)
    state = wrapped.reset(rng)
    action = jnp.zeros(2, dtype=jnp.float32)

    with self.assertRaisesRegex(
        reward_wrapper.RewardError, "Simulated failure in reward calculation"
    ):
      wrapped.step(state, action)


class BraxEvolvedRewardWrapperJitTest(absltest.TestCase):
  """Tests verifying BraxEvolvedRewardWrapper under JAX JIT compilation."""

  def setUp(self):
    super().setUp()
    self.base_env = DummyBraxEnv()

  def test_jax_jit_compilation_with_jax_reward_function(self):
    """Verifies that a JAX-compatible reward function successfully JIT-compiles."""

    def jax_reward_fn(env, prev_obs, action, obs, done, info, base_reward):
      del env, done, info
      # Pure JAX operations: dot products, sums, arithmetic
      movement = jnp.sum(obs - prev_obs)
      action_cost = 0.05 * jnp.sum(jnp.square(action))
      return base_reward + movement - action_cost

    wrapped = reward_wrapper.BraxEvolvedRewardWrapper(
        self.base_env, jax_reward_fn
    )

    # JIT compile the step function as Brax RL algorithms do
    jitted_step = jax.jit(wrapped.step)

    rng = jax.random.PRNGKey(42)
    state = wrapped.reset(rng)
    action = jnp.array([0.5, -0.5], dtype=jnp.float32)

    # First call: triggers JIT tracing and compilation
    next_state = jitted_step(state, action)

    # Expected:
    # prev_obs = [0, 0, 0, 0]
    # action sum = 0.0 -> obs = [0.5, 0.5, 0.5, 0.5]
    # movement = sum([0.5, 0.5, 0.5, 0.5]) = 2.0
    # action_cost = 0.05 * (0.25 + 0.25) = 0.025
    # base_reward = 1.0
    # expected = 1.0 + 2.0 - 0.025 = 2.975
    expected_reward = 1.0 + 2.0 - 0.025
    self.assertAlmostEqual(float(next_state.reward), expected_reward, places=4)

    # Second call: runs the compiled binary
    next_next_state = jitted_step(next_state, action)
    self.assertIsInstance(next_next_state.reward, jax.Array)

  def test_jax_jit_compilation_with_custom_done(self):
    """Verifies that JAX JIT handles custom boolean condition for termination."""

    def jax_reward_fn_with_done(
        env, prev_obs, action, obs, done, info, base_reward
    ):
      del env, prev_obs, action, done, info
      # Early termination condition using JAX boolean array
      custom_done = obs[0] > 1.0
      return base_reward + 2.0, custom_done

    wrapped = reward_wrapper.BraxEvolvedRewardWrapper(
        self.base_env, jax_reward_fn_with_done
    )
    jitted_step = jax.jit(wrapped.step)

    rng = jax.random.PRNGKey(0)
    state = wrapped.reset(rng)
    action = jnp.zeros(2, dtype=jnp.float32)

    # Step 1: obs becomes 0.5 -> obs[0] <= 1.0 -> done = 0.0
    state1 = jitted_step(state, action)
    self.assertEqual(float(state1.done), 0.0)
    self.assertEqual(float(state1.reward), 3.0)

    # Step 2: obs becomes 0.5 + 0.5 = 1.0 -> obs[0] <= 1.0 -> done = 0.0
    state2 = jitted_step(state1, action)
    self.assertEqual(float(state2.done), 0.0)

    # Step 3: obs becomes 1.0 + 0.5 = 1.5 -> obs[0] > 1.0 -> done = 1.0
    state3 = jitted_step(state2, action)
    self.assertEqual(float(state3.done), 1.0)
    self.assertEqual(float(state3.reward), 3.0)

  def test_jax_vmap_vectorized_execution(self):
    """Verifies that BraxEvolvedRewardWrapper can be vectorized with jax.vmap."""

    def custom_reward_fn(env, prev_obs, action, obs, done, info, base_reward):
      del env, prev_obs, done, info
      return base_reward + jnp.sum(obs) + jnp.sum(action)

    wrapped = reward_wrapper.BraxEvolvedRewardWrapper(
        self.base_env, custom_reward_fn
    )

    batch_size = 4
    vmapped_step = jax.jit(jax.vmap(wrapped.step))

    # Reset a batch of states
    rngs = jax.random.split(jax.random.PRNGKey(0), batch_size)
    states = jax.vmap(wrapped.reset)(rngs)
    actions = jnp.ones((batch_size, 2), dtype=jnp.float32)

    next_states = vmapped_step(states, actions)

    self.assertEqual(next_states.reward.shape, (batch_size,))
    self.assertEqual(next_states.done.shape, (batch_size,))


class FailureModesAndDiagnosticsTest(absltest.TestCase):
  """Tests documenting what happens when LLM-generated reward functions fail JIT compilation.

  These tests show:
  1. What errors occur when non-JIT compilable patterns are used.
  2. That BraxEvolvedRewardWrapper reports tracing errors as `RewardError`,
  which fails the training of the agent rather than the whole evolution run.
  """

  def setUp(self):
    super().setUp()
    self.base_env = DummyBraxEnv()

  def test_python_control_flow_fails_during_jit_tracing(self):
    """LLM writes `if obs[0] > 0:` -> raises TracerBoolConversionError.

    What it tells us:
      Directly calling this function under JAX tracing raises
      `jax.errors.TracerBoolConversionError`. The error message specifically
      instructs using `jnp.where` instead of Python `if`.
    """

    def bad_reward_fn_with_python_if(
        env, prev_obs, action, obs, done, info, base_reward
    ):
      del env, prev_obs, action, done, info, base_reward
      # Direct Python boolean evaluation of a JAX tracer raises
      # TracerBoolConversionError
      if obs[0] > 0.0:
        return 1.0
      return 0.0

    # Verify the exact error JAX produces when an LLM uses Python if on arrays
    with self.assertRaises(jax.errors.TracerBoolConversionError) as ctx:
      jax.jit(bad_reward_fn_with_python_if)(
          None, None, None, jnp.array([1.0]), None, None, 0.0
      )
    self.assertIn("TracerBoolConversionError", type(ctx.exception).__name__)

  def test_brax_wrapper_raises_reward_error_on_tracer_conversion_error(self):
    """Verifies a TracerBoolConversionError during JIT becomes a RewardError.

    Because `BraxEvolvedRewardWrapper.step` wraps `reward_fn` in `try ... except
    Exception:`, the `TracerBoolConversionError` raised while JAX traces `step`
    is re-raised as a `RewardError`.
    """

    def bad_reward_fn(env, prev_obs, action, obs, done, info, base_reward):
      del env, prev_obs, action, done, info, base_reward
      if obs[0] > 0.0:  # Triggers TracerBoolConversionError during JAX tracing
        return 10.0
      return 0.0

    wrapped = reward_wrapper.BraxEvolvedRewardWrapper(
        self.base_env, bad_reward_fn
    )

    jitted_step = jax.jit(wrapped.step)
    rng = jax.random.PRNGKey(0)
    state = wrapped.reset(rng)
    action = jnp.zeros(2, dtype=jnp.float32)

    with self.assertRaisesRegex(
        reward_wrapper.RewardError,
        "Failed to execute brax evolved reward function",
    ) as ctx:
      jitted_step(state, action)
    self.assertIsInstance(
        ctx.exception.__cause__, jax.errors.TracerBoolConversionError
    )

  def test_inplace_array_mutation_raises_reward_error(self):
    """LLM attempts in-place assignment (e.g.

    `obs[0] = 1.0`) -> raised as a RewardError.
    """

    def bad_mutation_fn(env, prev_obs, action, obs, done, info, base_reward):
      del env, prev_obs, action, done, info, base_reward
      obs[0] = 1.0  # JAX arrays are immutable! Raises TypeError
      return 10.0

    wrapped = reward_wrapper.BraxEvolvedRewardWrapper(
        self.base_env, bad_mutation_fn
    )
    jitted_step = jax.jit(wrapped.step)

    rng = jax.random.PRNGKey(0)
    state = wrapped.reset(rng)
    action = jnp.zeros(2, dtype=jnp.float32)

    with self.assertRaisesRegex(
        reward_wrapper.RewardError,
        "Failed to execute brax evolved reward function",
    ) as ctx:
      jitted_step(state, action)
    self.assertIsInstance(ctx.exception.__cause__, TypeError)

  def test_invalid_return_type_string_raises_jit_type_error(self):
    """LLM returns a non-array type (e.g.

    str) -> JAX JIT raises TypeError on PyTree compilation.
    """

    def string_reward_fn(env, prev_obs, action, obs, done, info, base_reward):
      del env, prev_obs, action, obs, done, info, base_reward
      return "invalid_string_reward"

    wrapped = reward_wrapper.BraxEvolvedRewardWrapper(
        self.base_env, string_reward_fn
    )

    # Under JIT, JAX cannot compile a string into a State PyTree leaf
    with self.assertRaises(TypeError):
      jitted_step = jax.jit(wrapped.step)
      rng = jax.random.PRNGKey(0)
      state = wrapped.reset(rng)
      jitted_step(state, jnp.zeros(2, dtype=jnp.float32))


class EvolvedRewardWrapperNonFiniteTest(parameterized.TestCase):
  """Tests that EvolvedRewardWrapper ends episodes on NaN or infinite values."""

  @parameterized.named_parameters(
      ("nan", "jnp.sqrt(-1.0)"),
      ("positive_infinity", "jnp.float32(1.0) / 0.0"),
      ("negative_infinity", "jnp.log(0.0)"),
  )
  def test_non_finite_reward_ends_episode_with_zero_reward(self, expression):
    reward_code = f"""
def compute_reward(env, state, action, next_state, done, info, base_reward):
  return {expression}
"""
    wrapped = reward_wrapper.EvolvedRewardWrapper(DummyDmEnv(), reward_code)
    wrapped.reset()

    ts = wrapped.step(0)

    self.assertTrue(ts.last())
    self.assertEqual(ts.reward, 0.0)
    self.assertEqual(ts.discount, 0.0)
    np.testing.assert_array_equal(ts.observation, [1.0, 0.0])

  def test_non_finite_observation_ends_episode_with_finite_values(self):
    reward_code = """
def compute_reward(env, state, action, next_state, done, info, base_reward):
  return next_state[0]
"""
    wrapped = reward_wrapper.EvolvedRewardWrapper(
        NanObservationDmEnv(), reward_code
    )
    wrapped.reset()

    ts = wrapped.step(0)

    self.assertTrue(ts.last())
    self.assertEqual(ts.reward, 0.0)
    np.testing.assert_array_equal(ts.observation, [0.0, 0.0])

  def test_pure_python_environment(self):
    reward_code = """
def compute_reward(env, state, action, next_state, done, info, base_reward):
  return float("nan") if next_state == 4 else 1.0
"""
    wrapped = reward_wrapper.EvolvedRewardWrapper(DiscreteDmEnv(), reward_code)
    wrapped.reset()

    finite_ts = wrapped.step(3)
    non_finite_ts = wrapped.step(4)

    self.assertFalse(finite_ts.last())
    self.assertEqual(finite_ts.reward, 1.0)
    self.assertTrue(non_finite_ts.last())
    self.assertEqual(non_finite_ts.reward, 0.0)
    self.assertEqual(non_finite_ts.observation, 4)


class BraxEvolvedRewardWrapperNonFiniteTest(parameterized.TestCase):
  """Tests that BraxEvolvedRewardWrapper ends episodes on NaN or infinities."""

  @parameterized.named_parameters(
      ("nan", jnp.nan),
      ("positive_infinity", jnp.inf),
      ("negative_infinity", -jnp.inf),
  )
  def test_non_finite_reward_ends_episode_with_zero_reward(self, bad_reward):
    def reward_fn(env, prev_obs, action, obs, done, info, base_reward):
      del env, prev_obs, action, obs, done, info, base_reward
      return jnp.float32(bad_reward), jnp.array(False)

    wrapped = reward_wrapper.BraxEvolvedRewardWrapper(DummyBraxEnv(), reward_fn)
    state = wrapped.reset(jax.random.PRNGKey(0))

    next_state = jax.jit(wrapped.step)(state, jnp.zeros(2))

    self.assertEqual(float(next_state.reward), 0.0)
    self.assertEqual(float(next_state.done), 1.0)
    self.assertEqual(_non_finite_terminations(next_state), 1.0)

  def test_non_finite_observation_ends_episode_with_finite_values(self):
    def reward_fn(env, prev_obs, action, obs, done, info, base_reward):
      del env, prev_obs, action, done, info, base_reward
      return jnp.sum(obs)

    wrapped = reward_wrapper.BraxEvolvedRewardWrapper(
        NanObservationBraxEnv(), reward_fn
    )
    state = wrapped.reset(jax.random.PRNGKey(0))

    next_state = jax.jit(wrapped.step)(state, jnp.zeros(2))

    np.testing.assert_array_equal(next_state.obs, np.zeros(4))
    self.assertEqual(float(next_state.reward), 0.0)
    self.assertEqual(float(next_state.done), 1.0)
    self.assertEqual(_non_finite_terminations(next_state), 1.0)

  def test_finite_transition_is_unchanged(self):
    def reward_fn(env, prev_obs, action, obs, done, info, base_reward):
      del env, prev_obs, action, obs, done, info
      return base_reward + 2.0

    wrapped = reward_wrapper.BraxEvolvedRewardWrapper(DummyBraxEnv(), reward_fn)
    state = wrapped.reset(jax.random.PRNGKey(0))

    next_state = jax.jit(wrapped.step)(state, jnp.zeros(2))

    np.testing.assert_allclose(next_state.obs, np.full(4, 0.5))
    self.assertEqual(float(next_state.reward), 3.0)
    self.assertEqual(float(next_state.done), 0.0)
    self.assertEqual(_non_finite_terminations(next_state), 0.0)

  def test_only_ends_episodes_with_non_finite_values_in_batch(self):
    def reward_fn(env, prev_obs, action, obs, done, info, base_reward):
      del env, prev_obs, action, done, info, base_reward
      return jnp.where(obs[0] > 0.6, jnp.nan, 1.0)

    wrapped = reward_wrapper.BraxEvolvedRewardWrapper(DummyBraxEnv(), reward_fn)
    states = jax.vmap(wrapped.reset)(jax.random.split(jax.random.PRNGKey(0)))
    # The first environment's observation becomes 0.7, the second's 0.5.
    actions = jnp.array([[1.0, 1.0], [0.0, 0.0]])

    next_states = jax.jit(jax.vmap(wrapped.step))(states, actions)

    np.testing.assert_array_equal(next_states.reward, [0.0, 1.0])
    np.testing.assert_array_equal(next_states.done, [1.0, 0.0])
    np.testing.assert_array_equal(
        _non_finite_terminations(next_states), [1.0, 0.0]
    )

  def test_reset_and_step_states_can_be_scanned(self):
    # Brax carries states through `jax.lax.scan`, which requires `reset` and
    # `step` to return states of the same structure and dtypes.
    def reward_fn(env, prev_obs, action, obs, done, info, base_reward):
      del env, prev_obs, action, obs, done, info
      return base_reward, jnp.array(False)

    wrapped = reward_wrapper.BraxEvolvedRewardWrapper(DummyBraxEnv(), reward_fn)

    def scan_step(state, _):
      return wrapped.step(state, jnp.zeros(2)), None

    final_state, _ = jax.lax.scan(
        scan_step, wrapped.reset(jax.random.PRNGKey(0)), None, length=3
    )

    self.assertEqual(_non_finite_terminations(final_state), 0.0)


class StressTestDummyRewardFunctionsTest(parameterized.TestCase):
  """Stress tests end-to-end flow with realistic dummy LLM-generated reward functions.

  Tests the entire path:
  reward_code string -> EvolvedRewardWrapper -> to_brax_env() ->
  BraxEvolvedRewardWrapper -> jax.jit(step)
  """

  @parameterized.named_parameters(
      (
          "potential_distance",
          """
import jax.numpy as jnp

def compute_reward(env, state, action, next_state, done, info, base_reward):
  # Potential-based shaping: negative norm of next observation
  dist = jnp.linalg.norm(next_state)
  return base_reward - 0.1 * dist
""",
      ),
      (
          "target_tracking_exponential",
          """
import jax.numpy as jnp

def compute_reward(env, state, action, next_state, done, info, base_reward):
  # Gaussian-style proximity to a target
  target = jnp.array([0.5, 0.5, 0.5, 0.5])
  error = jnp.sum(jnp.square(next_state - target))
  return base_reward + jnp.exp(-error)
""",
      ),
      (
          "control_effort_clipped",
          """
import jax.numpy as jnp

def compute_reward(env, state, action, next_state, done, info, base_reward):
  # Action magnitude penalty with clipping
  clipped_act = jnp.clip(action, -1.0, 1.0)
  action_penalty = 0.05 * jnp.sum(jnp.square(clipped_act))
  return base_reward - action_penalty
""",
      ),
      (
          "conditional_using_jnp_where",
          """
import jax.numpy as jnp

def compute_reward(env, state, action, next_state, done, info, base_reward):
  # The CORRECT way to write conditionals for JAX JIT
  is_positive = next_state[0] > 0.0
  bonus = jnp.where(is_positive, 2.0, -1.0)
  return base_reward + bonus
""",
      ),
      (
          "progress_velocity",
          """
import jax.numpy as jnp

def compute_reward(env, state, action, next_state, done, info, base_reward):
  # Velocity / progress reward
  dt = 0.02
  velocity = (next_state[0] - state[0]) / dt
  return base_reward + 0.1 * velocity
""",
      ),
      (
          "trigonometric_orientation",
          """
import jax.numpy as jnp

def compute_reward(env, state, action, next_state, done, info, base_reward):
  # Trigonometric stability reward
  upright = jnp.cos(next_state[0])
  spin = jnp.sin(next_state[1])
  return base_reward + upright - 0.2 * jnp.abs(spin)
""",
      ),
      (
          "tuple_custom_done",
          """
import jax.numpy as jnp

def compute_reward(env, state, action, next_state, done, info, base_reward):
  # Reward with dynamic termination threshold
  unstable = jnp.abs(next_state[0]) > 2.0
  return base_reward + 1.0, unstable
""",
      ),
  )
  def test_llm_dummy_reward_functions_jit_compile_and_step(
      self, reward_code: str
  ):
    """Stress tests that each LLM dummy reward compiles and runs under jax.jit."""
    # 1. Wrap dm_env with EvolvedRewardWrapper
    dm_brax_env = DummyBraxDmEnv()
    wrapper = reward_wrapper.EvolvedRewardWrapper(dm_brax_env, reward_code)

    # 2. Convert to Brax environment
    brax_wrapped = wrapper.to_brax_env()
    self.assertIsInstance(brax_wrapped, reward_wrapper.BraxEvolvedRewardWrapper)

    # 3. JIT compile the step function
    jitted_step = jax.jit(brax_wrapped.step)

    # 4. Reset environment
    rng = jax.random.PRNGKey(123)
    state = brax_wrapped.reset(rng)
    action = jnp.array([0.1, -0.2], dtype=jnp.float32)

    # 5. Run multiple steps to stress test compilation and steady execution
    for _ in range(5):
      state = jitted_step(state, action)
      self.assertTrue(jnp.isfinite(state.reward).all())
      self.assertTrue(jnp.isfinite(state.done).all())
      self.assertTrue(jnp.isfinite(state.obs).all())


if __name__ == "__main__":
  absltest.main()
