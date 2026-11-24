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

"""Neural network policy."""

import os
from typing import Any, cast, override

import flax.linen as nn
from habitats.policy import policy
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

State = dict[str, jnp.ndarray] | jnp.ndarray


def _get_activation(activation: str) -> Any:
  if activation == "swish":
    return jax.nn.swish
  elif activation == "tanh":
    return jnp.tanh
  elif activation == "relu":
    return jax.nn.relu
  raise ValueError(f"Unsupported activation: {activation}")


class MLPActor(nn.Module):
  """MLP Actor module that predicts action distribution logits."""

  num_dimensions: int
  hidden_sizes: tuple[int, ...] = (64, 64)
  activation: str = "swish"

  @nn.compact
  def __call__(self, obs: jax.Array) -> jax.Array:
    h = obs
    act_fn = _get_activation(self.activation)
    for size in self.hidden_sizes:
      h = nn.Dense(size)(h)
      h = act_fn(h)

    # Returns [loc, scale_raw] of shape [*batch, 2 * num_dimensions]
    return nn.Dense(2 * self.num_dimensions, name="logits")(h)


class MLPCritic(nn.Module):
  """MLP Critic module that predicts state value."""

  hidden_sizes: tuple[int, ...] = (64, 64)
  activation: str = "swish"

  @nn.compact
  def __call__(self, obs: jax.Array) -> jax.Array:
    h = obs
    act_fn = _get_activation(self.activation)
    for size in self.hidden_sizes:
      h = nn.Dense(size)(h)
      h = act_fn(h)

    value = nn.Dense(1, name="value")(h)
    return jnp.squeeze(value, axis=-1)


class MLPNetwork(nn.Module):
  """Composite MLP network combining MLPActor and MLPCritic."""

  actor: MLPActor
  critic: MLPCritic

  @nn.compact
  def __call__(
      self, obs: jax.Array
  ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    logits = self.actor(obs)
    loc, scale_raw = jnp.split(logits, 2, axis=-1)
    scale = jax.nn.softplus(scale_raw) + 1e-3
    value = self.critic(obs)
    return (loc, scale), value


class MLPPolicy(policy.Policy[State]):
  """Habitats policy wrapping MLPActor and MLPCritic."""

  def __init__(self, config: policy.ModelConfig, seed: int = 0):
    self.config: policy.ModelConfig = config
    activation = self.config.activation
    self.actor = MLPActor(
        num_dimensions=self.config.num_actions,
        hidden_sizes=self.config.policy_hidden_sizes,
        activation=activation,
    )
    self.critic = MLPCritic(
        hidden_sizes=self.config.value_hidden_sizes,
        activation=activation,
    )
    self.network = MLPNetwork(actor=self.actor, critic=self.critic)
    self.params: dict[str, Any] | None = None
    self._seed = seed
    self._rng_key = jax.random.PRNGKey(seed)

    def _sample_action(
        actor_params: Any,
        normalizer: dict[str, jax.Array] | None,
        obs: jax.Array,
        key: jax.Array,
        exploit: bool,
    ) -> jax.Array:
      if (
          normalizer is not None
          and "mean" in normalizer
          and "std" in normalizer
      ):
        mean = normalizer["mean"]
        std = normalizer["std"]
        obs = (obs - mean) / (std + 1e-8)
      logits = cast(jax.Array, self.actor.apply(actor_params, obs))
      loc, scale_raw = jnp.split(logits, 2, axis=-1)
      if exploit:
        action = jnp.tanh(loc)
      else:
        scale = jax.nn.softplus(scale_raw) + 1e-3
        action = jnp.tanh(loc + scale * jax.random.normal(key, loc.shape))
      return action[0]

    self.jit_sample_action = jax.jit(
        _sample_action,
        static_argnames=("exploit",),
    )

  @override
  def get_action(self, state: State, exploit: bool = False) -> np.ndarray:
    """Selects the action for the given state."""
    if self.params is None:
      raise ValueError(
          "Policy parameters are not set. Initialize or load params first."
      )

    # Preprocess state to add batch dimension if needed
    if isinstance(state, dict):
      obs = {k: jnp.expand_dims(jnp.array(v), axis=0) for k, v in state.items()}
    else:
      obs = jnp.expand_dims(jnp.array(state, dtype=jnp.float32), axis=0)

    self._rng_key, subkey = jax.random.split(self._rng_key)
    action = self.jit_sample_action(
        self.params["actor"],
        self.params.get("normalizer"),
        obs,
        subkey,
        exploit=exploit,
    )

    # Return as numpy array
    return np.asarray(action)

  @override
  def reset(self) -> None:
    """Resets the policy."""
    self._rng_key = jax.random.PRNGKey(self._seed)

  def get_params(self) -> Any:
    """Returns the policy parameters."""
    return self.params

  def set_params(self, params: Any) -> None:
    """Sets the policy parameters."""
    self.params = params

  def save(self, path: str) -> None:
    """Saves the policy parameters to the given path."""
    params_path = os.path.join(path, "params")
    if self.params is None:
      raise ValueError(
          "Policy parameters are not set. Cannot save empty policy."
      )
    with ocp.StandardCheckpointer() as checkpointer:
      checkpointer.save(params_path, self.params, force=True)

  def load(self, path: str) -> None:
    """Loads the policy parameters from the given path."""
    params_path = os.path.join(path, "params")
    with ocp.StandardCheckpointer() as checkpointer:
      self.params = checkpointer.restore(params_path)
