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

"""Policy interface."""

import abc
from collections.abc import Mapping, Sequence
import dataclasses
import enum
import os
from typing import Any, Generic, TypeVar, override

import dm_env
from dm_env import specs
from etils import epath
import numpy as np

T = TypeVar("T")
SubPolicyT = TypeVar("SubPolicyT", bound="Policy")
RouterPolicyT = TypeVar("RouterPolicyT", bound="Policy")


class PolicyName(enum.StrEnum):
  """Represents the name of the policy."""

  TABULAR = "tabular"
  TABULAR_META = "tabular_meta"
  MLP = "mlp"


def get_space_dimension(spec: Any) -> int:
  """Returns the scalar cardinality or flattened dimension of a spec."""
  if hasattr(spec, "num_values") and spec.num_values is not None:
    return int(spec.num_values)
  if hasattr(spec, "shape") and spec.shape is not None:
    return int(np.prod(spec.shape))
  raise ValueError(f"Unsupported spec type: {type(spec)}")


@dataclasses.dataclass(frozen=True)
class ModelConfig:
  """Represents the model config."""

  num_actions: int = 1
  num_states: int = 1
  action_shape: tuple[int, ...] = ()
  observation_shape: tuple[int, ...] = ()
  policy_hidden_sizes: tuple[int, ...] = (64, 64)
  value_hidden_sizes: tuple[int, ...] = (64, 64)
  activation: str = "swish"

  def apply_environment_spec(
      self,
      observations: Any,
      actions: Any,
  ) -> "ModelConfig":
    """Returns a new ModelConfig populated with dimensions from env specs."""
    return dataclasses.replace(
        self,
        num_actions=get_space_dimension(actions),
        num_states=get_space_dimension(observations),
        action_shape=tuple(actions.shape) if hasattr(actions, "shape") else (),
        observation_shape=tuple(observations.shape)
        if hasattr(observations, "shape")
        else (),
    )

  @classmethod
  def from_dict(cls, data: Mapping[str, Any]) -> "ModelConfig":
    """Creates a ModelConfig from a dictionary."""
    return cls(
        num_actions=data.get("num_actions", 1),
        num_states=data.get("num_states", 1),
        action_shape=tuple(data.get("action_shape", ())),
        observation_shape=tuple(data.get("observation_shape", ())),
        policy_hidden_sizes=tuple(data.get("policy_hidden_sizes", (64, 64))),
        value_hidden_sizes=tuple(data.get("value_hidden_sizes", (64, 64))),
        activation=data.get("activation", "swish"),
    )


@dataclasses.dataclass(frozen=True)
class MetaModelConfig:
  """Meta-policy config to define the meta-policy architecture."""

  # Policy config used to create the sub-policies.
  policy_config: ModelConfig = dataclasses.field(default_factory=ModelConfig)

  # Policy config used to create the router policy.
  router_config: ModelConfig = dataclasses.field(default_factory=ModelConfig)

  # Checkpoint paths used to initialize the frozen sub-policies.
  sub_policy_paths: tuple[str, ...] = ()

  def __post_init__(self) -> None:
    has_sub_policy_paths = bool(self.sub_policy_paths)
    if has_sub_policy_paths and self.nb_policies < 2:
      raise ValueError(
          "sub_policy_paths must contain at least 2 paths when provided, "
          f"got {len(self.sub_policy_paths)}."
      )

  @property
  def nb_policies(self) -> int:
    """Returns the number of sub-policies in the meta-policy."""
    return len(self.sub_policy_paths)

  def with_sub_policy_paths(
      self,
      sub_policy_paths: Sequence[str],
  ) -> "MetaModelConfig":
    """Returns a new MetaModelConfig configured with `sub_policy_paths`."""
    return dataclasses.replace(
        self,
        sub_policy_paths=tuple(sub_policy_paths),
    )

  def apply_environment_spec(
      self,
      observations: Any,
      actions: Any,
  ) -> "MetaModelConfig":
    """Returns a new MetaModelConfig with sub-policy specs populated."""
    if self.nb_policies < 2:
      raise ValueError(
          "MetaModelConfig requires at least 2 sub_policy_paths before "
          f"applying environment spec, got {self.nb_policies}."
      )
    return dataclasses.replace(
        self,
        policy_config=self.policy_config.apply_environment_spec(
            observations=observations,
            actions=actions,
        ),
        router_config=self.router_config.apply_environment_spec(
            observations=observations,
            actions=specs.DiscreteArray(
                num_values=self.nb_policies,
                dtype=int,
                name="meta_action",
            ),
        ),
    )

  @classmethod
  def from_dict(cls, data: Mapping[str, Any]) -> "MetaModelConfig":
    """Returns a MetaModelConfig from a dictionary."""
    return cls(
        policy_config=ModelConfig.from_dict(data.get("policy_config", {})),
        router_config=ModelConfig.from_dict(data.get("router_config", {})),
        sub_policy_paths=tuple(data.get("sub_policy_paths", ())),
    )


class Policy(abc.ABC, Generic[T]):
  """A policy for a habitat."""

  @abc.abstractmethod
  def get_action(self, state: T, exploit: bool = False) -> np.ndarray:
    """Returns the action for the given state."""
    pass

  @abc.abstractmethod
  def reset(self) -> None:
    """Resets the policy."""
    pass

  @abc.abstractmethod
  def get_params(self) -> Any:
    """Returns the policy parameters."""
    pass

  @abc.abstractmethod
  def set_params(self, params: Any) -> None:
    """Sets the policy parameters."""
    pass

  @abc.abstractmethod
  def save(self, path: str) -> None:
    """Saves the policy parameters to the given path."""
    pass

  @abc.abstractmethod
  def load(self, path: str) -> None:
    """Loads the policy parameters from the given path."""
    pass

  # Training algorithms interact with `trainable_policy` and `wrap_env(env)`
  # so they can train both flat policies and hierarchical meta-policies without
  # branching on the policy type:
  # - For a standard Policy, `wrap_env` returns the environment unchanged and
  #   `trainable_policy` returns `self`.
  # - For a MetaPolicy, `wrap_env` wraps the environment so its action space
  #   selects among the frozen sub-policies, and `trainable_policy` returns the
  #   inner router policy whose action space matches that wrapped environment.
  # Meanwhile, `policy.get_action()` and `policy.save()` continue to operate on
  # the full policy (e.g., for evaluation on the unwrapped environment and for
  # saving both router and sub-policy checkpoints).
  @property
  def trainable_policy(self) -> "Policy":
    """Returns the policy component whose parameters are optimized in training."""
    return self

  def wrap_env(self, env: dm_env.Environment) -> dm_env.Environment:
    """Adapts `env` for training `trainable_policy` (identity by default)."""
    return env


class MetaPolicyEnvWrapper(
    dm_env.Environment, Generic[T, SubPolicyT, RouterPolicyT]
):
  """Wraps a `dm_env.Environment` to expose sub-policy selection as actions.

  During meta-policy training, the router policy outputs a sub-policy index
  `k` in `[0, nb_policies)`. This wrapper exposes that discrete index space as
  its `action_spec()`, queries the `k`-th frozen sub-policy at the current
  environment state to obtain a primitive environment action, and steps the
  underlying environment with that primitive action.
  """

  def __init__(
      self,
      env: dm_env.Environment,
      meta_policy: "MetaPolicy[T, SubPolicyT, RouterPolicyT]",
  ):
    """Initializes the wrapper."""
    self._env = env
    self._meta_policy = meta_policy
    self._current_state: T | None = None

  @override
  def reset(self) -> dm_env.TimeStep:
    """Resets the underlying environment and records the initial state."""
    timestep = self._env.reset()
    self._current_state = timestep.observation
    return timestep

  @override
  def step(self, action: int) -> dm_env.TimeStep:
    """Routes meta-action through the selected sub-policy and steps env."""
    if self._current_state is None:
      raise RuntimeError("MetaPolicyEnvWrapper.step() called before reset().")
    primitive_action = self._meta_policy.get_primitive_action(
        self._current_state,
        action,
        exploit=True,
    )
    timestep = self._env.step(primitive_action)
    self._current_state = timestep.observation
    return timestep

  @override
  def observation_spec(self) -> specs.Array:
    """Returns the underlying environment's observation spec."""
    return self._env.observation_spec()

  @override
  def action_spec(self) -> specs.DiscreteArray:
    """Returns a discrete action spec over the sub-policy indices."""
    return specs.DiscreteArray(
        num_values=self._meta_policy.config.nb_policies,
        dtype=int,
        name="meta_action",
    )

  @override
  def close(self) -> None:
    """Closes the underlying environment."""
    self._env.close()


class MetaPolicy(Policy[T], Generic[T, SubPolicyT, RouterPolicyT]):
  """A meta-policy for routing states to frozen sub-policies."""

  def __init__(self, config: MetaModelConfig):
    """Initializes the meta-policy instance."""
    if config.nb_policies < 2:
      raise ValueError(
          "MetaPolicy requires at least 2 sub-policies, got"
          f" {config.nb_policies}."
      )
    self.config: MetaModelConfig = config
    self.router: RouterPolicyT = self._init_router()
    self.policies: list[SubPolicyT] = self._init_sub_policies()

  @property
  @override
  def trainable_policy(self) -> RouterPolicyT:
    """Returns the router policy trained to select among sub-policies."""
    return self.router

  @override
  def wrap_env(
      self, env: dm_env.Environment
  ) -> MetaPolicyEnvWrapper[T, SubPolicyT, RouterPolicyT]:
    """Wraps `env` so actions correspond to selecting a sub-policy."""
    return MetaPolicyEnvWrapper(env, self)

  @abc.abstractmethod
  def _init_router(self) -> RouterPolicyT:
    """Initializes and returns the meta-policy router."""

  @abc.abstractmethod
  def _init_sub_policies(self) -> list[SubPolicyT]:
    """Initializes and returns the sub-policy instances."""

  @abc.abstractmethod
  def get_meta_action(self, state: T, exploit: bool = False) -> int:
    """Returns the selected sub-policy index in [0, nb_policies)."""

  def get_primitive_action(
      self,
      state: T,
      meta_action: int,
      exploit: bool = True,
  ) -> np.ndarray:
    """Queries the selected sub-policy for a primitive environment action."""
    if not 0 <= meta_action < len(self.policies):
      raise ValueError(
          f"meta_action {meta_action} out of bounds for "
          f"{len(self.policies)} sub-policies."
      )
    return self.policies[meta_action].get_action(state, exploit=exploit)

  @override
  def get_action(self, state: T, exploit: bool = False) -> np.ndarray:
    """Routes state through the meta-policy to a sub-policy's action."""
    meta_action = self.get_meta_action(state, exploit=exploit)
    return self.get_primitive_action(state, meta_action, exploit=True)

  @override
  def reset(self) -> None:
    """Resets the router and sub-policies."""
    self.router.reset()
    for sub_policy in self.policies:
      sub_policy.reset()

  @override
  def get_params(self) -> dict[str, Any]:
    """Returns the router and sub-policy parameters."""
    return {
        "router": self.router.get_params(),
        "policies": [sub_policy.get_params() for sub_policy in self.policies],
    }

  @override
  def set_params(self, params: Mapping[str, Any]) -> None:
    """Sets the router and/or sub-policy parameters."""
    if "router" in params:
      self.router.set_params(params["router"])
    if "policies" in params:
      sub_params = params["policies"]
      if len(sub_params) != len(self.policies):
        raise ValueError(
            f"Expected {len(self.policies)} sub-policy parameter entries, "
            f"got {len(sub_params)}."
        )
      for sub_policy, param in zip(self.policies, sub_params, strict=True):
        sub_policy.set_params(param)

  @override
  def save(self, path: str) -> None:
    """Saves the router and all sub-policies to the given directory."""
    if path:
      epath.Path(path).mkdir(parents=True, exist_ok=True)
    self.router.save(path)
    for i, sub_policy in enumerate(self.policies):
      sub_dir = os.path.join(path, "sub_policies", f"policy_{i}")
      epath.Path(sub_dir).mkdir(parents=True, exist_ok=True)
      sub_policy.save(sub_dir)

  @override
  def load(self, path: str) -> None:
    """Loads the router and all sub-policies from the given directory."""
    sub_policies_dir = epath.Path(path) / "sub_policies"
    if not sub_policies_dir.exists():
      raise ValueError(
          f"Sub-policies directory {sub_policies_dir} not found in "
          f"MetaPolicy checkpoint path {path}."
      )

    num_saved_policies = 0
    for entry in sub_policies_dir.iterdir():
      if entry.name.startswith("policy_") and entry.is_dir():
        num_saved_policies += 1
    if num_saved_policies != len(self.policies):
      raise ValueError(
          f"Number of sub-policies in checkpoint ({num_saved_policies}) "
          "does not match number of sub-policies in policy "
          f"({len(self.policies)})."
      )

    self.router.load(path)
    for i, sub_policy in enumerate(self.policies):
      sub_dir = os.path.join(sub_policies_dir, f"policy_{i}")
      sub_policy.load(sub_dir)
