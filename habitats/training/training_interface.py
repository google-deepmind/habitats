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

"""Training interface."""

import abc
from collections.abc import Callable, Mapping
import dataclasses
import enum
from typing import Any, Generic, TypeVar

import dm_env
from dm_env import specs
from habitats.lib import numerics
from habitats.policy import policy as policy_lib
import numpy as np

PolicyLike = TypeVar("PolicyLike", bound=policy_lib.Policy)
EnvironmentFactory = Callable[[int, bool], dm_env.Environment]


class TrainingInterfaceName(enum.StrEnum):
  """Represents the name of the training interface."""

  Q_LEARNING = "q_learning"
  PPO = "ppo"


class TrainingDivergedError(RuntimeError):
  """Raised when training produces NaN or infinite parameters."""


def raise_if_diverged(
    params: Mapping[str, Any], *, env_steps: int | None = None
) -> None:
  """Raises `TrainingDivergedError` if any parameter is NaN or infinite.

  Args:
    params: Parameter trees, keyed by the name to report if they diverged, e.g.
      `{"policy": policy_params, "value": value_params}`.
    env_steps: Number of environment steps taken so far, to report if the
      parameters diverged. None if unknown.

  Raises:
    TrainingDivergedError: If any value in `params` is NaN or infinite.
  """
  diverged = [
      name for name, tree in params.items() if not numerics.all_finite(tree)
  ]
  if diverged:
    when = "" if env_steps is None else f" after {env_steps} environment steps"
    raise TrainingDivergedError(
        f"Training diverged{when}: {', '.join(diverged)} parameters contain"
        " NaN or infinite values."
    )


@dataclasses.dataclass(frozen=True)
class TrainingOutput:
  """Represents the output of the training procedure."""

  episode_rewards: np.ndarray
  episode_lengths: np.ndarray
  epsilons: np.ndarray
  # Why training failed, or None if it succeeded.
  error: str | None = None

  @property
  def mean_reward(self) -> float:
    return float(np.mean(self.episode_rewards))

  @property
  def mean_length(self) -> float:
    return float(np.mean(self.episode_lengths))

  @classmethod
  def empty(cls) -> "TrainingOutput":
    """Returns an empty TrainingOutput."""
    return cls(
        episode_rewards=np.array([]),
        episode_lengths=np.array([]),
        epsilons=np.array([]),
    )

  @classmethod
  def failure(cls, error: str) -> "TrainingOutput":
    """Returns the output of a training that failed because of `error`."""
    return dataclasses.replace(cls.empty(), error=error)

  def is_empty(self) -> bool:
    return len(self.episode_rewards) == 0


@dataclasses.dataclass(frozen=True)
class EvaluationOutput:
  """Represents the output of the evaluation procedure."""

  episode_rewards: np.ndarray
  episode_lengths: np.ndarray

  @property
  def mean_reward(self) -> float:
    return float(np.mean(self.episode_rewards))

  @property
  def mean_length(self) -> float:
    return float(np.mean(self.episode_lengths))


@dataclasses.dataclass(frozen=True, kw_only=True)
class TrainingInterfaceConfig:
  """Represents the training config used to train agents in an evolution step."""

  tag: str = "default"
  checkpoint_path: str = ""
  discount_factor: float = 0.99
  decay_rate: float = 0.999
  num_episodes: int = 10000
  seed: int = 42

  # Number of episodes when evaluating
  num_eval_episodes: int = 10

  # Maximum number of steps per evaluation episode.
  max_eval_steps: int = 1000

  # Maximum number of steps per training episode.
  max_steps: int = 1000
  epsilon_start: float = 1.0
  epsilon_min: float = 0.01
  epsilon_decay: float = 0.995
  # If set, epsilon decays from `epsilon_start` to `epsilon_min` over this
  # fraction of `num_episodes`, and `epsilon_decay` is ignored.
  exploration_fraction: float | None = None
  learning_rate: float = 0.1
  unroll_length: int = 16
  num_minibatches: int = 32
  num_timesteps: int = 50_000_000
  ppo_epochs: int = 10
  clip_value: bool = True
  value_clipping_epsilon: float = 0.2
  normalize_advantage: bool = True
  gae_lambda: float = 0.95
  log_every: int = 1000
  playground: bool = False
  num_actors: int = 1
  init_noise_std: float = 1.0
  reward_scaling: float = 1.0
  entropy_cost: float = 1e-2
  batch_size: int = 64
  normalize_observations: bool = False

  # Number of evaluations performed during training.
  num_train_evals: int = 10
  num_eval_envs: int = 128


class TrainingInterface(abc.ABC, Generic[PolicyLike]):
  """Represents the training procedure for an agent."""

  def __init__(self, config: "TrainingInterfaceConfig"):
    self.config = config

  @abc.abstractmethod
  def train(
      self, env_factory: EnvironmentFactory, policy: PolicyLike
  ) -> TrainingOutput:
    """Trains the policy."""
    pass

  @abc.abstractmethod
  def evaluate(
      self,
      env_factory: EnvironmentFactory,
      policy: PolicyLike,
      is_random_policy: bool = False,
  ) -> EvaluationOutput:
    """Evaluates the policy."""
    pass

  def verify_env(self, env: dm_env.Environment) -> None:
    """Verifies that the environment is compatible with the training interface."""


def is_discrete_env(env: dm_env.Environment) -> bool:
  """Checks if the environment has discrete observations and actions."""
  return isinstance(env.observation_spec(), specs.DiscreteArray) and isinstance(
      env.action_spec(), specs.DiscreteArray
  )
