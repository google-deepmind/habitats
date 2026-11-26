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

"""Q-learning policy."""

from typing import override

from etils import epath
from habitats.policy import policy
import numpy as np


class TabularPolicy(policy.Policy[int]):
  """A tabular policy."""

  def __init__(self, config: policy.ModelConfig):
    """Initializes the tabular policy."""
    self.config: policy.ModelConfig = config
    self.num_states = self.config.num_states
    self.num_actions = self.config.num_actions

    # Initialize Q-table with zeros
    self.q_table = np.zeros((self.num_states, self.num_actions))

  @property
  @override
  def trainable_policy(self) -> "TabularPolicy":
    """Returns this tabular policy as the trainable component."""
    return self

  @override
  def get_action(
      self,
      state: int,
      exploit: bool = False,
      legal_actions: np.ndarray | None = None,
  ) -> np.ndarray:
    """Selects an action using greedy tie-breaking over the Q-table.

    Args:
      state: The state index.
      exploit: Whether to break ties deterministically (first maximum) instead
        of uniformly at random.
      legal_actions: Optional boolean mask over actions. When given, only
        actions marked legal are considered; a mask with no legal action is
        ignored.

    Returns:
      The selected action.
    """
    q_values = self.q_table[state]
    if legal_actions is not None and legal_actions.any():
      q_values = np.where(legal_actions, q_values, -np.inf)
    if exploit:
      return np.array(np.argmax(q_values))
    max_q = np.max(q_values)
    actions = np.where(q_values == max_q)[0]
    return np.array(np.random.choice(actions))

  @override
  def reset(self) -> None:
    """Resets the policy."""
    self.q_table = np.zeros((self.num_states, self.num_actions))

  @override
  def get_params(self) -> np.ndarray:
    """Returns the policy parameters."""
    return self.q_table

  @override
  def set_params(self, params: np.ndarray) -> None:
    """Sets the policy parameters."""
    self.q_table = params

  @override
  def save(self, path: str) -> None:
    """Saves the policy parameters to the given path."""
    if path:
      epath.Path(path).mkdir(parents=True, exist_ok=True)
    q_table_path = epath.Path(path) / "q_table"
    with q_table_path.open("wb") as f:
      np.save(f, self.q_table)

  @override
  def load(self, path: str) -> None:
    """Loads the policy parameters from the given path."""
    q_table_path = epath.Path(path) / "q_table"
    with q_table_path.open("rb") as f:
      self.q_table = np.load(f)

    if self.num_states != self.q_table.shape[0]:
      raise ValueError(
          f"Number of states in loaded Q-table ({self.q_table.shape[0]}) does"
          f" not match number of states in policy ({self.num_states})."
      )
    if self.num_actions != self.q_table.shape[1]:
      raise ValueError(
          f"Number of actions in loaded Q-table ({self.q_table.shape[1]}) does"
          f" not match number of actions in policy ({self.num_actions})."
      )


class TabularMetaPolicy(policy.MetaPolicy[int, TabularPolicy, TabularPolicy]):
  """A tabular meta-policy that routes states to frozen TabularPolicy agents."""

  @override
  def _init_router(self) -> TabularPolicy:
    """Initializes the tabular router policy."""
    return TabularPolicy(self.config.router_config)

  @override
  def _init_sub_policies(self) -> list[TabularPolicy]:
    """Initializes and loads the frozen tabular sub-policies."""
    sub_policies = []
    for sub_path in self.config.sub_policy_paths:
      sub_policy = TabularPolicy(self.config.policy_config)
      sub_policy.load(sub_path)
      sub_policies.append(sub_policy)
    return sub_policies

  @property
  def q_table(self) -> np.ndarray:
    """Returns the router Q-table of shape (num_states, nb_policies)."""
    return self.router.q_table

  @q_table.setter
  def q_table(self, value: np.ndarray) -> None:
    """Sets the router Q-table."""
    self.router.q_table = value

  @property
  def num_states(self) -> int:
    """Returns the number of states."""
    return self.router.num_states

  @property
  def num_actions(self) -> int:
    """Returns the number of meta-actions (nb_policies)."""
    return self.router.num_actions

  @override
  def get_meta_action(self, state: int, exploit: bool = False) -> int:
    """Selects a sub-policy index in [0, nb_policies) from the router Q-table."""
    return int(self.router.get_action(state, exploit=exploit))
