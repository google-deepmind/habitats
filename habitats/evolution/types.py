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

"""Shared data types for evolution."""

import dataclasses
import json
from typing import Any, Callable

from habitats.training import training_interface
import numpy as np


TrainingOutput = training_interface.TrainingOutput


@dataclasses.dataclass(frozen=True, kw_only=True)
class AgentData:
  """Holds data for a single agent in the evolution.

  Attributes:
    id: Unique identifier for the agent.
    reward_code: The code defining the agent's reward function.
    generation: The generation number of this agent.
    generalization_score: The generalization score of this agent.
    training_output: The output of the training process for this agent.
    reasoning: The reasoning behind the agent's reward code.
    embedding: The embedding of the agent's reward code.
    path: The path to the agent's checkpoint.
    parent_ids: A list of parent agent IDs from the previous generation.
  """

  id: str
  reward_code: str
  generation: int
  generalization_score: float
  training_output: training_interface.TrainingOutput
  reasoning: str
  embedding: list[float]
  path: str
  parent_ids: list[str] = dataclasses.field(default_factory=list)

  @property
  def trained(self) -> bool:
    """Returns whether the agent has been trained."""
    return not self.training_output.is_empty()

  def to_json(self) -> str:
    """Returns the agent data as a JSON string."""

    agent_dict = dataclasses.asdict(self)
    def convert_arrays(v):
      return v.tolist() if isinstance(v, np.ndarray) else v

    agent_dict = map_nested(convert_arrays, agent_dict)
    return json.dumps(agent_dict)


@dataclasses.dataclass(frozen=True, kw_only=True)
class GenerationData:
  generation: int
  fitness_grid: list[list[float]]
  random_fitness: list[float]
  selected_agents: list[str]
  specialist_quality: float
  behavioral_diversity: float
  agents: dict[str, AgentData]
  generalization_scores: dict[str, float]
  # IDs of the agents whose training failed, mapped to the error. These agents
  # are left out of `agents`, of the fitness grid and of the selection.
  failed_agents: dict[str, str] = dataclasses.field(default_factory=dict)


def map_nested(
    func: Callable[[Any], Any],
    d: dict[str, Any],
) -> dict[str, Any]:
  return {
      k: map_nested(func, v) if isinstance(v, dict) else func(v)
      for k, v in d.items()
  }
