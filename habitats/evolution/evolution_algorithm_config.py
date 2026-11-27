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

"""Configuration for the evolution algorithm."""

import dataclasses
import json
from typing import Any

from habitats.config import training
from habitats.evolution import offspring_generator as og
from habitats.evolution import population_initializer as population_initializer_lib
from habitats.lib import llm_client


@dataclasses.dataclass(frozen=True, kw_only=True)
class EvolutionAlgorithmConfig:
  """Represents the config for the evolution algorithm."""

  training_config: training.TrainingConfig
  num_generations: int = 10
  initial_population_size: int = 4
  # Maximum number of new offspring per generation. If None, it is not limited.
  max_new_offspring_per_generation: int | None = 5
  num_eval_episodes: int = 10
  max_workers: int = 10
  max_population_size: int | None = None

  # Strategy used to create the rewards of the initial population.
  population_initializer: (
      population_initializer_lib.PopulationInitializerName
  ) = population_initializer_lib.PopulationInitializerName.SAME_REWARD

  offspring_generator: og.OffspringGeneratorName = (
      og.OffspringGeneratorName.PLANNED_MUTATION
  )

  # Number of past generations to consider when generating new offspring.
  past_generations_to_look_at: int = 4

  # Name of the LLM backend used for every LLM call (see
  # `habitats.lib.llm_client.get_backend`). The built-in `"genai"` backend
  # needs an API key when launching the experiment; other backends must be
  # registered by the launcher before the algorithm starts.
  llm_backend: str = llm_client.GENAI_BACKEND

  # Gemini model used by the mutator and offspring planner.
  llm_model: str = "gemini-4-hydrogen"

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "EvolutionAlgorithmConfig":
    """Creates an EvolutionAlgorithmConfig from a dictionary."""
    data_copy = dict(data)
    if isinstance(data_copy.get("training_config"), dict):
      data_copy["training_config"] = training.TrainingConfig.from_dict(
          data_copy["training_config"]
      )
    if isinstance(data_copy.get("offspring_generator"), str):
      data_copy["offspring_generator"] = og.OffspringGeneratorName(
          data_copy["offspring_generator"]
      )
    if isinstance(data_copy.get("population_initializer"), str):
      data_copy["population_initializer"] = (
          population_initializer_lib.PopulationInitializerName(
              data_copy["population_initializer"]
          )
      )
    return cls(**data_copy)

  @classmethod
  def from_json(cls, json_str: str) -> "EvolutionAlgorithmConfig":
    return cls.from_dict(json.loads(json_str))
