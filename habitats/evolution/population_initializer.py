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

"""Module to initialize a population of policies."""

import abc
from collections.abc import Callable, Sequence
import dataclasses
import enum

from habitats.lib import class_registry
from habitats.lib import llm_client
from habitats.lib import llm_parser
from habitats.lib import prompt_loader

class PopulationInitializerName(enum.StrEnum):
  """Names of the strategies that create the initial population's rewards."""

  SAME_REWARD = "same_reward"
  DIVERSE_REWARD = "diverse_reward"


@dataclasses.dataclass(frozen=True)
class PopulationInitializerConfig:
  """Configuration for the population initializer."""

  env_name: str
  model_name: str
  llm_config: llm_client.LLMConfig
  llm_backend: str = llm_client.GENAI_BACKEND


class PopulationInitializer(abc.ABC):
  """Initializes a population of policies with reward functions."""

  def __init__(
      self,
      config: PopulationInitializerConfig,
      name: str,
  ):
    """Initializes the population initializer."""
    self.config = config
    self.name = name

  @abc.abstractmethod
  def init_population(
      self,
      population_size: int,
      forbidden_codes: Sequence[str] | None = None,
      hints: Sequence[str | None] | None = None,
  ) -> list[str]:
    """Initializes a population of policies with reward functions.

    Args:
      population_size: The number of reward functions to initialize.
      forbidden_codes: Optional sequence of reward function codes that must not
        be duplicated.
      hints: Optional sequence of hints (one per reward function) to guide
        generation.

    Returns:
      A list of `population_size` newly generated reward function code strings.
    """


class SameRewardInitializer(PopulationInitializer):
  """Initializes a population of policies with the same reward function."""

  DEFAULT_REWARD_FUNCTION = """
def compute_reward(env, state, action, next_state, done, info, base_reward) -> tuple[float, bool]:
  # Default reward function: just return the base environment reward
  return base_reward, done
"""

  def __init__(self, config: PopulationInitializerConfig):
    """Initializes the same-reward initializer."""
    super().__init__(config, name="same_reward_initializer")

  def init_population(
      self,
      population_size: int,
      forbidden_codes: Sequence[str] | None = None,
      hints: Sequence[str | None] | None = None,
  ) -> list[str]:
    """Initializes a population of policies with the same reward function."""
    del forbidden_codes, hints
    return [self.DEFAULT_REWARD_FUNCTION] * population_size


class DiverseRewardInitializer(PopulationInitializer):
  """Initializes a population of policies with diverse reward functions."""

  def __init__(self, config: PopulationInitializerConfig):
    """Initializes the diverse-reward initializer."""
    super().__init__(config, name="diverse_reward_initializer")
    self.env_desc = prompt_loader.load_env_description(config.env_name)
    self.system_instruction = prompt_loader.load_population_initializer_prompt(
        "system_instruction"
    )
    self.prompt = prompt_loader.load_population_initializer_prompt(
        "user_prompt"
    )
    self.llm_config = config.llm_config.with_sub_role(
        "diverse_reward_initializer"
    )

    self.llm_client = llm_client.get_backend(
        config.llm_backend,
        config=self.llm_config,
    )

  def init_population(
      self,
      population_size: int,
      forbidden_codes: Sequence[str] | None = None,
      hints: Sequence[str | None] | None = None,
  ) -> list[str]:
    """Initializes a population of policies with diverse reward functions.

    Args:
      population_size: The number of reward functions to initialize.
      forbidden_codes: Optional sequence of reward function codes that must not
        be duplicated.
      hints: Optional sequence of hints (one per reward function) to guide
        generation.

    Returns:
      A list of `population_size` diverse reward functions.

    Raises:
      ValueError: If `hints` is provided and its length does not match
        `population_size`.
    """
    if hints is not None and len(hints) != population_size:
      raise ValueError(
          "Number of hints must match population size. Got"
          f" {len(hints)} hints and {population_size} population size."
      )

    if population_size <= 0:
      return []

    effective_hints: Sequence[str | None] = (
        hints if hints is not None else [None] * population_size
    )
    running_forbidden_codes = list(forbidden_codes) if forbidden_codes else []
    reward_functions: list[str] = []

    for hint in effective_hints:
      code = self._generate_diverse(
          forbidden_codes=running_forbidden_codes,
          hint=hint,
      )
      reward_functions.append(code)
      running_forbidden_codes.append(code)

    return reward_functions

  def _generate_diverse(
      self,
      forbidden_codes: Sequence[str],
      hint: str | None = None,
  ) -> str:
    """Generates a single diverse reward function avoiding `forbidden_codes`."""
    forbidden_codes_str = "".join(
        f"\n--- Forbidden {i+1} ---\n{code}\n"
        for i, code in enumerate(forbidden_codes)
    )
    prompt = self.prompt.format(
        env_desc=self.env_desc,
        forbidden_codes=forbidden_codes_str,
        hint=hint if hint else "N/A",
    )
    try:
      response = self.llm_client.generate(
          llm_input=llm_client.LLMInput(
              model=self.config.model_name,
              prompt=prompt,
              system_instruction=self.system_instruction,
          )
      )
      return llm_parser.extract_code_block(response.text)
    except Exception as e:
      raise RuntimeError(f"LLM generation failed: {e}") from e


POPULATION_INITIALIZER_REGISTRY: dict[
    PopulationInitializerName,
    Callable[[PopulationInitializerConfig], PopulationInitializer],
] = {}
register, get = class_registry.declare_registry(POPULATION_INITIALIZER_REGISTRY)

register(PopulationInitializerName.SAME_REWARD)(SameRewardInitializer)
register(PopulationInitializerName.DIVERSE_REWARD)(DiverseRewardInitializer)
