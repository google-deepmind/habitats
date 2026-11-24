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

"""Offspring generator strategies for evolutionary reward search."""

import abc
from collections.abc import Mapping, Sequence
import enum

from absl import logging
from habitats.evolution import mutator
from habitats.evolution import offspring_planner
from habitats.evolution import population_initializer
from habitats.evolution import types
from habitats.lib import class_registry
from habitats.lib import llm_client
from habitats.training import training_interface
import numpy as np

OffspringPlan = offspring_planner.OffspringPlan
OffspringPlanningError = offspring_planner.OffspringPlanningError


class OffspringGeneratorName(enum.StrEnum):
  INDEPENDENT = "independent"
  DIRECT_MUTATION = "direct_mutation"
  PLANNED_MUTATION = "planned_mutation"


OFFSPRING_GENERATOR_REGISTRY: dict[
    OffspringGeneratorName, type["OffspringGenerator"]
] = {}
register, get = class_registry.declare_registry(OFFSPRING_GENERATOR_REGISTRY)


class OffspringGenerator(abc.ABC):
  """Interface for generating offspring agents for the next generation."""

  def __init__(
      self,
      environment: str,
      llm_config: llm_client.LLMConfig,
      model_name: str,
      llm_backend: str = llm_client.GENAI_BACKEND,
      past_generations_to_look_at: int = 2,
  ):
    self.environment = environment
    self.llm_config = llm_config
    self.model_name = model_name
    self.llm_backend = llm_backend
    self.past_generations_to_look_at = past_generations_to_look_at
    # Whether the last `generate_offspring` call asked to end the evolution.
    self.stop_requested = False
    self.setup()

  @abc.abstractmethod
  def setup(self):
    """Sets up the offspring generator."""
    pass

  @abc.abstractmethod
  def generate_offspring(
      self,
      prev_gens: Sequence[types.GenerationData],
      max_new_offspring_count: int | None,
  ) -> list[types.AgentData]:
    """Generates offspring for the next generation.

    Args:
      prev_gens: The previous generations in chronological order.
      max_new_offspring_count: The maximum number of offspring to generate, or
        None to not limit it.

    Returns:
      A list of new offspring agents (with blank `id` fields to be assigned by
      the evolution coordinator).
    """

  def _create_agent_data_from_offspring(
      self,
      reward_code: str,
      parent_ids: list[str],
      reasoning: str,
      generation: int,
  ) -> types.AgentData:
    """Creates an AgentData object for the given reward code.

    Note that this function generates blank IDs that must be filled in later.

    Args:
      reward_code: The reward code of the agent.
      parent_ids: The parent IDs of the agent.
      reasoning: The reasoning behind the reward code.
      generation: The generation of the agent.

    Returns:
      An AgentData object for the given reward code.
    """
    return types.AgentData(
        id="",
        reward_code=reward_code,
        parent_ids=parent_ids,
        reasoning=reasoning,
        generation=generation,
        training_output=training_interface.TrainingOutput.empty(),
        generalization_score=0,
        embedding=[],
        path="",
    )

  def _last_n_generations_reward_codes(
      self,
      prev_gens: Sequence[types.GenerationData],
      n: int,
  ) -> list[str]:
    """Returns unique forbidden reward codes from the last `n` generations."""
    if n <= 0:
      return []
    forbidden_codes: set[str] = set()
    for gen in prev_gens[-n:]:
      for agent in gen.agents.values():
        forbidden_codes.add(agent.reward_code)
    return list(forbidden_codes)

  def _last_n_generations_agents(
      self,
      prev_gens: Sequence[types.GenerationData],
      n: int,
  ) -> dict[str, types.AgentData]:
    """Returns a mapping of agent ID to AgentData from the last `n` generations."""
    if n <= 0:
      return dict(prev_gens[-1].agents) if prev_gens else {}
    available_agents: dict[str, types.AgentData] = {}
    for gen in prev_gens[-n:]:
      available_agents.update(gen.agents)
    return available_agents

  def _sanitize_plans(
      self,
      plans: Sequence[OffspringPlan],
      available_agents: Mapping[str, types.AgentData],
  ) -> list[OffspringPlan]:
    """Removes unknown or hallucinated parent IDs from `plans`."""
    sanitized_plans: list[OffspringPlan] = []
    for plan in plans:
      valid_parent_ids = [
          pid for pid in plan.parent_ids if pid in available_agents
      ]
      if len(valid_parent_ids) != len(plan.parent_ids):
        logging.warning(
            "Dropping unknown parent IDs from plan %s (valid: %s).",
            plan.parent_ids,
            valid_parent_ids,
        )
      sanitized_plans.append(
          OffspringPlan(
              parent_ids=valid_parent_ids,
              reasoning=plan.reasoning,
              mutation_hint=plan.mutation_hint,
          )
      )
    return sanitized_plans

  def _num_offspring_per_survivor(
      self,
      prev_gens: Sequence[types.GenerationData],
      max_new_offspring_count: int | None,
  ) -> int:
    """Returns one offspring per survivor, up to `max_new_offspring_count`."""
    num_survivors = len(prev_gens[-1].selected_agents)
    if max_new_offspring_count is None:
      return num_survivors
    return min(num_survivors, max_new_offspring_count)


@register(OffspringGeneratorName.INDEPENDENT)
class IndependentGenerator(OffspringGenerator):
  """Offspring generator that uses independent evolution.

  Generates a new population of reward functions that are diverse and different
  from the last `past_generations_to_look_at` generations, with one offspring
  per survivor of the last generation, up to `max_new_offspring_count`.
  """

  def setup(self):
    """Sets up the IndependentGenerator."""
    self.llm_config = self.llm_config.with_sub_role("independent_generator")

    self.mutator = mutator.GeminiMutator(
        environment=self.environment,
        llm_config=self.llm_config,
        model_name=self.model_name,
        llm_backend=self.llm_backend,
    )

  def generate_offspring(
      self,
      prev_gens: Sequence[types.GenerationData],
      max_new_offspring_count: int | None,
  ) -> list[types.AgentData]:
    """Generates offspring for the next generation."""
    num_offspring = self._num_offspring_per_survivor(
        prev_gens, max_new_offspring_count
    )
    if num_offspring == 0:
      return []

    new_agents = []
    forbidden_codes = self._last_n_generations_reward_codes(
        prev_gens, self.past_generations_to_look_at
    )
    for _ in range(num_offspring):
      code = self.mutator.generate_diverse(
          forbidden_codes=forbidden_codes,
          hint=None,
      )
      new_agents.append(
          self._create_agent_data_from_offspring(
              reward_code=code,
              parent_ids=[],
              reasoning="",
              generation=prev_gens[-1].generation + 1,
          )
      )
      forbidden_codes.append(code)
    return new_agents


@register(OffspringGeneratorName.DIRECT_MUTATION)
class DirectMutationGenerator(OffspringGenerator):
  """Offspring generator that mutates surviving agents without a planner."""

  def setup(self):
    """Sets up the DirectMutationGenerator."""
    self.llm_config = self.llm_config.with_sub_role("direct_mutation_generator")

    self.mutator = mutator.GeminiMutator(
        environment=self.environment,
        llm_config=self.llm_config,
        model_name=self.model_name,
        llm_backend=self.llm_backend,
    )

    self.population_initializer = (
        population_initializer.DiverseRewardInitializer(
            population_initializer.PopulationInitializerConfig(
                env_name=self.environment,
                model_name=self.model_name,
                llm_config=self.llm_config,
                llm_backend=self.llm_backend,
            )
        )
    )

  def generate_offspring(
      self,
      prev_gens: Sequence[types.GenerationData],
      max_new_offspring_count: int | None,
  ) -> list[types.AgentData]:
    """Generates offspring for the next generation."""
    if max_new_offspring_count == 0:
      return []

    survivors = prev_gens[-1].selected_agents
    forbidden_codes = self._last_n_generations_reward_codes(
        prev_gens, self.past_generations_to_look_at
    )
    new_agents = []

    if not survivors:
      # If there are no survivors, generate new offspring from scratch while
      # avoiding forbidden codes from recent generations.
      new_offsprings = self.population_initializer.init_population(
          population_size=1,
          forbidden_codes=forbidden_codes,
      )
      for code in new_offsprings:
        new_agents.append(
            self._create_agent_data_from_offspring(
                reward_code=code,
                parent_ids=[],
                reasoning="",
                generation=prev_gens[-1].generation + 1,
            )
        )
      return new_agents

    # Mutates each survivor once, for up to `max_new_offspring_count` survivors.
    num_offspring = self._num_offspring_per_survivor(
        prev_gens, max_new_offspring_count
    )
    parent_ids = np.random.choice(survivors, size=num_offspring, replace=False)
    for parent_id in parent_ids.tolist():
      new_code = self.mutator.mutate(
          parent_codes=[prev_gens[-1].agents[parent_id].reward_code],
          forbidden_codes=forbidden_codes,
          mutation_hint=None,
      )
      forbidden_codes.append(new_code)

      new_agents.append(
          self._create_agent_data_from_offspring(
              reward_code=new_code,
              parent_ids=[parent_id],
              reasoning="",
              generation=prev_gens[-1].generation + 1,
          )
      )
    return new_agents


@register(OffspringGeneratorName.PLANNED_MUTATION)
class PlannedMutationGenerator(OffspringGenerator):
  """Offspring generator that uses an LLM planner and mutator.

  Analyzes up to `past_generations_to_look_at` previous generations to plan and
  generate the next generation of offspring.
  """

  def setup(self):
    """Sets up the PlannedMutationGenerator."""
    self.llm_config = self.llm_config.with_sub_role(
        "planned_mutation_generator"
    )

    self.planner = offspring_planner.OffspringPlanner(
        environment=self.environment,
        model_name=self.model_name,
        llm_config=self.llm_config,
        llm_backend=self.llm_backend,
        past_generations_to_look_at=self.past_generations_to_look_at,
    )

    self.mutator = mutator.GeminiMutator(
        environment=self.environment,
        llm_config=self.llm_config,
        model_name=self.model_name,
        llm_backend=self.llm_backend,
    )

    self.population_initializer = (
        population_initializer.DiverseRewardInitializer(
            population_initializer.PopulationInitializerConfig(
                env_name=self.environment,
                model_name=self.model_name,
                llm_config=self.llm_config,
                llm_backend=self.llm_backend,
            )
        )
    )

  def generate_offspring(
      self,
      prev_gens: Sequence[types.GenerationData],
      max_new_offspring_count: int | None,
  ) -> list[types.AgentData]:
    """Generates offspring for the next generation.

    An empty plan from the planner is its signal that the population has
    converged, so it sets `stop_requested`.

    Args:
      prev_gens: The previous generations in chronological order.
      max_new_offspring_count: The maximum number of offspring to generate, or
        None to let the planner choose freely.

    Returns:
      The new offspring agents.
    """
    self.stop_requested = False
    if max_new_offspring_count == 0:
      return []

    plans = self.planner.plan_generation(prev_gens, max_new_offspring_count)
    if not plans:
      logging.info(
          "LLM Planner returned 0 offspring plans: requesting to stop the"
          " evolution."
      )
      self.stop_requested = True
      return []

    has_too_many_plans = (
        max_new_offspring_count is not None
        and len(plans) > max_new_offspring_count
    )
    if has_too_many_plans:
      logging.warning(
          "LLM Planner returned %d offspring plans, more than the"
          " requested %d.",
          len(plans),
          max_new_offspring_count,
      )
      plans = plans[:max_new_offspring_count]

    forbidden_codes = self._last_n_generations_reward_codes(
        prev_gens, self.past_generations_to_look_at
    )
    available_agents = self._last_n_generations_agents(
        prev_gens, self.past_generations_to_look_at
    )
    plans = self._sanitize_plans(plans, available_agents)

    # Generate reward functions for agents with no parents.
    no_parent_plans = [plan for plan in plans if not plan.parent_ids]
    no_parents_reward_codes = self.population_initializer.init_population(
        population_size=len(no_parent_plans),
        forbidden_codes=forbidden_codes,
        hints=[plan.mutation_hint for plan in no_parent_plans],
    )
    new_agents = [
        self._create_agent_data_from_offspring(
            reward_code=reward_code,
            parent_ids=[],
            reasoning=plan.reasoning,
            generation=prev_gens[-1].generation + 1,
        )
        for reward_code, plan in zip(
            no_parents_reward_codes,
            no_parent_plans,
            strict=True,
        )
    ]
    forbidden_codes.extend(no_parents_reward_codes)

    # Generate reward functions for agents with parents.
    parents_plans = [plan for plan in plans if plan.parent_ids]
    for plan in parents_plans:
      logging.info(
          "Generating offspring with parents %s and hint: %s",
          plan.parent_ids,
          plan.mutation_hint,
      )
      new_code = self.mutator.mutate(
          parent_codes=[
              available_agents[pid].reward_code for pid in plan.parent_ids
          ],
          forbidden_codes=forbidden_codes,
          mutation_hint=plan.mutation_hint,
      )
      forbidden_codes.append(new_code)
      new_agents.append(
          self._create_agent_data_from_offspring(
              reward_code=new_code,
              parent_ids=plan.parent_ids,
              reasoning=plan.reasoning,
              generation=prev_gens[-1].generation + 1,
          )
      )

    return new_agents
