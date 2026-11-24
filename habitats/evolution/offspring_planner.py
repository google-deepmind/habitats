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

"""LLM-based offspring planner for evolutionary reward search."""

from collections.abc import Sequence
import dataclasses

from absl import logging
from habitats.evolution import types
from habitats.lib import llm_client
from habitats.lib import llm_parser
from habitats.lib import metrics
from habitats.lib import prompt_loader

@dataclasses.dataclass
class OffspringPlan:
  parent_ids: list[str]
  reasoning: str
  mutation_hint: str


class OffspringPlanningError(Exception):
  """Raised when offspring planning fails due to API or parsing errors."""


class OffspringPlanner:
  """Uses Gemini to plan the next generation of offspring."""

  def __init__(
      self,
      environment: str,
      model_name: str,
      llm_config: llm_client.LLMConfig,
      llm_backend: str = llm_client.GENAI_BACKEND,
      past_generations_to_look_at: int = 2,
  ):
    """Initializes the OffspringPlanner."""
    self.llm_config = llm_config
    self.model_name = model_name
    self._llm = llm_client.get_backend(
        llm_backend,
        config=llm_config.with_sub_role("planner"),
    )
    self.environment = environment
    self.env_desc = prompt_loader.load_env_description(self.environment)
    self.past_generations_to_look_at = past_generations_to_look_at

    self.planner_si = prompt_loader.load_offspring_generator_prompt(
        "planner/system_instruction"
    )
    self.planner_up = prompt_loader.load_offspring_generator_prompt(
        "planner/user_prompt"
    )

  def plan_generation(
      self,
      prev_gens: Sequence[types.GenerationData],
      max_new_offspring_count: int | None,
  ) -> list[OffspringPlan]:
    """Plans the next generation of offspring."""
    health_report = self._health_report(prev_gens)
    generation_report = self._generation_report(prev_gens)
    fitness_grid = self._fitness_grid(prev_gens)
    offspring_limit = ""
    if max_new_offspring_count is not None:
      offspring_limit = (
          f" It must include at most {max_new_offspring_count} new offsprings."
      )

    prompt = self.planner_up.format(
        env_desc=self.env_desc,
        generation=prev_gens[-1].generation,
        generation_report=generation_report,
        population_health_report=health_report,
        fitness_grid=fitness_grid,
        offspring_limit=offspring_limit,
    )

    response = self._llm.generate(
        llm_input=llm_client.LLMInput(
            model=self.model_name,
            prompt=prompt,
            system_instruction=self.planner_si,
            response_mime_type="application/json",
        )
    )
    return self._parse_response(response.text)

  def _health_report(
      self,
      prev_gens: Sequence[types.GenerationData],
  ) -> str:
    """Generates a health report for the previous generations."""
    last_gen = prev_gens[-1]
    return (
        f"Specialist Quality: {last_gen.specialist_quality:.4f}\n"
        f"Behavioral Diversity: {last_gen.behavioral_diversity:.4f}"
    )

  def _generation_report(
      self,
      prev_gens: Sequence[types.GenerationData],
  ) -> str:
    """Generates a survivor report for the previous generations."""
    generations_to_look_at = min(
        len(prev_gens), self.past_generations_to_look_at
    )
    gens = prev_gens[-generations_to_look_at:]
    generation_report = ""
    seen_agents: set[str] = set()
    for i, gen in enumerate(gens):
      # We put generation 0 first, then generation 1, etc, because we don't want
      # the planner to be aware that there were generations before generation 0.
      generation_report += f"Generation {i}:\n"
      for aid, agent in gen.agents.items():
        selected = (
            "Survivor" if aid in gen.selected_agents else "Non-survivor"
        )
        score = gen.generalization_scores.get(aid, agent.generalization_score)
        generalization = f"Generalization Score: {score:.4f}"
        if aid not in seen_agents:
          generation_report += (
              f"{aid} ({selected}, {generalization}): {agent.reward_code}\n"
          )
          seen_agents.add(aid)
        else:
          generation_report += f"{aid} ({selected}, {generalization})\n"
    generation_report += "\n"
    return generation_report

  def _fitness_grid(
      self,
      prev_gens: Sequence[types.GenerationData],
  ) -> str:
    """Generates the column-normalized fitness grid for the last generation."""
    last_gen = prev_gens[-1]
    agent_ids = list(last_gen.agents.keys())
    return metrics.format_fitness_grid(
        agent_ids=agent_ids,
        fitness_grid=last_gen.fitness_grid,
        random_fitness=last_gen.random_fitness,
    )

  def _parse_response(self, text: str) -> list[OffspringPlan]:
    """Parses the JSON response from Gemini into OffspringPlan objects."""
    try:
      parsed_payload = llm_parser.parse_json_object(text)
      plans = []
      for plan_dict in parsed_payload.get("offspring_plans", []):
        plans.append(
            OffspringPlan(
                parent_ids=plan_dict["parent_ids"],
                reasoning=plan_dict["reasoning"],
                mutation_hint=plan_dict["mutation_hint"],
            )
        )
      return plans
    except (ValueError, KeyError, TypeError) as e:
      logging.exception(
          "Failed to parse offspring plan JSON. Raw text: %s", text
      )
      raise OffspringPlanningError(
          f"Failed to parse offspring plan: {e}"
      ) from e
