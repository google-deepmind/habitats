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

"""Tests for the offspring generator."""

import json
from unittest import mock

from absl.testing import absltest
from habitats.evolution import offspring_generator
from habitats.evolution import types
from habitats.lib import llm_client
from habitats.training import training_interface
import numpy as np

def _make_generation(
    generation: int,
    agent_codes: dict[str, str],
    selected_agents: list[str],
) -> types.GenerationData:
  agents = {
      aid: types.AgentData(
          id=aid,
          reward_code=code,
          generation=generation,
          generalization_score=0.5,
          training_output=training_interface.TrainingOutput.empty(),
          reasoning="",
          embedding=[],
          path="",
          parent_ids=[],
      )
      for aid, code in agent_codes.items()
  }
  n = len(agents)
  return types.GenerationData(
      generation=generation,
      agents=agents,
      fitness_grid=[[1.0] * n for _ in range(n)],
      random_fitness=[0.0] * n,
      generalization_scores={aid: 0.5 for aid in agents},
      selected_agents=selected_agents,
      specialist_quality=0.8,
      behavioral_diversity=0.7,
  )


class OffspringGeneratorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(42)
    self.llm_config = llm_client.LLMConfig(role="test")

  @mock.patch.object(llm_client, "get_backend", autospec=True)
  def test_independent_generator_respects_bounds_and_last_n_gens(
      self, mock_get_backend
  ):
    mock_llm = mock.MagicMock()
    mock_llm.generate.side_effect = [
        llm_client.LLMOutput(text="```python\nreturn 10.0\n```"),
        llm_client.LLMOutput(text="```python\nreturn 20.0\n```"),
        llm_client.LLMOutput(text="```python\nreturn 30.0\n```"),
    ]
    mock_get_backend.return_value = mock_llm

    gen0 = _make_generation(0, {"agent_0": "code_gen0"}, ["agent_0"])
    gen1 = _make_generation(1, {"agent_1": "code_gen1"}, ["agent_1"])
    gen2 = _make_generation(2, {"agent_2": "code_gen2"}, ["agent_2"])

    generator = offspring_generator.IndependentGenerator(
        environment="cartpole",
        llm_config=self.llm_config,
        model_name="fake_model",
        past_generations_to_look_at=2,
    )

    # The last generation has a single survivor, so there is one offspring.
    offspring = generator.generate_offspring(
        prev_gens=[gen0, gen1, gen2],
        max_new_offspring_count=1,
    )
    self.assertLen(offspring, 1)
    self.assertEqual(offspring[0].reward_code, "return 10.0")
    self.assertEqual(offspring[0].generation, 3)

    prompt = mock_llm.generate.call_args.kwargs["llm_input"].prompt
    self.assertNotIn("code_gen0", prompt)
    self.assertIn("code_gen1", prompt)
    self.assertIn("code_gen2", prompt)

  @mock.patch.object(llm_client, "get_backend", autospec=True)
  def test_direct_mutation_generator_single_survivor_does_not_crash(
      self, mock_get_backend
  ):
    mock_llm = mock.MagicMock()
    mock_llm.generate.return_value = llm_client.LLMOutput(
        text="```python\nreturn 99.0\n```"
    )
    mock_get_backend.return_value = mock_llm

    gen0 = _make_generation(0, {"agent_0": "code_gen0"}, ["agent_0"])
    generator = offspring_generator.DirectMutationGenerator(
        environment="cartpole",
        llm_config=self.llm_config,
        model_name="fake_model",
        past_generations_to_look_at=2,
    )

    offspring = generator.generate_offspring(
        prev_gens=[gen0],
        max_new_offspring_count=1,
    )
    self.assertLen(offspring, 1)
    self.assertEqual(offspring[0].parent_ids, ["agent_0"])
    self.assertIsInstance(offspring[0].parent_ids, list)
    self.assertEqual(offspring[0].reward_code, "return 99.0")

  @mock.patch.object(llm_client, "get_backend", autospec=True)
  def test_independent_generator_creates_one_offspring_per_survivor(
      self, mock_get_backend
  ):
    mock_llm = mock.MagicMock()
    mock_llm.generate.return_value = llm_client.LLMOutput(
        text="```python\nreturn 1.0\n```"
    )
    mock_get_backend.return_value = mock_llm
    gen0 = _make_generation(
        0, {"agent_0": "code_0", "agent_1": "code_1"}, ["agent_0", "agent_1"]
    )
    generator = offspring_generator.IndependentGenerator(
        environment="cartpole",
        llm_config=self.llm_config,
        model_name="fake_model",
    )

    offspring = generator.generate_offspring(
        prev_gens=[gen0], max_new_offspring_count=None
    )

    self.assertLen(offspring, 2)

  @mock.patch.object(llm_client, "get_backend", autospec=True)
  def test_direct_mutation_generator_mutates_each_survivor_once(
      self, mock_get_backend
  ):
    mock_llm = mock.MagicMock()
    mock_llm.generate.return_value = llm_client.LLMOutput(
        text="```python\nreturn 7.0\n```"
    )
    mock_get_backend.return_value = mock_llm
    survivors = ["agent_0", "agent_1", "agent_2"]
    gen0 = _make_generation(
        0, {agent_id: f"code_{agent_id}" for agent_id in survivors}, survivors
    )
    generator = offspring_generator.DirectMutationGenerator(
        environment="cartpole",
        llm_config=self.llm_config,
        model_name="fake_model",
    )

    offspring = generator.generate_offspring(
        prev_gens=[gen0], max_new_offspring_count=None
    )

    self.assertCountEqual(
        [agent.parent_ids for agent in offspring],
        [["agent_0"], ["agent_1"], ["agent_2"]],
    )

  @mock.patch.object(llm_client, "get_backend", autospec=True)
  def test_direct_mutation_generator_no_survivors_passes_forbidden_codes(
      self, mock_get_backend
  ):
    mock_llm = mock.MagicMock()
    mock_llm.generate.return_value = llm_client.LLMOutput(
        text="```python\nreturn 42.0\n```"
    )
    mock_get_backend.return_value = mock_llm

    gen0 = _make_generation(0, {"agent_0": "forbidden_gen0"}, [])
    generator = offspring_generator.DirectMutationGenerator(
        environment="cartpole",
        llm_config=self.llm_config,
        model_name="fake_model",
        past_generations_to_look_at=2,
    )

    offspring = generator.generate_offspring(
        prev_gens=[gen0],
        max_new_offspring_count=1,
    )
    self.assertLen(offspring, 1)
    self.assertEqual(offspring[0].parent_ids, [])
    self.assertEqual(offspring[0].reward_code, "return 42.0")
    prompt = mock_llm.generate.call_args.kwargs["llm_input"].prompt
    self.assertIn("forbidden_gen0", prompt)

  @mock.patch.object(llm_client, "get_backend", autospec=True)
  def test_planned_mutation_generator_handles_parent_and_parentless_plans(
      self, mock_get_backend
  ):
    planner_llm = mock.MagicMock()
    mutator_llm = mock.MagicMock()
    initializer_llm = mock.MagicMock()

    def get_backend_side_effect(name, config):
      del name
      if config.role.endswith(":planner"):
        return planner_llm
      if config.role.endswith(":mutator"):
        return mutator_llm
      if config.role.endswith(":diverse_reward_initializer"):
        return initializer_llm
      raise ValueError(f"Unexpected role: {config.role}")

    mock_get_backend.side_effect = get_backend_side_effect

    planner_llm.generate.return_value = llm_client.LLMOutput(
        text=json.dumps({
            "offspring_plans": [
                {
                    "parent_ids": [],
                    "reasoning": "Explore fresh behavior",
                    "mutation_hint": "Try jumping",
                },
                {
                    "parent_ids": ["agent_0"],
                    "reasoning": "Refine survivor",
                    "mutation_hint": "Increase velocity weight",
                },
            ]
        })
    )
    initializer_llm.generate.return_value = llm_client.LLMOutput(
        text="```python\ndef compute_reward():\n  return 1.0\n```"
    )
    mutator_llm.generate.return_value = llm_client.LLMOutput(
        text="```python\ndef compute_reward():\n  return 2.0\n```"
    )

    gen0 = _make_generation(0, {"agent_0": "parent_code_0"}, ["agent_0"])
    generator = offspring_generator.PlannedMutationGenerator(
        environment="cartpole",
        llm_config=self.llm_config,
        model_name="fake_model",
        past_generations_to_look_at=2,
    )

    offspring = generator.generate_offspring(
        prev_gens=[gen0],
        max_new_offspring_count=2,
    )

    self.assertLen(offspring, 2)
    self.assertEqual(offspring[0].parent_ids, [])
    self.assertEqual(
        offspring[0].reward_code, "def compute_reward():\n  return 1.0"
    )
    self.assertEqual(offspring[1].parent_ids, ["agent_0"])
    self.assertEqual(
        offspring[1].reward_code, "def compute_reward():\n  return 2.0"
    )

    # Verify initializer received forbidden_codes from gen0
    init_prompt = initializer_llm.generate.call_args.kwargs["llm_input"].prompt
    self.assertIn("parent_code_0", init_prompt)

    # Verify mutator received both gen0 code and the newly initialized code in
    # forbidden_codes
    mutate_prompt = mutator_llm.generate.call_args.kwargs["llm_input"].prompt
    self.assertIn("parent_code_0", mutate_prompt)
    self.assertIn("def compute_reward():\n  return 1.0", mutate_prompt)

  @mock.patch.object(llm_client, "get_backend", autospec=True)
  def test_planned_mutation_generator_sanitizes_older_and_unknown_parents(
      self, mock_get_backend
  ):
    planner_llm = mock.MagicMock()
    mutator_llm = mock.MagicMock()
    initializer_llm = mock.MagicMock()

    def get_backend_side_effect(name, config):
      del name
      if config.role.endswith(":planner"):
        return planner_llm
      if config.role.endswith(":mutator"):
        return mutator_llm
      if config.role.endswith(":diverse_reward_initializer"):
        return initializer_llm
      raise ValueError(f"Unexpected role: {config.role}")

    mock_get_backend.side_effect = get_backend_side_effect

    planner_llm.generate.return_value = llm_client.LLMOutput(
        text=json.dumps({
            "offspring_plans": [
                {
                    "parent_ids": ["agent_from_gen0", "hallucinated_agent"],
                    "reasoning": "Use gen0 agent and drop hallucinated ID",
                    "mutation_hint": "Keep balance",
                },
                {
                    "parent_ids": ["completely_unknown_agent"],
                    "reasoning": "Falls back to parentless generation",
                    "mutation_hint": "Fresh exploration",
                },
            ]
        })
    )
    initializer_llm.generate.return_value = llm_client.LLMOutput(
        text="```python\ndef compute_reward():\n  return 10.0\n```"
    )
    mutator_llm.generate.return_value = llm_client.LLMOutput(
        text="```python\ndef compute_reward():\n  return 20.0\n```"
    )

    gen0 = _make_generation(
        0, {"agent_from_gen0": "code_gen0"}, ["agent_from_gen0"]
    )
    gen1 = _make_generation(
        1, {"agent_from_gen1": "code_gen1"}, ["agent_from_gen1"]
    )

    generator = offspring_generator.PlannedMutationGenerator(
        environment="cartpole",
        llm_config=self.llm_config,
        model_name="fake_model",
        past_generations_to_look_at=2,
    )

    offspring = generator.generate_offspring(
        prev_gens=[gen0, gen1],
        max_new_offspring_count=2,
    )

    self.assertLen(offspring, 2)
    # First returned agent is the one whose unknown parent was dropped to []
    self.assertEqual(offspring[0].parent_ids, [])
    self.assertEqual(
        offspring[0].reward_code, "def compute_reward():\n  return 10.0"
    )
    # Second returned agent kept the valid parent from gen0
    self.assertEqual(offspring[1].parent_ids, ["agent_from_gen0"])
    self.assertEqual(
        offspring[1].reward_code, "def compute_reward():\n  return 20.0"
    )

  @mock.patch.object(llm_client, "get_backend", autospec=True)
  def test_deduplicates_forbidden_codes_across_generations(
      self, mock_get_backend
  ):
    del mock_get_backend
    gen0 = _make_generation(
        0,
        {"agent_0": "code_0", "agent_1": "code_1"},
        ["agent_0"],
    )
    gen1 = _make_generation(
        1,
        {"agent_0": "code_0", "agent_2": "code_2"},
        ["agent_0", "agent_2"],
    )
    gen2 = _make_generation(
        2,
        {"agent_0": "code_0", "agent_2": "code_2", "agent_3": "code_3"},
        ["agent_0"],
    )
    generator = offspring_generator.IndependentGenerator(
        environment="cartpole",
        llm_config=self.llm_config,
        model_name="fake_model",
        past_generations_to_look_at=3,
    )

    forbidden = generator._last_n_generations_reward_codes(
        [gen0, gen1, gen2], 3
    )
    self.assertLen(forbidden, 4)
    self.assertCountEqual(forbidden, ["code_0", "code_1", "code_2", "code_3"])

  @mock.patch.object(llm_client, "get_backend", autospec=True)
  def test_planned_mutation_generator_empty_plan_requests_stop(
      self, mock_get_backend
  ):
    planner_llm = mock.MagicMock()
    other_llm = mock.MagicMock()
    mock_get_backend.side_effect = lambda name, config: (
        planner_llm if config.role.endswith(":planner") else other_llm
    )
    planner_llm.generate.side_effect = [
        llm_client.LLMOutput(text=json.dumps({"offspring_plans": []})),
        llm_client.LLMOutput(
            text=json.dumps({
                "offspring_plans": [{
                    "parent_ids": ["agent_0"],
                    "reasoning": "Refine survivor",
                    "mutation_hint": "Keep balance",
                }]
            })
        ),
    ]
    other_llm.generate.return_value = llm_client.LLMOutput(
        text="```python\ndef compute_reward():\n  return 1.0\n```"
    )
    gen0 = _make_generation(0, {"agent_0": "parent_code_0"}, ["agent_0"])
    generator = offspring_generator.PlannedMutationGenerator(
        environment="cartpole",
        llm_config=self.llm_config,
        model_name="fake_model",
    )
    self.assertFalse(generator.stop_requested)

    with self.subTest("empty_plan"):
      offspring = generator.generate_offspring([gen0], 2)
      self.assertEmpty(offspring)
      self.assertTrue(generator.stop_requested)

    with self.subTest("next_call_with_plans_resets_the_request"):
      offspring = generator.generate_offspring([gen0], 2)
      self.assertLen(offspring, 1)
      self.assertFalse(generator.stop_requested)

  @mock.patch.object(llm_client, "get_backend", autospec=True)
  def test_planned_mutation_generator_no_capacity_does_not_request_stop(
      self, mock_get_backend
  ):
    planner_llm = mock.MagicMock()
    mock_get_backend.return_value = planner_llm
    gen0 = _make_generation(0, {"agent_0": "parent_code_0"}, ["agent_0"])
    generator = offspring_generator.PlannedMutationGenerator(
        environment="cartpole",
        llm_config=self.llm_config,
        model_name="fake_model",
    )

    offspring = generator.generate_offspring([gen0], 0)

    self.assertEmpty(offspring)
    self.assertFalse(generator.stop_requested)
    planner_llm.generate.assert_not_called()


if __name__ == "__main__":
  absltest.main()
