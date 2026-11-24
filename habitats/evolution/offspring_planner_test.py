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

"""Tests for the offspring planner."""

import json
from unittest import mock

from absl.testing import absltest
from habitats.evolution import offspring_planner
from habitats.evolution import types
from habitats.lib import llm_client
from habitats.training import training_interface

def _make_agent(
    agent_id: str, code: str, generation: int = 1
) -> types.AgentData:
  """Creates a test AgentData."""
  return types.AgentData(
      id=agent_id,
      reward_code=code,
      generation=generation,
      generalization_score=0.0,
      training_output=training_interface.TrainingOutput.empty(),
      reasoning="test",
      embedding=[],
      path="",
      parent_ids=[],
  )


class OffspringPlannerTest(absltest.TestCase):

  @mock.patch.object(llm_client, "get_backend", autospec=True)
  def test_plan_generation_basic_and_fenced_json(self, mock_get_backend):
    mock_llm = mock.MagicMock()
    mock_llm.generate.return_value = llm_client.LLMOutput(
        text=(
            "```json\n"
            + json.dumps({
                "offspring_plans": [
                    {
                        "parent_ids": ["agent_0"],
                        "reasoning": "Test reasoning 1",
                        "mutation_hint": "Test hint 1",
                    },
                    {
                        "parent_ids": ["agent_0", "agent_1"],
                        "reasoning": "Test reasoning 2",
                        "mutation_hint": "Test hint 2",
                    },
                ]
            })
            + "\n```"
        )
    )
    mock_get_backend.return_value = mock_llm

    planner = offspring_planner.OffspringPlanner(
        environment="cartpole",
        model_name="fake_model",
        llm_config=llm_client.LLMConfig(role="test"),
        llm_backend="genai",
        past_generations_to_look_at=2,
    )

    agents = {
        "agent_0": _make_agent("agent_0", "return 1.0"),
        "agent_1": _make_agent("agent_1", "return 2.0"),
    }
    gen_data = types.GenerationData(
        generation=1,
        agents=agents,
        fitness_grid=[[1.0, 0.0], [0.0, 1.0]],
        random_fitness=[0.1, 0.1],
        generalization_scores={"agent_0": 0.75, "agent_1": 0.25},
        selected_agents=["agent_0", "agent_1"],
        specialist_quality=0.9,
        behavioral_diversity=0.8,
    )

    plans = planner.plan_generation(
        prev_gens=[gen_data],
        max_new_offspring_count=4,
    )

    self.assertLen(plans, 2)
    self.assertEqual(plans[0].parent_ids, ["agent_0"])
    self.assertEqual(plans[0].reasoning, "Test reasoning 1")
    self.assertEqual(plans[0].mutation_hint, "Test hint 1")
    self.assertEqual(plans[1].parent_ids, ["agent_0", "agent_1"])

    mock_llm.generate.assert_called_once()
    llm_input = mock_llm.generate.call_args.kwargs["llm_input"]
    self.assertEqual(llm_input.model, "fake_model")
    self.assertEqual(llm_input.response_mime_type, "application/json")
    self.assertIn("Generalization Score: 0.7500", llm_input.prompt)

  @mock.patch.object(llm_client, "get_backend", autospec=True)
  def test_plan_generation_invalid_json_raises_error(self, mock_get_backend):
    mock_llm = mock.MagicMock()
    mock_llm.generate.return_value = llm_client.LLMOutput(text="invalid json")
    mock_get_backend.return_value = mock_llm

    planner = offspring_planner.OffspringPlanner(
        environment="cartpole",
        model_name="fake_model",
        llm_config=llm_client.LLMConfig(role="test"),
    )

    gen_data = types.GenerationData(
        generation=0,
        agents={"agent_0": _make_agent("agent_0", "return 1.0", generation=0)},
        fitness_grid=[[1.0]],
        random_fitness=[0.0],
        generalization_scores={"agent_0": 0.5},
        selected_agents=["agent_0"],
        specialist_quality=1.0,
        behavioral_diversity=0.0,
    )

    with self.assertRaises(offspring_planner.OffspringPlanningError):
      planner.plan_generation(prev_gens=[gen_data], max_new_offspring_count=2)

  @mock.patch.object(llm_client, "get_backend", autospec=True)
  def test_plan_generation_without_limit_omits_offspring_limit(
      self, mock_get_backend
  ):
    mock_llm = mock.MagicMock()
    mock_llm.generate.return_value = llm_client.LLMOutput(
        text=json.dumps({"offspring_plans": []})
    )
    mock_get_backend.return_value = mock_llm
    planner = offspring_planner.OffspringPlanner(
        environment="cartpole",
        model_name="fake_model",
        llm_config=llm_client.LLMConfig(role="test"),
    )
    gen_data = types.GenerationData(
        generation=0,
        agents={"agent_0": _make_agent("agent_0", "return 1.0", generation=0)},
        fitness_grid=[[1.0]],
        random_fitness=[0.0],
        generalization_scores={"agent_0": 0.5},
        selected_agents=["agent_0"],
        specialist_quality=1.0,
        behavioral_diversity=0.0,
    )

    planner.plan_generation(prev_gens=[gen_data], max_new_offspring_count=None)

    prompt = mock_llm.generate.call_args.kwargs["llm_input"].prompt
    self.assertNotIn("at most", prompt)

  @mock.patch.object(llm_client, "get_backend", autospec=True)
  def test_fitness_grid_normalizes_columns_using_random_fitness(
      self, mock_get_backend
  ):
    mock_llm = mock.MagicMock()
    mock_llm.generate.return_value = llm_client.LLMOutput(
        text=json.dumps({"offspring_plans": []})
    )
    mock_get_backend.return_value = mock_llm

    planner = offspring_planner.OffspringPlanner(
        environment="cartpole",
        model_name="fake_model",
        llm_config=llm_client.LLMConfig(role="test"),
    )

    agents = {
        "agent_0": _make_agent("agent_0", "return 1.0"),
        "agent_1": _make_agent("agent_1", "return 2.0"),
    }
    gen_data = types.GenerationData(
        generation=1,
        agents=agents,
        fitness_grid=[[10.0, 100.0], [20.0, 50.0]],
        random_fitness=[0.0, 0.0],
        generalization_scores={"agent_0": 0.75, "agent_1": 0.75},
        selected_agents=["agent_0", "agent_1"],
        specialist_quality=0.5,
        behavioral_diversity=0.2,
    )

    planner.plan_generation(prev_gens=[gen_data], max_new_offspring_count=2)
    prompt = mock_llm.generate.call_args.kwargs["llm_input"].prompt
    self.assertIn("0.5000", prompt)
    self.assertIn("1.0000", prompt)
    self.assertNotIn("100.0000", prompt)

  @mock.patch.object(llm_client, "get_backend", autospec=True)
  def test_generation_report_prints_reward_code_only_once_per_agent(
      self, mock_get_backend
  ):
    mock_llm = mock.MagicMock()
    mock_llm.generate.return_value = llm_client.LLMOutput(
        text=json.dumps({"offspring_plans": []})
    )
    mock_get_backend.return_value = mock_llm

    planner = offspring_planner.OffspringPlanner(
        environment="cartpole",
        model_name="fake_model",
        llm_config=llm_client.LLMConfig(role="test"),
        past_generations_to_look_at=2,
    )

    gen0 = types.GenerationData(
        generation=0,
        agents={
            "agent_0": _make_agent("agent_0", "return 1.0", generation=0),
        },
        fitness_grid=[[1.0]],
        random_fitness=[0.0],
        generalization_scores={"agent_0": 1.0},
        selected_agents=["agent_0"],
        specialist_quality=1.0,
        behavioral_diversity=0.0,
    )
    gen1 = types.GenerationData(
        generation=1,
        agents={
            "agent_0": _make_agent("agent_0", "return 1.0", generation=1),
            "agent_1": _make_agent("agent_1", "return 2.0", generation=1),
        },
        fitness_grid=[[1.0, 0.0], [0.0, 1.0]],
        random_fitness=[0.0, 0.0],
        generalization_scores={"agent_0": 0.5, "agent_1": 0.5},
        selected_agents=["agent_0", "agent_1"],
        specialist_quality=1.0,
        behavioral_diversity=1.0,
    )

    planner.plan_generation(prev_gens=[gen0, gen1], max_new_offspring_count=2)
    prompt = mock_llm.generate.call_args.kwargs["llm_input"].prompt
    self.assertEqual(prompt.count("return 1.0"), 1)
    self.assertEqual(prompt.count("return 2.0"), 1)
    self.assertIn(
        "agent_0 (Survivor, Generalization Score: 1.0000): return 1.0\n",
        prompt,
    )
    self.assertIn(
        "agent_0 (Survivor, Generalization Score: 0.5000)\n",
        prompt,
    )
    self.assertIn(
        "agent_1 (Survivor, Generalization Score: 0.5000): return 2.0\n",
        prompt,
    )


if __name__ == "__main__":
  absltest.main()
