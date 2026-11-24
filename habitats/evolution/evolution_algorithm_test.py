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

"""Tests for evolution_algorithm."""

import dataclasses
from unittest import mock

from absl.testing import absltest
from habitats.config import training
from habitats.evolution import evolution_algorithm
from habitats.evolution import evolution_algorithm_config
from habitats.evolution import population_initializer
from habitats.evolution import types
from habitats.lib import errors
from habitats.lib import llm_client
from habitats.training import training_interface
from habitats.training import training_pool as training_pool_lib
import numpy as np


def _agent(agent_id: str) -> types.AgentData:
  return types.AgentData(
      id=agent_id,
      reward_code="",
      generation=0,
      generalization_score=0.0,
      training_output=training_interface.TrainingOutput.empty(),
      reasoning="",
      embedding=[],
      path="",
  )


def _failed_agent(agent_id: str, error: str) -> types.AgentData:
  return dataclasses.replace(
      _agent(agent_id),
      training_output=training_interface.TrainingOutput.failure(error),
  )


def _generation(
    agent_ids: list[str], failed_agents: dict[str, str]
) -> types.GenerationData:
  return types.GenerationData(
      generation=0,
      fitness_grid=[],
      random_fitness=[],
      selected_agents=[],
      specialist_quality=0.0,
      behavioral_diversity=0.0,
      agents={agent_id: _agent(agent_id) for agent_id in agent_ids},
      generalization_scores={},
      failed_agents=failed_agents,
  )


class ComputeGenerationMetricsTest(absltest.TestCase):

  def test_computes_population_and_fitness_metrics(self):
    gen_data = types.GenerationData(
        generation=3,
        fitness_grid=[[10.0, 2.0], [4.0, 6.0]],
        random_fitness=[1.0, 3.0],
        selected_agents=["agent_0"],
        specialist_quality=0.75,
        behavioral_diversity=0.25,
        agents={"agent_0": _agent("agent_0"), "agent_1": _agent("agent_1")},
        generalization_scores={"agent_0": 0.8, "agent_1": 0.4},
        failed_agents={"agent_2": "TrainingDivergedError: NaN parameters."},
    )

    scalars = evolution_algorithm.compute_generation_metrics(
        gen_data,
        norm_fitness_grid=np.array([[1.0, 0.0], [0.5, 1.0]]),
        norm_random_fitness=np.array([0.0, 0.5]),
        num_trained=3,
    )

    expected = {
        "specialist_quality": 0.75,
        "behavioral_diversity": 0.25,
        "num_agents": 2.0,
        "num_survivors": 1.0,
        "num_trained": 3.0,
        "num_failed": 1.0,
        "survival_rate": 0.5,
        "generalization_mean": 0.6,
        "generalization_max": 0.8,
        "generalization_min": 0.4,
        "specialist_margin_mean": 0.75,
        "specialist_margin_min": 0.5,
        "self_fitness_mean": 8.0,
        "self_fitness_max": 10.0,
        "random_fitness_mean": 2.0,
    }
    self.assertSameElements(scalars.keys(), expected.keys())
    for key, value in expected.items():
      self.assertAlmostEqual(scalars[key], value, msg=key)

  def test_empty_generation(self):
    gen_data = types.GenerationData(
        generation=0,
        fitness_grid=[],
        random_fitness=[],
        selected_agents=[],
        specialist_quality=0.0,
        behavioral_diversity=0.0,
        agents={},
        generalization_scores={},
    )

    scalars = evolution_algorithm.compute_generation_metrics(
        gen_data,
        norm_fitness_grid=np.zeros((0, 0)),
        norm_random_fitness=np.zeros(0),
        num_trained=0,
    )

    self.assertEqual(
        scalars, {"num_agents": 0.0, "num_trained": 0.0, "num_failed": 0.0}
    )


class DropFailedAgentsTest(absltest.TestCase):

  def test_drops_failed_agents_and_keeps_order(self):
    agents = {
        "agent_0": _agent("agent_0"),
        "agent_1": _failed_agent("agent_1", "RewardError: Bad reward."),
        "agent_2": _agent("agent_2"),
        "agent_3": _failed_agent("agent_3", "TrainingDivergedError: NaN."),
    }

    trained, failed = evolution_algorithm.drop_failed_agents(
        agents, generation=1
    )

    self.assertEqual(list(trained), ["agent_0", "agent_2"])
    self.assertEqual(
        failed,
        {
            "agent_1": "RewardError: Bad reward.",
            "agent_3": "TrainingDivergedError: NaN.",
        },
    )

  def test_keeps_agents_without_failures(self):
    agents = {"agent_0": _agent("agent_0"), "agent_1": _agent("agent_1")}

    trained, failed = evolution_algorithm.drop_failed_agents(
        agents, generation=1
    )

    self.assertEqual(trained, agents)
    self.assertEmpty(failed)

  def test_raises_if_training_failed_for_every_agent(self):
    agents = {"agent_0": _failed_agent("agent_0", "RewardError: Bad reward.")}

    with self.assertRaisesRegex(
        errors.NoTrainedAgentsError,
        "Training failed for all 1 agents of generation 2",
    ):
      evolution_algorithm.drop_failed_agents(agents, generation=2)


class SelectSurvivorsTest(absltest.TestCase):

  def test_selects_distinct_specialists_that_beat_random(self):
    selected = evolution_algorithm.select_survivors(
        agent_ids=["agent_0", "agent_1"],
        norm_fitness_grid=np.array([[1.0, 0.0], [0.0, 1.0]]),
        norm_random_fitness=np.array([0.0, 0.0]),
    )
    self.assertEqual(selected, ["agent_0", "agent_1"])

  def test_breaks_ties_in_favor_of_incumbent_survivor_preventing_fratricide(
      self,
  ):
    selected = evolution_algorithm.select_survivors(
        agent_ids=["agent_0", "agent_1"],
        norm_fitness_grid=np.array([[1.0, 1.0], [1.0, 1.0]]),
        norm_random_fitness=np.array([0.0, 0.0]),
    )
    self.assertEqual(selected, ["agent_0"])

  def test_rejects_agents_not_strictly_beating_random_with_fallback(self):
    selected = evolution_algorithm.select_survivors(
        agent_ids=["agent_0", "agent_1"],
        norm_fitness_grid=np.array([[1.0, 1.0], [1.0, 1.0]]),
        norm_random_fitness=np.array([1.0, 1.0]),
    )
    self.assertEqual(selected, ["agent_0"])


class NextAgentIndexTest(absltest.TestCase):

  def test_no_generations(self):
    self.assertEqual(evolution_algorithm.next_agent_index([]), 0)

  def test_follows_all_agents(self):
    generations = [
        _generation(["agent_0", "agent_1"], failed_agents={}),
        _generation(["agent_1", "agent_3"], failed_agents={}),
    ]

    self.assertEqual(evolution_algorithm.next_agent_index(generations), 4)

  def test_counts_failed_agents(self):
    generations = [
        _generation(["agent_0", "agent_1"], failed_agents={}),
        _generation(["agent_1"], failed_agents={"agent_2": "RewardError."}),
    ]

    self.assertEqual(evolution_algorithm.next_agent_index(generations), 3)


class EvolvePopulationTest(absltest.TestCase):

  def test_preserves_survivors_and_assigns_offspring_ids(self):
    config = evolution_algorithm_config.EvolutionAlgorithmConfig(
        training_config=training.get_config("test"),
        max_population_size=3,
        max_new_offspring_per_generation=5,
    )
    mock_worker = mock.create_autospec(
        training_pool_lib.TrainingWorkerInterface, instance=True
    )
    pool = training_pool_lib.TrainingPool([mock_worker])

    with mock.patch.object(llm_client, "get_backend", autospec=True):
      algo = evolution_algorithm.EvolutionAlgorithm(
          config=config,
          run_id="test_run",
          training_pool=pool,
      )

    survivor = dataclasses.replace(_agent("agent_0"), reward_code="return 1.0")
    non_survivor = dataclasses.replace(
        _agent("agent_1"), reward_code="return 0.0"
    )
    gen0 = types.GenerationData(
        generation=0,
        fitness_grid=[[1.0, 0.0], [0.0, 0.5]],
        random_fitness=[0.0, 0.0],
        selected_agents=["agent_0"],
        specialist_quality=1.0,
        behavioral_diversity=0.5,
        agents={"agent_0": survivor, "agent_1": non_survivor},
        generalization_scores={"agent_0": 0.5, "agent_1": 0.25},
    )
    algo.generations_data = [gen0]
    algo._next_agent_id = 2

    new_offspring = dataclasses.replace(
        _agent(""),
        reward_code="return 2.0",
        generation=1,
        parent_ids=["agent_0"],
    )
    with mock.patch.object(
        algo.offspring_generator,
        "generate_offspring",
        return_value=[new_offspring],
    ) as mock_generate:
      next_agents = algo.evolve_population()

    # max_population_size (3) - 1 survivor = 2 max new offspring
    mock_generate.assert_called_once_with([gen0], 2)
    self.assertEqual(list(next_agents.keys()), ["agent_0", "agent_2"])
    self.assertEqual(next_agents["agent_0"].id, "agent_0")
    self.assertEqual(next_agents["agent_0"].generation, 1)
    self.assertEqual(next_agents["agent_0"].reward_code, "return 1.0")
    self.assertEqual(next_agents["agent_2"].id, "agent_2")
    self.assertEqual(next_agents["agent_2"].generation, 1)
    self.assertEqual(next_agents["agent_2"].reward_code, "return 2.0")

  def test_uses_diverse_reward_initializer_for_initial_population(self):
    config = evolution_algorithm_config.EvolutionAlgorithmConfig(
        training_config=training.get_config("test"),
        initial_population_size=2,
        population_initializer=(
            population_initializer.PopulationInitializerName.DIVERSE_REWARD
        ),
    )
    mock_worker = mock.create_autospec(
        training_pool_lib.TrainingWorkerInterface, instance=True
    )
    pool = training_pool_lib.TrainingPool([mock_worker])

    with mock.patch.object(llm_client, "get_backend", autospec=True):
      algo = evolution_algorithm.EvolutionAlgorithm(
          config=config,
          run_id="test_run",
          training_pool=pool,
      )

    self.assertIsInstance(
        algo.population_initializer,
        population_initializer.DiverseRewardInitializer,
    )
    with mock.patch.object(
        algo.population_initializer,
        "init_population",
        return_value=["return 1.0", "return 2.0"],
    ) as mock_init:
      agents = algo.initialize_population()

    mock_init.assert_called_once_with(2)
    self.assertEqual(list(agents.keys()), ["agent_0", "agent_1"])
    self.assertEqual(agents["agent_0"].reward_code, "return 1.0")
    self.assertEqual(agents["agent_1"].reward_code, "return 2.0")

  def test_zero_offspring_requests_equilibrium_stop(self):
    config = evolution_algorithm_config.EvolutionAlgorithmConfig(
        training_config=training.get_config("test"),
        max_new_offspring_per_generation=2,
    )
    mock_worker = mock.create_autospec(
        training_pool_lib.TrainingWorkerInterface, instance=True
    )
    pool = training_pool_lib.TrainingPool([mock_worker])

    with mock.patch.object(llm_client, "get_backend", autospec=True):
      algo = evolution_algorithm.EvolutionAlgorithm(
          config=config,
          run_id="test_run",
          training_pool=pool,
      )

    survivor = dataclasses.replace(_agent("agent_0"), reward_code="return 1.0")
    gen0 = types.GenerationData(
        generation=0,
        fitness_grid=[[1.0]],
        random_fitness=[0.0],
        selected_agents=["agent_0"],
        specialist_quality=1.0,
        behavioral_diversity=0.0,
        agents={"agent_0": survivor},
        generalization_scores={"agent_0": 1.0},
    )
    algo.generations_data = [gen0]
    algo._next_agent_id = 1

    self.assertFalse(algo.check_equilibrium())
    with mock.patch.object(
        algo.offspring_generator,
        "generate_offspring",
        return_value=[],
    ):
      algo.evolve_population()

    self.assertTrue(algo.check_equilibrium())


class RunTest(absltest.TestCase):

  def _make_algorithm(
      self,
  ) -> tuple[evolution_algorithm.EvolutionAlgorithm, mock.MagicMock]:
    config = evolution_algorithm_config.EvolutionAlgorithmConfig(
        training_config=training.get_config("test"),
        num_generations=3,
        initial_population_size=2,
    )
    pool = mock.create_autospec(training_pool_lib.TrainingPool, instance=True)
    pool.train_agents.side_effect = lambda agents, base_checkpoint_path: agents

    def evaluate_agents(agents, prev_gen):
      del prev_gen
      n = len(agents)
      # Each policy is best on its own reward and beats the random policy.
      return np.eye(n) + 1.0, np.zeros(n)

    pool.evaluate_agents.side_effect = evaluate_agents
    with mock.patch.object(llm_client, "get_backend", autospec=True):
      algo = evolution_algorithm.EvolutionAlgorithm(
          config=config,
          run_id="test_run",
          training_pool=pool,
      )
    return algo, pool

  def test_stops_right_after_planner_requests_it(self):
    algo, pool = self._make_algorithm()

    def generate_offspring(prev_gens, max_new_offspring_count):
      del prev_gens, max_new_offspring_count
      algo.offspring_generator.stop_requested = True
      return []

    with mock.patch.object(
        algo.offspring_generator,
        "generate_offspring",
        side_effect=generate_offspring,
    ) as mock_generate:
      algo.run()

    # Generation 0 is trained and evaluated, the planner asks to stop, and the
    # unchanged survivors are not trained or evaluated again.
    mock_generate.assert_called_once()
    pool.train_agents.assert_called_once()
    pool.evaluate_agents.assert_called_once()
    self.assertLen(algo.generations_data, 1)
    self.assertEqual(algo.status, "completed")
    self.assertEqual(algo.current_generation, 1)

  def test_runs_all_generations_without_stop_request(self):
    algo, pool = self._make_algorithm()

    # One new offspring per generation keeps the run going (no offspring would
    # request a stop).
    with mock.patch.object(
        algo.offspring_generator,
        "generate_offspring",
        return_value=[_agent("offspring")],
    ) as mock_generate:
      algo.run()

    self.assertEqual(mock_generate.call_count, 2)
    self.assertEqual(pool.evaluate_agents.call_count, 3)
    self.assertLen(algo.generations_data, 3)
    self.assertEqual(algo.status, "completed")
    self.assertEqual(algo.current_generation, 3)


if __name__ == "__main__":
  absltest.main()
