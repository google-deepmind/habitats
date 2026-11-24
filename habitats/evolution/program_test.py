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

"""Tests for the evolution program."""

import dataclasses
import json
from unittest import mock

from absl.testing import absltest
from habitats.config import evolution as evolution_config
from habitats.config import training
from habitats.evolution import evolution_algorithm
from habitats.evolution import evolution_algorithm_config
from habitats.evolution import program as program_lib
from habitats.training import training_pool as training_pool_lib


class ProgramTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.config = evolution_config.EvolutionConfig(
        evolution_algorithm_config=evolution_algorithm_config.EvolutionAlgorithmConfig(
            training_config=training.get_config("ppo_cartpole"),
            num_generations=2,
            initial_population_size=2,
            num_eval_episodes=2,
        ),
        run_id="run",
    )

  def test_evolution_algorithm_config_serialization_roundtrip(self):
    # The algorithm config must survive a round trip through its JSON dict form.
    original_config = self.config.evolution_algorithm_config
    json_str = json.dumps(dataclasses.asdict(original_config))
    restored_config = (
        evolution_algorithm_config.EvolutionAlgorithmConfig.from_json(json_str)
    )
    self.assertEqual(restored_config, original_config)

    q_config = evolution_algorithm_config.EvolutionAlgorithmConfig(
        training_config=training.get_config("q_learning_tic_tac_toe"),
        num_generations=5,
        initial_population_size=10,
    )
    json_str_q = json.dumps(dataclasses.asdict(q_config))
    restored_q = evolution_algorithm_config.EvolutionAlgorithmConfig.from_json(
        json_str_q
    )
    self.assertEqual(restored_q, q_config)

  def _coordinator(
      self, config: evolution_config.EvolutionConfig | None = None
  ) -> program_lib.EvolutionCoordinator:
    worker = mock.create_autospec(
        training_pool_lib.TrainingWorkerInterface, instance=True
    )
    return program_lib.EvolutionCoordinator(
        config=config or self.config, worker_clients=[worker]
    )

  def test_coordinator_runs_algorithm_and_stops_program(self):
    coordinator = self._coordinator()
    mock_algo = mock.MagicMock()

    with (
        mock.patch.object(
            evolution_algorithm, "EvolutionAlgorithm", return_value=mock_algo
        ) as mock_cls,
    ):
      coordinator.run()

    kwargs = mock_cls.call_args.kwargs
    self.assertEqual(kwargs["run_id"], "run")
    self.assertIsInstance(
        kwargs["training_pool"], training_pool_lib.TrainingPool
    )
    mock_algo.run.assert_called_once()

  def test_coordinator_failure_stops_program_and_reraises(self):
    coordinator = self._coordinator()
    mock_algo = mock.MagicMock()
    mock_algo.run.side_effect = RuntimeError("boom")

    with (
        mock.patch.object(
            evolution_algorithm, "EvolutionAlgorithm", return_value=mock_algo
        ),
    ):
      with self.assertRaisesRegex(RuntimeError, "boom"):
        coordinator.run()

  def test_random_run_id_when_unset(self):
    config = evolution_config.EvolutionConfig(
        evolution_algorithm_config=self.config.evolution_algorithm_config
    )
    coordinator = self._coordinator(config)

    with (
        mock.patch.object(
            evolution_algorithm, "EvolutionAlgorithm"
        ) as mock_cls,
    ):
      coordinator.run()

    self.assertLen(mock_cls.call_args.kwargs["run_id"], 16)


if __name__ == "__main__":
  absltest.main()
