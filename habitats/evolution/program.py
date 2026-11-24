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

"""Evolution program: runs the evolution algorithm with a training pool.

A program consists of `num_workers` `TrainingWorker` nodes and one 
`EvolutionCoordinator` node that runs the `EvolutionAlgorithm` against them.
"""

from collections.abc import Sequence
import dataclasses
import uuid

from absl import logging
from habitats.config import evolution as evolution_config
from habitats.evolution import evolution_algorithm
from habitats.training import training_pool as training_pool_lib


class EvolutionCoordinator:
  """Coordinator node: runs the evolution loop on the worker pool."""

  def __init__(
      self,
      config: evolution_config.EvolutionConfig,
      worker_clients: Sequence[training_pool_lib.LocalTrainingWorkerInterface],
  ):
    self._config = config
    self._worker_clients = list(worker_clients)

  def _resolve_run_id(self) -> str:
    """Returns the run ID to use for this coordinator's run."""
    return self._config.run_id or uuid.uuid4().hex[:16]

  def _create_algorithm(
      self,
      config: evolution_config.EvolutionConfig,
      training_pool: training_pool_lib.TrainingPool,
  ) -> evolution_algorithm.EvolutionAlgorithm:
    """Creates and prepares the `EvolutionAlgorithm` instance."""
    return evolution_algorithm.EvolutionAlgorithm(
        config=config.evolution_algorithm_config,
        run_id=config.run_id,
        api_key=config.api_key,
        training_pool=training_pool,
    )

  def _stop_on_failure(self, message: str) -> None:
    """Stops the running program, if any, after a coordinator error."""
    del message

  def run(self) -> None:
    """Runs the evolution loop using the worker pool."""
    logging.info(
        "Starting EvolutionCoordinator with %d worker clients.",
        len(self._worker_clients),
    )
    try:
      config = dataclasses.replace(self._config, run_id=self._resolve_run_id())
      logging.info("Resolved run ID: %s", config.run_id)
      training_pool = training_pool_lib.TrainingPool(self._worker_clients)
      algo = self._create_algorithm(config, training_pool)
      logging.info(
          "Evolution algorithm initialized. Starting evolution loop..."
      )
      algo.run()
    except Exception as e:  # pylint: disable=broad-except
      logging.exception("An error occurred during evolution: %s", e)
      self._stop_on_failure(f"Evolution failed: {type(e).__name__}: {e}")
      raise
    logging.info("Evolution loop completed successfully.")

  def __call__(self) -> None:
    return self.run()
