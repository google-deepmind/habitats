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

"""Runs a Habitats evolution experiment locally.

Launches one coordinator and `--num_workers` training workers, e.g.:

  evolve --config=habitats_test --checkpoint_path=/tmp/habitats/ckpt \\
      --api_key=$GEMINI_API_KEY

Each agent's reward code and checkpoints end up under
`<checkpoint_path>/<run_id>/<agent_id>/`; per-generation metrics are logged.
"""

from collections.abc import Sequence
import dataclasses
import os

from absl import app
from absl import flags
from absl import logging
from habitats.config import evolution
from habitats.config import sweep
from habitats.evolution import program as program_lib
from habitats.training import training_pool

_CONFIG = flags.DEFINE_string(
    "config",
    None,
    "Name of the registered evolution config to run.",
    required=True,
)
_CHECKPOINT_PATH = flags.DEFINE_string(
    "checkpoint_path",
    None,
    "Directory for agent checkpoints (one subdirectory per run ID).",
    required=True,
)
_API_KEY = flags.DEFINE_string(
    "api_key",
    None,
    "Gemini API key for the 'genai' LLM backend. Defaults to the"
    " GEMINI_API_KEY environment variable.",
)
_LLM_BACKEND = flags.DEFINE_string(
    "llm_backend",
    None,
    "LLM backend name registered in habitats.lib.llm_client. Defaults to the"
    " config's value ('genai' unless overridden).",
)
_RUN_ID = flags.DEFINE_string(
    "run_id",
    None,
    "Run ID, used as the checkpoint subdirectory. A random one is generated"
    " if unset.",
)
_NUM_WORKERS = flags.DEFINE_integer(
    "num_workers", 2, "Number of training workers."
)
_SWEEP = flags.DEFINE_multi_string(
    "sweep",
    [],
    "Sweep axis as '<dotted.path>=<v1>,<v2>,...' over `EvolutionConfig`"
    " fields. Must expand to a single work unit (worker) for a local launch.",
)
_OVERRIDE = flags.DEFINE_multi_string(
    "override",
    [],
    "Override a field of the evolution config, e.g."
    " 'evolution_algorithm_config.num_generations=3'. Can be repeated.",
)


def _configure_launch(
    config: evolution.EvolutionConfig,
) -> evolution.EvolutionConfig:
  """Populates launch-time settings on `config` from CLI flags."""
  algo_config = config.evolution_algorithm_config
  if _LLM_BACKEND.value is not None:
    algo_config = dataclasses.replace(
        algo_config, llm_backend=_LLM_BACKEND.value
    )
  training_config = dataclasses.replace(
      algo_config.training_config,
      training_interface_config=dataclasses.replace(
          algo_config.training_config.training_interface_config,
          checkpoint_path=_CHECKPOINT_PATH.value,
      ),
  )
  return dataclasses.replace(
      config,
      api_key=_API_KEY.value or os.environ.get("GEMINI_API_KEY"),
      run_id=_RUN_ID.value or "",
      evolution_algorithm_config=dataclasses.replace(
          algo_config, training_config=training_config
      ),
  )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  config = _configure_launch(evolution.get_config(_CONFIG.value))
  work_units, _ = sweep.expand_configs(
      {_CONFIG.value: config}, _SWEEP.value, _OVERRIDE.value
  )
  if len(work_units) != 1:
    raise app.UsageError(
        f"Configuration yields {len(work_units)} work units, but a local"
        " launch runs a single program; drop or narrow --sweep."
    )
  work_unit = work_units[0]
  logging.info(
      "Launching evolution config '%s' with %d workers, params=%s",
      work_unit.config_name,
      _NUM_WORKERS.value,
      work_unit.display_params,
  )
  if _NUM_WORKERS.value <= 0:
    raise ValueError("num_workers must be positive.")
  training_cfg = work_unit.config.evolution_algorithm_config.training_config
  workers = [
      training_pool.TrainingWorker(training_cfg)
      for _ in range(_NUM_WORKERS.value)
  ]
  program_lib.EvolutionCoordinator(work_unit.config, workers).run()


if __name__ == "__main__":
  app.run(main)
