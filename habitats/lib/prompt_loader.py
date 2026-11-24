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

"""Library for loading LLM prompts from external files."""

from collections.abc import Mapping
import dataclasses
import enum

from etils import epath

# `prompts/` is a data-only directory next to this package, so resolve it
# relative to the `habitats` package root.
_PROMPTS_SUBDIR = "prompts"

# Prompt files registered by callers outside this package (keyed by the same
# relative filename used in `_load_file`, e.g. "mutator/foo.md").
# Registered paths take precedence over the bundled `prompts/` directory.
_EXTRA_PROMPT_FILES: dict[str, epath.Path] = {}

# Environment descriptions registered by callers outside this package, keyed
# by environment name. Consulted when the name is not a member of
# `EnvironmentPrompt`.
_EXTRA_ENV_PROMPTS: dict[str, epath.Path] = {}


@dataclasses.dataclass(frozen=True)
class PromptInfo:
  """Information about an environment's prompt."""

  env_name: str
  filename: str


class EnvironmentPrompt(enum.Enum):
  """Enum mapping environments to their prompt info."""

  TIC_TAC_TOE = PromptInfo("tic_tac_toe", "environments/tictactoe.md")
  CARTPOLE = PromptInfo("cartpole", "environments/cartpole.md")
  CHEETAH = PromptInfo("cheetah", "environments/cheetah.md")
  HOPPER = PromptInfo("hopper", "environments/hopper.md")
  QUADRUPED = PromptInfo("quadruped", "environments/quadruped.md")
  HUMANOID = PromptInfo("humanoid", "environments/humanoid.md")
  MINI_CARD_DECK = PromptInfo(
      "mini_card_deck", "environments/mini_card_deck.md"
  )

  @classmethod
  def from_env_name(cls, env_name: str) -> "EnvironmentPrompt":
    for env in cls:
      if env.value.env_name == env_name:
        return env
    raise ValueError(f"Unknown environment: {env_name}")


def register_prompt_files(files: Mapping[str, epath.Path]) -> None:
  """Registers additional prompt files, keyed by relative prompt name.

  Lets code outside this package (e.g. a legacy or experimental track) supply
  prompts that the shared loaders resolve by name, such as
  `"mutator/diverse_initial.md"`, without bundling them here.

  Args:
    files: Mapping from relative prompt filename to the file's path.
  """
  _EXTRA_PROMPT_FILES.update(files)


def register_env_descriptions(envs: Mapping[str, epath.Path]) -> None:
  """Registers environment description prompts for extra environment names.

  Args:
    envs: Mapping from environment name to the description file's path.
  """
  _EXTRA_ENV_PROMPTS.update(envs)


def load_system_instruction(name: str) -> str:
  """Loads a system instruction prompt by name (e.g., 'mutator')."""
  if name == "mutator":
    return _load_file("mutator/system_instruction.md")
  raise ValueError(f"Unknown system instruction: {name}")


def load_mutator_prompt(name: str) -> str:
  """Loads a mutator prompt by name."""
  filename = f"mutator/{name}.md"
  return _load_file(filename)


def load_env_description(env_name: str) -> str:
  """Loads the environment description prompt for the given environment."""
  if env_name in _EXTRA_ENV_PROMPTS:
    return _read(_EXTRA_ENV_PROMPTS[env_name])
  try:
    env = EnvironmentPrompt.from_env_name(env_name)
  except ValueError as e:
    raise ValueError(f"Unknown environment: {env_name}") from e
  return _load_file(env.value.filename)


def load_population_initializer_prompt(name: str) -> str:
  """Loads a population initializer prompt by name."""
  return _load_file(f"population_initializer/{name}.md")


def load_offspring_generator_prompt(name: str) -> str:
  """Loads an offspring generator prompt by name."""
  return _load_file(f"offspring_generator/{name}.md")


def _load_file(filename: str) -> str:
  if filename in _EXTRA_PROMPT_FILES:
    return _read(_EXTRA_PROMPT_FILES[filename])
  return _read(epath.resource_path("habitats") / _PROMPTS_SUBDIR / filename)


def _read(path: epath.Path) -> str:
  return path.read_text(encoding="utf-8").strip()
