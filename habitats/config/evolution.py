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

"""Configuration for Habitats environments."""

from collections.abc import Callable
import dataclasses

from habitats.config import training
from habitats.evolution import evolution_algorithm_config
from habitats.evolution import offspring_generator as og
from habitats.evolution import population_initializer

@dataclasses.dataclass(frozen=True)
class EvolutionConfig:
  """Represents an experiment config."""

  evolution_algorithm_config: (
      evolution_algorithm_config.EvolutionAlgorithmConfig
  )
  run_id: str = ''
  api_key: str | None = None


_EVOLUTION_REGISTRY: dict[str, Callable[..., EvolutionConfig]] = {}


def register(
    f: Callable[..., EvolutionConfig],
) -> Callable[..., EvolutionConfig]:
  """Helper function to access local registry."""
  name = getattr(f, '__name__', None)
  if name is None:
    raise ValueError(f'Function {f} has no __name__.')

  if name in _EVOLUTION_REGISTRY:
    raise ValueError(f'{name} is already registered.')
  _EVOLUTION_REGISTRY[name] = f
  return f


def get_config(name: str) -> EvolutionConfig:
  """Returns the evolution config for the given name."""
  return _EVOLUTION_REGISTRY[name]()


# ------------------------------------------------------------------------------
# Presets
# ------------------------------------------------------------------------------


@register
def habitats() -> EvolutionConfig:
  """Returns the base Habitats evolution config preset."""
  return EvolutionConfig(
      evolution_algorithm_config=evolution_algorithm_config.EvolutionAlgorithmConfig(
          training_config=training.get_config('test'),
          num_generations=10,
          initial_population_size=1,
          num_eval_episodes=10,
          offspring_generator=og.OffspringGeneratorName.PLANNED_MUTATION,
      ),
  )


# ------------------------------------------------------------------------------
# Mini Card Deck
# ------------------------------------------------------------------------------


@register
def habitats_mini_card_deck() -> EvolutionConfig:
  """Returns the Habitats config for mini card deck."""
  preset = get_config('habitats')
  return dataclasses.replace(
      preset,
      evolution_algorithm_config=dataclasses.replace(
          preset.evolution_algorithm_config,
          training_config=training.get_config('q_learning_mini_card_deck'),
          num_generations=30,
          initial_population_size=3,
          max_new_offspring_per_generation=10,
          population_initializer=(
              population_initializer.PopulationInitializerName.SAME_REWARD
          ),
      ),
  )


# ------------------------------------------------------------------------------
# Hopper
# ------------------------------------------------------------------------------


@register
def habitats_hopper() -> EvolutionConfig:
  """Returns the Habitats config for hopper."""
  preset = get_config('habitats')
  return dataclasses.replace(
      preset,
      evolution_algorithm_config=dataclasses.replace(
          preset.evolution_algorithm_config,
          training_config=training.get_config('ppo_hopper'),
      ),
  )


# ------------------------------------------------------------------------------
# Cheetah
# ------------------------------------------------------------------------------


@register
def habitats_cheetah() -> EvolutionConfig:
  """Returns the Habitats config for cheetah."""
  preset = get_config('habitats')
  return dataclasses.replace(
      preset,
      evolution_algorithm_config=dataclasses.replace(
          preset.evolution_algorithm_config,
          training_config=training.get_config('ppo_cheetah'),
      ),
  )


# ------------------------------------------------------------------------------
# Quadruped
# ------------------------------------------------------------------------------


@register
def habitats_quadruped() -> EvolutionConfig:
  """Returns the Habitats config for quadruped."""
  preset = get_config('habitats')
  return dataclasses.replace(
      preset,
      evolution_algorithm_config=dataclasses.replace(
          preset.evolution_algorithm_config,
          training_config=training.get_config('ppo_quadruped'),
      ),
  )


@register
def direct_mutation_quadruped() -> EvolutionConfig:
  """Returns the direct mutation config for quadruped."""
  preset = get_config('habitats')
  return dataclasses.replace(
      preset,
      evolution_algorithm_config=dataclasses.replace(
          preset.evolution_algorithm_config,
          training_config=training.get_config('ppo_quadruped'),
          offspring_generator=og.OffspringGeneratorName.DIRECT_MUTATION,
      ),
  )


@register
def independent_quadruped() -> EvolutionConfig:
  """Returns the independent config for quadruped."""
  preset = get_config('habitats')
  return dataclasses.replace(
      preset,
      evolution_algorithm_config=dataclasses.replace(
          preset.evolution_algorithm_config,
          training_config=training.get_config('ppo_quadruped'),
          offspring_generator=og.OffspringGeneratorName.INDEPENDENT,
      ),
  )


# ------------------------------------------------------------------------------
# Cartpole
# ------------------------------------------------------------------------------


@register
def habitats_cartpole() -> EvolutionConfig:
  """Returns the Habitats config for cartpole."""
  preset = get_config('habitats')
  return dataclasses.replace(
      preset,
      evolution_algorithm_config=dataclasses.replace(
          preset.evolution_algorithm_config,
          training_config=training.get_config('ppo_cartpole'),
      ),
  )


# ------------------------------------------------------------------------------
# Humanoid
# ------------------------------------------------------------------------------


@register
def habitats_humanoid() -> EvolutionConfig:
  """Returns the Habitats config for humanoid."""
  preset = get_config('habitats')
  return dataclasses.replace(
      preset,
      evolution_algorithm_config=dataclasses.replace(
          preset.evolution_algorithm_config,
          training_config=training.get_config('ppo_humanoid'),
      ),
  )


# ------------------------------------------------------------------------------
# Test
# ------------------------------------------------------------------------------


@register
def habitats_test() -> EvolutionConfig:
  """Returns the Habitats config for test."""
  preset = get_config('habitats')
  return dataclasses.replace(
      preset,
      evolution_algorithm_config=dataclasses.replace(
          preset.evolution_algorithm_config,
          training_config=training.get_config('test'),
          num_generations=3,
          initial_population_size=2,
          num_eval_episodes=2,
      ),
  )
