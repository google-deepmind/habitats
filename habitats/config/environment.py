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

"""Environment registry."""

from collections.abc import Callable
import enum

import dm_env
from habitats.environment import mini_card_deck_env
from habitats.environment import mjx_env
from habitats.environment import tictactoe_env
from habitats.lib import class_registry


class EnvironmentName(enum.StrEnum):
  """Represents the name of the environment."""

  CARTPOLE = "cartpole"
  TIC_TAC_TOE = "tic_tac_toe"
  MINI_CARD_DECK = "mini_card_deck"
  ASCENDING_SORT_GAME = "ascending_sort_game"
  DESCENDING_SORT_GAME = "descending_sort_game"
  TRIPLET_MATCHING_GAME = "triplet_matching_game"
  TARGET_SUM_GAME = "target_sum_game"
  HIGH_CARD_MAXIMIZER_GAME = "high_card_maximizer_game"
  LOW_CARD_MINIMALIST_GAME = "low_card_minimalist_game"
  EVEN_ODD_PATTERN_GAME = "even_odd_pattern_game"
  SANDWICH_PAIR_GAME = "sandwich_pair_game"
  RAPID_CLEAR_DISCARD_GAME = "rapid_clear_discard_game"
  FULL_HOUSE_TRIPLE_PAIR_GAME = "full_house_triple_pair_game"
  HOPPER = "hopper"
  CHEETAH = "cheetah"
  QUADRUPED = "quadruped"
  HUMANOID = "humanoid"


EnvironmentFactory = Callable[[int, bool], dm_env.Environment]

ENVIRONMENT_REGISTRY: dict[EnvironmentName, EnvironmentFactory] = {}
register, get = class_registry.declare_registry(ENVIRONMENT_REGISTRY)


@register(EnvironmentName.CARTPOLE)
def cartpole(seed: int, playground: bool = False) -> dm_env.Environment:
  del playground
  return mjx_env.MJXEnv(
      domain_name="cartpole",
      task_name="balance",
      seed=seed,
      backend="mjx",
  )


@register(EnvironmentName.QUADRUPED)
def quadruped(
    seed: int, playground: bool = False, backend: str = "mjx"
) -> dm_env.Environment:
  del playground
  return mjx_env.MJXEnv(
      domain_name="quadruped",
      task_name="run",
      seed=seed,
      backend=backend,
  )


@register(EnvironmentName.CHEETAH)
def cheetah(
    seed: int, playground: bool = False, backend: str = "mjx"
) -> dm_env.Environment:
  del playground
  return mjx_env.MJXEnv(
      domain_name="cheetah",
      task_name="run",
      seed=seed,
      backend=backend,
  )


@register(EnvironmentName.HOPPER)
def hopper(
    seed: int, playground: bool = False, backend: str = "mjx"
) -> dm_env.Environment:
  del playground
  return mjx_env.MJXEnv(
      domain_name="hopper",
      task_name="hop",
      seed=seed,
      backend=backend,
  )


@register(EnvironmentName.HUMANOID)
def humanoid(
    seed: int, playground: bool = False, backend: str = "mjx"
) -> dm_env.Environment:
  del playground
  return mjx_env.MJXEnv(
      domain_name="humanoid",
      task_name="run",
      seed=seed,
      backend=backend,
  )


@register(EnvironmentName.TIC_TAC_TOE)
def tic_tac_toe(seed: int, playground: bool = False) -> dm_env.Environment:
  del seed
  return tictactoe_env.TicTacToeEnv(playground=playground)


@register(EnvironmentName.MINI_CARD_DECK)
def mini_card_deck(seed: int, playground: bool = False) -> dm_env.Environment:
  del playground
  return mini_card_deck_env.MiniCardDeckEnv(seed=seed)


@register(EnvironmentName.ASCENDING_SORT_GAME)
def ascending_sort_game(
    seed: int, playground: bool = False
) -> dm_env.Environment:
  del playground
  return mini_card_deck_env.AscendingSortGame(seed=seed)


@register(EnvironmentName.DESCENDING_SORT_GAME)
def descending_sort_game(
    seed: int, playground: bool = False
) -> dm_env.Environment:
  del playground
  return mini_card_deck_env.DescendingSortGame(seed=seed)


@register(EnvironmentName.TRIPLET_MATCHING_GAME)
def triplet_matching_game(
    seed: int, playground: bool = False
) -> dm_env.Environment:
  del playground
  return mini_card_deck_env.TripletMatchingGame(seed=seed)


@register(EnvironmentName.TARGET_SUM_GAME)
def target_sum_game(seed: int, playground: bool = False) -> dm_env.Environment:
  del playground
  return mini_card_deck_env.TargetSumGame(seed=seed)


@register(EnvironmentName.HIGH_CARD_MAXIMIZER_GAME)
def high_card_maximizer_game(
    seed: int, playground: bool = False
) -> dm_env.Environment:
  del playground
  return mini_card_deck_env.HighCardMaximizerGame(seed=seed)


@register(EnvironmentName.LOW_CARD_MINIMALIST_GAME)
def low_card_minimalist_game(
    seed: int, playground: bool = False
) -> dm_env.Environment:
  del playground
  return mini_card_deck_env.LowCardMinimalistGame(seed=seed)


@register(EnvironmentName.EVEN_ODD_PATTERN_GAME)
def even_odd_pattern_game(
    seed: int, playground: bool = False
) -> dm_env.Environment:
  del playground
  return mini_card_deck_env.EvenOddPatternGame(seed=seed)


@register(EnvironmentName.SANDWICH_PAIR_GAME)
def sandwich_pair_game(
    seed: int, playground: bool = False
) -> dm_env.Environment:
  del playground
  return mini_card_deck_env.SandwichPairGame(seed=seed)


@register(EnvironmentName.RAPID_CLEAR_DISCARD_GAME)
def rapid_clear_discard_game(
    seed: int, playground: bool = False
) -> dm_env.Environment:
  del playground
  return mini_card_deck_env.RapidClearDiscardGame(seed=seed)


@register(EnvironmentName.FULL_HOUSE_TRIPLE_PAIR_GAME)
def full_house_triple_pair_game(
    seed: int, playground: bool = False
) -> dm_env.Environment:
  del playground
  return mini_card_deck_env.FullHouseTriplePairGame(seed=seed)
