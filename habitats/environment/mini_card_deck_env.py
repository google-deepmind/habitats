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

"""Single-Player Card Sorting Sandbox Environment."""

from typing import Any
import dm_env
from dm_env import specs
import numpy as np


class MiniCardDeckEnv(dm_env.Environment):
  """A single-player mini card deck environment.

  The player interacts with a deck of 32 cards (4 copies of each card 1-8), a
  hand of max size
  1,
  and a table with 3 slots. There are no built-in rules or winning conditions.

  Action Space:
    0: Pass (do nothing)
    1: Draw (from deck to hand)
    2: Play to Slot 0
    3: Play to Slot 1
    4: Play to Slot 2
    5: Take from Slot 0
    6: Take from Slot 1
    7: Take from Slot 2
    8: Discard (remove card in hand from game)

  State Space:
    Discrete(6561) encoded as:
      hand * 729 + slot0 * 81 + slot1 * 9 + slot2
    where each component is 0 (empty) or 1-8 (card value).
  """

  def __init__(self, seed: int | None = None, max_steps: int = 500):
    self._max_steps = max_steps
    self._action_spec = specs.DiscreteArray(9, dtype=int, name="action")
    self._observation_spec = specs.DiscreteArray(
        6561, dtype=int, name="observation"
    )
    self._rng = np.random.RandomState(seed)

    self._deck = []
    self._hand = 0  # 0 means empty
    self._table = [0, 0, 0]  # 0 means empty
    self._step_count = 0

  def reset(self) -> dm_env.TimeStep:
    self._deck = list(range(1, 9)) * 4
    self._rng.shuffle(self._deck)
    self._hand = 0
    self._table = [0, 0, 0]
    self._step_count = 0
    return dm_env.restart(self._get_state_index())

  def step(self, action: int) -> dm_env.TimeStep:
    self._step_count += 1
    reward = 0.0  # Playground step reward
    invalid_action = False

    if action == 0:  # Pass
      pass

    elif action == 1:  # Draw
      if self._hand == 0 and len(self._deck) > 0:
        self._hand = self._deck.pop(0)
      else:
        invalid_action = True

    elif action in (2, 3, 4):  # Play to Slot 0, 1, 2
      slot_idx = action - 2
      if self._hand != 0 and self._table[slot_idx] == 0:
        self._table[slot_idx] = self._hand
        self._hand = 0
      else:
        invalid_action = True

    elif action in (5, 6, 7):  # Take from Slot 0, 1, 2
      slot_idx = action - 5
      if self._hand == 0 and self._table[slot_idx] != 0:
        self._hand = self._table[slot_idx]
        self._table[slot_idx] = 0
      else:
        invalid_action = True

    elif action == 8:  # Discard
      if self._hand != 0:
        self._hand = 0
      else:
        invalid_action = True
    else:
      invalid_action = True

    if invalid_action:
      reward = -0.5

    state_idx = self._get_state_index()

    if self._step_count >= self._max_steps:
      return dm_env.termination(reward, state_idx)
    else:
      return dm_env.transition(reward, state_idx)

  def legal_actions(self) -> np.ndarray:
    """Returns which actions `step` would accept in the current state.

    Unlike the state index, this accounts for the deck: drawing is illegal once
    the deck is empty, which the observation does not reveal.

    Returns:
      A boolean array with one entry per action.
    """
    hand_empty = self._hand == 0
    mask = np.zeros(self._action_spec.num_values, dtype=bool)
    mask[0] = True  # Pass
    mask[1] = hand_empty and bool(self._deck)  # Draw
    for slot_idx, card in enumerate(self._table):
      mask[2 + slot_idx] = not hand_empty and card == 0  # Play
      mask[5 + slot_idx] = hand_empty and card != 0  # Take
    mask[8] = not hand_empty  # Discard
    return mask

  def action_spec(self) -> specs.DiscreteArray:
    return self._action_spec

  def observation_spec(self) -> specs.DiscreteArray:
    return self._observation_spec

  @property
  def action_names(self) -> list[str]:
    return [
        "Pass 💤",
        "Draw 🎴",
        "Play Slot 0 ⬇️0",
        "Play Slot 1 ⬇️1",
        "Play Slot 2 ⬇️2",
        "Take Slot 0 ⬆️0",
        "Take Slot 1 ⬆️1",
        "Take Slot 2 ⬆️2",
        "Discard 🗑️",
    ]

  def _get_state_index(self) -> int:
    return (
        self._hand * 729
        + self._table[0] * 81
        + self._table[1] * 9
        + self._table[2]
    )

  def decode(self, state_index: int) -> dict[str, Any]:
    """Decodes a state index back into structured state components.

    Args:
      state_index: The integer state index.

    Returns:
      A dictionary containing:
        - "hand": int (0-8, 0=empty)
        - "table": list of 3 ints (0-8, 0=empty)
    """
    hand = state_index // 729
    rem = state_index % 729
    slot0 = rem // 81
    rem = rem % 81
    slot1 = rem // 9
    slot2 = rem % 9
    return {
        "hand": hand,
        "table": [slot0, slot1, slot2],
    }

  @property
  def hand(self) -> int:
    return self._hand

  @property
  def table(self) -> list[int]:
    return list(self._table)

  @property
  def deck(self) -> list[int]:
    return list(self._deck)


class MiniCardDeckRewardWrapper(dm_env.Environment):
  """Wraps MiniCardDeckEnv to apply the card sorting reward objective."""

  def __init__(self, env: MiniCardDeckEnv):
    self.env = env

  def __getattr__(self, name: str) -> Any:
    return getattr(self.env, name)

  def reset(self) -> dm_env.TimeStep:
    return self.env.reset()

  def legal_actions(self) -> np.ndarray:
    """Returns which actions the wrapped environment would accept."""
    return self.env.legal_actions()

  def step(self, action: Any) -> dm_env.TimeStep:
    timestep = self.env.step(action)
    base_reward = timestep.reward
    if base_reward is None:
      base_reward = 0.0

    # If base reward < -0.2 (invalid action, returns -0.5), penalize -2.0
    if base_reward < -0.2:
      reward = -2.0
      custom_done = False
    else:
      reward = base_reward
      custom_done = False
      table = self.env.table
      # Check if table is full and sorted (ascending)
      if table[0] != 0 and table[1] != 0 and table[2] != 0:
        if table[0] < table[1] < table[2]:
          reward += 10.0
          custom_done = True

    if custom_done and not timestep.last():
      return dm_env.termination(float(reward), timestep.observation)
    else:
      return dm_env.TimeStep(
          step_type=timestep.step_type,
          reward=float(reward),
          discount=timestep.discount,
          observation=timestep.observation,
      )

  def action_spec(self) -> specs.Array:
    return self.env.action_spec()

  def observation_spec(self) -> specs.Array:
    return self.env.observation_spec()

  def close(self) -> None:
    self.env.close()


class BaseCardDeckGameWrapper(dm_env.Environment):
  """Base wrapper for MiniCardDeckEnv to apply custom game objectives."""

  def __init__(self, seed: int | None = None, max_steps: int = 200):
    self.env = MiniCardDeckEnv(seed=seed, max_steps=max_steps)

  def __getattr__(self, name: str) -> Any:
    return getattr(self.env, name)

  def reset(self) -> dm_env.TimeStep:
    return self.env.reset()

  def legal_actions(self) -> np.ndarray:
    """Returns which actions the wrapped environment would accept."""
    return self.env.legal_actions()

  def action_spec(self) -> specs.Array:
    return self.env.action_spec()

  def observation_spec(self) -> specs.Array:
    return self.env.observation_spec()

  def close(self) -> None:
    self.env.close()

  def step(self, action: Any) -> dm_env.TimeStep:
    timestep = self.env.step(action)
    base_reward = timestep.reward if timestep.reward is not None else 0.0

    # Handle invalid action penalty
    if base_reward < -0.2:
      reward = -2.0
      custom_done = False
    else:
      reward, custom_done = self.compute_game_reward_and_done(timestep)

    if custom_done and not timestep.last():
      return dm_env.termination(float(reward), timestep.observation)
    else:
      return dm_env.TimeStep(
          step_type=timestep.step_type,
          reward=float(reward),
          discount=timestep.discount,
          observation=timestep.observation,
      )

  def compute_game_reward_and_done(
      self, timestep: dm_env.TimeStep
  ) -> tuple[float, bool]:
    """Computes reward and completion status for valid transitions."""
    raise NotImplementedError


class AscendingSortGame(BaseCardDeckGameWrapper):
  """Game 1: Table slots full and sorted ascending (1 < 2 < 3)."""

  def compute_game_reward_and_done(
      self, timestep: dm_env.TimeStep
  ) -> tuple[float, bool]:
    table = self.env.table
    if table[0] != 0 and table[1] != 0 and table[2] != 0:
      if table[0] < table[1] < table[2]:
        return 10.0, True
    return 0.0, False


class DescendingSortGame(BaseCardDeckGameWrapper):
  """Game 2: Table slots full and sorted descending (3 > 2 > 1)."""

  def compute_game_reward_and_done(
      self, timestep: dm_env.TimeStep
  ) -> tuple[float, bool]:
    table = self.env.table
    if table[0] != 0 and table[1] != 0 and table[2] != 0:
      if table[0] > table[1] > table[2]:
        return 10.0, True
    return 0.0, False


class TripletMatchingGame(BaseCardDeckGameWrapper):
  """Game 3: All 3 table slots occupied and equal (x, x, x)."""

  def compute_game_reward_and_done(
      self, timestep: dm_env.TimeStep
  ) -> tuple[float, bool]:
    table = self.env.table
    if table[0] != 0 and table[0] == table[1] == table[2]:
      return 10.0, True
    return 0.0, False


class TargetSumGame(BaseCardDeckGameWrapper):
  """Game 4: Table slots full and sum to exactly 12."""

  def __init__(self, seed: int, target_sum: int = 12, max_steps: int = 200):
    super().__init__(seed=seed, max_steps=max_steps)
    self.target_sum = target_sum

  def compute_game_reward_and_done(
      self, timestep: dm_env.TimeStep
  ) -> tuple[float, bool]:
    table = self.env.table
    if table[0] != 0 and table[1] != 0 and table[2] != 0:
      if sum(table) == self.target_sum:
        return 10.0, True
    return 0.0, False


class HighCardMaximizerGame(BaseCardDeckGameWrapper):
  """Game 5: Fill table slots with high cards (>= 6)."""

  def compute_game_reward_and_done(
      self, timestep: dm_env.TimeStep
  ) -> tuple[float, bool]:
    table = self.env.table
    if table[0] >= 6 and table[1] >= 6 and table[2] >= 6:
      return 10.0 + sum(table), True
    return 0.0, False


class LowCardMinimalistGame(BaseCardDeckGameWrapper):
  """Game 6: Fill table slots with low cards (<= 3)."""

  def compute_game_reward_and_done(
      self, timestep: dm_env.TimeStep
  ) -> tuple[float, bool]:
    table = self.env.table
    if 0 < table[0] <= 3 and 0 < table[1] <= 3 and 0 < table[2] <= 3:
      return 10.0, True
    return 0.0, False


class EvenOddPatternGame(BaseCardDeckGameWrapper):
  """Game 7: Alternating parity on table (even, odd, even)."""

  def compute_game_reward_and_done(
      self, timestep: dm_env.TimeStep
  ) -> tuple[float, bool]:
    table = self.env.table
    if table[0] != 0 and table[1] != 0 and table[2] != 0:
      if table[0] % 2 == 0 and table[1] % 2 == 1 and table[2] % 2 == 0:
        return 10.0, True
    return 0.0, False


class SandwichPairGame(BaseCardDeckGameWrapper):
  """Game 8: Outer table slots equal, middle slot different (x, y, x)."""

  def compute_game_reward_and_done(
      self, timestep: dm_env.TimeStep
  ) -> tuple[float, bool]:
    table = self.env.table
    if table[0] != 0 and table[1] != 0 and table[2] != 0:
      if table[0] == table[2] and table[0] != table[1]:
        return 10.0, True
    return 0.0, False


class RapidClearDiscardGame(BaseCardDeckGameWrapper):
  """Game 9: Reward for discarding high cards (>= 5)."""

  def compute_game_reward_and_done(
      self, timestep: dm_env.TimeStep
  ) -> tuple[float, bool]:
    # Discard action is action 8
    # Step penalty to encourage speed
    reward = -0.05
    done = False
    if len(self.env.deck) == 0 and self.env.hand == 0:
      reward += 10.0
      done = True
    return reward, done


class FullHouseTriplePairGame(BaseCardDeckGameWrapper):
  """Game 10: Pair on slots 0-1 and max card 8 on slot 2."""

  def compute_game_reward_and_done(
      self, timestep: dm_env.TimeStep
  ) -> tuple[float, bool]:
    table = self.env.table
    if table[0] != 0 and table[0] == table[1] and table[2] == 8:
      return 10.0, True
    return 0.0, False
