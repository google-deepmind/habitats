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

"""Tests for CardSandboxEnv, verifying card game mechanics and state transitions."""

from absl.testing import absltest
from habitats.environment import mini_card_deck_env


class MiniCardDeckEnvTest(absltest.TestCase):

  def test_initial_state(self):
    env = mini_card_deck_env.MiniCardDeckEnv(seed=42)
    timestep = env.reset()

    self.assertTrue(timestep.first())
    self.assertEqual(env.hand, 0)
    self.assertEqual(env.table, [0, 0, 0])
    self.assertLen(env.deck, 32)
    # Check state index calculation: 0*729 + 0*81 + 0*9 + 0 = 0
    self.assertEqual(timestep.observation, 0)

  def test_valid_draw(self):
    env = mini_card_deck_env.MiniCardDeckEnv(seed=42)
    env.reset()
    initial_deck = env.deck.copy()

    # Draw card
    timestep = env.step(1)

    self.assertEqual(env.hand, initial_deck[0])
    self.assertLen(env.deck, 31)
    self.assertEqual(timestep.reward, 0.0)
    # State index: hand*729
    self.assertEqual(timestep.observation, env.hand * 729)

  def test_invalid_draw_hand_full(self):
    env = mini_card_deck_env.MiniCardDeckEnv(seed=42)
    env.reset()

    # Draw once (valid)
    env.step(1)
    hand_before = env.hand

    # Draw again (invalid)
    timestep = env.step(1)

    self.assertEqual(env.hand, hand_before)  # Hand unchanged
    self.assertEqual(timestep.reward, -0.5)  # Invalid action penalty

  def test_valid_play(self):
    env = mini_card_deck_env.MiniCardDeckEnv(seed=42)
    env.reset()

    # Draw (valid)
    env.step(1)
    drawn_card = env.hand

    # Play to Slot 1 (action 3)
    timestep = env.step(3)

    self.assertEqual(env.hand, 0)
    self.assertEqual(env.table, [0, drawn_card, 0])
    self.assertEqual(timestep.reward, 0.0)
    # State index: slot1 * 9
    self.assertEqual(timestep.observation, drawn_card * 9)

  def test_invalid_play_hand_empty(self):
    env = mini_card_deck_env.MiniCardDeckEnv(seed=42)
    env.reset()

    # Play to Slot 0 (action 2) without drawing
    timestep = env.step(2)

    self.assertEqual(env.table, [0, 0, 0])
    self.assertEqual(timestep.reward, -0.5)

  def test_invalid_play_slot_full(self):
    env = mini_card_deck_env.MiniCardDeckEnv(seed=42)
    env.reset()

    # Draw and Play to Slot 0
    env.step(1)
    card1 = env.hand
    env.step(2)

    # Draw another card
    env.step(1)
    card2 = env.hand

    # Try to play to Slot 0 again (full)
    timestep = env.step(2)

    self.assertEqual(env.table, [card1, 0, 0])
    self.assertEqual(env.hand, card2)
    self.assertEqual(timestep.reward, -0.5)

  def test_valid_take(self):
    env = mini_card_deck_env.MiniCardDeckEnv(seed=42)
    env.reset()

    # Draw and play to Slot 2 (action 4)
    env.step(1)
    card = env.hand
    env.step(4)

    # Take from Slot 2 (action 7)
    timestep = env.step(7)

    self.assertEqual(env.hand, card)
    self.assertEqual(env.table, [0, 0, 0])
    self.assertEqual(timestep.reward, 0.0)
    # State index: hand * 729
    self.assertEqual(timestep.observation, card * 729)

  def test_invalid_take_hand_full(self):
    env = mini_card_deck_env.MiniCardDeckEnv(seed=42)
    env.reset()

    # Draw (hand full)
    env.step(1)
    # Try to take from Slot 0 (empty anyway, but hand full is also check)
    timestep = env.step(5)

    self.assertEqual(timestep.reward, -0.5)

  def test_invalid_take_slot_empty(self):
    env = mini_card_deck_env.MiniCardDeckEnv(seed=42)
    env.reset()

    # Try to take from Slot 0 (empty, hand empty)
    timestep = env.step(5)

    self.assertEqual(timestep.reward, -0.5)

  def test_valid_discard(self):
    env = mini_card_deck_env.MiniCardDeckEnv(seed=42)
    env.reset()

    # Draw
    env.step(1)
    # Discard (action 8)
    timestep = env.step(8)

    self.assertEqual(env.hand, 0)
    self.assertEqual(timestep.reward, 0.0)
    self.assertEqual(timestep.observation, 0)

  def test_invalid_discard_hand_empty(self):
    env = mini_card_deck_env.MiniCardDeckEnv(seed=42)
    env.reset()

    # Discard with empty hand
    timestep = env.step(8)

    self.assertEqual(timestep.reward, -0.5)

  def test_max_steps(self):
    env = mini_card_deck_env.MiniCardDeckEnv(max_steps=3)
    env.reset()

    # 3 steps
    timestep = env.step(0)  # Pass
    self.assertFalse(timestep.last())
    timestep = env.step(0)  # Pass
    self.assertFalse(timestep.last())
    timestep = env.step(0)  # Pass
    self.assertTrue(timestep.last())

  def test_decode(self):
    env = mini_card_deck_env.MiniCardDeckEnv()
    # Test initial state (0)
    decoded = env.decode(0)
    self.assertEqual(decoded["hand"], 0)
    self.assertEqual(decoded["table"], [0, 0, 0])

    # Test some random state index
    # hand=1, slot0=2, slot1=3, slot2=4 -> index 922
    decoded = env.decode(922)
    self.assertEqual(decoded["hand"], 1)
    self.assertEqual(decoded["table"], [2, 3, 4])


class MiniCardDeckRewardWrapperTest(absltest.TestCase):

  def test_invalid_action_penalty(self):
    base_env = mini_card_deck_env.MiniCardDeckEnv()
    env = mini_card_deck_env.MiniCardDeckRewardWrapper(base_env)
    env.reset()

    # Try to play slot 0 (invalid because hand is empty)
    timestep = env.step(2)
    self.assertEqual(timestep.reward, -2.0)
    self.assertFalse(timestep.last())

  def test_valid_step_zero_reward(self):
    base_env = mini_card_deck_env.MiniCardDeckEnv()
    env = mini_card_deck_env.MiniCardDeckRewardWrapper(base_env)
    env.reset()

    # Draw (valid)
    timestep = env.step(1)
    self.assertEqual(timestep.reward, 0.0)
    self.assertFalse(timestep.last())

  def test_sorting_win_condition(self):
    base_env = mini_card_deck_env.MiniCardDeckEnv()
    env = mini_card_deck_env.MiniCardDeckRewardWrapper(base_env)
    env.reset()

    # Manually set deck to guarantee ascending order [1, 2, 3]
    env.env._deck = [1, 2, 3]

    # Draw and play to slot 0
    env.step(1)  # hand = 1
    env.step(2)  # slot0 = 1

    # Draw and play to slot 1
    env.step(1)  # hand = 2
    env.step(3)  # slot1 = 2

    # Draw and play to slot 2
    env.step(1)  # hand = 3
    # Playing to slot 2 should trigger the win condition
    timestep = env.step(4)  # slot2 = 3

    # Total reward for the last step should be base_reward (0.0) + 10.0 = 10.0
    self.assertEqual(timestep.reward, 10.0)
    self.assertTrue(timestep.last())

  def test_full_table_not_sorted_no_win(self):
    base_env = mini_card_deck_env.MiniCardDeckEnv()
    env = mini_card_deck_env.MiniCardDeckRewardWrapper(base_env)
    env.reset()

    # Manually set deck to guarantee descending order [3, 2, 1]
    env.env._deck = [3, 2, 1]

    # Draw and play to slot 0
    env.step(1)  # hand = 3
    env.step(2)  # slot0 = 3

    # Draw and play to slot 1
    env.step(1)  # hand = 2
    env.step(3)  # slot1 = 2

    # Draw and play to slot 2
    env.step(1)  # hand = 1
    # Playing to slot 2 should NOT trigger the win condition
    timestep = env.step(4)  # slot2 = 1

    self.assertEqual(timestep.reward, 0.0)
    self.assertFalse(timestep.last())


if __name__ == "__main__":
  absltest.main()
