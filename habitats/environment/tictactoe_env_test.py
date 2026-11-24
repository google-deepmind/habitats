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

"""Unit tests for TicTacToeEnv custom dm_env environment."""

import random

from absl.testing import absltest
from habitats.environment import tictactoe_env



class TicTacToeEnvTest(absltest.TestCase):

  def test_initialization_default_playground(self):
    env = tictactoe_env.TicTacToeEnv()
    self.assertTrue(env.playground)
    self.assertEqual(env.action_spec().num_values, 9)
    self.assertEqual(env.observation_spec().num_values, 19683)
    self.assertEqual(env.board, [0] * 9)

  def test_initialization_standard(self):
    env = tictactoe_env.TicTacToeEnv(playground=False)
    self.assertFalse(env.playground)

  def test_reset(self):
    env = tictactoe_env.TicTacToeEnv()
    timestep = env.reset()
    self.assertTrue(timestep.first())
    self.assertEqual(timestep.observation, 0)
    self.assertEqual(env.board, [0] * 9)

  def test_valid_step(self):
    env = tictactoe_env.TicTacToeEnv()
    random.seed(42)  # Set seed for reproducible opponent action
    env.reset()

    # Agent plays at 4 (center)
    timestep = env.step(4)
    # Agent board state has 1 at index 4
    self.assertEqual(env.board[4], 1)
    # Opponent should have played somewhere else (value 2)
    self.assertEqual(sum(1 for v in env.board if v == 2), 1)
    self.assertEqual(sum(1 for v in env.board if v == 1), 1)
    self.assertEqual(sum(1 for v in env.board if v == 0), 7)
    self.assertEqual(timestep.reward, 0.0)
    self.assertFalse(timestep.last())

  def test_invalid_step(self):
    env = tictactoe_env.TicTacToeEnv(playground=False)
    env.reset()
    env.step(4)  # Agent plays 4. Opponent plays somewhere.
    # Agent tries to play at 4 again
    timestep = env.step(4)
    self.assertEqual(timestep.reward, -10.0)
    self.assertTrue(timestep.last())

  def test_agent_wins(self):
    env = tictactoe_env.TicTacToeEnv(playground=False)
    env.reset()
    # Force state where agent is close to win
    # Board:
    # X X .
    # O O .
    # . . .
    env.board = [1, 1, 0, 2, 2, 0, 0, 0, 0]
    # Agent plays at 2 to win
    timestep = env.step(2)
    self.assertEqual(timestep.reward, 10.0)
    self.assertTrue(timestep.last())
    self.assertEqual(env.board[2], 1)

  def test_opponent_wins(self):
    env = tictactoe_env.TicTacToeEnv(playground=False)
    random.seed(
        1
    )  # Seed makes opponent play O at index 5 when given board below
    env.reset()
    # Board:
    # X . .
    # O O .
    # X . .
    # Opponent needs index 5 to win.
    env.board = [1, 0, 0, 2, 2, 0, 1, 0, 0]
    # Agent plays index 1 (safe move, doesn't win or block opponent at 5)
    timestep = env.step(1)
    # Opponent should play at 5 and win
    self.assertEqual(env.board[5], 2)
    self.assertEqual(timestep.reward, -10.0)
    self.assertTrue(timestep.last())

  def test_draw(self):
    env = tictactoe_env.TicTacToeEnv()
    env.reset()
    # Force draw board state (only 1 cell remaining)
    # X O X
    # X O O
    # O X . -> wait, changed to O O X? No:
    # index:
    # 0: X, 1: O, 2: X
    # 3: X, 4: O, 5: O
    # 6: O, 7: X, 8: .
    # Board: [1, 2, 1, 1, 2, 2, 2, 1, 0]
    # Agent plays 8
    env.board = [1, 2, 1, 1, 2, 2, 2, 1, 0]
    timestep = env.step(8)
    self.assertEqual(timestep.reward, 0.0)
    self.assertTrue(timestep.last())
    self.assertEqual(env.board[8], 1)

  def test_playground_initialization_explicit(self):
    env = tictactoe_env.TicTacToeEnv(playground=True)
    self.assertTrue(env.playground)

  def test_playground_invalid_step(self):
    env = tictactoe_env.TicTacToeEnv(playground=True)
    env.reset()
    env.step(4)
    timestep = env.step(4)
    self.assertEqual(timestep.reward, -1.0)
    self.assertTrue(timestep.last())

  def test_playground_agent_wins_does_not_terminate(self):
    env = tictactoe_env.TicTacToeEnv(playground=True)
    env.reset()
    env.board = [1, 1, 0, 2, 2, 0, 0, 0, 0]
    timestep = env.step(2)
    self.assertEqual(timestep.reward, 0.0)
    self.assertFalse(timestep.last())
    self.assertEqual(env.board[2], 1)

  def test_playground_opponent_wins_does_not_terminate(self):
    env = tictactoe_env.TicTacToeEnv(playground=True)
    random.seed(1)
    env.reset()
    env.board = [1, 0, 0, 2, 2, 0, 1, 0, 0]
    timestep = env.step(1)
    self.assertEqual(env.board[5], 2)
    self.assertEqual(timestep.reward, 0.0)
    self.assertFalse(timestep.last())

  def test_playground_draw_terminates(self):
    env = tictactoe_env.TicTacToeEnv(playground=True)
    env.reset()
    env.board = [1, 2, 1, 1, 2, 2, 2, 1, 0]
    timestep = env.step(8)
    self.assertEqual(timestep.reward, 0.0)
    self.assertTrue(timestep.last())
    self.assertEqual(env.board[8], 1)


if __name__ == "__main__":
  absltest.main()
