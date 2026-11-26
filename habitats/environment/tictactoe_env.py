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

"""Custom dm_env environment for Tic Tac Toe."""

import random
import dm_env
from dm_env import specs
import numpy as np


class TicTacToeEnv(dm_env.Environment):
  """A custom 3x3 Tic Tac Toe environment for tabular Q-learning.

  The agent plays as 'X' (1) and moves first.
  The environment's internal opponent plays as 'O' (2) and moves second.
  Empty cells are represented by 0.

  State Space:
    Discrete(19683) - Base-3 representation of the 3x3 board layout.

  Action Space:
    Discrete(9) - cell index from 0 to 8:
      0 1 2
      3 4 5
      6 7 8
  """

  metadata = {"render_modes": ["ansi"]}

  def __init__(self, playground: bool = True):
    self.playground = playground
    self._action_spec = specs.DiscreteArray(9, dtype=int, name="action")
    self._observation_spec = specs.DiscreteArray(
        19683, dtype=int, name="observation"
    )  # 3^9
    self.board = [0] * 9
    self.reset()

  def action_spec(self) -> specs.DiscreteArray:
    return self._action_spec

  def observation_spec(self) -> specs.DiscreteArray:
    return self._observation_spec

  @property
  def action_names(self) -> list[str]:
    return [
        "Top-Left (0)",
        "Top-Mid (1)",
        "Top-Right (2)",
        "Mid-Left (3)",
        "Center (4)",
        "Mid-Right (5)",
        "Bot-Left (6)",
        "Bot-Mid (7)",
        "Bot-Right (8)",
    ]

  def legal_actions(self) -> np.ndarray:
    """Returns which cells are empty, one boolean per action."""
    return np.array([cell == 0 for cell in self.board], dtype=bool)

  def reset(self) -> dm_env.TimeStep:
    self.board = [0] * 9
    state = self._board_to_state(self.board)
    return dm_env.restart(state)

  def step(self, action: int) -> dm_env.TimeStep:
    # 1. Check if action is valid (cell must be empty)
    if self.board[action] != 0:
      state = self._board_to_state(self.board)
      reward = -1.0 if self.playground else -10.0
      return dm_env.termination(reward, state)

    # 2. Agent (X) makes move
    self.board[action] = 1

    if not self.playground:
      # --- Standard Mode ---
      # Check if Agent won
      if self._check_win(self.board, 1):
        state = self._board_to_state(self.board)
        return dm_env.termination(10.0, state)

      # Check for draw
      if 0 not in self.board:
        state = self._board_to_state(self.board)
        return dm_env.termination(0.0, state)

      # 3. Opponent (O) makes move
      empty_cells = [i for i, v in enumerate(self.board) if v == 0]
      opponent_action = random.choice(empty_cells)
      self.board[opponent_action] = 2

      # Check if Opponent won
      if self._check_win(self.board, 2):
        state = self._board_to_state(self.board)
        return dm_env.termination(-10.0, state)

      # Check for draw
      if 0 not in self.board:
        state = self._board_to_state(self.board)
        return dm_env.termination(0.0, state)

      # Game continues
      state = self._board_to_state(self.board)
      return dm_env.transition(0.0, state)

    else:
      # --- Playground Mode ---
      # Only terminate when board is full
      if 0 not in self.board:
        state = self._board_to_state(self.board)
        return dm_env.termination(0.0, state)

      # Opponent (O) makes move
      empty_cells = [i for i, v in enumerate(self.board) if v == 0]
      opponent_action = random.choice(empty_cells)
      self.board[opponent_action] = 2

      # Check for draw again after opponent's move
      if 0 not in self.board:
        state = self._board_to_state(self.board)
        return dm_env.termination(0.0, state)

      # Game continues
      state = self._board_to_state(self.board)
      return dm_env.transition(0.0, state)

  def _board_to_state(self, board: list[int]) -> int:
    state = 0
    for i, v in enumerate(board):
      state += int(v * (3**i))
    return state

  def _state_to_board(self, state: int) -> list[int]:
    board = [0] * 9
    temp = state
    for i in range(9):
      board[i] = temp % 3
      temp //= 3
    return board

  def _check_win(self, board: list[int], player: int) -> bool:
    wins = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],  # rows
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],  # cols
        [0, 4, 8],
        [2, 4, 6],  # diags
    ]
    for line in wins:
      if all(board[cell] == player for cell in line):
        return True
    return False

  def render(self) -> str:
    """Renders the environment to an ANSI string."""
    symbols = {0: ".", 1: "X", 2: "O"}
    outfile = []
    for r in range(3):
      for c in range(3):
        outfile.append(symbols[self.board[r * 3 + c]])
      outfile.append("\n")
    return "".join(outfile)
