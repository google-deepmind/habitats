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

"""Utility functions for metric calculation and normalization in evolutionary search."""

import numpy as np


def normalize_by_columns(
    fitness_grid: list[list[float]] | np.ndarray,
    random_fitness: list[float] | np.ndarray,
    precision: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
  """Normalizes the fitness grid and optional random fitness column-wise to [0, 1].

  Normalization is done per column (reward function) across all policies
  (including the random policy if provided). If all values in a column are
  equal,
  they are all normalized to 1.0. Values are rounded to `precision` decimal
  places.

  Args:
    fitness_grid: Matrix of shape (N, M) or list of rows, where grid[i][j]
      is fitness of Policy i on Reward j.
    random_fitness: List of size M of random policy fitness on each
      Reward j.
    precision: Number of decimal places to round normalized values to.

  Returns:
    A tuple of (normalized_fitness_grid, normalized_random_fitness).
  """
  grid_arr = np.asarray(fitness_grid, dtype=float)
  if grid_arr.size == 0:
    return grid_arr, np.array(random_fitness, dtype=float)

  if grid_arr.ndim != 2:
    raise ValueError(
        f"fitness_grid must be 2D, got {grid_arr.ndim}D with shape"
        f" {grid_arr.shape}"
    )

  num_rows, num_cols = grid_arr.shape

  random_arr = np.asarray(random_fitness, dtype=float)
  if random_arr.ndim != 1 or len(random_arr) != num_cols:
    raise ValueError(
        f"random_fitness must be 1D of length {num_cols}, got shape"
        f" {random_arr.shape}"
    )

  norm_grid = np.zeros_like(grid_arr)
  norm_random = np.zeros(num_cols, dtype=float)

  for j in range(num_cols):
    col_values = [float(grid_arr[i, j]) for i in range(num_rows)]
    if random_arr is not None:
      col_values.append(float(random_arr[j]))

    min_val = min(col_values)
    max_val = max(col_values)

    if max_val == min_val:
      norm_col = [1.0 for _ in col_values]
    else:
      norm_col = [
          round((v - min_val) / (max_val - min_val), precision)
          for v in col_values
      ]

    for i in range(num_rows):
      norm_grid[i, j] = norm_col[i]
      norm_random[j] = norm_col[-1]

  return norm_grid, norm_random


def format_fitness_grid(
    agent_ids: list[str],
    fitness_grid: list[list[float]] | np.ndarray,
    random_fitness: list[float] | np.ndarray,
) -> str:
  """Formats a fitness grid into a normalized, aligned ASCII table string.

  Args:
    agent_ids: list of agent IDs corresponding to the rows/columns.
    fitness_grid: 2D array or nested list of fitness scores.
    random_fitness: Random baseline scores for column normalization.

  Returns:
    Formatted string table with columns and rows labeled by agent IDs.
  """
  if not agent_ids:
    return ""

  norm_grid, _ = normalize_by_columns(
      fitness_grid, random_fitness=random_fitness, precision=4
  )
  if norm_grid.size == 0:
    return ""

  grid_str = " " * 10 + " ".join(f"{aid:>10}" for aid in agent_ids) + "\n"
  for i, aid_i in enumerate(agent_ids):
    grid_str += f"{aid_i:<10}"
    for j in range(len(agent_ids)):
      grid_str += f" {norm_grid[i, j]:10.4f}"
    grid_str += "\n"
  return grid_str


def calculate_behavioral_diversity(
    fitness_grid: list[list[float]] | np.ndarray,
) -> float:
  """Calculates the average pairwise cosine distance between policy profiles.

  Profiles with a zero norm (e.g. a policy normalized to 0 under every reward
  because it is worse than the random policy everywhere) have no direction, so
  they are excluded instead of counting as maximally distant from every other
  profile.

  Args:
    fitness_grid: n x n grid where grid[i] is the performance profile of Policy
      i.

  Returns:
    The average pairwise cosine distance (0.0 to 1.0) between non-zero
    profiles, or 0.0 if fewer than two profiles are non-zero.
  """
  matrix = np.asarray(fitness_grid, dtype=float)
  if matrix.size == 0 or matrix.ndim != 2:
    return 0.0

  norms = np.linalg.norm(matrix, axis=1, keepdims=True)
  non_zero = norms[:, 0] > 0
  matrix = matrix[non_zero]
  norms = norms[non_zero]
  n = matrix.shape[0]
  if n <= 1:
    return 0.0
  normalized_matrix = matrix / norms

  # Pairwise cosine similarity: dot product of normalized vectors
  similarity_matrix = np.dot(normalized_matrix, normalized_matrix.T)

  # Cosine distance = 1 - similarity.
  # Clip to [0, 2] to handle potential numerical precision issues,
  # but since fitness is non-negative [0, 1], similarity is in [0, 1]
  # and distance is in [0, 1].
  distance_matrix = np.clip(1.0 - similarity_matrix, 0.0, 1.0)

  # We only care about the upper triangle (excluding diagonal)
  triu_indices = np.triu_indices(n, k=1)
  pairwise_distances = distance_matrix[triu_indices]

  return float(np.mean(pairwise_distances))


def calculate_specialist_quality(
    fitness_grid: list[list[float]] | np.ndarray,
) -> float:
  """Calculates the average specialist quality (mean of diagonal).

  Args:
    fitness_grid: n x n grid where grid[i][i] is the fitness of Policy i on its
      own Reward i.

  Returns:
    The average specialist quality.
  """
  matrix = np.asarray(fitness_grid, dtype=float)
  if matrix.size == 0 or matrix.ndim != 2:
    return 0.0
  n = min(matrix.shape)
  if n == 0:
    return 0.0
  return float(np.mean(np.diag(matrix)))
