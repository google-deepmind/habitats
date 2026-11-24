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

"""Tests for metrics."""

from absl.testing import absltest
from habitats.lib import metrics
import numpy as np



class MetricsTest(absltest.TestCase):

  def test_normalize_by_columns(self):
    grid = [
        [10.0, 100.0],
        [20.0, 50.0],
    ]
    random_fitness = [0.0, 0.0]
    norm_grid, norm_random = metrics.normalize_by_columns(
        grid, random_fitness=random_fitness
    )
    self.assertIsNotNone(norm_random)
    expected_grid = np.array([
        [0.5, 1.0],
        [1.0, 0.5],
    ])
    expected_random = np.array([0.0, 0.0])
    np.testing.assert_array_almost_equal(norm_grid, expected_grid)
    np.testing.assert_array_almost_equal(norm_random, expected_random)

  def test_format_fitness_grid(self):
    agent_ids = ["agent_0", "agent_1"]
    grid = [
        [10.0, 100.0],
        [20.0, 50.0],
    ]
    random_fitness = [0.0, 0.0]
    table = metrics.format_fitness_grid(
        agent_ids=agent_ids, fitness_grid=grid, random_fitness=random_fitness
    )

    lines = table.strip().split("\n")
    self.assertLen(lines, 3)
    # Header check
    self.assertIn("agent_0", lines[0])
    self.assertIn("agent_1", lines[0])
    # Row 0: agent_0 (0.5000, 1.0000)
    self.assertTrue(lines[1].startswith("agent_0"))
    self.assertIn("0.5000", lines[1])
    self.assertIn("1.0000", lines[1])
    # Row 1: agent_1 (1.0000, 0.5000)
    self.assertTrue(lines[2].startswith("agent_1"))
    self.assertIn("1.0000", lines[2])
    self.assertIn("0.5000", lines[2])

  def test_format_fitness_grid_empty(self):
    self.assertEqual(metrics.format_fitness_grid([], [], []), "")

  def test_calculate_behavioral_diversity(self):
    self.assertEqual(metrics.calculate_behavioral_diversity([]), 0.0)
    self.assertEqual(metrics.calculate_behavioral_diversity([[1.0, 0.0]]), 0.0)

    orthogonal = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    self.assertAlmostEqual(
        metrics.calculate_behavioral_diversity(orthogonal), 1.0
    )

    identical = np.array([
        [1.0, 1.0],
        [1.0, 1.0],
    ])
    self.assertAlmostEqual(
        metrics.calculate_behavioral_diversity(identical), 0.0
    )

  def test_calculate_behavioral_diversity_collapsed_population(self):
    # Every policy is worse than the random policy under every reward, so its
    # normalized profile is all zeros: there is no diversity to measure.
    collapsed = np.zeros((3, 3))
    self.assertEqual(metrics.calculate_behavioral_diversity(collapsed), 0.0)

  def test_calculate_behavioral_diversity_excludes_zero_profiles(self):
    # Two identical policies plus one zero profile: only the identical pair is
    # compared, instead of the zero profile counting as distance 1 to both.
    grid = np.array([
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    self.assertAlmostEqual(metrics.calculate_behavioral_diversity(grid), 0.0)

    # A single non-zero profile has no pair to compare.
    single = np.array([
        [1.0, 0.5],
        [0.0, 0.0],
    ])
    self.assertEqual(metrics.calculate_behavioral_diversity(single), 0.0)

  def test_calculate_specialist_quality(self):
    self.assertEqual(metrics.calculate_specialist_quality([]), 0.0)
    grid = np.array([
        [1.0, 0.2],
        [0.4, 0.8],
    ])
    self.assertAlmostEqual(metrics.calculate_specialist_quality(grid), 0.9)


if __name__ == "__main__":
  absltest.main()
