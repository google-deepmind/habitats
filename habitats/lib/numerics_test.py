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

"""Tests for numerics."""

from absl.testing import absltest
from absl.testing import parameterized
from habitats.lib import numerics
import jax.numpy as jnp
import numpy as np



class AllFiniteTest(parameterized.TestCase):

  # Trees are built lazily: JAX can't be called before `absltest.main()`.
  @parameterized.named_parameters(
      ("python_float", lambda: 1.5),
      ("numpy_array", lambda: np.array([0.0, -2.0], dtype=np.float32)),
      ("jax_array", lambda: jnp.array([1.0, 2.0])),
      ("nested_tree", lambda: {"a": [np.ones(3)], "b": (jnp.zeros(2), 0.5)}),
      ("empty_tree", lambda: {}),
  )
  def test_finite_values(self, make_tree):
    self.assertTrue(numerics.all_finite(make_tree()))

  @parameterized.named_parameters(
      ("nan_python_float", lambda: float("nan")),
      ("inf_numpy_array", lambda: np.array([0.0, np.inf])),
      ("negative_inf_jax_array", lambda: jnp.array([-jnp.inf])),
      ("nan_in_nested_tree", lambda: {"a": 1.0, "b": (jnp.array([np.nan]),)}),
  )
  def test_non_finite_values(self, make_tree):
    self.assertFalse(numerics.all_finite(make_tree()))

  def test_ignores_leaves_that_cannot_be_non_finite(self):
    tree = {
        "count": np.uint32(7),
        "flag": True,
        "board_cell": 3,
        "name": "agent_0",
    }
    self.assertTrue(numerics.all_finite(tree))


if __name__ == "__main__":
  absltest.main()
