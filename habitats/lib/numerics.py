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

"""Numerical checks shared by Habitats environments and trainers."""

from typing import Any

import jax
import numpy as np


def all_finite(tree: Any) -> bool:
  """Returns whether every floating-point value in `tree` is finite.

  The check runs on the host with NumPy. It accepts Python scalars, NumPy
  arrays and concrete JAX arrays (which are copied to the host), so it works
  for both JAX-based and pure-Python environments. It can't be used on traced
  values inside `jax.jit`. Leaves that can't hold NaN or infinity, such as
  integers, booleans or strings, are ignored.

  Args:
    tree: A pytree, e.g. an observation, a reward or network parameters.
  """
  arrays = (np.asarray(leaf) for leaf in jax.tree.leaves(tree))
  return all(
      np.isfinite(array).all()
      for array in arrays
      if np.issubdtype(array.dtype, np.inexact)
  )
