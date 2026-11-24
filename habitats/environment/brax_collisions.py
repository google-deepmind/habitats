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

"""Collision settings for Brax systems.

MuJoCo only creates a contact between two geoms when

    (contype_a & conaffinity_b) | (contype_b & conaffinity_a) != 0.

The Brax locomotion assets (ant, humanoid, hopper, ...) default every geom to
`contype=0, conaffinity=0` and only re-enable a few of them (mostly the feet),
while the floor has `conaffinity=1`. Every other body part therefore passes
straight through the floor, which the stock Brax tasks hide by ending episodes
as soon as the torso leaves a "healthy" height range.
"""

import copy

from brax import base
import numpy as np

# Geoms attached to the world body are static scenery, such as the floor.
_WORLD_BODY_ID = 0
# The floor has `conaffinity=1`, so a geom collides with it once bit 0 of its
# `contype` is set.
_FLOOR_CONTACT_BIT = 1


def enable_floor_collisions(sys: base.System) -> base.System:
  """Returns a copy of `sys` in which every body part collides with the floor.

  Sets bit 0 of `contype` on every geom attached to a moving body.
  `conaffinity` is left untouched: body geoms in the Brax locomotion assets
  have `conaffinity=0`, so body parts still don't collide with each other. That
  avoids jamming parts that overlap by design (e.g. the ant's hips) and keeps
  the number of contacts, hence the simulation cost, low.

  Args:
    sys: A Brax system loaded from MJCF. It is not modified.

  Returns:
    The updated system, including its `mj_model` (used for rendering).
  """
  is_body_geom = sys.geom_bodyid != _WORLD_BODY_ID
  contype = np.where(
      is_body_geom, sys.geom_contype | _FLOOR_CONTACT_BIT, sys.geom_contype
  )
  # Update a copy of the MuJoCo model so that the input system is untouched.
  mj_model = copy.copy(sys.mj_model)
  mj_model.geom_contype[:] = contype
  return sys.replace(geom_contype=contype, mj_model=mj_model)
