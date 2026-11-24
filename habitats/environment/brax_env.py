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

"""Interface for environments that can provide an on-device Brax environment."""

import abc
from brax import envs


class BraxEnv(abc.ABC):
  """Interface for dm_env environments that can provide an on-device Brax environment."""

  @abc.abstractmethod
  def to_brax_env(self) -> envs.Env:
    """Returns the underlying Brax environment for fast JAX execution."""
