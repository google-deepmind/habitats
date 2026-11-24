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

"""Marks environments whose own termination the evolved reward replaces."""

import dm_env


class ReplaceableTermination:
  """Marks an environment whose termination is part of the base task.

  The evolved reward replaces the base task, so for an environment marked with
  this class only the evolved reward ends an episode, in training and in
  evaluation; see `EvolvedRewardWrapper`. Use it for tasks that can go on past
  their own termination, such as a MuJoCo body that has fallen over. Leave it
  off for environments whose episodes can't continue past it, such as a
  finished game: their termination also ends the episode.
  """


def ignores_base_termination(env: dm_env.Environment) -> bool:
  """Returns whether only the evolved reward ends episodes of `env`."""
  return isinstance(env, ReplaceableTermination)
