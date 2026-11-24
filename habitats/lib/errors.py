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

"""Exception hierarchy for the Habitats libraries.

Modules that wrap external infrastructure (LLM backends, run-logging sinks, 
...) catch that infrastructure's exceptions at their boundary and
re-raise one of the classes below.
"""


class HabitatsError(Exception):
  """Base class for all errors raised by the Habitats libraries."""


class LoggingError(HabitatsError):
  """Raised when a run-logging backend fails."""


class LLMError(HabitatsError):
  """Raised when an LLM backend fails to generate content."""


class NoTrainedAgentsError(HabitatsError):
  """Raised when training failed for every agent of a generation."""
