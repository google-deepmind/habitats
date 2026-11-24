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

"""Utilities for parsing structured outputs (code blocks and JSON) from LLMs."""

import json
from typing import Any


def extract_code_block(text: str, language: str = "python") -> str:
  """Extracts a code block from an LLM markdown response.

  Prefers a fenced block tagged with `language` (e.g. ```python ... ```), falls
  back to a generic fenced block (``` ... ```), and returns `text.strip()` if no
  fences are present.

  Args:
    text: Raw text returned by the LLM.
    language: Expected markdown language tag on the opening code fence.

  Returns:
    The extracted code string with leading and trailing whitespace stripped.
  """
  lang_marker = f"```{language}"
  end_marker = "```"

  start_idx = text.find(lang_marker)
  if start_idx != -1:
    start_idx += len(lang_marker)
  else:
    generic_idx = text.find(end_marker)
    if generic_idx == -1:
      return text.strip()
    start_idx = generic_idx + len(end_marker)
    # If the opening fence has another language tag on the same line, skip past
    # the newline only when there is no space on the fence line.
    newline_idx = text.find("\n", start_idx)
    closing_idx = text.find(end_marker, start_idx)
    if (
        newline_idx != -1
        and (closing_idx == -1 or newline_idx < closing_idx)
        and text[start_idx:newline_idx].strip().isalpha()
    ):
      start_idx = newline_idx + 1

  end_idx = text.find(end_marker, start_idx)
  if end_idx == -1:
    return text[start_idx:].strip()

  return text[start_idx:end_idx].strip()


def parse_json_object(text: str) -> dict[str, Any]:
  """Extracts and parses a JSON object (`dict`) from an LLM response.

  Handles raw JSON strings, markdown-fenced JSON blocks (```json ... ```), and
  JSON objects embedded within surrounding prose.

  Args:
    text: Raw text returned by the LLM.

  Returns:
    The parsed JSON dictionary.

  Raises:
    ValueError: If `text` does not contain a valid JSON object.
  """
  candidate = extract_code_block(text, language="json")
  start_idx = candidate.find("{")
  end_idx = candidate.rfind("}")
  if start_idx != -1 and end_idx != -1 and start_idx <= end_idx:
    candidate = candidate[start_idx : end_idx + 1]

  parsed = json.loads(candidate)
  if not isinstance(parsed, dict):
    raise ValueError(
        f"Expected a JSON object (dict), got {type(parsed).__name__}."
    )
  return parsed
