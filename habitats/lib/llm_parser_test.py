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

"""Tests for LLM output parsing utilities."""

from absl.testing import absltest
from absl.testing import parameterized
from habitats.lib import llm_parser



class LLMParserTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="python_fenced_block",
          text=(
              "Here is the reward:\n```python\ndef compute_reward():\n  return"
              " 1.0\n```\nDone."
          ),
          expected="def compute_reward():\n  return 1.0",
      ),
      dict(
          testcase_name="generic_fenced_block",
          text="```\ndef compute_reward():\n  return 2.0\n```",
          expected="def compute_reward():\n  return 2.0",
      ),
      dict(
          testcase_name="unclosed_python_fence",
          text="```python\ndef compute_reward():\n  return 3.0\n",
          expected="def compute_reward():\n  return 3.0",
      ),
      dict(
          testcase_name="no_fences_raw_code",
          text="  def compute_reward():\n    return 4.0  ",
          expected="def compute_reward():\n    return 4.0",
      ),
  )
  def test_extract_code_block(self, text: str, expected: str):
    self.assertEqual(llm_parser.extract_code_block(text), expected)

  @parameterized.named_parameters(
      dict(
          testcase_name="raw_json",
          text='{"offspring_plans": [{"parent_ids": ["agent_0"]}]}',
          expected={"offspring_plans": [{"parent_ids": ["agent_0"]}]},
      ),
      dict(
          testcase_name="fenced_json",
          text=(
              "Plan below:\n```json\n"
              '{"offspring_plans": []}\n'
              "```\nEnd of plan."
          ),
          expected={"offspring_plans": []},
      ),
      dict(
          testcase_name="json_with_surrounding_prose",
          text='Sure! Here is the plan: {"offspring_plans": []} Hope it helps.',
          expected={"offspring_plans": []},
      ),
  )
  def test_parse_json_object(self, text: str, expected: dict[str, object]):
    self.assertEqual(llm_parser.parse_json_object(text), expected)

  @parameterized.named_parameters(
      dict(testcase_name="invalid_json", text="not valid json at all"),
      dict(testcase_name="json_array_not_object", text="[1, 2, 3]"),
  )
  def test_parse_json_object_raises_value_error(self, text: str):
    with self.assertRaises(ValueError):
      llm_parser.parse_json_object(text)


if __name__ == "__main__":
  absltest.main()
