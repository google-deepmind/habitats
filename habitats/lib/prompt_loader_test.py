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

"""Tests for the prompt loader library, ensuring prompts are loaded correctly."""

from absl.testing import absltest
from habitats.lib import prompt_loader


class PromptLoaderTest(absltest.TestCase):

  def test_load_system_instruction_mutator(self):
    content = prompt_loader.load_system_instruction("mutator")
    self.assertIn(
        "You are an expert RL researcher and evolutionary operator.", content
    )

  def test_load_system_instruction_invalid(self):
    with self.assertRaises(ValueError):
      prompt_loader.load_system_instruction("invalid")

  def test_load_env_description_invalid(self):
    with self.assertRaises(ValueError):
      prompt_loader.load_env_description("InvalidEnv-v0")

  def test_load_env_description_humanoid(self):
    content = prompt_loader.load_env_description("humanoid")
    self.assertIn("Environment: MuJoCo Humanoid Playground", content)
    self.assertIn("def compute_reward(", content)
    self.assertEqual(
        prompt_loader.EnvironmentPrompt.from_env_name("humanoid"),
        prompt_loader.EnvironmentPrompt.HUMANOID,
    )

  def test_load_mutator_prompt_diverse_initial(self):
    content = prompt_loader.load_mutator_prompt("diverse_initial")
    self.assertIn("Generate an initial reward function", content)

  def test_load_mutator_prompt_diverse_subsequent(self):
    content = prompt_loader.load_mutator_prompt("diverse_subsequent")
    self.assertIn("<already_generated_reward_functions>", content)
    self.assertIn("CONCEPTUALLY DIFFERENT reward function", content)

  def test_load_mutator_prompt_mutate_with_forbidden(self):
    content = prompt_loader.load_mutator_prompt("mutate_with_forbidden")
    self.assertIn(
        "Already existing/generated reward functions in this run", content
    )

  def test_load_mutator_prompt_mutate_without_forbidden(self):
    content = prompt_loader.load_mutator_prompt("mutate_without_forbidden")
    self.assertIn("Generate a new mutated/combined reward function", content)

  def test_load_population_initializer_prompt(self):
    content = prompt_loader.load_population_initializer_prompt(
        "system_instruction"
    )
    self.assertNotEmpty(content)

  def test_load_offspring_generator_prompt(self):
    content = prompt_loader.load_offspring_generator_prompt(
        "planner/user_prompt"
    )
    self.assertNotEmpty(content)


if __name__ == "__main__":
  absltest.main()
