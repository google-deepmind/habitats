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

"""Gemini-based mutator for reward functions."""

from collections.abc import Sequence

from habitats.lib import llm_client
from habitats.lib import llm_parser
from habitats.lib import prompt_loader


class GeminiMutator:
  """Uses Gemini to generate and mutate reward functions."""

  def __init__(
      self,
      llm_config: llm_client.LLMConfig,
      environment: str,
      model_name: str = "gemini-3.5-flash",
      llm_backend: str = llm_client.GENAI_BACKEND,
  ):
    """Initializes the GeminiMutator."""
    self._llm = llm_client.get_backend(
        llm_backend,
        config=llm_config.with_sub_role("mutator"),
    )
    self.model_name = model_name
    self._system_instruction = prompt_loader.load_system_instruction("mutator")
    self.environment = environment
    self.env_desc = prompt_loader.load_env_description(self.environment)

  def _get_system_instruction(self) -> str:
    return self._system_instruction

  def generate_diverse(
      self,
      forbidden_codes: Sequence[str] | None = None,
      hint: str | None = None,
  ) -> str:
    """Generates a new reward function that is diverse from `forbidden_codes`."""

    if forbidden_codes:
      template = prompt_loader.load_mutator_prompt("diverse_subsequent")
      forbidden_codes_str = "".join(
          f"\n--- Forbidden {i+1} ---\n{code}\n"
          for i, code in enumerate(forbidden_codes)
      )
      prompt = template.format(
          env_desc=self.env_desc, forbidden_codes=forbidden_codes_str
      )
    else:
      template = prompt_loader.load_mutator_prompt("diverse_initial")
      prompt = template.format(env_desc=self.env_desc)

    if hint:
      prompt += f"\n\nHint/Direction for the new reward function:\n{hint}"

    try:
      response = self._llm.generate(
          llm_input=llm_client.LLMInput(
              model=self.model_name,
              prompt=prompt,
              system_instruction=self._get_system_instruction(),
          )
      )
      return llm_parser.extract_code_block(response.text)
    except Exception as e:
      raise RuntimeError(f"Gemini generation failed: {e}") from e

  def mutate(
      self,
      parent_codes: Sequence[str],
      forbidden_codes: Sequence[str] | None = None,
      mutation_hint: str | None = None,
  ) -> str:
    """Generates a mutated reward function based on parents, avoiding `forbidden_codes`."""

    parent_codes_str = "".join(
        f"\n--- Parent {i+1} ---\n{code}\n"
        for i, code in enumerate(parent_codes)
    )

    hint_str = ""
    if mutation_hint:
      hint_str = f"\n\nMutation Instruction/Hint:\n{mutation_hint}"

    if forbidden_codes:
      template = prompt_loader.load_mutator_prompt("mutate_with_forbidden")
      forbidden_codes_str = "".join(
          f"\n--- Forbidden {i+1} ---\n{code}\n"
          for i, code in enumerate(forbidden_codes)
      )
      prompt = template.format(
          env_desc=self.env_desc,
          parent_codes=parent_codes_str,
          forbidden_codes=forbidden_codes_str,
          mutation_hint=hint_str,
      )
    else:
      template = prompt_loader.load_mutator_prompt("mutate_without_forbidden")
      prompt = template.format(
          env_desc=self.env_desc,
          parent_codes=parent_codes_str,
          mutation_hint=hint_str,
      )

    try:
      response = self._llm.generate(
          llm_input=llm_client.LLMInput(
              model=self.model_name,
              prompt=prompt,
              system_instruction=self._get_system_instruction(),
          )
      )
      return llm_parser.extract_code_block(response.text)
    except Exception as e:
      raise RuntimeError(f"Gemini mutation failed: {e}") from e
