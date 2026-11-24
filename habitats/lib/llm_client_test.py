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

"""Unit tests for the portable LLM client: call logging, retries, registry."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from google import genai
from google.genai import errors as genai_errors
from habitats.lib import errors
from habitats.lib import llm_client

_INPUT = llm_client.LLMInput(
    model="gemini-2.5-flash",
    prompt="Generate reward",
    system_instruction="You are an RL expert",
    response_mime_type="application/json",
)


class _FakeLLM(llm_client.LLM):
  """LLM whose backend is a mock, so `generate()` can be tested in isolation."""

  def __init__(self, config: llm_client.LLMConfig, backend: mock.Mock):
    super().__init__(config)
    self._backend = backend

  def _generate(self, llm_input: llm_client.LLMInput) -> llm_client.LLMOutput:
    return self._backend(llm_input)


class _FakeHttpxTimeout(Exception):
  """Stands in for `httpx.ReadTimeout` without depending on httpx."""

  __module__ = "httpx"


def _api_error(code: int) -> genai_errors.APIError:
  return genai_errors.APIError(
      code=code, response_json={"error": {"code": code, "message": "m"}}
  )


class CallLoggingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.logger = mock.Mock()
    self.config = llm_client.LLMConfig(role="mutator", call_logger=self.logger)

  def test_generate_logs_input_output_and_latency(self):
    backend = mock.Mock(return_value=llm_client.LLMOutput(text="def f(): ..."))

    with mock.patch.object(
        llm_client.time, "monotonic", side_effect=[1.0, 1.25]
    ):
      output = _FakeLLM(self.config, backend).generate(_INPUT)

    self.assertEqual(output.text, "def f(): ...")
    self.logger.log_call.assert_called_once_with(
        _INPUT,
        llm_client.LLMOutput(text="def f(): ..."),
        role="mutator",
        backend="_FakeLLM",
        latency_ms=250.0,
        error_message=None,
    )

  def test_generate_logs_failure_and_reraises(self):
    def failing_backend(unused_llm_input):
      try:
        raise TimeoutError("deadline exceeded")
      except TimeoutError as err:
        raise errors.LLMError("GenAI generate_content failed") from err

    llm = _FakeLLM(self.config, mock.Mock(side_effect=failing_backend))
    with self.assertRaisesRegex(errors.LLMError, "generate_content failed"):
      llm.generate(_INPUT)

    kwargs = self.logger.log_call.call_args.kwargs
    self.assertIsNone(self.logger.log_call.call_args.args[1])
    self.assertEqual(
        kwargs["error_message"], "TimeoutError('deadline exceeded')"
    )

  def test_logging_failure_does_not_fail_the_llm_call(self):
    self.logger.log_call.side_effect = errors.LoggingError("sink unavailable")
    backend = mock.Mock(return_value=llm_client.LLMOutput(text="ok"))

    output = _FakeLLM(self.config, backend).generate(_INPUT)  # No raise.

    self.assertEqual(output.text, "ok")
    self.logger.log_call.assert_called_once()

  def test_logging_is_off_by_default(self):
    backend = mock.Mock(return_value=llm_client.LLMOutput(text="ok"))

    output = _FakeLLM(llm_client.LLMConfig(role="mutator"), backend).generate(
        _INPUT
    )

    self.assertEqual(output.text, "ok")
    self.logger.log_call.assert_not_called()

  def test_sub_role_keeps_the_logger(self):
    backend = mock.Mock(return_value=llm_client.LLMOutput(text="ok"))

    _FakeLLM(self.config.with_sub_role("planner"), backend).generate(_INPUT)

    self.assertEqual(
        self.logger.log_call.call_args.kwargs["role"], "mutator:planner"
    )


class GenAiLLMTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.client = mock.create_autospec(genai.Client, instance=True)
    self.enter_context(
        mock.patch.object(genai, "Client", return_value=self.client)
    )
    self.sleep = self.enter_context(mock.patch("time.sleep", autospec=True))

  def _llm(self) -> llm_client.GenAiLLM:
    return llm_client.GenAiLLM(llm_client.LLMConfig(role="mutator", key="k"))

  def test_requires_api_key(self):
    with self.assertRaises(ValueError):
      llm_client.GenAiLLM(llm_client.LLMConfig(role="mutator"))

  def test_generate_success(self):
    self.client.models.generate_content.return_value = mock.Mock(text="hi")

    self.assertEqual(self._llm().generate(_INPUT).text, "hi")

    _, kwargs = self.client.models.generate_content.call_args
    self.assertEqual(kwargs["model"], "gemini-2.5-flash")
    self.assertEqual(kwargs["contents"], "Generate reward")
    self.assertEqual(
        kwargs["config"].system_instruction, "You are an RL expert"
    )
    self.assertEqual(kwargs["config"].response_mime_type, "application/json")

  @parameterized.parameters(408, 429, 500, 502, 503, 504)
  def test_retries_transient_api_errors(self, status_code):
    self.client.models.generate_content.side_effect = [
        _api_error(status_code),
        _api_error(status_code),
        mock.Mock(text="success"),
    ]

    self.assertEqual(self._llm().generate(_INPUT).text, "success")
    self.assertEqual(self.client.models.generate_content.call_count, 3)
    self.assertEqual(self.sleep.call_count, 2)

  @parameterized.parameters(
      _FakeHttpxTimeout("The read operation timed out"),
      TimeoutError("Operation timed out"),
  )
  def test_retries_timeouts(self, timeout_exc):
    self.client.models.generate_content.side_effect = [
        timeout_exc,
        timeout_exc,
        mock.Mock(text="success"),
    ]

    self.assertEqual(self._llm().generate(_INPUT).text, "success")
    self.assertEqual(self.client.models.generate_content.call_count, 3)
    self.assertEqual(self.sleep.call_count, 2)

  @parameterized.parameters(400, 403, 404)
  def test_no_retry_on_non_transient_errors(self, status_code):
    api_error = _api_error(status_code)
    self.client.models.generate_content.side_effect = api_error

    with self.assertRaises(errors.LLMError) as raised:
      self._llm().generate(_INPUT)

    self.assertIs(raised.exception.__cause__, api_error)
    self.client.models.generate_content.assert_called_once()
    self.sleep.assert_not_called()

  def test_unrelated_errors_propagate_unchanged(self):
    self.client.models.generate_content.side_effect = KeyError("bug")

    with self.assertRaises(KeyError):
      self._llm().generate(_INPUT)
    self.sleep.assert_not_called()

  def test_raises_llm_error_after_retries(self):
    api_error = _api_error(503)
    self.client.models.generate_content.side_effect = api_error

    with self.assertRaisesRegex(
        errors.LLMError, "GenAI generate_content failed for gemini-2.5-flash"
    ) as raised:
      self._llm().generate(_INPUT)

    self.assertIs(raised.exception.__cause__, api_error)
    # 1 initial attempt + 5 retries.
    self.assertEqual(self.client.models.generate_content.call_count, 6)

  def test_backoff_is_exponential_and_capped(self):
    self.client.models.generate_content.side_effect = _api_error(503)
    with mock.patch.object(llm_client.random, "uniform", return_value=1.0):
      with self.assertRaises(errors.LLMError):
        self._llm().generate(_INPUT)

    self.assertEqual(
        [c.args[0] for c in self.sleep.call_args_list],
        [1.0, 2.0, 4.0, 8.0, 16.0],
    )


class BackendRegistryTest(absltest.TestCase):

  def test_genai_backend_is_builtin(self):
    config = llm_client.LLMConfig(role="mutator", key="k")
    self.assertIsInstance(
        llm_client.get_backend(llm_client.GENAI_BACKEND, config),
        llm_client.GenAiLLM,
    )

  def test_registered_backend_is_used(self):
    factory = mock.Mock(return_value=mock.create_autospec(llm_client.LLM))
    llm_client.register_backend("test_backend")(factory)
    self.addCleanup(llm_client._BACKENDS.pop, "test_backend")  # pylint: disable=protected-access
    config = llm_client.LLMConfig(role="mutator")

    llm = llm_client.get_backend("test_backend", config)

    factory.assert_called_once_with(config)
    self.assertIs(llm, factory.return_value)

  def test_registering_a_name_twice_fails(self):
    with self.assertRaisesRegex(ValueError, "already registered"):
      llm_client.register_backend(llm_client.GENAI_BACKEND)(mock.Mock())

  def test_unknown_backend(self):
    with self.assertRaisesRegex(ValueError, "'nope' is not registered"):
      llm_client.get_backend("nope", llm_client.LLMConfig(role="mutator"))


if __name__ == "__main__":
  absltest.main()
