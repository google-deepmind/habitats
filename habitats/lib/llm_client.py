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

"""LLM client abstraction for Habitats with a pluggable backend registry.

`LLM` is the interface the evolution code programs against. `GenAiLLM` is the
built-in backend on top of the `google.genai` SDK. Additional backends register
themselves with the `register_backend` decorator and are selected by name
through `get_backend`.

Every `LLM.generate()` call is recorded by the `CallLogger` carried in its
`LLMConfig`, if any; this is how run-logging systems hook in without the core
depending on them.
"""

import abc
from collections.abc import Callable
import dataclasses
import functools
import random
import time
from typing import Protocol

from absl import logging
from google import genai
from google.api_core import exceptions as google_exceptions
from google.genai import errors as genai_errors
from google.genai import types
from habitats.lib import class_registry
from habitats.lib import errors

# Exceptions raised by the GenAI SDK that `GenAiLLM` retries on and, once
# retries are exhausted, translates into `errors.LLMError`. Transport errors
# from the SDK's HTTP client (the `httpx` package) are recognised by module
# name in `_is_http_client_error` so that this package does not depend on it.
_GENAI_ERRORS = (
    genai_errors.APIError,
    TimeoutError,
    google_exceptions.GoogleAPIError,
)
_HTTP_CLIENT_MODULE = "httpx"

# Retry schedule for transient GenAI errors: fuzzed exponential backoff.
_GENAI_NUM_RETRIES = 5
_GENAI_INITIAL_DELAY_SEC = 1.0
_GENAI_MAX_DELAY_SEC = 60.0

GENAI_BACKEND = "genai"


@dataclasses.dataclass(frozen=True)
class LLMOutput:
  text: str


@dataclasses.dataclass(frozen=True)
class LLMInput:
  model: str
  prompt: str
  system_instruction: str
  thinking_budget: int | None = None
  response_mime_type: str | None = None


class CallLogger(Protocol):
  """Receives one record per `LLM.generate()` call."""

  def log_call(
      self,
      llm_input: LLMInput,
      llm_output: LLMOutput | None,
      *,
      role: str,
      backend: str,
      latency_ms: float,
      error_message: str | None,
  ) -> None:
    """Records a call.

    Args:
      llm_input: Request as sent to the backend.
      llm_output: Backend response, or None if the call raised.
      role: Caller role from `LLMConfig.role`, e.g. "mutator" or "planner".
      backend: LLM class name, e.g. "GenAiLLM".
      latency_ms: Wall-clock duration of the call, including retries.
      error_message: Exception message if the call raised; None on success.

    Raises:
      errors.LoggingError: If the record could not be written. `LLM.generate`
        logs and swallows it so that logging never fails an LLM call.
    """


@dataclasses.dataclass(frozen=True)
class LLMConfig:
  """LLM configuration."""

  # Role of the LLM (mutator, etc). This is mainly used for logging purposes.
  role: str
  key: str | None = None
  # Per-call RPC/HTTP timeout in milliseconds (10 min as default).
  timeout_ms: int = 600_000
  # Receives one record per `generate()` call; None disables call logging.
  # Carried in the config so that it reaches every LLM created from it, e.g.
  # the per-role clients derived with `with_sub_role`.
  call_logger: CallLogger | None = None

  def with_sub_role(self, sub_role: str) -> "LLMConfig":
    return dataclasses.replace(
        self,
        role=f"{self.role}:{sub_role}",
    )


def _is_http_client_error(e: Exception) -> bool:
  """Returns True for transport errors raised by the SDK's HTTP client."""
  return type(e).__module__.split(".")[0] == _HTTP_CLIENT_MODULE


def _is_backend_error(e: Exception) -> bool:
  """Returns True if `e` is an error from the GenAI backend or its transport."""
  return isinstance(e, _GENAI_ERRORS) or _is_http_client_error(e)


def is_transient_api_error(e: Exception) -> bool:
  """Returns True if `e` is a transient Gemini API or network error."""
  if isinstance(e, genai_errors.APIError):
    return e.code in (408, 429, 500, 502, 503, 504)
  return isinstance(
      e, (TimeoutError, google_exceptions.GoogleAPIError)
  ) or _is_http_client_error(e)


class LLM(abc.ABC):
  """Abstract base class for LLM clients.

  `generate()` is the public entry point: it times the call, delegates to the
  backend-specific `_generate()`, and records the request, response (or
  error), and latency through the config's `CallLogger`, if any.
  """

  def __init__(self, config: LLMConfig):
    self._config = config

  @property
  def config(self) -> LLMConfig:
    return self._config

  def generate(self, llm_input: LLMInput) -> LLMOutput:
    """Generates content using the LLM and logs the call.

    Args:
      llm_input: Model, prompt, and generation settings.

    Returns:
      The backend response.

    Raises:
      errors.LLMError: If the backend call fails, after the failure is logged.
    """
    start_time = time.monotonic()
    try:
      llm_output = self._generate(llm_input)
    except errors.LLMError as err:
      # Record the backend's own exception when there is one.
      cause = err if err.__cause__ is None else err.__cause__
      self._log_call(llm_input, None, start_time, error_message=repr(cause))
      raise
    self._log_call(llm_input, llm_output, start_time)
    return llm_output

  @abc.abstractmethod
  def _generate(self, llm_input: LLMInput) -> LLMOutput:
    """Backend-specific generation; raises `errors.LLMError` on failure."""

  def _log_call(
      self,
      llm_input: LLMInput,
      llm_output: LLMOutput | None,
      start_time: float,
      error_message: str | None = None,
  ) -> None:
    """Records the call; a logging failure must never fail the LLM call."""
    if self._config.call_logger is None:
      return
    try:
      self._config.call_logger.log_call(
          llm_input,
          llm_output,
          role=self._config.role,
          backend=type(self).__name__,
          latency_ms=(time.monotonic() - start_time) * 1000.0,
          error_message=error_message,
      )
    except errors.LoggingError as err:
      logging.warning(
          "[%s] Dropping call log record for role %r: %s (%s)",
          type(self).__name__,
          self._config.role,
          err,
          err.__cause__,
      )


# Backend name -> LLM factory. `GenAiLLM` registers itself below; backends
# outside this package register themselves on import with the decorator.
_BACKENDS: dict[str, Callable[[LLMConfig], LLM]] = {}
register_backend, _get_backend_factory = class_registry.declare_registry(
    _BACKENDS
)


def get_backend(name: str, config: LLMConfig) -> LLM:
  """Returns an LLM instance for the named backend."""
  return _get_backend_factory(name)(config)


@register_backend(GENAI_BACKEND)
class GenAiLLM(LLM):
  """Wrapper around genai.Client with centralized retries."""

  def __init__(self, config: LLMConfig):
    super().__init__(config=config)

    if not config.key:
      raise ValueError("API key must be provided in LLMConfig for GenAiLLM.")

  @functools.cached_property
  def _client(self) -> genai.Client:
    return genai.Client(
        api_key=self._config.key,
        http_options=types.HttpOptions(timeout=self._config.timeout_ms),
    )

  def _generate(self, llm_input: LLMInput) -> LLMOutput:
    """Generates content using genai.Client with retries."""
    try:
      return _retry_transient(lambda: self._generate_once(llm_input))
    except Exception as err:
      if not _is_backend_error(err):
        raise
      raise errors.LLMError(
          f"GenAI generate_content failed for {llm_input.model}"
      ) from err

  def _generate_once(self, llm_input: LLMInput) -> LLMOutput:
    response = self._client.models.generate_content(
        model=llm_input.model,
        contents=llm_input.prompt,
        config=types.GenerateContentConfig(
            system_instruction=llm_input.system_instruction,
            response_mime_type=llm_input.response_mime_type,
        ),
    )
    return LLMOutput(text=response.text or "")


def _retry_transient(fn: Callable[[], LLMOutput]) -> LLMOutput:
  """Calls `fn`, retrying transient GenAI SDK errors with fuzzed backoff.

  Non-transient errors and the last transient error are re-raised unchanged.

  Args:
    fn: Zero-argument callable performing one request.

  Returns:
    The result of the first successful call.
  """
  delay = _GENAI_INITIAL_DELAY_SEC
  for attempt in range(_GENAI_NUM_RETRIES + 1):
    try:
      return fn()
    except Exception as err:  # pylint: disable=broad-exception-caught
      if attempt == _GENAI_NUM_RETRIES or not is_transient_api_error(err):
        raise
      sleep_sec = random.uniform(0.5, 1.5) * delay
      logging.warning(
          "Transient LLM error (%r); retry %d/%d in %.1fs",
          err,
          attempt + 1,
          _GENAI_NUM_RETRIES,
          sleep_sec,
      )
      time.sleep(sleep_sec)
      delay = min(delay * 2, _GENAI_MAX_DELAY_SEC)
  raise AssertionError("unreachable")  # pragma: no cover
