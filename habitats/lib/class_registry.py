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

"""Utilities for registering classes for use in configs."""

from collections.abc import Callable
import functools
from typing import TypeVar

T = TypeVar("T")
# Registry key: a `StrEnum` member for closed sets of names, or a plain `str`
# for registries that other packages extend.
N = TypeVar("N", bound=str)


def register(
    name: N,
    registry: dict[N, T],
) -> Callable[[T], T]:
  """Registers a class or callable under `name` in `registry`."""

  def _decorator(target: T) -> T:
    if name in registry:
      raise ValueError(f"Class or callable '{name}' is already registered.")
    registry[name] = target
    return target

  return _decorator


def get(
    name: N,
    registry: dict[N, T],
) -> T:
  """Returns the item with the given name from `registry`."""
  if name not in registry:
    raise ValueError(f"Class or callable '{name}' is not registered.")
  return registry[name]


def declare_registry(
    registry: dict[N, T],
) -> tuple[
    Callable[[N], Callable[[T], T]],
    Callable[[N], T],
]:
  """Declares a typed `(register, get)` pair bound to `registry`."""
  register_fn = functools.partial(register, registry=registry)
  get_fn = functools.partial(get, registry=registry)
  return register_fn, get_fn
