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

"""Utilities to sweep over fields of (nested, frozen) dataclass configs.

A sweep is specified as a list of `<dotted.field.path>=<v1>,<v2>,...` strings,
one per axis. Paths are always the full dotted path of the field inside the
root config, e.g.
`evolution_algorithm_config.training_config.training_interface_config.seed`.
The cartesian product of all axes yields one set of overrides per work unit.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import enum
import itertools
import types
import typing
from typing import Any, Generic, TypeVar

_T = TypeVar("_T")

_TRUE_STRINGS = frozenset({"true", "1", "yes"})
_FALSE_STRINGS = frozenset({"false", "0", "no"})
_NONE_STRING = "none"


@dataclasses.dataclass(frozen=True)
class SweepAxis:
  """A config field and the values it takes in the sweep.

  Attributes:
    path: Full dotted path of the field inside the root config.
    values: Values of the field, already converted to the field's type.
  """

  path: str
  values: tuple[Any, ...]


@dataclasses.dataclass(frozen=True)
class WorkUnitSpec(Generic[_T]):
  """A resolved config and display parameters for a single work unit.

  Attributes:
    config_name: Name of the base config.
    config: The config instance with sweep overrides applied.
    display_params: JSON-friendly hyperparameter mapping, for logging and
      display.
  """

  config_name: str
  config: _T
  display_params: dict[str, Any]


def expand_configs(
    configs: Mapping[str, _T],
    specs: Sequence[str],
    overrides: Sequence[str] = (),
) -> tuple[list[WorkUnitSpec[_T]], list[str]]:
  """Expands one or more base configs across all `specs` sweep axes.

  Args:
    configs: Mapping from config name to base dataclass config instance.
    specs: Sweep specifications (`<dotted.path>=<v1>,<v2>,...`).
    overrides: Overrides (`<dotted.path>=<value>`) applied to every work unit.

  Returns:
    A tuple `(work_units, group_by)` where `work_units` contains one
    `WorkUnitSpec` per `(config, sweep combination)` in Cartesian product order,
    and `group_by` lists the hyperparameter names that vary across work units
    (including `"config"` when multiple configs are given).

  Raises:
    ValueError: If `configs` is empty, any sweep or override spec is invalid
      for any of the provided configs, or a path is both swept and overridden.
  """
  if not configs:
    raise ValueError("At least one config must be provided.")

  include_config = len(configs) > 1
  work_units: list[WorkUnitSpec[_T]] = []
  axes: list[SweepAxis] = []

  for config_name, base_config in configs.items():
    axes = parse_sweep(base_config, specs)
    fixed_overrides = parse_overrides(base_config, overrides)
    swept_and_overridden = sorted(
        {axis.path for axis in axes}.intersection(fixed_overrides)
    )
    if swept_and_overridden:
      raise ValueError(
          f"Paths {swept_and_overridden} are both swept and overridden."
      )
    for sweep_values in expand(axes):
      display_params = (
          {"config": config_name, **to_display_params(sweep_values)}
          if include_config
          else to_display_params(sweep_values)
      )
      config = apply_overrides(base_config, {**sweep_values, **fixed_overrides})
      work_units.append(
          WorkUnitSpec(
              config_name=config_name,
              config=config,
              display_params=display_params,
          )
      )

  group_by = (["config"] if include_config else []) + [
      axis.path for axis in axes
  ]
  return work_units, group_by


def parse_sweep(root: Any, specs: Sequence[str]) -> list[SweepAxis]:
  """Parses `<path>=<v1>,<v2>,...` specs against the fields of `root`.

  Args:
    root: The base config (a dataclass instance) the sweep applies to.
    specs: Sweep specifications, one per axis.

  Returns:
    One axis per spec, in order, with values converted to the field's type.

  Raises:
    ValueError: If a spec is malformed, a path does not exist or does not
      point to a scalar field, a path is repeated, or a value cannot be
      converted to the field's type.
  """
  axes = []
  seen_paths = set()
  for spec in specs:
    path, sep, raw_values = spec.partition("=")
    path = path.strip()
    if not sep or not path or not raw_values:
      raise ValueError(
          f"Invalid sweep spec {spec!r}; expected '<dotted.path>=<v1>,<v2>'."
      )
    if path in seen_paths:
      raise ValueError(f"Sweep path {path!r} is specified more than once.")
    seen_paths.add(path)

    field_type = _resolve_field_type(root, path)
    values = tuple(
        _parse_value(raw.strip(), field_type, path)
        for raw in raw_values.split(",")
    )
    axes.append(SweepAxis(path=path, values=values))
  return axes


def parse_overrides(root: Any, specs: Sequence[str]) -> dict[str, Any]:
  """Parses `<path>=<value>` specs against the fields of `root`.

  Args:
    root: The base config (a dataclass instance) the overrides apply to.
    specs: Override specifications, one per override.

  Returns:
    A mapping from full dotted field path to the new value.

  Raises:
    ValueError: If a spec is invalid or has more than one value.
  """
  overrides = {}
  for axis in parse_sweep(root, specs):
    if len(axis.values) != 1:
      raise ValueError(
          f"Override {axis.path!r} must have exactly one value, got"
          f" {len(axis.values)}."
      )
    overrides[axis.path] = axis.values[0]
  return overrides


def expand(axes: Sequence[SweepAxis]) -> list[dict[str, Any]]:
  """Returns the cartesian product of `axes` as a list of overrides.

  Args:
    axes: The sweep axes.

  Returns:
    One `{path: value}` mapping per combination. A single empty mapping if
    there are no axes, so that a sweep-less launch yields one work unit.
  """
  paths = [axis.path for axis in axes]
  return [
      dict(zip(paths, combination))
      for combination in itertools.product(*(axis.values for axis in axes))
  ]


def apply_overrides(root: _T, overrides: Mapping[str, Any]) -> _T:
  """Returns a copy of `root` with each dotted-path field set to its value.

  Args:
    root: A (possibly frozen) dataclass instance.
    overrides: Mapping from full dotted field path to the new value.

  Returns:
    A new config with the overrides applied; `root` is not modified.
  """
  for path, value in overrides.items():
    root = _replace_path(root, path.split("."), value)
  return root


def to_display_params(overrides: Mapping[str, Any]) -> dict[str, Any]:
  """Converts overrides to JSON-friendly values for experiment parameters."""
  return {
      path: value.value if isinstance(value, enum.Enum) else value
      for path, value in overrides.items()
  }


def _replace_path(obj: Any, parts: Sequence[str], value: Any) -> Any:
  head, rest = parts[0], parts[1:]
  if rest:
    value = _replace_path(getattr(obj, head), rest, value)
  return dataclasses.replace(obj, **{head: value})


def _resolve_field_type(root: Any, path: str) -> Any:
  """Returns the declared type of the field at `path`, validating the path."""
  obj = root
  field_type: Any = None
  parts = path.split(".")
  for i, part in enumerate(parts):
    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
      parent = ".".join(parts[:i])
      raise ValueError(
          f"Sweep path {path!r}: {parent!r} is not a dataclass config."
      )
    field_names = [f.name for f in dataclasses.fields(obj)]
    if part not in field_names:
      raise ValueError(
          f"Sweep path {path!r}: {type(obj).__name__} has no field"
          f" {part!r}. Available fields: {', '.join(field_names)}."
      )
    field_type = typing.get_type_hints(type(obj))[part]
    obj = getattr(obj, part)

  if dataclasses.is_dataclass(obj):
    raise ValueError(
        f"Sweep path {path!r} points to a nested config; sweep one of its"
        " fields instead."
    )
  return field_type


def _parse_value(raw: str, field_type: Any, path: str) -> Any:
  """Converts `raw` to `field_type`, which may be an `X | None` union."""
  origin = typing.get_origin(field_type)
  if origin in (typing.Union, types.UnionType):
    members = typing.get_args(field_type)
    if type(None) in members and raw.lower() == _NONE_STRING:
      return None
    errors = []
    for member in members:
      if member is type(None):
        continue
      try:
        return _parse_scalar(raw, member, path)
      except ValueError as e:
        errors.append(str(e))
    raise ValueError(
        f"Sweep path {path!r}: cannot parse {raw!r} as {field_type}:"
        f" {'; '.join(errors)}"
    )
  return _parse_scalar(raw, field_type, path)


def _parse_scalar(raw: str, field_type: Any, path: str) -> Any:
  """Converts `raw` to a scalar `field_type` (enum, bool, int, float, str)."""
  if isinstance(field_type, type) and issubclass(field_type, enum.Enum):
    valid = [str(member.value) for member in field_type]
    for member in field_type:
      if str(member.value) == raw:
        return member
    raise ValueError(
        f"Sweep path {path!r}: {raw!r} is not a valid"
        f" {field_type.__name__}. Valid values: {', '.join(valid)}."
    )
  if field_type is bool:
    lowered = raw.lower()
    if lowered in _TRUE_STRINGS:
      return True
    if lowered in _FALSE_STRINGS:
      return False
    raise ValueError(f"Sweep path {path!r}: {raw!r} is not a valid bool.")
  if field_type in (int, float):
    try:
      return field_type(raw)
    except ValueError as e:
      raise ValueError(
          f"Sweep path {path!r}: {raw!r} is not a valid {field_type.__name__}."
      ) from e
  if field_type is str:
    return raw
  raise ValueError(
      f"Sweep path {path!r}: fields of type {field_type} cannot be swept."
  )
