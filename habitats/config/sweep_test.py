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

"""Tests for sweep."""

import dataclasses
import enum

from absl.testing import absltest
from habitats.config import sweep


class _Algo(enum.StrEnum):
  A = "a"
  B = "b"


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Inner:
  seed: int = 0
  rate: float = 0.1
  enabled: bool = False
  limit: int | None = None


@dataclasses.dataclass(frozen=True)
class _Root:
  inner: _Inner
  algo: _Algo = _Algo.A
  model: str = "m0"


_ROOT = _Root(inner=_Inner())


class ParseSweepTest(absltest.TestCase):

  def test_parses_values_to_field_types(self):
    axes = sweep.parse_sweep(
        _ROOT,
        [
            "algo=a,b",
            "inner.seed=0,1",
            "inner.rate=0.5",
            "inner.enabled=true,false",
            "inner.limit=none,3",
            "model=gemini-3.8-flash,gemini-3.5-flash",
        ],
    )

    self.assertEqual(
        axes,
        [
            sweep.SweepAxis("algo", (_Algo.A, _Algo.B)),
            sweep.SweepAxis("inner.seed", (0, 1)),
            sweep.SweepAxis("inner.rate", (0.5,)),
            sweep.SweepAxis("inner.enabled", (True, False)),
            sweep.SweepAxis("inner.limit", (None, 3)),
            sweep.SweepAxis("model", ("gemini-3.8-flash", "gemini-3.5-flash")),
        ],
    )

  def test_rejects_unknown_field(self):
    with self.assertRaisesRegex(ValueError, "has no field 'sed'"):
      sweep.parse_sweep(_ROOT, ["inner.sed=1"])

  def test_rejects_short_name(self):
    with self.assertRaisesRegex(ValueError, "has no field 'seed'"):
      sweep.parse_sweep(_ROOT, ["seed=1"])

  def test_rejects_nested_config(self):
    with self.assertRaisesRegex(ValueError, "nested config"):
      sweep.parse_sweep(_ROOT, ["inner=x"])

  def test_rejects_invalid_enum_value(self):
    with self.assertRaisesRegex(ValueError, "Valid values: a, b"):
      sweep.parse_sweep(_ROOT, ["algo=c"])

  def test_rejects_invalid_int(self):
    with self.assertRaisesRegex(ValueError, "not a valid int"):
      sweep.parse_sweep(_ROOT, ["inner.seed=x"])

  def test_rejects_malformed_spec(self):
    with self.assertRaisesRegex(ValueError, "Invalid sweep spec"):
      sweep.parse_sweep(_ROOT, ["inner.seed"])

  def test_rejects_duplicate_path(self):
    with self.assertRaisesRegex(ValueError, "more than once"):
      sweep.parse_sweep(_ROOT, ["inner.seed=1", "inner.seed=2"])


class ExpandTest(absltest.TestCase):

  def test_cartesian_product(self):
    overrides = sweep.expand([
        sweep.SweepAxis("algo", (_Algo.A, _Algo.B)),
        sweep.SweepAxis("inner.seed", (0, 1, 2)),
    ])

    self.assertLen(overrides, 6)
    self.assertEqual(overrides[0], {"algo": _Algo.A, "inner.seed": 0})
    self.assertEqual(overrides[-1], {"algo": _Algo.B, "inner.seed": 2})

  def test_no_axes_yields_single_empty_override(self):
    self.assertEqual(sweep.expand([]), [{}])


class ApplyOverridesTest(absltest.TestCase):

  def test_applies_nested_overrides_without_mutating_root(self):
    config = sweep.apply_overrides(
        _ROOT, {"algo": _Algo.B, "inner.seed": 3, "model": "x"}
    )

    self.assertEqual(
        config, _Root(inner=_Inner(seed=3), algo=_Algo.B, model="x")
    )
    self.assertEqual(_ROOT, _Root(inner=_Inner()))


class ToDisplayParamsTest(absltest.TestCase):

  def test_converts_enums_to_values(self):
    self.assertEqual(
        sweep.to_display_params({"algo": _Algo.B, "inner.seed": 1}),
        {"algo": "b", "inner.seed": 1},
    )


class ExpandConfigsTest(absltest.TestCase):

  def test_single_config_without_sweep(self):
    work_units, group_by = sweep.expand_configs({"cfg_a": _ROOT}, [])

    self.assertEqual(group_by, [])
    self.assertEqual(
        work_units,
        [
            sweep.WorkUnitSpec(
                config_name="cfg_a", config=_ROOT, display_params={}
            )
        ],
    )

  def test_multiple_configs_with_sweep(self):
    root_b = _Root(inner=_Inner(rate=0.5), algo=_Algo.B)
    work_units, group_by = sweep.expand_configs(
        {"cfg_a": _ROOT, "cfg_b": root_b},
        ["inner.seed=1,2"],
    )

    self.assertEqual(group_by, ["config", "inner.seed"])
    self.assertLen(work_units, 4)
    self.assertEqual(
        work_units[0],
        sweep.WorkUnitSpec(
            config_name="cfg_a",
            config=_Root(inner=_Inner(seed=1)),
            display_params={"config": "cfg_a", "inner.seed": 1},
        ),
    )
    self.assertEqual(
        work_units[3],
        sweep.WorkUnitSpec(
            config_name="cfg_b",
            config=_Root(inner=_Inner(seed=2, rate=0.5), algo=_Algo.B),
            display_params={"config": "cfg_b", "inner.seed": 2},
        ),
    )

  def test_applies_overrides_to_every_work_unit(self):
    root_b = _Root(inner=_Inner(rate=0.5), algo=_Algo.B)
    work_units, group_by = sweep.expand_configs(
        {"cfg_a": _ROOT, "cfg_b": root_b}, ["inner.seed=1,2"], ["inner.limit=3"]
    )

    self.assertEqual(group_by, ["config", "inner.seed"])
    self.assertEqual([wu.config.inner.limit for wu in work_units], [3, 3, 3, 3])

  def test_rejects_override_with_multiple_values(self):
    with self.assertRaisesRegex(ValueError, "exactly one value"):
      sweep.expand_configs({"cfg_a": _ROOT}, [], ["inner.seed=1,2"])

  def test_rejects_path_both_swept_and_overridden(self):
    with self.assertRaisesRegex(ValueError, "both swept and overridden"):
      sweep.expand_configs(
          {"cfg_a": _ROOT}, ["inner.seed=1,2"], ["inner.seed=3"]
      )

  def test_rejects_empty_configs(self):
    with self.assertRaisesRegex(ValueError, "At least one config"):
      sweep.expand_configs({}, [])


if __name__ == "__main__":
  absltest.main()
