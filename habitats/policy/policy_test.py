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

"""Tests for policy base classes and configurations."""

import dataclasses
import os
import tempfile
from typing import override

from absl.testing import absltest
from dm_env import specs
from habitats.policy import policy as policy_lib
import numpy as np



class _DummyPolicy(policy_lib.Policy[int]):
  """Dummy policy for testing MetaPolicy base contract."""

  def __init__(self, config: policy_lib.ModelConfig, default_action: int = 0):
    self.config: policy_lib.ModelConfig = config
    self.action_val = default_action

  @override
  def get_action(
      self,
      state: int,
      exploit: bool = False,
      legal_actions: np.ndarray | None = None,
  ) -> np.ndarray:
    return np.array(self.action_val)

  @override
  def reset(self) -> None:
    self.action_val = 0

  @override
  def get_params(self) -> int:
    return self.action_val

  @override
  def set_params(self, params: int) -> None:
    self.action_val = int(params)

  @override
  def save(self, path: str) -> None:
    with open(os.path.join(path, "param.txt"), "w") as f:
      f.write(str(self.action_val))

  @override
  def load(self, path: str) -> None:
    with open(os.path.join(path, "param.txt"), "r") as f:
      self.action_val = int(f.read().strip())


class _DummyMetaPolicy(policy_lib.MetaPolicy[int, _DummyPolicy, _DummyPolicy]):
  """Concrete MetaPolicy over _DummyPolicy for testing."""

  @override
  def _init_router(self) -> _DummyPolicy:
    return _DummyPolicy(self.config.router_config, default_action=0)

  @override
  def _init_sub_policies(self) -> list[_DummyPolicy]:
    sub_policies = []
    for sub_path in self.config.sub_policy_paths:
      sub_policy = _DummyPolicy(self.config.policy_config)
      sub_policy.load(sub_path)
      sub_policies.append(sub_policy)
    return sub_policies

  @override
  def get_meta_action(self, state: int, exploit: bool = False) -> int:
    return int(self.router.get_action(state, exploit=exploit))


class PolicyTest(absltest.TestCase):

  def test_model_config_apply_environment_spec(self):
    obs_spec = specs.DiscreteArray(num_values=15)
    act_spec = specs.DiscreteArray(num_values=4)
    cfg = policy_lib.ModelConfig().apply_environment_spec(
        observations=obs_spec,
        actions=act_spec,
    )
    self.assertEqual(cfg.num_states, 15)
    self.assertEqual(cfg.num_actions, 4)

  def test_meta_policy_config_validation(self):
    with self.assertRaisesRegex(
        ValueError, "sub_policy_paths must contain at least 2 paths"
    ):
      policy_lib.MetaModelConfig(sub_policy_paths=("/ckpt/0",))

    with self.assertRaisesRegex(
        ValueError, "requires at least 2 sub_policy_paths"
    ):
      policy_lib.MetaModelConfig().apply_environment_spec(
          observations=specs.DiscreteArray(num_values=4),
          actions=specs.DiscreteArray(num_values=2),
      )

  def test_meta_policy_config_apply_environment_spec_and_from_dict(self):
    obs_spec = specs.DiscreteArray(num_values=12)
    act_spec = specs.DiscreteArray(num_values=6)
    base_meta_cfg = policy_lib.MetaModelConfig().with_sub_policy_paths(
        ["/ckpt/0", "/ckpt/1", "/ckpt/2"]
    )

    updated_cfg = base_meta_cfg.apply_environment_spec(
        observations=obs_spec,
        actions=act_spec,
    )
    self.assertEqual(updated_cfg.nb_policies, 3)
    self.assertEqual(
        updated_cfg.sub_policy_paths,
        ("/ckpt/0", "/ckpt/1", "/ckpt/2"),
    )
    self.assertEqual(updated_cfg.policy_config.num_states, 12)
    self.assertEqual(updated_cfg.policy_config.num_actions, 6)
    self.assertEqual(updated_cfg.router_config.num_states, 12)
    self.assertEqual(updated_cfg.router_config.num_actions, 3)

    restored_cfg = policy_lib.MetaModelConfig.from_dict(
        dataclasses.asdict(updated_cfg)
    )
    self.assertEqual(restored_cfg, updated_cfg)

  def test_meta_policy_routing_save_load_and_sub_policy_paths_init(self):
    policy_cfg = policy_lib.ModelConfig(num_states=10, num_actions=5)

    with tempfile.TemporaryDirectory() as tmpdir:
      sub_paths = []
      for idx, action_val in enumerate([0, 10, 20]):
        sub_dir = os.path.join(tmpdir, f"standalone_{idx}")
        os.makedirs(sub_dir, exist_ok=True)
        standalone = _DummyPolicy(policy_cfg, default_action=action_val)
        standalone.save(sub_dir)
        sub_paths.append(sub_dir)

      meta_cfg = (
          policy_lib.MetaModelConfig(policy_config=policy_cfg)
          .with_sub_policy_paths(sub_paths[:2])
          .apply_environment_spec(
              observations=specs.DiscreteArray(num_values=10),
              actions=specs.DiscreteArray(num_values=5),
          )
      )
      meta_policy = _DummyMetaPolicy(meta_cfg)
      meta_policy.router.set_params(1)
      self.assertEqual(int(meta_policy.get_action(0, exploit=True)), 10)

      with self.assertRaisesRegex(ValueError, "out of bounds"):
        meta_policy.get_primitive_action(0, meta_action=2)

      ckpt_dir = os.path.join(tmpdir, "meta_ckpt")
      meta_policy.save(ckpt_dir)

      loaded_meta = _DummyMetaPolicy(meta_cfg)
      loaded_meta.load(ckpt_dir)
      self.assertEqual(loaded_meta.get_params(), meta_policy.get_params())

      cfg_with_3_subs = meta_cfg.with_sub_policy_paths(
          sub_paths
      ).apply_environment_spec(
          observations=specs.DiscreteArray(num_values=10),
          actions=specs.DiscreteArray(num_values=5),
      )
      meta_from_3_subs = _DummyMetaPolicy(cfg_with_3_subs)
      self.assertEqual(meta_from_3_subs.config.nb_policies, 3)
      self.assertEqual(meta_from_3_subs.config.router_config.num_actions, 3)
      self.assertLen(meta_from_3_subs.policies, 3)
      meta_from_3_subs.router.set_params(2)
      self.assertEqual(int(meta_from_3_subs.get_action(0, exploit=True)), 20)

      with self.assertRaisesRegex(
          ValueError,
          r"Number of sub-policies in checkpoint \(2\) does not match",
      ):
        meta_from_3_subs.load(ckpt_dir)


if __name__ == "__main__":
  absltest.main()
