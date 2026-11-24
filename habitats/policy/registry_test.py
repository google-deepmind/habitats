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

"""Tests for policy registry."""

import os

from absl.testing import absltest
from habitats.policy import neural
from habitats.policy import policy
from habitats.policy import registry
from habitats.policy import tabular


class PolicyRegistryTest(absltest.TestCase):

  def test_every_policy_name_is_registered_exactly_once(self):
    for name in policy.PolicyName:
      is_registered_as_policy = name in registry.POLICY_REGISTRY
      is_registered_as_meta_policy = name in registry.META_POLICY_REGISTRY
      self.assertNotEqual(
          is_registered_as_policy,
          is_registered_as_meta_policy,
          msg=f"{name} must be registered in exactly one registry.",
      )

  def test_is_meta_policy(self):
    self.assertTrue(registry.is_meta_policy(policy.PolicyName.TABULAR_META))
    self.assertFalse(registry.is_meta_policy(policy.PolicyName.TABULAR))
    self.assertFalse(registry.is_meta_policy(policy.PolicyName.MLP))

  def test_get_registered_classes(self):
    self.assertIs(
        registry.get_policy(policy.PolicyName.TABULAR), tabular.TabularPolicy
    )
    self.assertIs(registry.get_policy(policy.PolicyName.MLP), neural.MLPPolicy)
    self.assertIs(
        registry.get_meta_policy(policy.PolicyName.TABULAR_META),
        tabular.TabularMetaPolicy,
    )

  def test_create_builds_policies(self):
    model_config = policy.ModelConfig(num_states=3, num_actions=2)

    self.assertIsInstance(
        registry.create(policy.PolicyName.TABULAR, model_config),
        tabular.TabularPolicy,
    )
    self.assertIsInstance(
        registry.create(policy.PolicyName.MLP, model_config),
        neural.MLPPolicy,
    )

  def test_create_builds_meta_policies(self):
    model_config = policy.ModelConfig(num_states=3, num_actions=2)
    temp_dir = self.create_tempdir().full_path
    sub_policy_paths = []
    for idx in range(2):
      sub_policy_path = os.path.join(temp_dir, f"sub_policy_{idx}")
      tabular.TabularPolicy(model_config).save(sub_policy_path)
      sub_policy_paths.append(sub_policy_path)
    meta_model_config = policy.MetaModelConfig(
        policy_config=model_config,
        router_config=model_config,
        sub_policy_paths=tuple(sub_policy_paths),
    )

    self.assertIsInstance(
        registry.create(policy.PolicyName.TABULAR_META, meta_model_config),
        tabular.TabularMetaPolicy,
    )

  def test_create_raises_on_mismatched_config_type(self):
    with self.assertRaisesRegex(TypeError, "requires a MetaModelConfig"):
      registry.create(policy.PolicyName.TABULAR_META, policy.ModelConfig())
    with self.assertRaisesRegex(TypeError, "requires a ModelConfig"):
      registry.create(policy.PolicyName.TABULAR, policy.MetaModelConfig())


if __name__ == "__main__":
  absltest.main()
