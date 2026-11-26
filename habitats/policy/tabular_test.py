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

"""Tests for tabular and tabular meta-policies."""

import os
import tempfile

from absl.testing import absltest
import dm_env
from dm_env import specs
from habitats.policy import policy as policy_lib
from habitats.policy import tabular
import numpy as np



class TabularPolicyTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = self.create_tempdir().full_path
    self.num_states = 6
    self.num_actions = 4
    self.model_config = policy_lib.ModelConfig(
        num_states=self.num_states,
        num_actions=self.num_actions,
    )

  def _make_meta_config(self, nb_policies: int) -> policy_lib.MetaModelConfig:
    sub_paths = []
    for idx in range(nb_policies):
      sub_policy = tabular.TabularPolicy(self.model_config)
      sub_path = os.path.join(self.temp_dir, f"init_sub_{idx}")
      sub_policy.save(sub_path)
      sub_paths.append(sub_path)
    return (
        policy_lib.MetaModelConfig(policy_config=self.model_config)
        .with_sub_policy_paths(sub_paths)
        .apply_environment_spec(
            observations=specs.DiscreteArray(num_values=self.num_states),
            actions=specs.DiscreteArray(num_values=self.num_actions),
        )
    )

  def test_tabular_meta_policy_initialization_and_routing(self):
    meta_config = self._make_meta_config(nb_policies=2)
    meta_policy = tabular.TabularMetaPolicy(meta_config)

    self.assertEqual(meta_policy.q_table.shape, (self.num_states, 2))
    self.assertEqual(meta_policy.num_states, self.num_states)
    self.assertEqual(meta_policy.num_actions, 2)
    self.assertLen(meta_policy.policies, 2)
    for sub_policy in meta_policy.policies:
      self.assertEqual(
          sub_policy.q_table.shape, (self.num_states, self.num_actions)
      )

    # Set sub-policy 0 to prefer primitive action 1 at state 0
    meta_policy.policies[0].q_table[0, 1] = 10.0
    # Set sub-policy 1 to prefer primitive action 3 at state 0
    meta_policy.policies[1].q_table[0, 3] = 10.0

    # Route state 0 -> sub-policy 0 -> primitive action 1
    meta_policy.q_table[0, 0] = 5.0
    meta_policy.q_table[0, 1] = 1.0
    self.assertEqual(meta_policy.get_meta_action(0, exploit=True), 0)
    self.assertEqual(int(meta_policy.get_action(0, exploit=True)), 1)

    # Route state 0 -> sub-policy 1 -> primitive action 3
    meta_policy.q_table[0, 1] = 20.0
    self.assertEqual(meta_policy.get_meta_action(0, exploit=True), 1)
    self.assertEqual(int(meta_policy.get_action(0, exploit=True)), 3)

  def test_tabular_meta_policy_save_and_load_roundtrip(self):
    meta_config = self._make_meta_config(nb_policies=2)
    meta_policy = tabular.TabularMetaPolicy(meta_config)
    meta_policy.q_table = np.arange(self.num_states * 2, dtype=float).reshape(
        self.num_states, 2
    )
    meta_policy.policies[0].q_table = np.ones(
        (self.num_states, self.num_actions)
    )
    meta_policy.policies[1].q_table = (
        np.ones((self.num_states, self.num_actions)) * 2.0
    )

    with tempfile.TemporaryDirectory() as tmpdir:
      ckpt_path = os.path.join(tmpdir, "meta_policy_ckpt")
      meta_policy.save(ckpt_path)

      restored = tabular.TabularMetaPolicy(meta_config)
      restored.load(ckpt_path)

      np.testing.assert_array_equal(restored.q_table, meta_policy.q_table)
      for orig_sub, rest_sub in zip(
          meta_policy.policies, restored.policies, strict=True
      ):
        np.testing.assert_array_equal(rest_sub.q_table, orig_sub.q_table)

  def test_tabular_meta_policy_init_with_sub_policy_paths(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      paths = []
      for idx in range(3):
        sub = tabular.TabularPolicy(self.model_config)
        sub.q_table[:, idx] = 100.0
        sub_path = os.path.join(tmpdir, f"agent_{idx}")
        sub.save(sub_path)
        paths.append(sub_path)

      meta_config = (
          policy_lib.MetaModelConfig()
          .with_sub_policy_paths(paths)
          .apply_environment_spec(
              observations=specs.DiscreteArray(num_values=self.num_states),
              actions=specs.DiscreteArray(num_values=self.num_actions),
          )
      )
      meta_policy = tabular.TabularMetaPolicy(meta_config)
      self.assertEqual(meta_policy.config.nb_policies, 3)
      self.assertEqual(meta_policy.num_actions, 3)
      self.assertEqual(meta_policy.q_table.shape, (self.num_states, 3))
      self.assertLen(meta_policy.policies, 3)

      for idx in range(3):
        meta_policy.q_table[2, :] = 0.0
        meta_policy.q_table[2, idx] = 1.0
        self.assertEqual(int(meta_policy.get_action(2, exploit=True)), idx)

  def test_tabular_policy_exploit_breaks_ties_deterministically(self):
    policy = tabular.TabularPolicy(self.model_config)
    # With all Q-values tied at 0.0, exploit=True must deterministically pick 0.
    actions = [int(policy.get_action(0, exploit=True)) for _ in range(20)]
    self.assertEqual(actions, [0] * 20)

    # Tie between action 2 and action 3 at max_q = 5.0 -> picks first max (2).
    policy.q_table[1, 2] = 5.0
    policy.q_table[1, 3] = 5.0
    tied_actions = [int(policy.get_action(1, exploit=True)) for _ in range(20)]
    self.assertEqual(tied_actions, [2] * 20)

  def test_tabular_policy_only_returns_legal_actions(self):
    policy = tabular.TabularPolicy(self.model_config)
    policy.q_table[0] = [1.0, 5.0, 3.0, 0.0]
    legal_actions = np.array([True, False, True, True])

    # Action 1 has the highest value but is illegal: the best legal one wins.
    self.assertEqual(
        int(policy.get_action(0, exploit=True, legal_actions=legal_actions)), 2
    )
    explored = {
        int(policy.get_action(0, exploit=False, legal_actions=legal_actions))
        for _ in range(20)
    }
    self.assertEqual(explored, {2})

  def test_tabular_policy_breaks_ties_among_legal_actions_only(self):
    policy = tabular.TabularPolicy(self.model_config)
    legal_actions = np.array([False, True, False, True])

    # All Q-values are tied at 0.0, so every legal action is a candidate.
    self.assertEqual(
        int(policy.get_action(0, exploit=True, legal_actions=legal_actions)), 1
    )
    np.random.seed(0)
    explored = {
        int(policy.get_action(0, exploit=False, legal_actions=legal_actions))
        for _ in range(50)
    }
    self.assertEqual(explored, {1, 3})

  def test_tabular_policy_ignores_mask_without_legal_actions(self):
    policy = tabular.TabularPolicy(self.model_config)
    policy.q_table[0] = [1.0, 5.0, 3.0, 0.0]

    action = policy.get_action(
        0, exploit=True, legal_actions=np.zeros(self.num_actions, dtype=bool)
    )

    self.assertEqual(int(action), 1)

  def test_tabular_meta_policy_masks_sub_policy_not_router(self):
    meta_policy = tabular.TabularMetaPolicy(self._make_meta_config(2))
    meta_policy.policies[0].q_table[0] = [0.0, 10.0, 4.0, 0.0]
    meta_policy.policies[1].q_table[0] = [0.0, 0.0, 0.0, 10.0]
    # The router prefers sub-policy 0, whose favorite action 1 is illegal.
    meta_policy.q_table[0] = [5.0, 1.0]
    legal = np.array([True, False, True, True])

    action = meta_policy.get_action(0, exploit=True, legal_actions=legal)

    # Sub-policy 0 is still the one selected, and falls back to action 2.
    self.assertEqual(int(action), 2)
    self.assertEqual(
        int(meta_policy.get_primitive_action(0, 1, legal_actions=legal)),
        3,
    )


class _FixedMaskEnv(dm_env.Environment):
  """Single-state environment that records actions and reports a fixed mask."""

  def __init__(self, legal_actions: list[bool]):
    self._legal_actions = legal_actions
    self.actions: list[int] = []

  def legal_actions(self) -> np.ndarray:
    return np.array(self._legal_actions, dtype=bool)

  def reset(self) -> dm_env.TimeStep:
    return dm_env.restart(0)

  def step(self, action: int) -> dm_env.TimeStep:
    self.actions.append(int(action))
    return dm_env.transition(reward=0.0, observation=0)

  def observation_spec(self) -> specs.DiscreteArray:
    return specs.DiscreteArray(num_values=6)

  def action_spec(self) -> specs.DiscreteArray:
    return specs.DiscreteArray(num_values=len(self._legal_actions))


class _NoMaskEnv(dm_env.Environment):
  """Single-state environment that records actions and reports no mask."""

  def __init__(self):
    self.actions: list[int] = []

  def reset(self) -> dm_env.TimeStep:
    return dm_env.restart(0)

  def step(self, action: int) -> dm_env.TimeStep:
    self.actions.append(int(action))
    return dm_env.transition(reward=0.0, observation=0)

  def observation_spec(self) -> specs.DiscreteArray:
    return specs.DiscreteArray(num_values=6)

  def action_spec(self) -> specs.DiscreteArray:
    return specs.DiscreteArray(num_values=4)


class MetaPolicyEnvWrapperMaskingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    model_config = policy_lib.ModelConfig(num_states=6, num_actions=4)
    sub_paths = []
    # Sub-policy 0 ranks the actions 1 > 2 > 0 = 3; sub-policy 1 prefers 3.
    for idx, q_values in enumerate(
        ([0.0, 10.0, 4.0, 0.0], [0.0, 0.0, 0.0, 10.0])
    ):
      sub_policy = tabular.TabularPolicy(model_config)
      sub_policy.q_table[0] = q_values
      sub_path = os.path.join(self.create_tempdir().full_path, f"sub_{idx}")
      sub_policy.save(sub_path)
      sub_paths.append(sub_path)
    meta_config = (
        policy_lib.MetaModelConfig(policy_config=model_config)
        .with_sub_policy_paths(sub_paths)
        .apply_environment_spec(
            observations=specs.DiscreteArray(num_values=6),
            actions=specs.DiscreteArray(num_values=4),
        )
    )
    self.meta_policy = tabular.TabularMetaPolicy(meta_config)

  def test_unmasked_wrapper_steps_with_illegal_sub_policy_action(self):
    env = _FixedMaskEnv([True, False, True, True])
    wrapped = self.meta_policy.wrap_env(env)
    wrapped.reset()

    wrapped.step(0)

    self.assertEqual(env.actions, [1])

  def test_masked_wrapper_steps_with_best_legal_sub_policy_action(self):
    env = _FixedMaskEnv([True, False, True, True])
    wrapped = self.meta_policy.wrap_env(env, mask_illegal_actions=True)
    wrapped.reset()

    wrapped.step(0)
    wrapped.step(1)

    self.assertEqual(env.actions, [2, 3])
    # Every sub-policy stays selectable: the wrapper reports no mask itself.
    self.assertFalse(hasattr(wrapped, "legal_actions"))
    self.assertEqual(wrapped.action_spec().num_values, 2)

  def test_masked_wrapper_keeps_sub_policy_action_without_env_mask(self):
    env = _NoMaskEnv()
    wrapped = self.meta_policy.wrap_env(env, mask_illegal_actions=True)
    wrapped.reset()

    wrapped.step(0)
    wrapped.step(1)

    # The environment reports no legal actions, so nothing is restricted.
    self.assertEqual(env.actions, [1, 3])


if __name__ == "__main__":
  absltest.main()
