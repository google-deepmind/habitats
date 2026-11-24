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

"""Tests for QLearningInterface and MetaPolicy training/evolution loading."""

import os
from typing import Any, override

from absl.testing import absltest
import dm_env
from dm_env import specs
from habitats.policy import policy as policy_lib
from habitats.policy import tabular
from habitats.training import qlearning
from habitats.training import training_interface
import numpy as np


class _TwoStageRoutingEnv(dm_env.Environment):
  """Two-state discrete environment with 4 primitive actions.

  - State 0: primitive action 1 gives reward +1.0 and transitions to State 1;
    any other action gives -1.0 and transitions to State 1.
  - State 1: primitive action 3 gives reward +1.0 and terminates;
    any other action gives -1.0 and terminates.
  """

  def __init__(self):
    self._state = 0

  @override
  def reset(self) -> dm_env.TimeStep:
    self._state = 0
    return dm_env.restart(np.int64(self._state))

  @override
  def step(self, action: Any) -> dm_env.TimeStep:
    act = int(action)
    if self._state == 0:
      reward = 1.0 if act == 1 else -1.0
      self._state = 1
      return dm_env.transition(reward=reward, observation=np.int64(self._state))
    reward = 1.0 if act == 3 else -1.0
    return dm_env.termination(reward=reward, observation=np.int64(self._state))

  @override
  def observation_spec(self) -> specs.DiscreteArray:
    return specs.DiscreteArray(num_values=2, dtype=np.int64)

  @override
  def action_spec(self) -> specs.DiscreteArray:
    return specs.DiscreteArray(num_values=4, dtype=np.int64)


class QLearningInterfaceTest(absltest.TestCase):

  def test_exploration_fraction_sets_epsilon_schedule(self):
    interface = qlearning.QLearningInterface(
        training_interface.TrainingInterfaceConfig(
            num_episodes=100,
            exploration_fraction=0.5,
            epsilon_min=0.01,
            max_steps=5,
            num_eval_episodes=1,
            log_every=1000,
        )
    )
    policy = tabular.TabularPolicy(
        policy_lib.ModelConfig(num_states=2, num_actions=4)
    )
    env_factory = lambda seed, playground: _TwoStageRoutingEnv()

    output = interface.train(env_factory, policy)

    # Epsilon reaches its minimum after half of the episodes and stays there.
    np.testing.assert_allclose(output.epsilons[49], 0.01)
    self.assertEqual(output.epsilons[-1], 0.01)

  def test_evaluate_stops_after_max_eval_steps(self):
    interface = qlearning.QLearningInterface(
        training_interface.TrainingInterfaceConfig(
            num_eval_episodes=2, max_eval_steps=1
        )
    )
    policy = tabular.TabularPolicy(
        policy_lib.ModelConfig(num_states=2, num_actions=4)
    )
    env_factory = lambda seed, playground: _TwoStageRoutingEnv()

    output = interface.evaluate(env_factory, policy)

    np.testing.assert_array_equal(output.episode_lengths, [1, 1])


class QLearningMetaPolicyTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = self.create_tempdir().full_path
    self.sub_model_config = policy_lib.ModelConfig(num_states=2, num_actions=4)
    sub_paths = []
    for idx, action_idx in enumerate([1, 3]):
      sub_policy = tabular.TabularPolicy(self.sub_model_config)
      sub_policy.q_table[:, action_idx] = 5.0
      sub_path = os.path.join(self.temp_dir, f"sub_{idx}")
      sub_policy.save(sub_path)
      sub_paths.append(sub_path)
    self.meta_config = (
        policy_lib.MetaModelConfig(policy_config=self.sub_model_config)
        .with_sub_policy_paths(sub_paths)
        .apply_environment_spec(
            observations=specs.DiscreteArray(num_values=2, dtype=np.int64),
            actions=specs.DiscreteArray(num_values=4, dtype=np.int64),
        )
    )

  def test_meta_policy_env_wrapper_spec_and_stepping(self):
    base_env = _TwoStageRoutingEnv()
    meta_policy = tabular.TabularMetaPolicy(self.meta_config)
    # Sub-policy 0 always picks primitive action 1; sub-policy 1 picks 3.
    meta_policy.policies[0].q_table[:, 1] = 10.0
    meta_policy.policies[1].q_table[:, 3] = 10.0

    wrapped = meta_policy.wrap_env(base_env)
    self.assertIsInstance(wrapped, policy_lib.MetaPolicyEnvWrapper)
    self.assertIs(meta_policy.trainable_policy, meta_policy.router)
    self.assertEqual(wrapped.observation_spec().num_values, 2)
    self.assertEqual(wrapped.action_spec().num_values, 2)

    ts0 = wrapped.reset()
    self.assertEqual(int(ts0.observation), 0)
    # Choose sub-policy 0 at state 0 -> primitive action 1 -> reward +1.0
    ts1 = wrapped.step(0)
    self.assertEqual(float(ts1.reward), 1.0)
    self.assertEqual(int(ts1.observation), 1)
    # Choose sub-policy 1 at state 1 -> primitive action 3 -> reward +1.0
    ts2 = wrapped.step(1)
    self.assertEqual(float(ts2.reward), 1.0)
    self.assertTrue(ts2.last())
    wrapped.close()

  def test_train_and_evaluate_tabular_meta_policy(self):
    meta_policy = tabular.TabularMetaPolicy(self.meta_config)
    # Sub-policy 0 is specialist for state 0 (action 1)
    meta_policy.policies[0].q_table[:, 1] = 5.0
    # Sub-policy 1 is specialist for state 1 (action 3)
    meta_policy.policies[1].q_table[:, 3] = 5.0

    sub0_before = meta_policy.policies[0].q_table.copy()
    sub1_before = meta_policy.policies[1].q_table.copy()

    ckpt_path = os.path.join(self.temp_dir, "meta_ckpt")
    interface = qlearning.QLearningInterface(
        training_interface.TrainingInterfaceConfig(
            checkpoint_path=ckpt_path,
            num_episodes=150,
            num_eval_episodes=10,
            max_steps=5,
            learning_rate=0.2,
            epsilon_start=0.5,
            epsilon_min=0.05,
            epsilon_decay=0.98,
            log_every=1000,
        )
    )

    env_factory = lambda seed, playground: _TwoStageRoutingEnv()
    train_out = interface.train(env_factory, meta_policy)
    self.assertLen(train_out.episode_rewards, 150)

    # Sub-policy Q-tables must remain frozen during meta-policy training
    np.testing.assert_array_equal(meta_policy.policies[0].q_table, sub0_before)
    np.testing.assert_array_equal(meta_policy.policies[1].q_table, sub1_before)

    # Router should have learned state 0 -> sub-policy 0, state 1 -> sub-policy 1
    self.assertEqual(meta_policy.get_meta_action(0, exploit=True), 0)
    self.assertEqual(meta_policy.get_meta_action(1, exploit=True), 1)

    eval_out = interface.evaluate(
        env_factory, meta_policy, is_random_policy=False
    )
    self.assertAlmostEqual(eval_out.mean_reward, 2.0)

    # Random baseline evaluation must sample over all 4 primitive env actions
    random_eval_out = interface.evaluate(
        env_factory, meta_policy, is_random_policy=True
    )
    self.assertLen(random_eval_out.episode_rewards, 10)

    # Verify saved meta-policy checkpoint roundtrip
    restored_meta = tabular.TabularMetaPolicy(self.meta_config)
    restored_meta.load(ckpt_path)
    np.testing.assert_array_almost_equal(
        restored_meta.q_table, meta_policy.q_table
    )
    np.testing.assert_array_equal(
        restored_meta.policies[0].q_table, sub0_before
    )
    np.testing.assert_array_equal(
        restored_meta.policies[1].q_table, sub1_before
    )


if __name__ == "__main__":
  absltest.main()
