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

import os
from typing import override
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import dm_env
from dm_env import specs
from habitats.policy import neural
from habitats.policy import policy as policy_lib
from habitats.training import ppo
from habitats.training import training_interface
import jax
import numpy as np


class _ContinuousPointEnv(dm_env.Environment):
  """Minimal non-Brax continuous environment exercising `PPOInterface._train_acme`."""

  def __init__(self, seed: int = 0, max_steps: int = 6):
    self._rng = np.random.default_rng(seed)
    self._max_steps = max_steps
    self._state = np.zeros((3,), dtype=np.float32)
    self._steps = 0

  @override
  def reset(self) -> dm_env.TimeStep:
    self._steps = 0
    self._state = self._rng.uniform(-0.5, 0.5, size=(3,)).astype(np.float32)
    return dm_env.restart(self._state.copy())

  @override
  def step(self, action: np.ndarray) -> dm_env.TimeStep:
    self._steps += 1
    act = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    self._state = (self._state + 0.2 * np.pad(act, (0, 1))).astype(np.float32)
    reward = float(-np.sum(np.square(self._state)))
    if self._steps >= self._max_steps:
      return dm_env.termination(reward=reward, observation=self._state.copy())
    return dm_env.transition(
        reward=reward, observation=self._state.copy(), discount=0.99
    )

  @override
  def observation_spec(self) -> specs.Array:
    return specs.Array(shape=(3,), dtype=np.float32, name="obs")

  @override
  def action_spec(self) -> specs.BoundedArray:
    return specs.BoundedArray(
        shape=(2,), dtype=np.float32, minimum=-1.0, maximum=1.0, name="act"
    )


class PPOInterfaceTest(parameterized.TestCase):

  def test_train_acme_saves_checkpoint_and_evaluates(self):
    ckpt_dir = os.path.join(self.create_tempdir().full_path, "acme_ckpt")
    env_factory = lambda seed, playground: _ContinuousPointEnv(seed=seed)
    probe_env = env_factory(0, False)
    model_cfg = policy_lib.ModelConfig(
        policy_hidden_sizes=(8, 8),
        value_hidden_sizes=(8, 8),
    ).apply_environment_spec(
        actions=probe_env.action_spec(),
        observations=probe_env.observation_spec(),
    )
    probe_env.close()

    batch_size = 2 * jax.local_device_count()
    cfg = training_interface.TrainingInterfaceConfig(
        checkpoint_path=ckpt_dir,
        num_timesteps=12 * batch_size,
        num_actors=1,
        unroll_length=4,
        batch_size=batch_size,
        num_minibatches=2,
        ppo_epochs=1,
        learning_rate=1e-3,
        num_eval_episodes=2,
        max_eval_steps=6,
        max_steps=6,
        seed=0,
    )
    interface = ppo.PPOInterface(cfg)
    policy = neural.MLPPolicy(model_cfg, seed=0)
    train_out = interface.train(env_factory, policy)
    self.assertIsNone(train_out.error)

    restored = neural.MLPPolicy(model_cfg, seed=1)
    restored.load(ckpt_dir)
    eval_out = interface.evaluate(env_factory, restored, is_random_policy=False)
    self.assertLen(eval_out.episode_rewards, 2)
    self.assertTrue(np.all(np.isfinite(eval_out.episode_rewards)))

  @parameterized.named_parameters(
      dict(testcase_name="smaller_than_num_devices", batch_size=2),
      dict(testcase_name="not_divisible_by_num_devices", batch_size=6),
      dict(testcase_name="non_positive", batch_size=0),
  )
  def test_train_acme_rejects_invalid_batch_size(self, batch_size: int):
    env_factory = lambda seed, playground: _ContinuousPointEnv(seed=seed)
    probe_env = env_factory(0, False)
    model_cfg = policy_lib.ModelConfig(
        policy_hidden_sizes=(8, 8),
        value_hidden_sizes=(8, 8),
    ).apply_environment_spec(
        actions=probe_env.action_spec(),
        observations=probe_env.observation_spec(),
    )
    probe_env.close()

    policy = neural.MLPPolicy(model_cfg, seed=0)
    interface = ppo.PPOInterface(
        training_interface.TrainingInterfaceConfig(batch_size=batch_size)
    )
    four_devices = jax.local_devices()[:1] * 4
    with mock.patch.object(
        jax, "local_devices", autospec=True, return_value=four_devices
    ):
      with self.assertRaisesRegex(
          ValueError,
          "positive multiple of the number of local JAX devices",
      ):
        interface.train(env_factory, policy)


if __name__ == "__main__":
  absltest.main()
