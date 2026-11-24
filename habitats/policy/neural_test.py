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

"""Tests for neural network policy."""

import tempfile
from typing import cast

from absl.testing import absltest
from absl.testing import parameterized
from habitats.policy import neural
from habitats.policy import policy as policy_lib
import jax
import jax.numpy as jnp
import numpy as np


class NeuralPolicyTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.num_actions = 4
    self.num_states = 12
    self.config = policy_lib.ModelConfig(
        num_actions=self.num_actions,
        num_states=self.num_states,
        action_shape=(self.num_actions,),
        observation_shape=(self.num_states,),
        policy_hidden_sizes=(32, 32),
        value_hidden_sizes=(32, 32),
        activation="swish",
    )

  def test_mlp_actor_forward(self):
    actor = neural.MLPActor(
        num_dimensions=self.num_actions,
        hidden_sizes=(32, 32),
        activation="swish",
    )
    key = jax.random.PRNGKey(0)
    dummy_obs = jnp.zeros((1, self.num_states))

    variables = actor.init(key, dummy_obs)
    logits = cast(jax.Array, actor.apply(variables, dummy_obs))
    self.assertEqual(logits.shape, (1, 2 * self.num_actions))

    loc, scale_raw = jnp.split(logits, 2, axis=-1)
    scale = jax.nn.softplus(scale_raw) + 1e-3
    self.assertEqual(loc.shape, (1, self.num_actions))
    self.assertEqual(scale.shape, (1, self.num_actions))
    self.assertTrue(jnp.all(scale > 0))

  def test_mlp_critic_forward(self):
    critic = neural.MLPCritic(
        hidden_sizes=(32, 32),
        activation="swish",
    )
    key = jax.random.PRNGKey(0)
    dummy_obs = jnp.zeros((1, self.num_states))

    variables = critic.init(key, dummy_obs)
    value = cast(jax.Array, critic.apply(variables, dummy_obs))
    self.assertEqual(value.shape, (1,))

  def test_mlp_policy_get_action_without_normalizer(self):
    policy = neural.MLPPolicy(self.config, seed=42)
    key = jax.random.PRNGKey(0)
    dummy_obs = jnp.zeros((1, self.num_states))

    actor_vars = policy.actor.init(key, dummy_obs)
    critic_vars = policy.critic.init(key, dummy_obs)

    policy.set_params({
        "actor": actor_vars,
        "critic": critic_vars,
        "normalizer": None,
    })

    obs = np.ones(self.num_states, dtype=np.float32)
    action_exploit = policy.get_action(obs, exploit=True)
    self.assertEqual(action_exploit.shape, (self.num_actions,))
    self.assertTrue(np.all(action_exploit >= -1.0))
    self.assertTrue(np.all(action_exploit <= 1.0))

    action_sample = policy.get_action(obs, exploit=False)
    self.assertEqual(action_sample.shape, (self.num_actions,))

  def test_mlp_policy_get_action_with_normalizer(self):
    policy = neural.MLPPolicy(self.config, seed=42)
    key = jax.random.PRNGKey(0)
    dummy_obs = jnp.zeros((1, self.num_states))

    actor_vars = policy.actor.init(key, dummy_obs)
    critic_vars = policy.critic.init(key, dummy_obs)

    normalizer = {
        "mean": jnp.ones((self.num_states,), dtype=jnp.float32) * 5.0,
        "std": jnp.ones((self.num_states,), dtype=jnp.float32) * 2.0,
    }

    policy.set_params({
        "actor": actor_vars,
        "critic": critic_vars,
        "normalizer": normalizer,
    })

    obs = np.ones(self.num_states, dtype=np.float32) * 5.0
    action = policy.get_action(obs, exploit=True)
    self.assertEqual(action.shape, (self.num_actions,))

  def test_save_and_load_roundtrip(self):
    policy = neural.MLPPolicy(self.config, seed=42)
    key = jax.random.PRNGKey(0)
    dummy_obs = jnp.zeros((1, self.num_states))

    actor_vars = policy.actor.init(key, dummy_obs)
    critic_vars = policy.critic.init(key, dummy_obs)
    normalizer = {
        "mean": jnp.ones((self.num_states,), dtype=jnp.float32) * 2.0,
        "std": jnp.ones((self.num_states,), dtype=jnp.float32) * 1.5,
    }

    policy.set_params({
        "actor": actor_vars,
        "critic": critic_vars,
        "normalizer": normalizer,
    })

    obs = np.random.normal(size=(self.num_states,)).astype(np.float32)
    action_before = policy.get_action(obs, exploit=True)

    with tempfile.TemporaryDirectory() as tmpdir:
      policy.save(tmpdir)
      policy.save(tmpdir)

      loaded_policy = neural.MLPPolicy(self.config, seed=42)
      loaded_policy.load(tmpdir)

      action_after = loaded_policy.get_action(obs, exploit=True)
      np.testing.assert_allclose(action_before, action_after, rtol=1e-5)


if __name__ == "__main__":
  absltest.main()
