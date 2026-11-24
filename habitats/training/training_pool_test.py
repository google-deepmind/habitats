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

"""Tests for training_pool."""

import contextlib
import dataclasses
import os
from unittest import mock

from absl.testing import absltest
from habitats.config import environment as env_config_lib
from habitats.config import training as training_config_lib
from habitats.evolution import types
from habitats.policy import policy as policy_lib
from habitats.policy import tabular
from habitats.training import registry as training_interface_registry
from habitats.training import training_interface
from habitats.training import training_pool
import numpy as np


_DUMMY_REWARD_CODE = """
def compute_reward(env, state, action, next_state, done, info, base_reward) -> float:
  return float(base_reward)
"""


def _make_dummy_agent(
    agent_id: str,
    ckpt_path: str = "",
    reward_code: str = _DUMMY_REWARD_CODE,
) -> types.AgentData:
  return types.AgentData(
      id=agent_id,
      reward_code=reward_code,
      generation=0,
      generalization_score=0.0,
      training_output=training_interface.TrainingOutput.empty(),
      reasoning="test",
      embedding=[0.0],
      path=ckpt_path,
      parent_ids=[],
  )


class TrainingPoolTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = self.create_tempdir().full_path
    base_config = training_config_lib.get_config("q_learning_tic_tac_toe")
    # Use very few episodes for quick unit testing
    self.training_config = training_config_lib.TrainingConfig(
        model_config=base_config.model_config,
        training_interface_config=training_interface.TrainingInterfaceConfig(
            num_episodes=2,
            num_eval_episodes=2,
            max_steps=5,
            log_every=100,
        ),
        training_interface=base_config.training_interface,
        environment=base_config.environment,
        policy=base_config.policy,
    )

  def test_worker_train_and_evaluate(self):
    worker = training_pool.TrainingWorker(self.training_config)
    agent = _make_dummy_agent("agent_0")
    ckpt_path = os.path.join(self.temp_dir, "agent_0")

    # 1. Test train
    train_out = worker.train(agent=agent, checkpoint_path=ckpt_path)
    self.assertIsInstance(train_out, training_interface.TrainingOutput)
    self.assertNotEmpty(train_out.episode_rewards)

    # Update agent with saved path
    trained_agent = _make_dummy_agent("agent_0", ckpt_path=ckpt_path)

    # 2. Test evaluate with policy
    eval_out = worker.evaluate(
        reward_agent=trained_agent,
        policy_agent=trained_agent,
    )
    self.assertIsInstance(eval_out, training_interface.EvaluationOutput)
    self.assertNotEmpty(eval_out.episode_rewards)

    # 3. Test evaluate with random policy
    random_eval_out = worker.evaluate(
        reward_agent=trained_agent,
        policy_agent=None,
    )
    self.assertIsInstance(random_eval_out, training_interface.EvaluationOutput)
    self.assertNotEmpty(random_eval_out.episode_rewards)

  def test_worker_keeps_base_termination_of_games(self):
    worker = training_pool.TrainingWorker(self.training_config)

    env = worker._make_reward_env_factory(_DUMMY_REWARD_CODE)(0, False)

    self.assertFalse(env.ignore_base_termination)

  def test_worker_returns_failure_when_reward_code_is_invalid(self):
    worker = training_pool.TrainingWorker(self.training_config)
    agent = _make_dummy_agent("agent_0", reward_code="def compute_reward(:")

    train_out = worker.train(
        agent=agent, checkpoint_path=os.path.join(self.temp_dir, "agent_0")
    )

    self.assertTrue(train_out.is_empty())
    self.assertStartsWith(train_out.error, "RewardCompilationError: ")

  def test_worker_returns_failure_when_training_diverges(self):
    diverging_interface = mock.create_autospec(
        training_interface.TrainingInterface, instance=True
    )
    diverging_interface.train.side_effect = (
        training_interface.TrainingDivergedError("NaN parameters.")
    )

    train_out = self._train_with_interface(diverging_interface)

    self.assertTrue(train_out.is_empty())
    self.assertEqual(train_out.error, "TrainingDivergedError: NaN parameters.")

  def test_worker_raises_errors_not_caused_by_the_agent(self):
    failing_interface = mock.create_autospec(
        training_interface.TrainingInterface, instance=True
    )
    failing_interface.train.side_effect = OSError("Disk full.")

    with self.assertRaisesRegex(OSError, "Disk full."):
      self._train_with_interface(failing_interface)

  def _train_with_interface(
      self, interface: training_interface.TrainingInterface
  ) -> training_interface.TrainingOutput:
    """Trains an agent with a TrainingWorker that uses `interface`."""
    worker = training_pool.TrainingWorker(self.training_config)
    with mock.patch.object(
        training_interface_registry,
        "get",
        return_value=mock.Mock(return_value=interface),
    ):
      return worker.train(
          agent=_make_dummy_agent("agent_0"),
          checkpoint_path=os.path.join(self.temp_dir, "agent_0"),
      )

  def test_pool_train_agents_parallel(self):
    worker1 = training_pool.TrainingWorker(self.training_config)
    worker2 = training_pool.TrainingWorker(self.training_config)
    pool = training_pool.TrainingPool([worker1, worker2])
    self.assertEqual(pool.num_workers, 2)

    agents = {f"agent_{i}": _make_dummy_agent(f"agent_{i}") for i in range(4)}

    trained_agents = pool.train_agents(agents, self.temp_dir)
    self.assertLen(trained_agents, 4)
    for agent_id, agent in trained_agents.items():
      self.assertEqual(agent.id, agent_id)
      self.assertFalse(agent.training_output.is_empty())
      self.assertTrue(os.path.exists(agent.path))

  def test_pool_traces_training_and_evaluation(self):
    labels = []

    @contextlib.contextmanager
    def trace(label):
      labels.append(label)
      yield

    worker = training_pool.TrainingWorker(self.training_config)
    pool = training_pool.TrainingPool([worker])
    agents = {"agent_0": _make_dummy_agent("agent_0")}

    with mock.patch.object(training_pool, "trace", trace):
      trained = pool.train_agents(agents, self.temp_dir)
      pool.evaluate_agents(trained, prev_gen=None)

    self.assertEqual(
        labels, ["agent agent_0: Training", "Evaluating agents ['agent_0']"]
    )

  def test_pool_evaluate_agents_parallel(self):
    worker1 = training_pool.TrainingWorker(self.training_config)
    worker2 = training_pool.TrainingWorker(self.training_config)
    pool = training_pool.TrainingPool([worker1, worker2])

    raw_agents = {
        f"agent_{i}": _make_dummy_agent(f"agent_{i}") for i in range(4)
    }
    trained_all = pool.train_agents(raw_agents, self.temp_dir)
    agents = {f"agent_{i}": trained_all[f"agent_{i}"] for i in range(3)}

    # Initial evaluation without previous generation
    fitness_grid, random_fitness = pool.evaluate_agents(agents, prev_gen=None)
    self.assertEqual(fitness_grid.shape, (3, 3))
    self.assertEqual(random_fitness.shape, (3,))

    # Evaluation with previous generation caching
    prev_gen = types.GenerationData(
        generation=0,
        fitness_grid=fitness_grid.tolist(),
        random_fitness=random_fitness.tolist(),
        selected_agents=["agent_0", "agent_1"],
        specialist_quality=1.0,
        behavioral_diversity=0.5,
        agents=agents,
        generalization_scores={"agent_0": 1.0, "agent_1": 1.0, "agent_2": 1.0},
    )

    # Next generation with an added offspring
    next_agents = dict(agents)
    next_agents["agent_3"] = trained_all["agent_3"]

    next_grid, next_random = pool.evaluate_agents(
        next_agents, prev_gen=prev_gen
    )
    self.assertEqual(next_grid.shape, (4, 4))
    self.assertEqual(next_random.shape, (4,))
    # Existing subgrid and random fitness should match cached values
    np.testing.assert_array_almost_equal(next_grid[:3, :3], fitness_grid)
    np.testing.assert_array_almost_equal(next_random[:3], random_fitness)

  def test_pool_worker_reclamation_on_error(self):
    mock_worker = mock.create_autospec(
        training_pool.TrainingWorkerInterface, instance=True
    )
    mock_worker.train.side_effect = RuntimeError("Worker training failure")

    pool = training_pool.TrainingPool([mock_worker])
    self.assertEqual(pool._idle_workers.qsize(), 1)

    agents = {"agent_0": _make_dummy_agent("agent_0")}
    with self.assertRaises(RuntimeError):
      pool.train_agents(agents, self.temp_dir)

    # Worker must be returned to queue even after error
    self.assertEqual(pool._idle_workers.qsize(), 1)

  def test_worker_train_and_evaluate_meta_policy(self):
    base_meta_cfg = training_config_lib.get_config(
        "q_learning_meta_tic_tac_toe"
    )
    env = env_config_lib.get(base_meta_cfg.environment)(0, False)
    sub_model_config = policy_lib.ModelConfig().apply_environment_spec(
        observations=env.observation_spec(),
        actions=env.action_spec(),
    )
    env.close()

    sub_paths = []
    for idx in range(2):
      sub_dir = os.path.join(self.temp_dir, f"sub_policy_{idx}")
      sub_policy = tabular.TabularPolicy(sub_model_config)
      sub_policy.save(sub_dir)
      sub_paths.append(sub_dir)

    assert isinstance(base_meta_cfg.model_config, policy_lib.MetaModelConfig)
    meta_training_config = training_config_lib.TrainingConfig(
        model_config=base_meta_cfg.model_config.with_sub_policy_paths(
            sub_paths
        ),
        training_interface_config=training_interface.TrainingInterfaceConfig(
            num_episodes=2,
            num_eval_episodes=2,
            max_steps=5,
            log_every=100,
        ),
        training_interface=base_meta_cfg.training_interface,
        environment=base_meta_cfg.environment,
        policy=base_meta_cfg.policy,
    )
    restored_cfg = training_config_lib.TrainingConfig.from_dict(
        dataclasses.asdict(meta_training_config)
    )
    self.assertIsInstance(
        restored_cfg.model_config, policy_lib.MetaModelConfig
    )
    self.assertEqual(restored_cfg, meta_training_config)

    worker = training_pool.TrainingWorker(meta_training_config)
    agent = _make_dummy_agent("meta_agent_0")
    ckpt_path = os.path.join(self.temp_dir, "meta_agent_0")

    train_out = worker.train(agent=agent, checkpoint_path=ckpt_path)
    self.assertFalse(train_out.is_empty())
    self.assertTrue(os.path.exists(os.path.join(ckpt_path, "q_table")))
    self.assertTrue(
        os.path.exists(
            os.path.join(ckpt_path, "sub_policies", "policy_0", "q_table")
        )
    )

    trained_agent = _make_dummy_agent("meta_agent_0", ckpt_path=ckpt_path)
    eval_out = worker.evaluate(
        reward_agent=trained_agent,
        policy_agent=trained_agent,
    )
    self.assertNotEmpty(eval_out.episode_rewards)

    random_eval_out = worker.evaluate(
        reward_agent=trained_agent,
        policy_agent=None,
    )
    self.assertNotEmpty(random_eval_out.episode_rewards)


if __name__ == "__main__":
  absltest.main()
