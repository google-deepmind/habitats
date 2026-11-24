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

"""Q-learning training interface."""

from collections.abc import Sequence
from typing import override

from absl import logging
from clu import metric_writers
import dm_env
from dm_env import specs

from habitats.policy import tabular
from habitats.training import training_interface
import numpy as np

TabularPolicyLike = tabular.TabularPolicy | tabular.TabularMetaPolicy


class QLearningInterface(
    training_interface.TrainingInterface[TabularPolicyLike]
):
  """Represents an interface that uses Q-learning for training/evaluation."""

  @override
  def train(
      self,
      env_factory: training_interface.EnvironmentFactory,
      policy: TabularPolicyLike,
  ) -> training_interface.TrainingOutput:
    """Trains the agent."""

    base_env = env_factory(self.config.seed, self.config.playground)
    self.verify_env(base_env)

    # Adapt the environment and extract the trainable policy component so flat
    # policies and hierarchical meta-policies are trained identically.
    env = policy.wrap_env(base_env)
    trainable_policy = policy.trainable_policy
    actor_writer, learner_writer, eval_writer = self._create_metric_writers()

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    epsilons: list[float] = []
    epsilon = self.config.epsilon_start
    epsilon_decay = self._epsilon_decay()
    total_env_steps = 0

    try:
      for episode in range(self.config.num_episodes):
        total_reward, steps, mean_td_loss = self._run_training_episode(
            env, trainable_policy, epsilon
        )
        total_env_steps += steps
        training_interface.raise_if_diverged(
            {"q_table": trainable_policy.q_table}, env_steps=total_env_steps
        )

        if self._should_log(episode):
          eval_output = self.evaluate(env_factory, policy)
          self._log_episode_metrics(
              episode=episode,
              total_reward=total_reward,
              steps=steps,
              mean_td_loss=mean_td_loss,
              epsilon=epsilon,
              eval_output=eval_output,
              actor_writer=actor_writer,
              learner_writer=learner_writer,
              eval_writer=eval_writer,
          )

        epsilon = max(self.config.epsilon_min, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        epsilons.append(epsilon)
    finally:
      env.close()
      self._close_metric_writers((actor_writer, learner_writer, eval_writer))

    if self.config.checkpoint_path:
      policy.save(self.config.checkpoint_path)
    return training_interface.TrainingOutput(
        episode_rewards=np.array(episode_rewards),
        episode_lengths=np.array(episode_lengths),
        epsilons=np.array(epsilons),
    )

  def _run_training_episode(
      self,
      env: dm_env.Environment,
      trainable_policy: tabular.TabularPolicy,
      epsilon: float,
  ) -> tuple[float, int, float]:
    """Runs a single Q-learning episode and returns (reward, steps, mean_td_loss)."""
    timestep = env.reset()
    state = int(timestep.observation)
    total_reward = 0.0
    total_td_loss = 0.0
    steps = 0

    for _ in range(self.config.max_steps):
      if np.random.rand() < epsilon:
        action = int(np.random.randint(trainable_policy.num_actions))
      else:
        action = int(trainable_policy.get_action(state, exploit=False))
      timestep = env.step(action)
      next_state = int(timestep.observation)
      reward = float(timestep.reward)

      best_next_q = np.max(trainable_policy.q_table[next_state])
      target = (
          reward
          + (1.0 - float(timestep.last()))
          * self.config.discount_factor
          * best_next_q
      )

      td_error = target - trainable_policy.q_table[state, action]
      trainable_policy.q_table[state, action] += (
          self.config.learning_rate * td_error
      )

      state = next_state
      total_reward += reward
      total_td_loss += float(td_error**2)
      steps += 1

      if timestep.last():
        break

    mean_td_loss = total_td_loss / max(1, steps)
    return total_reward, steps, mean_td_loss

  def _should_log(self, episode: int) -> bool:
    """Returns whether metrics should be logged for `episode`."""
    return (
        episode % self.config.log_every == 0
        or episode == self.config.num_episodes - 1
    )

  def _epsilon_decay(self) -> float:
    """Returns the factor applied to epsilon after each training episode."""
    if self.config.exploration_fraction is None:
      return self.config.epsilon_decay
    num_decay_episodes = max(
        1.0, self.config.exploration_fraction * self.config.num_episodes
    )
    return (self.config.epsilon_min / self.config.epsilon_start) ** (
        1.0 / num_decay_episodes
    )

  def _create_metric_writers(
      self,
  ) -> tuple[
      metric_writers.MetricWriter,
      metric_writers.MetricWriter,
      metric_writers.MetricWriter,
  ]:
    """Creates actor, learner, and evaluator metric writers."""
    prefix = f"{self.config.tag}_" if self.config.tag else ""
    logdir = self.config.checkpoint_path or None
    actor_writer = metric_writers.create_default_writer(
        logdir, asynchronous=True, collection=f"{prefix}actor"
    )
    learner_writer = metric_writers.create_default_writer(
        logdir, asynchronous=True, collection=f"{prefix}learner"
    )
    eval_writer = metric_writers.create_default_writer(
        logdir, asynchronous=True, collection=f"{prefix}evaluator"
    )
    return actor_writer, learner_writer, eval_writer

  def _log_episode_metrics(
      self,
      *,
      episode: int,
      total_reward: float,
      steps: int,
      mean_td_loss: float,
      epsilon: float,
      eval_output: training_interface.EvaluationOutput,
      actor_writer: metric_writers.MetricWriter,
      learner_writer: metric_writers.MetricWriter,
      eval_writer: metric_writers.MetricWriter,
  ) -> None:
    """Writes training and evaluation scalars to the metric writers."""
    actor_writer.write_scalars(
        step=episode,
        scalars={
            "episode_return": total_reward,
            "episode_length": float(steps),
            "epsilon": epsilon,
        },
    )
    learner_writer.write_scalars(
        step=episode,
        scalars={
            "loss": mean_td_loss,
            "episode_return": total_reward,
        },
    )
    eval_writer.write_scalars(
        step=episode,
        scalars={
            "episode_return": eval_output.mean_reward,
            "episode_length": eval_output.mean_length,
        },
    )
    logging.info(
        "Episode %d: reward=%.2f, length=%d, eval_return=%.2f, epsilon=%.4f",
        episode,
        total_reward,
        steps,
        eval_output.mean_reward,
        epsilon,
    )

  def _close_metric_writers(
      self, writers: Sequence[metric_writers.MetricWriter]
  ) -> None:
    """Flushes and closes all metric writers."""
    for writer in writers:
      writer.flush()
      writer.close()

  @override
  def evaluate(
      self,
      env_factory: training_interface.EnvironmentFactory,
      policy: TabularPolicyLike,
      is_random_policy: bool = False,
  ) -> training_interface.EvaluationOutput:
    """Evaluates the agent."""
    env = env_factory(self.config.seed, self.config.playground)
    self.verify_env(env)
    action_spec = env.action_spec()
    assert isinstance(action_spec, specs.DiscreteArray)
    num_env_actions = int(action_spec.num_values)  # pyrefly: ignore[attribute-error]

    total_rewards = []
    lengths = []

    for _ in range(self.config.num_eval_episodes):
      timestep = env.reset()
      state = int(timestep.observation)
      total_reward = 0.0
      steps = 0

      for _ in range(self.config.max_eval_steps):
        if is_random_policy:
          action = int(np.random.randint(num_env_actions))
        else:
          action = int(policy.get_action(state, exploit=True))
        timestep = env.step(action)
        next_state = int(timestep.observation)
        reward = float(timestep.reward)

        state = next_state
        total_reward += reward
        steps += 1

        if timestep.last():
          break

      total_rewards.append(total_reward)
      lengths.append(steps)

    env.close()
    return training_interface.EvaluationOutput(
        episode_rewards=np.array(total_rewards),
        episode_lengths=np.array(lengths),
    )

  @override
  def verify_env(self, env: dm_env.Environment) -> None:
    """Verifies that the environment is compatible with Q-learning."""
    if not training_interface.is_discrete_env(env):
      raise ValueError(
          "QLearningInterface only supports discrete environments. This implies"
          " that both the observation and action spaces must be discrete."
      )
