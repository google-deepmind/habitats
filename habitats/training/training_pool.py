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

"""Training pool interface and worker node."""

from collections.abc import Iterator, Mapping, Sequence
import concurrent.futures
import contextlib
import dataclasses
import os
import queue
import random
import time
from typing import Protocol

from absl import logging
import dm_env
from etils import epath
from habitats.config import environment as env_config_lib
from habitats.config import training as training_config_lib
from habitats.environment import termination
from habitats.evolution import reward_wrapper
from habitats.evolution import types
from habitats.policy import registry as policy_registry
from habitats.training import registry as training_interface_registry
from habitats.training import training_interface
import jax
import numpy as np

# Errors caused by the reward code of the agent being trained, rather than by
# the infrastructure. `TrainingWorker.train` returns them as a failed
# `TrainingOutput` instead of raising them: raised by a remote worker, they
# could reach the coordinator as untyped errors, indistinguishable from the
# failures that should stop the run.
_AGENT_ERRORS = (
    reward_wrapper.RewardError,
    training_interface.TrainingDivergedError,
)


class TrainingWorkerInterface(Protocol):
  """Protocol defining the methods exposed by a TrainingWorker."""

  def train(
      self,
      agent: types.AgentData,
      checkpoint_path: str,
  ) -> training_interface.TrainingOutput:
    """Trains a policy using the agent's evolved reward function.

    Args:
      agent: The agent to train.
      checkpoint_path: Where to save the trained policy.

    Returns:
      The training output, which holds the error if the agent's reward code
      made training fail.
    """
    pass

  def evaluate(
      self,
      reward_agent: types.AgentData,
      policy_agent: types.AgentData | None = None,
  ) -> training_interface.EvaluationOutput:
    """Evaluates a policy on an environment wrapped with reward_agent's reward."""
    pass


LocalTrainingWorkerInterface = TrainingWorkerInterface


@dataclasses.dataclass(frozen=True)
class GridEvaluationTask:
  """Task to evaluate a specific policy against a specific reward function."""

  i: int
  j: int
  policy_agent: types.AgentData
  reward_agent: types.AgentData


@dataclasses.dataclass(frozen=True)
class RandomEvaluationTask:
  """Task to evaluate a random baseline policy against a reward function."""

  i: int
  reward_agent: types.AgentData


class TrainingWorker:
  """A worker node providing TrainingInterface execution (local or remote)."""

  def __init__(
      self,
      training_config: training_config_lib.TrainingConfig,
  ):
    self._training_config = training_config
    self._base_interface_config = training_config.training_interface_config
    self._env_factory = env_config_lib.get(training_config.environment)
    # Probe environment spec to compute model dimensions
    probe_env = self._env_factory(
        self._base_interface_config.seed,
        self._base_interface_config.playground,
    )
    self._model_config = training_config.model_config.apply_environment_spec(
        actions=probe_env.action_spec(),
        observations=probe_env.observation_spec(),
    )
    probe_env.close()

  def _make_reward_env_factory(
      self, reward_code: str
  ) -> training_interface.EnvironmentFactory:
    """Wraps the base environment factory with the evolved reward function."""

    def factory(seed: int, playground: bool) -> dm_env.Environment:
      base_env = self._env_factory(seed, playground)
      return reward_wrapper.EvolvedRewardWrapper(
          base_env,
          reward_code,
          termination.ignores_base_termination(base_env),
      )

    return factory

  def train(
      self,
      agent: types.AgentData,
      checkpoint_path: str,
  ) -> training_interface.TrainingOutput:
    """See `TrainingWorkerInterface.train`."""
    seed = self._base_interface_config.seed
    random.seed(seed)
    np.random.seed(seed)

    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
      epath.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    interface_config = dataclasses.replace(
        self._base_interface_config,
        checkpoint_path=checkpoint_path,
        tag=agent.id,
    )
    interface = training_interface_registry.get(
        self._training_config.training_interface
    )(interface_config)

    policy = policy_registry.create(
        self._training_config.policy, self._model_config
    )
    env_factory = self._make_reward_env_factory(agent.reward_code)
    try:
      return interface.train(env_factory, policy)
    except _AGENT_ERRORS as e:
      logging.exception("Training failed for agent %s.", agent.id)
      return training_interface.TrainingOutput.failure(
          f"{type(e).__name__}: {e}"
      )

  def evaluate(
      self,
      reward_agent: types.AgentData,
      policy_agent: types.AgentData | None = None,
  ) -> training_interface.EvaluationOutput:
    """Evaluates a policy on an environment wrapped with reward_agent's reward."""
    seed = self._base_interface_config.seed
    random.seed(seed)
    np.random.seed(seed)

    interface = training_interface_registry.get(
        self._training_config.training_interface
    )(self._base_interface_config)

    policy = policy_registry.create(
        self._training_config.policy, self._model_config
    )
    if policy_agent:
      logging.info("Loading policy from %s", policy_agent.path)
      policy.load(policy_agent.path)

    env_factory = self._make_reward_env_factory(reward_agent.reward_code)
    return interface.evaluate(env_factory, policy, policy_agent is None)


@contextlib.contextmanager
def trace(label: str) -> Iterator[None]:
  """Scope wrapping a long-running task, e.g. one agent's training.

  Logs the start and the duration of the task and annotates it in JAX profiler
  traces.

  Args:
    label: Name of the task, e.g. "agent agent_0: Training".

  Yields:
    Nothing.
  """
  logging.info("Start - Task: %s", label)
  start = time.monotonic()
  try:
    with jax.profiler.TraceAnnotation(label):
      yield
  finally:
    logging.info(
        "Finish - Task: %s, Elapsed: %.2fs", label, time.monotonic() - start
    )


class TrainingPool:
  """Manages distribution of training and evaluation tasks across workers."""

  def __init__(
      self,
      workers: Sequence[LocalTrainingWorkerInterface],
  ):
    """Initializes the pool.

    Args:
      workers: Workers to distribute training and evaluation over.
    """
    if not workers:
      raise ValueError("TrainingPool must have at least one worker.")
    self._workers = list(workers)
    self._idle_workers: queue.Queue[LocalTrainingWorkerInterface] = queue.Queue()
    for worker in self._workers:
      self._idle_workers.put(worker)
    self._executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=len(self._workers)
    )

  @property
  def num_workers(self) -> int:
    return len(self._workers)

  def train_agents(
      self,
      agents: Mapping[str, types.AgentData],
      base_checkpoint_path: str,
  ) -> dict[str, types.AgentData]:
    """Trains multiple agents in parallel across available workers."""
    trained_agents = dict[str, types.AgentData]()
    if not agents:
      return trained_agents

    def _train_single(
        agent_id: str, agent: types.AgentData
    ) -> tuple[str, types.AgentData]:
      worker = self._idle_workers.get()
      try:
        agent_ckpt_path = os.path.join(base_checkpoint_path, "train", agent_id)

        # If the coordinator was preempted mid-generation and restarted, an
        # earlier attempt may have already written checkpoints or event logs
        # under this agent ID. Delete any existing directory so we start clean.
        agent_ckpt_dir = epath.Path(agent_ckpt_path)
        if agent_ckpt_dir.exists():
          agent_ckpt_dir.rmtree()
        agent_ckpt_dir.mkdir(parents=True)
        (agent_ckpt_dir / "data.json").write_text(agent.to_json())

        with trace(f"agent {agent_id}: Training"):
          training_output = worker.train(
              agent=agent,
              checkpoint_path=agent_ckpt_path,
          )

        updated_agent = dataclasses.replace(
            agent,
            training_output=training_output,
            path=agent_ckpt_path,
        )
        (agent_ckpt_dir / "data.json").write_text(updated_agent.to_json())
        return agent_id, updated_agent
      finally:
        self._idle_workers.put(worker)

    futures = [
        self._executor.submit(_train_single, agent_id, agent)
        for agent_id, agent in agents.items()
    ]
    for future in concurrent.futures.as_completed(futures):
      agent_id, updated_agent = future.result()
      trained_agents[agent_id] = updated_agent

    return trained_agents

  def evaluate_agents(
      self,
      agents: Mapping[str, types.AgentData],
      prev_gen: types.GenerationData | None,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Evaluates cross-reward matrix and random baselines in parallel."""
    agent_ids = list(agents.keys())
    n = len(agent_ids)

    fitness_grid = np.zeros((n, n), dtype=np.float64)
    random_fitness = np.zeros(n, dtype=np.float64)

    # 1. Identify which evaluations can be reused from cache vs new tasks
    grid_tasks, random_tasks = self._plan_evaluation_tasks(
        agents=agents,
        prev_gen=prev_gen,
        fitness_grid=fitness_grid,
        random_fitness=random_fitness,
    )

    logging.info(
        "Evaluating population: %d grid tasks (%d cached), %d random tasks"
        " (%d cached)",
        len(grid_tasks),
        (n * n) - len(grid_tasks),
        len(random_tasks),
        n - len(random_tasks),
    )

    # 2. Dispatch tasks to worker pool in parallel
    with trace(f"Evaluating agents {agent_ids}"):
      self._execute_evaluation_tasks(
          grid_tasks=grid_tasks,
          random_tasks=random_tasks,
          fitness_grid=fitness_grid,
          random_fitness=random_fitness,
      )

    return fitness_grid, random_fitness

  def _plan_evaluation_tasks(
      self,
      agents: Mapping[str, types.AgentData],
      prev_gen: types.GenerationData | None,
      fitness_grid: np.ndarray,
      random_fitness: np.ndarray,
  ) -> tuple[list[GridEvaluationTask], list[RandomEvaluationTask]]:
    """Identifies which grid cells must be computed vs reused from previous generation."""
    agent_ids = list(agents.keys())
    prev_agent_to_idx = (
        {aid: idx for idx, aid in enumerate(prev_gen.agents.keys())}
        if prev_gen
        else {}
    )
    raw_grid = prev_gen.fitness_grid if prev_gen else None
    raw_rand = (
        prev_gen.random_fitness
        if prev_gen and prev_gen.random_fitness
        else None
    )

    grid_tasks: list[GridEvaluationTask] = []
    for i, policy_id in enumerate(agent_ids):
      for j, reward_id in enumerate(agent_ids):
        # Check if this exact (policy, reward) pair was already evaluated in
        # prev_gen
        if (
            raw_grid
            and policy_id in prev_agent_to_idx
            and reward_id in prev_agent_to_idx
        ):
          prev_i = prev_agent_to_idx[policy_id]
          prev_j = prev_agent_to_idx[reward_id]
          fitness_grid[i, j] = raw_grid[prev_i][prev_j]
        else:
          grid_tasks.append(
              GridEvaluationTask(
                  i=i,
                  j=j,
                  policy_agent=agents[policy_id],
                  reward_agent=agents[reward_id],
              )
          )

    random_tasks: list[RandomEvaluationTask] = []
    for i, agent_id in enumerate(agent_ids):
      if raw_rand and agent_id in prev_agent_to_idx:
        prev_idx = prev_agent_to_idx[agent_id]
        random_fitness[i] = raw_rand[prev_idx]
      else:
        random_tasks.append(
            RandomEvaluationTask(i=i, reward_agent=agents[agent_id])
        )
    return grid_tasks, random_tasks

  def _execute_evaluation_tasks(
      self,
      grid_tasks: Sequence[GridEvaluationTask],
      random_tasks: Sequence[RandomEvaluationTask],
      fitness_grid: np.ndarray,
      random_fitness: np.ndarray,
  ) -> None:
    """Dispatches evaluation tasks to idle workers and records results."""

    def _run_grid_task(task: GridEvaluationTask) -> tuple[int, int, float]:
      worker = self._idle_workers.get()
      try:
        out = worker.evaluate(
            reward_agent=task.reward_agent,
            policy_agent=task.policy_agent,
        )
        return task.i, task.j, out.mean_reward
      finally:
        self._idle_workers.put(worker)

    def _run_random_task(task: RandomEvaluationTask) -> tuple[int, float]:
      worker = self._idle_workers.get()
      try:
        out = worker.evaluate(
            reward_agent=task.reward_agent,
            policy_agent=None,
        )
        return task.i, out.mean_reward
      finally:
        self._idle_workers.put(worker)

    grid_futures = [
        self._executor.submit(_run_grid_task, task) for task in grid_tasks
    ]
    random_futures = [
        self._executor.submit(_run_random_task, task) for task in random_tasks
    ]

    for future in concurrent.futures.as_completed(grid_futures):
      i, j, mean_reward = future.result()
      fitness_grid[i, j] = mean_reward

    for future in concurrent.futures.as_completed(random_futures):
      i, mean_reward = future.result()
      random_fitness[i] = mean_reward
