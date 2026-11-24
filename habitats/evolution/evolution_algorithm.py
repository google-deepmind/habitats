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

"""Evolution algorithm: LLM-evolved reward functions over generations.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import os
import random
import time
import traceback

from absl import logging
from clu import metric_writers
from habitats.config import environment
from habitats.evolution import evolution_algorithm_config
from habitats.evolution import offspring_generator as og
from habitats.evolution import population_initializer
from habitats.evolution import types
from habitats.lib import errors
from habitats.lib import llm_client
from habitats.lib import metrics
from habitats.training import registry as training_interface_registry
from habitats.training import training_interface
from habitats.training import training_pool as training_pool_lib
import numpy as np


class EvolutionAlgorithm:
  """Represents an interface that uses an evolution algorithm for training."""

  def __init__(
      self,
      config: evolution_algorithm_config.EvolutionAlgorithmConfig,
      run_id: str,
      api_key: str | None = None,
      training_pool: training_pool_lib.TrainingPool | None = None,
  ):
    self.config = config
    self.run_id = run_id
    self.status = "initialized"
    self.current_generation = 0
    self.generations_data: list[types.GenerationData] = []

    self._validate_config()

    self.llm_config = llm_client.LLMConfig(
        role="",
        key=api_key,
    )

    self.api_key = api_key

    self.population_initializer = population_initializer.get(
        config.population_initializer
    )(
        population_initializer.PopulationInitializerConfig(
            env_name=config.training_config.environment,
            model_name=config.llm_model,
            llm_config=self.llm_config,
            llm_backend=config.llm_backend,
        )
    )
    self.offspring_generator = og.get(config.offspring_generator)(
        environment=config.training_config.environment,
        llm_config=self.llm_config,
        model_name=config.llm_model,
        llm_backend=config.llm_backend,
        past_generations_to_look_at=config.past_generations_to_look_at,
    )

    # Initialize training pool: use provided pool or fallback to local
    # in-process worker
    if training_pool is None:
      logging.info("No training pool provided. Using local in-process worker.")
      local_worker = training_pool_lib.TrainingWorker(config.training_config)
      self.training_pool = training_pool_lib.TrainingPool([local_worker])
    else:
      logging.info(
          "Using provided training pool of %d workers.",
          training_pool.num_workers,
      )
      self.training_pool = training_pool

    self._stop_requested = False
    self._next_agent_id = 0

  def save_state(self, progress_message: str | None = None):
    """Reports the current run state."""
    logging.info("Saving state: %s", progress_message)

  def run(self) -> None:
    """Runs the evolution algorithm."""
    random.seed(self.config.training_config.training_interface_config.seed)
    np.random.seed(self.config.training_config.training_interface_config.seed)
    self.save_state("Starting...")

    metrics_writer = metric_writers.LoggingWriter()

    try:
      population_start = time.monotonic()
      start_gen, current_agents = self.get_last_generation_population()
      population_seconds = time.monotonic() - population_start

      if start_gen >= self.config.num_generations:
        logging.info("Run %s is already completed.", self.run_id)
        self.status = "completed"
        self.save_state(progress_message="Evolution completed.")
        return
      # When resuming, evolving the last saved generation can also end the run.
      if self.check_equilibrium():
        self.status = "completed"
        self.current_generation = start_gen
        self.save_state(f"Equilibrium reached at generation {start_gen - 1}.")
        return

      for gen_idx in range(start_gen, self.config.num_generations):
        generation_start = time.monotonic()
        current_agents = self._sort_agents(current_agents)
        self.current_generation = gen_idx
        self.save_state(f"Starting Generation {gen_idx}")
        logging.info(
            "Agents: %s ", ", ".join(list(current_agents.keys())),
        )

        # Train agents
        training_start = time.monotonic()
        with training_pool_lib.trace(f"gen {gen_idx}: Training"):
          prev_gen = (
              self.generations_data[-1] if self.generations_data else None
          )
          to_train = dict[str, types.AgentData]()
          for agent_id, agent in current_agents.items():
            is_survivor = prev_gen and agent_id in prev_gen.selected_agents
            if is_survivor and agent.trained:
              logging.info(
                  "Gen %d: Agent %s is a survivor and has already been"
                  " trained.",
                  gen_idx,
                  agent_id,
              )
              continue
            to_train[agent_id] = agent

          if to_train:
            logging.info(
                "Gen %d: Training %s agents",
                gen_idx,
                ", ".join(list(to_train.keys())),
            )
            self.save_state(
                progress_message=(
                    f"Gen {gen_idx}: Training {len(to_train)} agents"
                )
            )
            newly_trained_agents = self._train_agents(to_train)
            current_agents.update(newly_trained_agents)
        current_agents, failed_agents = drop_failed_agents(
            current_agents, gen_idx
        )
        training_seconds = time.monotonic() - training_start

        self.save_state(f"Gen {gen_idx}: Evaluating agents...")

        evaluation_start = time.monotonic()
        with training_pool_lib.trace(f"gen {gen_idx}: Evaluating"):
          fitness_grid, random_fitness = self.training_pool.evaluate_agents(
              agents=current_agents, prev_gen=prev_gen
          )
        evaluation_seconds = time.monotonic() - evaluation_start

        self.save_state(f"Gen {gen_idx}: Selecting survivors...")
        logging.info("Gen %d: Selecting survivors", gen_idx)

        agent_ids = list(current_agents.keys())
        norm_fitness_grid, norm_random_fitness = metrics.normalize_by_columns(
            fitness_grid, random_fitness
        )
        specialist_quality = float(np.mean(np.diag(norm_fitness_grid)))
        behavioral_diversity = metrics.calculate_behavioral_diversity(
            norm_fitness_grid
        )
        logging.info(
            "Gen %d QD Metrics - Specialist Quality: %.4f, Behavioral"
            " Diversity: %.4f",
            gen_idx,
            specialist_quality,
            behavioral_diversity,
        )

        selected_ids = select_survivors(
            agent_ids=agent_ids,
            norm_fitness_grid=norm_fitness_grid,
            norm_random_fitness=norm_random_fitness,
        )

        generalization_scores = {
            agent_id: float(np.mean(norm_fitness_grid[i]))
            for i, agent_id in enumerate(agent_ids)
        }

        gen_data = types.GenerationData(
            generation=gen_idx,
            agents=current_agents,
            fitness_grid=fitness_grid.tolist(),
            generalization_scores=generalization_scores,
            selected_agents=selected_ids,
            random_fitness=random_fitness.tolist(),
            specialist_quality=specialist_quality,
            behavioral_diversity=behavioral_diversity,
            failed_agents=failed_agents,
        )
        self.generations_data.append(gen_data)
        self.save_state("Computing generation metrics...")

        scalars = compute_generation_metrics(
            gen_data,
            norm_fitness_grid=norm_fitness_grid,
            norm_random_fitness=norm_random_fitness,
            num_trained=len(to_train),
        )
        scalars.update(
            total_agents_created=float(self._next_agent_id),
            population_seconds=population_seconds,
            training_seconds=training_seconds,
            evaluation_seconds=evaluation_seconds,
            generation_seconds=time.monotonic() - generation_start,
        )
        metrics_writer.write_scalars(gen_idx, scalars)
        metrics_writer.flush()

        if gen_idx < self.config.num_generations - 1:
          # 5. Evolve next generation
          self.save_state(f"Evolving generation {gen_idx + 1}...")
          population_start = time.monotonic()
          with training_pool_lib.trace(f"gen {gen_idx}: Evolving"):
            current_agents = self.evolve_population()
          population_seconds = time.monotonic() - population_start

          # Stop as soon as the planner asks, rather than training and
          # evaluating the unchanged survivors for one more generation.
          if self.check_equilibrium():
            self.status = "completed"
            self.current_generation = gen_idx + 1
            self.save_state(f"Equilibrium reached at generation {gen_idx}.")
            return

      self.status = "completed"
      self.current_generation = self.config.num_generations
      self.save_state("Evolution complete.")

    # Mark the run as failed and report the error before re-raising.
    except Exception as e:  # pylint: disable=broad-except
      logging.exception("Evolution run failed")
      self.status = "failed"
      self.save_state(f"Error: {e}\n{traceback.format_exc()}")
      raise e
    finally:
      metrics_writer.close()

  def _train_agents(
      self,
      agents: dict[str, types.AgentData],
  ) -> dict[str, types.AgentData]:
    """Trains the given agents."""
    # Checkpoints are stored per run under the base path.
    checkpoint_base = os.path.join(
        self.config.training_config.training_interface_config.checkpoint_path,
        self.run_id,
    )
    logging.info("Checkpoint base path for training: %s", checkpoint_base)
    return self.training_pool.train_agents(
        agents=agents,
        base_checkpoint_path=checkpoint_base,
    )

  def _validate_config(self) -> None:
    """Validates that the training interface and environment factory are compatible."""
    interface = training_interface_registry.get(
        self.config.training_config.training_interface
    )(self.config.training_config.training_interface_config)

    env = environment.get(self.config.training_config.environment)(
        self.config.training_config.training_interface_config.seed,
        self.config.training_config.training_interface_config.playground,
    )

    interface.verify_env(env)

  def check_equilibrium(self) -> bool:
    """Checks if the evolution algorithm has reached equilibrium."""
    return self._stop_requested

  def _sort_agents(
      self, agents: dict[str, types.AgentData]
  ) -> dict[str, types.AgentData]:
    """Sorts agents dict by numeric agent ID."""

    def _agent_sort_key(agent_id: str) -> int:
      try:
        return int(agent_id.split("_")[-1])
      except (ValueError, IndexError):
        return 0

    return {k: agents[k] for k in sorted(agents.keys(), key=_agent_sort_key)}

  def get_last_generation_population(
      self,
  ) -> tuple[int, dict[str, types.AgentData]]:
    """Returns the last generation population."""
    if not self.generations_data:
      current_agents = self.initialize_population()
      current_agents = self._sort_agents(current_agents)
      self.save_state("Initializing population...")
      return 0, current_agents

    # Resume from last completed generation
    last_gen = self.generations_data[-1]
    start_gen = last_gen.generation + 1

    if start_gen >= self.config.num_generations:
      return start_gen, dict[str, types.AgentData]()

    self.save_state(
        f"Resuming: evolving generation {start_gen} from generation"
        f" {last_gen.generation}..."
    )
    current_agents = self.evolve_population()
    current_agents = self._sort_agents(current_agents)
    return start_gen, current_agents

  def evolve_population(self) -> dict[str, types.AgentData]:
    """Evolves the population to the next generation."""
    prev_gen = self.generations_data[-1]
    next_agents = dict[str, types.AgentData]()

    # Keep survivors (reward functions and IDs are preserved exactly)
    for survivor_id in prev_gen.selected_agents:
      survivor_agent = prev_gen.agents[survivor_id]
      next_agents[survivor_id] = dataclasses.replace(
          survivor_agent,
          generation=prev_gen.generation + 1,
      )

    max_num_offspring = self.config.max_new_offspring_per_generation
    if self.config.max_population_size is not None:
      remaining_capacity = self.config.max_population_size - len(next_agents)
      max_num_offspring = (
          remaining_capacity
          if max_num_offspring is None
          else min(max_num_offspring, remaining_capacity)
      )

    offspring = self.offspring_generator.generate_offspring(
        self.generations_data, max_num_offspring
    )
    self._stop_requested = self.offspring_generator.stop_requested

    can_add_offspring = max_num_offspring is None or max_num_offspring > 0
    if can_add_offspring and not offspring:
      logging.info("Offspring generator returned 0 offspring; requesting stop.")
      self._stop_requested = True
    for agent in offspring:
      agent_id = self.generate_agent_id()
      next_agents[agent_id] = dataclasses.replace(agent, id=agent_id)

    return self._sort_agents(next_agents)

  def generate_agent_id(self) -> str:
    """Generates and returns the next unique agent ID."""
    agent_id = f"agent_{self._next_agent_id}"
    self._next_agent_id += 1
    return agent_id

  def initialize_population(self) -> dict[str, types.AgentData]:
    """Initializes the population with diverse reward functions."""
    logging.info("Initializing population...")
    agents = {}
    self.save_state("Initializing population...")
    reward_codes = self.population_initializer.init_population(
        self.config.initial_population_size
    )
    for reward_code in reward_codes:
      agent_id = self.generate_agent_id()
      agents[agent_id] = types.AgentData(
          id=agent_id,
          generation=0,
          reward_code=reward_code,
          training_output=training_interface.TrainingOutput.empty(),
          generalization_score=0,
          parent_ids=[],
          embedding=[],
          reasoning="",
          path="",
      )

    logging.info("Initialized %d agents", len(agents))
    return agents


def _get_prev_agent_to_idx(
    prev_gen: types.GenerationData | None,
) -> dict[str, int]:
  """Returns a mapping from agent ID to its index in the previous generation."""
  if not prev_gen:
    return {}
  return {aid: idx for idx, aid in enumerate(prev_gen.agents.keys())}


def drop_failed_agents(
    agents: Mapping[str, types.AgentData], generation: int
) -> tuple[dict[str, types.AgentData], dict[str, str]]:
  """Removes the agents whose training failed.

  Training fails if an agent's evolved reward breaks it, e.g. by making it
  diverge. Such an agent is dropped from its generation instead of failing the
  whole run.

  Args:
    agents: The agents of a generation after training, keyed by ID.
    generation: Index of the generation, for logging.

  Returns:
    The agents whose training succeeded, and the IDs of the dropped agents
    mapped to their error. Both keep the order of `agents`.

  Raises:
    errors.NoTrainedAgentsError: If training failed for every agent.
  """
  trained_agents = {}
  failed_agents = {}
  for agent_id, agent in agents.items():
    error = agent.training_output.error
    if error is None:
      trained_agents[agent_id] = agent
    else:
      logging.warning(
          "Gen %d: dropping agent %s, whose training failed: %s",
          generation,
          agent_id,
          error,
      )
      failed_agents[agent_id] = error
  if failed_agents and not trained_agents:
    raise errors.NoTrainedAgentsError(
        f"Training failed for all {len(failed_agents)} agents of generation"
        f" {generation}: {failed_agents}"
    )
  return trained_agents, failed_agents


def select_survivors(
    agent_ids: Sequence[str],
    norm_fitness_grid: np.ndarray,
    norm_random_fitness: np.ndarray,
) -> list[str]:
  """Selects surviving specialist agents from the column-normalized fitness grid.

  An agent `i` survives if its policy is the top performer on its own reward
  column `i` and strictly outperforms the random baseline on reward `i`. Ties
  on column `i` are broken in favor of earlier (incumbent) agents (`k < i`) so
  that a redundant clone offspring cannot mutually annihilate its parent, while
  still rejecting duplicate later agents. If no agent qualifies, falls back to
  the single agent with the highest margin over random.

  Args:
    agent_ids: Ordered sequence of agent IDs corresponding to grid rows/columns.
    norm_fitness_grid: Column-normalized fitness matrix of shape (N, N).
    norm_random_fitness: Column-normalized random policy fitness of shape (N,).

  Returns:
    List of surviving agent IDs.
  """
  if not agent_ids:
    return []

  selected_ids: list[str] = []
  for i, agent_id in enumerate(agent_ids):
    self_fitness = norm_fitness_grid[i][i]

    is_best = True
    for k in range(len(agent_ids)):
      if k < i and norm_fitness_grid[k][i] >= self_fitness:
        is_best = False
        break
      if k > i and norm_fitness_grid[k][i] > self_fitness:
        is_best = False
        break

    beats_random = self_fitness > norm_random_fitness[i]

    if is_best and beats_random:
      selected_ids.append(agent_id)

  if not selected_ids:
    margins = [
        norm_fitness_grid[i][i] - norm_random_fitness[i]
        for i in range(len(agent_ids))
    ]
    best_idx = int(np.argmax(margins))
    selected_ids = [agent_ids[best_idx]]
    logging.info("Extinction fallback: selected %s", selected_ids[0])

  return selected_ids


def next_agent_index(generations: Sequence[types.GenerationData]) -> int:
  """Returns the index following those of all the agents of `generations`.

  Failed agents count too, so that their IDs, which also name their checkpoint
  directories, are never reused.

  Args:
    generations: The generations of a run.
  """
  indices = [
      int(agent_id.split("_")[-1])
      for generation in generations
      for agent_id in (*generation.agents, *generation.failed_agents)
  ]
  return max(indices, default=-1) + 1


def compute_generation_metrics(
    gen_data: types.GenerationData,
    *,
    norm_fitness_grid: np.ndarray,
    norm_random_fitness: np.ndarray,
    num_trained: int,
) -> dict[str, float]:
  """Computes the scalar metrics logged for one generation.

  Args:
    gen_data: The evaluated generation.
    norm_fitness_grid: Column-normalized fitness grid, where entry [i][j] is the
      fitness of policy i on reward j.
    norm_random_fitness: Column-normalized random-policy fitness per reward.
    num_trained: Number of agents trained during this generation.

  Returns:
    A mapping from metric name to value; these are the per-generation scalars
    written to the metrics writer.
  """
  num_agents = len(gen_data.agents)
  num_failed = float(len(gen_data.failed_agents))
  if num_agents == 0:
    return {
        "num_agents": 0.0,
        "num_trained": float(num_trained),
        "num_failed": num_failed,
    }

  num_survivors = len(gen_data.selected_agents)
  raw_self_fitness = np.diag(np.asarray(gen_data.fitness_grid, dtype=float))
  random_fitness = np.asarray(gen_data.random_fitness, dtype=float)
  specialist_margins = np.diag(norm_fitness_grid) - norm_random_fitness
  generalization = np.asarray(
      list(gen_data.generalization_scores.values()), dtype=float
  )

  return {
      "specialist_quality": gen_data.specialist_quality,
      "behavioral_diversity": gen_data.behavioral_diversity,
      "num_agents": float(num_agents),
      "num_survivors": float(num_survivors),
      "num_trained": float(num_trained),
      "num_failed": num_failed,
      "survival_rate": num_survivors / num_agents,
      "generalization_mean": float(np.mean(generalization)),
      "generalization_max": float(np.max(generalization)),
      "generalization_min": float(np.min(generalization)),
      "specialist_margin_mean": float(np.mean(specialist_margins)),
      "specialist_margin_min": float(np.min(specialist_margins)),
      "self_fitness_mean": float(np.mean(raw_self_fitness)),
      "self_fitness_max": float(np.max(raw_self_fitness)),
      "random_fitness_mean": float(np.mean(random_fitness)),
  }
