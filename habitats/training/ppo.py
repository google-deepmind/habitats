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

"""PPO training interface."""

from collections.abc import Iterator, Mapping
from concurrent import futures
import queue
import tempfile
import threading
import time
from typing import Any, NamedTuple, cast, override

import sys
import types


class _StubModule(types.ModuleType):
  def __getattr__(self, name: str) -> Any:
    if name.startswith("__"):
      raise AttributeError(name)
    return self

  __call__ = lambda self, *_, **__: self


# Stub C++ `launchpad`/`reverb` plus `gym` (`acme.jax.networks` ->
# `gym_wrapper`), which `dm-acme` imports at load time even for JAX-only PPO.
for _mod_name in ("gym", "launchpad", "reverb"):
  try:
    __import__(_mod_name)
  except ModuleNotFoundError:
    sys.modules[_mod_name] = _StubModule(_mod_name)

from absl import logging
from acme import adders
from acme import environment_loop
from acme import specs as acme_specs
from acme.adders import reverb as adders_reverb
from acme.agents.jax import ppo
from acme.jax import experiments
from acme.jax import networks
from acme.jax import savers
from acme.jax import utils as jax_utils
from acme.jax.experiments import config as experiment_config
from acme.utils import counting
from acme.utils import loggers
from brax import envs
from brax.training import distribution as brax_distribution
from brax.training import networks as brax_networks
from brax.training import types as brax_types
from brax.training.agents.ppo import networks as brax_ppo_networks
from brax.training.agents.ppo import train as brax_ppo
from clu import metric_writers
import dm_env
from habitats.environment import brax_env

from habitats.policy import neural
from habitats.training import training_interface
import jax
import jax.numpy as jnp
import numpy as np

# Metric key renamings from Brax to Acme standard terms.
_METRIC_RENAMING: Mapping[str, str] = {
    # Eval metrics (Brax evaluator -> Acme)
    "episode_reward": "episode_return",
    "avg_episode_length": "episode_length",
    # Rollout metrics (Brax EpisodeMetricsLogger -> Acme)
    "sum_reward": "episode_return",
    "reward": "episode_return",
    "length": "episode_length",
    # Learner metrics (Brax SGD -> Acme)
    "total_loss": "loss",
}


def run_distributed_experiment(
    experiment: experiments.ExperimentConfig,
    num_actors: int,
    num_evaluators: int = 1,
    *,
    unroll_length: int = 8,
    batch_size: int = 256,
) -> Any:
  """Runs a multi-actor training loop in-process without Launchpad or Reverb.

  Args:
    experiment: The experiment config.
    num_actors: The number of actors.
    num_evaluators: The number of evaluators.
    unroll_length: PPO rollout unroll length per sequence chunk.
    batch_size: Number of sequences per learner update batch.

  Returns:
    The trained policy parameters.

  Raises:
    ValueError: If `batch_size` is not a positive multiple of the number of
      local JAX devices.
  """
  devices = jax.local_devices()
  per_device_batch_size, ragged = divmod(batch_size, len(devices))
  if per_device_batch_size <= 0 or ragged:
    raise ValueError(
        f"batch_size ({batch_size}) must be a positive multiple of the number"
        f" of local JAX devices ({len(devices)})."
    )

  key = jax.random.PRNGKey(experiment.seed)

  dummy_seed = 1
  spec = experiment.environment_spec or acme_specs.make_environment_spec(
      experiment.environment_factory(dummy_seed)
  )
  network = experiment.network_factory(spec)

  stop_event = threading.Event()
  seq_queue: queue.SimpleQueue[Any] = queue.SimpleQueue()

  _Sample = NamedTuple("_Sample", [("data", Any)])

  class _QueueAdder(adders.Adder):
    """Buffers `unroll_length` steps plus a bootstrap observation."""

    def __init__(self):
      self._steps: list[Any] = []
      self._obs: Any = None

    @override
    def reset(self) -> None:
      self._obs = None

    @override
    def add_first(self, timestep: dm_env.TimeStep) -> None:
      self._obs = timestep.observation

    @override
    def add(
        self, action: Any, next_timestep: dm_env.TimeStep, extras: Any = ()
    ) -> None:
      discount = (
          0.0
          if next_timestep.last() or next_timestep.discount is None
          else next_timestep.discount
      )
      self._steps.append(
          adders_reverb.Step(
              self._obs,
              action,
              np.float32(next_timestep.reward or 0.0),
              np.float32(discount),
              False,
              extras,
          )
      )
      self._obs = next_timestep.observation
      if len(self._steps) == unroll_length:
        while seq_queue.qsize() >= batch_size and not stop_event.is_set():
          time.sleep(0.001)
        boot = self._steps[-1]._replace(observation=self._obs)
        seq_queue.put(
            jax.tree_util.tree_map(lambda *xs: np.stack(xs), *self._steps, boot)
        )
        self._steps.clear()

  def _batches() -> Iterator[_Sample]:
    while True:
      items = [seq_queue.get() for _ in range(per_device_batch_size)]
      yield _Sample(jax.tree_util.tree_map(lambda *xs: np.stack(xs), *items))

  dataset = iter(jax_utils.multi_device_put(_batches(), devices))

  # 2. Build Parent Counter (make_distributed_experiment.py:222-237)
  parent_counter = counting.Counter(time_delta=0.0)

  # 3. Build Learner (make_distributed_experiment.py:239-285)
  learner_key, key = jax.random.split(key)
  learner_counter = counting.Counter(
      parent_counter, prefix="learner", time_delta=0.0
  )
  learner = experiment.builder.make_learner(
      random_key=learner_key,
      networks=network,
      dataset=cast(Iterator[Any], dataset),
      logger_fn=experiment.logger_factory,
      environment_spec=spec,
      replay_client=None,
      counter=learner_counter,
  )

  # Wrap learner with checkpointing (make_distributed_experiment.py:265-278).
  # Keep a reference to the raw learner for use as variable_source by actors,
  # since CheckpointingRunner doesn't implement VariableSource.
  # In the original code, actors access the learner via handles which hide the
  # type; here we pass the raw learner directly.
  variable_source = learner
  if experiment.checkpointing:
    checkpointing = experiment.checkpointing
    learner = savers.CheckpointingRunner(
        learner,
        key="learner",
        subdirectory="learner",
        time_delta_minutes=checkpointing.time_delta_minutes,
        directory=checkpointing.directory,
        add_uid=checkpointing.add_uid,
        max_to_keep=checkpointing.max_to_keep,
        keep_checkpoint_every_n_hours=checkpointing.keep_checkpoint_every_n_hours,
        checkpoint_ttl_seconds=checkpointing.checkpoint_ttl_seconds,
    )

  # 4. Build Actors & EnvironmentLoops
  *actor_keys, key = jax.random.split(key, num_actors + 1)
  actor_loops: list[environment_loop.EnvironmentLoop] = []
  for actor_id in range(num_actors):
    env_key, a_key = jax.random.split(actor_keys[actor_id])
    env = experiment.environment_factory(jax_utils.sample_uint32(env_key))
    env_spec = acme_specs.make_environment_spec(env)
    net = experiment.network_factory(env_spec)
    pol_net = experiment_config.make_policy(
        experiment=experiment,
        networks=net,
        environment_spec=env_spec,
        evaluation=False,
    )
    actor = experiment.builder.make_actor(
        a_key,
        pol_net,
        env_spec,
        variable_source=variable_source,
        adder=_QueueAdder(),
    )
    actor_counter = counting.Counter(
        parent_counter, prefix="actor", time_delta=0.0
    )
    logger = experiment.logger_factory(
        "actor", actor_counter.get_steps_key(), actor_id
    )
    loop = environment_loop.EnvironmentLoop(
        env,
        actor,
        counter=actor_counter,
        logger=logger,
        observers=experiment.observers,
    )
    actor_loops.append(loop)
  logging.info("Number of actors: %d", len(actor_loops))

  # Build Evaluators
  *eval_keys, _ = jax.random.split(key, num_evaluators + 1)
  eval_loops: list[environment_loop.EnvironmentLoop] = []
  for eval_id in range(num_evaluators):
    eval_env_key, eval_key = jax.random.split(eval_keys[eval_id])
    eval_env = experiment.environment_factory(
        jax_utils.sample_uint32(eval_env_key)
    )
    eval_env_spec = acme_specs.make_environment_spec(eval_env)
    eval_net = experiment.network_factory(eval_env_spec)
    eval_counter = counting.Counter(
        parent_counter, prefix="evaluator", time_delta=0.0
    )
    eval_logger = experiment.logger_factory(
        "evaluator", eval_counter.get_steps_key(), eval_id
    )
    eval_policy = experiment_config.make_policy(
        experiment=experiment,
        networks=eval_net,
        environment_spec=eval_env_spec,
        evaluation=True,
    )

    eval_actor = experiment.builder.make_actor(
        random_key=eval_key,
        policy=eval_policy,
        environment_spec=eval_env_spec,
        variable_source=variable_source,
    )

    eval_loop = environment_loop.EnvironmentLoop(
        eval_env,
        eval_actor,
        counter=eval_counter,
        logger=eval_logger,
        observers=experiment.observers,
    )
    eval_loops.append(eval_loop)
  logging.info("Number of evaluators: %d", len(eval_loops))

  # Thread Execution & Steps Limiter
  executor = futures.ThreadPoolExecutor(
      max_workers=num_actors + len(eval_loops) + 1
  )

  def run_actor(loop: environment_loop.EnvironmentLoop) -> None:
    """Runs actor episodes until stop_event is set."""
    while not stop_event.is_set():
      episode_start = time.time()
      result = loop.run_episode()
      result = {**result, "episode_duration": time.time() - episode_start}
      loop._logger.write(result)  # pylint: disable=protected-access

  def run_evaluator(loop: environment_loop.EnvironmentLoop) -> None:
    """Runs evaluator episodes until stop_event is set."""
    while not stop_event.is_set():
      episode_start = time.time()
      result = loop.run_episode()
      result = {**result, "episode_duration": time.time() - episode_start}
      loop._logger.write(result)  # pylint: disable=protected-access

  def run_learner_loop() -> None:
    """Runs learner steps whenever a full batch is queued."""
    while not stop_event.is_set():
      if seq_queue.qsize() >= batch_size:
        learner.step()
      else:
        time.sleep(0.001)

  learner_future = executor.submit(run_learner_loop)
  actor_futures = [executor.submit(run_actor, loop) for loop in actor_loops]
  eval_futures = [executor.submit(run_evaluator, loop) for loop in eval_loops]
  all_futures = actor_futures + eval_futures + [learner_future]

  # Monitor total actor steps across all actors (like StepsLimiter)
  try:
    while True:
      # Check for thread failures early.
      for f in all_futures:
        if f.done():
          f.result()  # Re-raises any exception from the thread.
      counts = parent_counter.get_counts()
      actor_steps = counts.get("actor_steps", 0)
      if (
          experiment.max_num_actor_steps
          and actor_steps >= experiment.max_num_actor_steps
      ):
        break
      time.sleep(0.1)
  finally:
    stop_event.set()
    for f in all_futures:
      f.result()  # Re-raises any exception from the thread.
    executor.shutdown(wait=True)
    if isinstance(learner, savers.CheckpointingRunner):
      learner._checkpointer.save(force=True)  # pylint: disable=protected-access
    for loop in actor_loops + eval_loops:
      loop._environment.close()  # pylint: disable=protected-access

  # PPOLearner.get_variables(["params"])[0] returns an instance of
  # acme.agents.jax.ppo.networks.PPOParams(model_params=..., num_sgd_steps=...).
  # We extract model_params (the actual Flax network weights dict) which is what
  # MLPPolicy.apply() expects and what policy.save() / policy.load() operates
  # on.
  params = variable_source.get_variables(["params"])[0]
  return params.model_params


def _raise_if_params_diverged(
    current_step: int, make_policy: Any, params: tuple[Any, Any, Any]
) -> None:
  """Brax `policy_params_fn` stopping training as soon as it diverges.

  Brax calls it on the host after every evaluation. Without it, NaN parameters
  would only be detected after the last training step, by an assertion without
  a message in `brax.training.pmap.assert_is_replicated`.

  Args:
    current_step: Number of environment steps taken so far.
    make_policy: Brax's policy factory. Unused.
    params: The normalizer, policy and value parameters.

  Raises:
    TrainingDivergedError: If any of the parameters is NaN or infinite.
  """
  del make_policy  # Unused.
  normalizer_params, policy_params, value_params = params
  training_interface.raise_if_diverged(
      {
          "normalizer": normalizer_params,
          "policy": policy_params,
          "value": value_params,
      },
      env_steps=current_step,
  )


class PPOInterface(training_interface.TrainingInterface[neural.MLPPolicy]):
  """Represents an interface that uses PPO for training/evaluation."""

  @override
  def train(
      self,
      env_factory: training_interface.EnvironmentFactory,
      policy: neural.MLPPolicy,
  ) -> training_interface.TrainingOutput:
    """Trains the agent."""

    probe_env = env_factory(self.config.seed, self.config.playground)
    brax_environment: envs.Env | None = None
    if isinstance(probe_env, brax_env.BraxEnv):
      try:
        brax_environment = probe_env.to_brax_env()
        logging.info(
            "Environment '%s' implements BraxEnv; using on-device Brax PPO"
            " training.",
            type(probe_env).__name__,
        )
      except NotImplementedError as e:
        logging.info(
            "Environment '%s' implements BraxEnv but conversion failed: %s."
            " Falling back to Acme PPO.",
            type(probe_env).__name__,
            e,
        )
    else:
      logging.info(
          "Environment '%s' does not implement BraxEnv. Using Acme PPO"
          " training.",
          type(probe_env).__name__,
      )
    probe_env.close()

    if brax_environment is not None:
      return self._train_mjx(brax_environment, policy)
    else:
      return self._train_acme(env_factory, policy)

  def _train_mjx(
      self,
      brax_environment: envs.Env,
      policy: neural.MLPPolicy,
  ) -> training_interface.TrainingOutput:
    """Fast on-device vectorized PPO training loop using Brax."""

    tagged_actor_label = (
        f"{self.config.tag}_actor" if self.config.tag else "actor"
    )
    tagged_eval_label = (
        f"{self.config.tag}_evaluator" if self.config.tag else "evaluator"
    )
    tagged_learner_label = (
        f"{self.config.tag}_learner" if self.config.tag else "learner"
    )

    actor_writer = metric_writers.create_default_writer(
        self.config.checkpoint_path,
        just_logging=jax.process_index() > 0,
        # Must be asynchronous because progress_fn is called from inside a JAX
        # host callback (jax.debug.callback). Synchronous metric writing can
        # trigger device-to-host transfers (ToLiteral) which PJRT forbids inside
        # host callbacks to prevent deadlocks.
        asynchronous=True,
        collection=tagged_actor_label,
    )

    eval_writer = metric_writers.create_default_writer(
        self.config.checkpoint_path,
        just_logging=jax.process_index() > 0,
        asynchronous=True,
        collection=tagged_eval_label,
    )

    learner_writer = metric_writers.create_default_writer(
        self.config.checkpoint_path,
        just_logging=jax.process_index() > 0,
        asynchronous=True,
        collection=tagged_learner_label,
    )

    num_devices = jax.local_device_count()
    num_envs = self.config.num_actors
    num_envs = max(num_devices, (num_envs // num_devices) * num_devices)

    eval_episode_rewards: list[float] = []
    eval_episode_lengths: list[float] = []

    def progress_fn(num_steps: int, metrics: Mapping[str, Any]) -> None:
      eval_scalars: dict[str, float] = {}
      actor_scalars: dict[str, float] = {}
      learner_scalars: dict[str, float] = {}

      for key, raw_val in metrics.items():
        try:
          val = float(raw_val)
        except (ValueError, TypeError):
          continue

        prefix, _, raw_name = key.partition("/")
        name = _METRIC_RENAMING.get(raw_name, raw_name)

        if prefix == "eval":
          eval_scalars[name] = val
          if name == "episode_return":
            eval_episode_rewards.append(val)
          elif name == "episode_length":
            eval_episode_lengths.append(val)
        elif prefix == "episode":
          actor_scalars[name] = val
        elif prefix == "training" or prefix == "params":
          learner_scalars[name] = val

      if eval_scalars:
        eval_writer.write_scalars(step=num_steps, scalars=eval_scalars)
        if "episode_return" in eval_scalars:
          logging.info(
              "[Eval] Step %d: return=%.2f, length=%.1f",
              num_steps,
              eval_scalars["episode_return"],
              eval_scalars.get("episode_length", 0.0),
          )

      if actor_scalars:
        actor_writer.write_scalars(step=num_steps, scalars=actor_scalars)
        if "episode_return" in actor_scalars:
          logging.info(
              "[Actor] Step %d: return=%.2f, length=%.1f",
              num_steps,
              actor_scalars["episode_return"],
              actor_scalars.get("episode_length", 0.0),
          )

      if learner_scalars:
        learner_writer.write_scalars(step=num_steps, scalars=learner_scalars)

    def custom_network_factory(
        observation_size: brax_types.ObservationSize,
        action_size: int,
        preprocess_observations_fn: brax_types.PreprocessObservationFn = brax_types.identity_observation_preprocessor,
    ) -> brax_ppo_networks.PPONetworks:
      obs_shape = (
          (observation_size,)
          if isinstance(observation_size, int)
          else tuple(observation_size)
      )
      dummy_obs = jnp.zeros((1,) + obs_shape)

      def policy_init(key: jax.Array) -> Any:
        return policy.actor.init(key, dummy_obs)

      def policy_apply(
          processor_params: Any, policy_params: Any, obs: jax.Array
      ) -> Any:
        obs = preprocess_observations_fn(obs, processor_params)
        return policy.actor.apply(policy_params, obs)

      def value_init(key: jax.Array) -> Any:
        return policy.critic.init(key, dummy_obs)

      def value_apply(
          processor_params: Any, value_params: Any, obs: jax.Array
      ) -> Any:
        obs = preprocess_observations_fn(obs, processor_params)
        return policy.critic.apply(value_params, obs)

      return brax_ppo_networks.PPONetworks(
          policy_network=brax_networks.FeedForwardNetwork(
              init=policy_init, apply=policy_apply
          ),
          value_network=brax_networks.FeedForwardNetwork(
              init=value_init, apply=value_apply
          ),
          parametric_action_distribution=brax_distribution.NormalTanhDistribution(
              event_size=action_size
          ),
      )

    logging.info(
        "Starting Brax MJX PPO training (%d envs, %d timesteps, %d unroll)",
        num_envs,
        self.config.num_timesteps,
        self.config.unroll_length,
    )
    start_time = time.time()
    try:
      _, trained_params, _ = brax_ppo.train(
          environment=brax_environment,
          num_timesteps=self.config.num_timesteps,
          num_envs=num_envs,
          episode_length=self.config.max_steps,
          unroll_length=self.config.unroll_length,
          num_minibatches=self.config.num_minibatches,
          batch_size=self.config.batch_size,
          num_updates_per_batch=self.config.ppo_epochs,
          learning_rate=self.config.learning_rate,
          discounting=self.config.discount_factor,
          gae_lambda=self.config.gae_lambda,
          clipping_epsilon=self.config.value_clipping_epsilon,
          normalize_advantage=self.config.normalize_advantage,
          seed=self.config.seed,
          num_evals=self.config.num_train_evals,
          num_eval_envs=self.config.num_eval_envs,
          network_factory=custom_network_factory,
          progress_fn=progress_fn,
          policy_params_fn=_raise_if_params_diverged,
          run_evals=True,
          save_checkpoint_path=None,
          log_training_metrics=True,
          training_metrics_steps=self.config.log_every,
          reward_scaling=self.config.reward_scaling,
          normalize_observations=self.config.normalize_observations,
          entropy_cost=self.config.entropy_cost,
      )
    finally:
      actor_writer.flush()
      actor_writer.close()
      eval_writer.flush()
      eval_writer.close()
      learner_writer.flush()
      learner_writer.close()
    end_time = time.time()
    logging.info("Brax PPO training complete in %.2fs", end_time - start_time)

    normalizer_state = trained_params[0]
    actor_params = trained_params[1]
    critic_params = trained_params[2]

    normalizer_dict = None
    if self.config.normalize_observations and hasattr(normalizer_state, "mean"):
      normalizer_dict = {
          "mean": np.asarray(normalizer_state.mean),
          "std": np.asarray(normalizer_state.std),
      }

    policy.set_params({
        "actor": actor_params,
        "critic": critic_params,
        "normalizer": normalizer_dict,
    })

    if self.config.checkpoint_path:
      logging.info("Saving policy to %s", self.config.checkpoint_path)
      policy.save(self.config.checkpoint_path)

    return training_interface.TrainingOutput(
        episode_rewards=np.array(eval_episode_rewards),
        episode_lengths=np.array(eval_episode_lengths),
        epsilons=np.array([]),
    )

  def _train_acme(
      self,
      env_factory: training_interface.EnvironmentFactory,
      policy: neural.MLPPolicy,
  ) -> training_interface.TrainingOutput:
    """Acme training fallback for non-vectorized/discrete environments."""
    actor_logger = loggers.InMemoryLogger()
    evaluator_logger = loggers.InMemoryLogger()
    learner_logger = loggers.InMemoryLogger()

    def custom_logger(
        label: str,
        steps_key: str | None = None,
        task_id: int | None = None,
    ) -> loggers.Logger:
      del task_id
      tagged_label = f"{self.config.tag}_{label}" if self.config.tag else label
      default_logger = loggers.make_default_logger(
          label=tagged_label,
          save_data=False,
          steps_key=steps_key or "steps",
      )
      if label == "learner":
        return loggers.Dispatcher([default_logger, learner_logger])
      if label == "actor":
        return loggers.Dispatcher([default_logger, actor_logger])
      if label == "evaluator":
        return loggers.Dispatcher([default_logger, evaluator_logger])
      return default_logger

    ppo_config = ppo.PPOConfig(
        learning_rate=self.config.learning_rate,
        unroll_length=self.config.unroll_length,
        num_minibatches=self.config.num_minibatches,
        num_epochs=self.config.ppo_epochs,
        batch_size=self.config.batch_size,
        entropy_cost=self.config.entropy_cost,
        clip_value=self.config.clip_value,
        value_clipping_epsilon=self.config.value_clipping_epsilon,
        normalize_advantage=self.config.normalize_advantage,
        gae_lambda=self.config.gae_lambda,
        discount=self.config.discount_factor,
    )
    builder = ppo.PPOBuilder(ppo_config)

    with tempfile.TemporaryDirectory() as tmpdir:
      checkpointing_config = experiments.CheckpointingConfig(
          directory=tmpdir,
          time_delta_minutes=5,
          add_uid=False,
      )
      logging.info("Checkpointing config: %s", checkpointing_config)
      logging.info("Saving to %s", tmpdir)

      # Compute the total number of actor steps.

      total_actor_steps = self.config.num_timesteps
      logging.info("Total number of actor steps: %d", total_actor_steps)

      config = experiments.ExperimentConfig(
          builder=builder,
          environment_factory=lambda seed: env_factory(
              seed, self.config.playground
          ),
          network_factory=self._make_network_factory(policy),
          seed=self.config.seed,
          max_num_actor_steps=total_actor_steps,
          checkpointing=checkpointing_config,
          logger_factory=custom_logger,  # pyrefly: ignore[bad-argument-type]
      )

      start_time = time.time()
      trained_params = run_distributed_experiment(
          config,
          num_actors=self.config.num_actors,
          unroll_length=self.config.unroll_length,
          batch_size=self.config.batch_size,
      )
      end_time = time.time()
      logging.info("PPO Training took %s seconds", end_time - start_time)
      training_interface.raise_if_diverged({
          "actor": trained_params["actor"],
          "critic": trained_params["critic"],
      })

      policy.set_params({
          "actor": {"params": trained_params["actor"]},
          "critic": {"params": trained_params["critic"]},
          "normalizer": None,
      })
      if self.config.checkpoint_path:
        logging.info("Saving policy to %s", self.config.checkpoint_path)
        policy.save(self.config.checkpoint_path)

    episode_rewards = []
    episode_lengths = []
    for entry in evaluator_logger.data:
      if "episode_return" in entry:
        episode_rewards.append(float(entry["episode_return"]))
      if "episode_length" in entry:
        episode_lengths.append(float(entry["episode_length"]))

    return training_interface.TrainingOutput(
        episode_rewards=np.array(episode_rewards),
        episode_lengths=np.array(episode_lengths),
        epsilons=np.array([]),
    )

  def evaluate(
      self,
      env_factory: training_interface.EnvironmentFactory,
      policy: neural.MLPPolicy,
      is_random_policy: bool = False,
  ) -> training_interface.EvaluationOutput:
    """Evaluates the policy."""
    env = env_factory(self.config.seed, self.config.playground)
    total_rewards = []
    total_lengths = []
    action_spec = env.action_spec()

    try:
      for _ in range(self.config.num_eval_episodes):
        timestep = env.reset()
        episode_reward = 0.0
        steps = 0
        while not timestep.last() and steps < self.config.max_eval_steps:
          if is_random_policy:
            action_np = np.random.normal(size=action_spec.shape).astype(
                action_spec.dtype
            )
          else:
            action_np = policy.get_action(timestep.observation, exploit=True)
          timestep = env.step(action_np)
          episode_reward += float(timestep.reward or 0.0)
          steps += 1
        total_rewards.append(episode_reward)
        total_lengths.append(steps)
    finally:
      env.close()

    return training_interface.EvaluationOutput(
        episode_rewards=np.array(total_rewards),
        episode_lengths=np.array(total_lengths),
    )

  def _make_network_factory(
      self,
      policy: neural.MLPPolicy,
  ) -> experiments.NetworkFactory[ppo.PPONetworks]:
    """Returns a network factory for the given environment spec."""

    def network_factory(
        environment_spec: acme_specs.EnvironmentSpec,
    ) -> ppo.PPONetworks:
      model = policy.network
      dummy_observation = jax_utils.zeros_like(environment_spec.observations)
      dummy_observation = jax_utils.add_batch_dim(dummy_observation)

      def init_fn(key: jax.Array) -> networks.Params:
        variables = model.init(key, dummy_observation)
        return variables["params"]

      def apply_fn(
          params: networks.Params,
          observation: Any,
      ) -> tuple[ppo.networks.TanhNormalParams, Any]:
        output = cast(
            tuple[tuple[jax.Array, jax.Array], jax.Array],
            model.apply({"params": params}, observation),
        )
        (loc, scale), value = output
        policy_output = ppo.networks.TanhNormalParams(
            loc=loc,
            scale=scale,
        )
        return policy_output, value

      network = networks.FeedForwardNetwork(
          init=init_fn,
          apply=apply_fn,
      )
      return ppo.make_tanh_normal_ppo_networks(network)

    return network_factory
