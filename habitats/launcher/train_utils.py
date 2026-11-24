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

"""Utilities for initializing and running policy training."""

import random

from absl import logging
from habitats.config import environment
from habitats.config import training
from habitats.policy import registry as policy_registry
from habitats.training import registry as training_interface_registry
from habitats.training import training_interface
import numpy as np


def run_training(
    config: training.TrainingConfig,
) -> training_interface.TrainingOutput:
  """Initializes the environment and policy for `config` and runs training.

  Args:
    config: The training configuration to execute.

  Returns:
    The resulting `TrainingOutput`.
  """
  random.seed(config.training_interface_config.seed)
  np.random.seed(config.training_interface_config.seed)

  interface = training_interface_registry.get(
      config.training_interface
  )(config.training_interface_config)

  environment_factory = environment.get(config.environment)
  env = environment_factory(
      config.training_interface_config.seed,
      config.training_interface_config.playground,
  )
  try:
    interface.verify_env(env)
    model_config = config.model_config.apply_environment_spec(
        actions=env.action_spec(),
        observations=env.observation_spec(),
    )
  finally:
    env.close()

  policy = policy_registry.create(config.policy, model_config)
  if config.init_ckpt_path:
    policy.load(config.init_ckpt_path)

  training_output = interface.train(environment_factory, policy)
  logging.info("Training finished.")
  logging.info("Average episode reward: %s", training_output.mean_reward)
  logging.info("Average episode length: %s", training_output.mean_length)
  return training_output
