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

"""Training config registry."""

from collections.abc import Callable, Mapping
import dataclasses
from typing import Any

from habitats.config import environment
from habitats.policy import policy as policy_lib
from habitats.policy import registry as policy_registry
from habitats.training import training_interface

TrainingInterfaceName = training_interface.TrainingInterfaceName
PolicyName = policy_lib.PolicyName


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
  """Represents the training config."""

  model_config: policy_lib.ModelConfig | policy_lib.MetaModelConfig
  training_interface_config: training_interface.TrainingInterfaceConfig
  training_interface: TrainingInterfaceName
  environment: environment.EnvironmentName
  policy: PolicyName
  init_ckpt_path: str = ''

  @classmethod
  def from_dict(cls, data: Mapping[str, Any]) -> 'TrainingConfig':
    """Creates a TrainingConfig from a dictionary."""
    policy_name = data['policy']
    if isinstance(policy_name, str):
      policy_name = PolicyName(policy_name)

    model_cfg = data.get('model_config', {})
    if isinstance(model_cfg, Mapping):
      is_meta_policy = policy_registry.is_meta_policy(policy_name)
      if is_meta_policy:
        model_cfg = policy_lib.MetaModelConfig.from_dict(model_cfg)
      else:
        model_cfg = policy_lib.ModelConfig.from_dict(model_cfg)

    ti_cfg = data.get('training_interface_config', {})
    if isinstance(ti_cfg, Mapping):
      ti_cfg = training_interface.TrainingInterfaceConfig(**ti_cfg)

    ti_name = data['training_interface']
    if isinstance(ti_name, str):
      ti_name = TrainingInterfaceName(ti_name)

    env_name = data['environment']
    if isinstance(env_name, str):
      env_name = environment.EnvironmentName(env_name)

    return cls(
        model_config=model_cfg,
        training_interface_config=ti_cfg,
        training_interface=ti_name,
        environment=env_name,
        policy=policy_name,
        init_ckpt_path=data.get('init_ckpt_path', ''),
    )


TRAINING_REGISTRY: dict[str, Callable[..., TrainingConfig]] = {}


def register(
    f: Callable[..., TrainingConfig],
) -> Callable[..., TrainingConfig]:
  """Helper function to access local registry."""
  name = getattr(f, '__name__', None)
  if name is None:
    raise ValueError(f'Function {f} has no __name__.')

  if name in TRAINING_REGISTRY:
    raise ValueError(f'{name} is already registered.')
  TRAINING_REGISTRY[name] = f
  return f


def get_config(name: str) -> TrainingConfig:
  """Returns the training config for the given name."""
  return TRAINING_REGISTRY[name]()


@register
def ppo_cartpole() -> TrainingConfig:
  """Returns the PPO config."""
  return TrainingConfig(
      model_config=policy_lib.ModelConfig(),
      training_interface_config=training_interface.TrainingInterfaceConfig(
          learning_rate=3e-4,
          unroll_length=5,
          num_minibatches=32,
          clip_value=True,
          value_clipping_epsilon=0.2,
          normalize_advantage=True,
          gae_lambda=0.95,
          discount_factor=0.97,
          num_timesteps=10_000_000,
          num_actors=2048,
          batch_size=1024,
          num_train_evals=20,
          ppo_epochs=4,
          normalize_observations=True,
          num_eval_episodes=20,
          entropy_cost=1e-2,
          reward_scaling=10.0,
      ),
      training_interface=TrainingInterfaceName.PPO,
      environment=environment.EnvironmentName.CARTPOLE,
      policy=PolicyName.MLP,
  )


@register
def ppo_hopper() -> TrainingConfig:
  """Returns the PPO config."""
  return TrainingConfig(
      model_config=policy_lib.ModelConfig(),
      training_interface_config=training_interface.TrainingInterfaceConfig(
          learning_rate=6e-4,
          unroll_length=64,
          num_minibatches=32,
          clip_value=True,
          value_clipping_epsilon=0.2,
          normalize_advantage=True,
          gae_lambda=0.95,
          discount_factor=0.997,
          num_timesteps=50_000_000,
          num_actors=2048,
          batch_size=512,
          num_eval_episodes=20,
          reward_scaling=10.0,
          normalize_observations=True,
      ),
      training_interface=TrainingInterfaceName.PPO,
      environment=environment.EnvironmentName.HOPPER,
      policy=PolicyName.MLP,
  )


@register
def ppo_cheetah() -> TrainingConfig:
  """Returns the PPO config."""
  return TrainingConfig(
      model_config=policy_lib.ModelConfig(),
      training_interface_config=training_interface.TrainingInterfaceConfig(
          learning_rate=3e-4,
          unroll_length=20,
          num_minibatches=32,
          clip_value=True,
          value_clipping_epsilon=0.2,
          normalize_advantage=True,
          gae_lambda=0.95,
          discount_factor=0.95,
          num_timesteps=50_000_000,
          num_actors=2048,
          batch_size=512,
          num_eval_episodes=20,
          reward_scaling=10.0,
          normalize_observations=True,
          entropy_cost=1e-3,
      ),
      training_interface=TrainingInterfaceName.PPO,
      environment=environment.EnvironmentName.CHEETAH,
      policy=PolicyName.MLP,
  )


@register
def ppo_quadruped() -> TrainingConfig:
  """Returns the PPO config."""
  return TrainingConfig(
      model_config=policy_lib.ModelConfig(
          policy_hidden_sizes=(32, 32, 32, 32),
          value_hidden_sizes=(256, 256, 256, 256, 256),
      ),
      training_interface_config=training_interface.TrainingInterfaceConfig(
          learning_rate=3e-4,
          unroll_length=5,
          num_minibatches=32,
          clip_value=True,
          value_clipping_epsilon=0.3,
          normalize_advantage=True,
          normalize_observations=True,
          gae_lambda=0.95,
          discount_factor=0.97,
          num_timesteps=50_000_000,
          num_actors=4096,
          batch_size=2048,
          ppo_epochs=4,
          reward_scaling=10.0,
          entropy_cost=1e-2,
          num_train_evals=20,
          init_noise_std=1.0,
          seed=1,
      ),
      training_interface=TrainingInterfaceName.PPO,
      environment=environment.EnvironmentName.QUADRUPED,
      policy=PolicyName.MLP,
  )


@register
def ppo_humanoid() -> TrainingConfig:
  """Returns the PPO config."""
  return TrainingConfig(
      model_config=policy_lib.ModelConfig(
          policy_hidden_sizes=(64, 64, 64, 64),
          value_hidden_sizes=(256, 256, 256, 256, 256),
      ),
      training_interface_config=training_interface.TrainingInterfaceConfig(
          learning_rate=3e-4,
          unroll_length=10,
          num_minibatches=32,
          clip_value=True,
          value_clipping_epsilon=0.3,
          normalize_advantage=True,
          normalize_observations=True,
          gae_lambda=0.95,
          discount_factor=0.97,
          num_timesteps=50_000_000,
          num_actors=2048,
          batch_size=1024,
          ppo_epochs=8,
          reward_scaling=0.1,
          entropy_cost=1e-3,
          num_train_evals=10,
          init_noise_std=1.0,
          seed=1,
      ),
      training_interface=TrainingInterfaceName.PPO,
      environment=environment.EnvironmentName.HUMANOID,
      policy=PolicyName.MLP,
  )


@register
def q_learning_tic_tac_toe() -> TrainingConfig:
  """Returns the Q-learning config."""
  return TrainingConfig(
      model_config=policy_lib.ModelConfig(),
      training_interface_config=training_interface.TrainingInterfaceConfig(),
      training_interface=TrainingInterfaceName.Q_LEARNING,
      environment=environment.EnvironmentName.TIC_TAC_TOE,
      policy=PolicyName.TABULAR,
  )


@register
def q_learning_mini_card_deck() -> TrainingConfig:
  """Returns the Q-learning config."""
  return TrainingConfig(
      model_config=policy_lib.ModelConfig(),
      training_interface_config=training_interface.TrainingInterfaceConfig(
          num_episodes=20_000,
          learning_rate=0.05,
          discount_factor=0.95,
          epsilon_decay=0.9995,
          max_eval_steps=100,
      ),
      training_interface=TrainingInterfaceName.Q_LEARNING,
      environment=environment.EnvironmentName.MINI_CARD_DECK,
      policy=PolicyName.TABULAR,
  )


def get_qlearning_card_deck_interface_config() -> (
    training_interface.TrainingInterfaceConfig
):
  """Returns the Q-learning interface config for card deck games."""
  return training_interface.TrainingInterfaceConfig(
      num_episodes=5000,
      log_every=10,
      learning_rate=0.05,
      discount_factor=0.95,
      exploration_fraction=0.8,
      num_eval_episodes=50,
      max_eval_steps=100,
  )


def _make_card_deck_q_learning_config(
    env_name: environment.EnvironmentName,
) -> TrainingConfig:
  """Returns a tabular Q-learning TrainingConfig for a card deck game."""
  return TrainingConfig(
      model_config=policy_lib.ModelConfig(),
      training_interface_config=get_qlearning_card_deck_interface_config(),
      training_interface=TrainingInterfaceName.Q_LEARNING,
      environment=env_name,
      policy=PolicyName.TABULAR,
  )


@register
def q_learning_ascending_sort_game() -> TrainingConfig:
  """Returns the Q-learning config for AscendingSortGame."""
  return _make_card_deck_q_learning_config(
      environment.EnvironmentName.ASCENDING_SORT_GAME
  )


@register
def q_learning_descending_sort_game() -> TrainingConfig:
  """Returns the Q-learning config for DescendingSortGame."""
  return _make_card_deck_q_learning_config(
      environment.EnvironmentName.DESCENDING_SORT_GAME
  )


@register
def q_learning_triplet_matching_game() -> TrainingConfig:
  """Returns the Q-learning config for TripletMatchingGame."""
  return _make_card_deck_q_learning_config(
      environment.EnvironmentName.TRIPLET_MATCHING_GAME
  )


@register
def q_learning_target_sum_game() -> TrainingConfig:
  """Returns the Q-learning config for TargetSumGame."""
  return _make_card_deck_q_learning_config(
      environment.EnvironmentName.TARGET_SUM_GAME
  )


@register
def q_learning_high_card_maximizer_game() -> TrainingConfig:
  """Returns the Q-learning config for HighCardMaximizerGame."""
  return _make_card_deck_q_learning_config(
      environment.EnvironmentName.HIGH_CARD_MAXIMIZER_GAME
  )


@register
def q_learning_low_card_minimalist_game() -> TrainingConfig:
  """Returns the Q-learning config for LowCardMinimalistGame."""
  return _make_card_deck_q_learning_config(
      environment.EnvironmentName.LOW_CARD_MINIMALIST_GAME
  )


@register
def q_learning_even_odd_pattern_game() -> TrainingConfig:
  """Returns the Q-learning config for EvenOddPatternGame."""
  return _make_card_deck_q_learning_config(
      environment.EnvironmentName.EVEN_ODD_PATTERN_GAME
  )


@register
def q_learning_sandwich_pair_game() -> TrainingConfig:
  """Returns the Q-learning config for SandwichPairGame."""
  return _make_card_deck_q_learning_config(
      environment.EnvironmentName.SANDWICH_PAIR_GAME
  )


@register
def q_learning_rapid_clear_discard_game() -> TrainingConfig:
  """Returns the Q-learning config for RapidClearDiscardGame."""
  return _make_card_deck_q_learning_config(
      environment.EnvironmentName.RAPID_CLEAR_DISCARD_GAME
  )


@register
def q_learning_full_house_triple_pair_game() -> TrainingConfig:
  """Returns the Q-learning config for FullHouseTriplePairGame."""
  return _make_card_deck_q_learning_config(
      environment.EnvironmentName.FULL_HOUSE_TRIPLE_PAIR_GAME
  )


@register
def q_learning_meta_tic_tac_toe() -> TrainingConfig:
  """Returns the Q-learning meta-policy config for Tic-Tac-Toe."""
  return TrainingConfig(
      model_config=policy_lib.MetaModelConfig(
          policy_config=policy_lib.ModelConfig(),
          router_config=policy_lib.ModelConfig(),
      ),
      training_interface_config=training_interface.TrainingInterfaceConfig(
          log_every=10,
      ),
      training_interface=TrainingInterfaceName.Q_LEARNING,
      environment=environment.EnvironmentName.TIC_TAC_TOE,
      policy=PolicyName.TABULAR_META,
  )


@register
def q_learning_meta_mini_card_deck() -> TrainingConfig:
  """Returns the Q-learning meta-policy config for Mini Card Deck."""
  return _make_card_deck_meta_q_learning_config(
      environment.EnvironmentName.MINI_CARD_DECK
  )


def _make_card_deck_meta_q_learning_config(
    env_name: environment.EnvironmentName,
) -> TrainingConfig:
  """Returns a tabular Q-learning MetaPolicy TrainingConfig for a card deck game."""
  return TrainingConfig(
      model_config=policy_lib.MetaModelConfig(
          policy_config=policy_lib.ModelConfig(),
          router_config=policy_lib.ModelConfig(),
      ),
      training_interface_config=get_qlearning_card_deck_interface_config(),
      training_interface=TrainingInterfaceName.Q_LEARNING,
      environment=env_name,
      policy=PolicyName.TABULAR_META,
  )


@register
def q_learning_meta_ascending_sort_game() -> TrainingConfig:
  """Returns the Q-learning meta-policy config for AscendingSortGame."""
  return _make_card_deck_meta_q_learning_config(
      environment.EnvironmentName.ASCENDING_SORT_GAME
  )


@register
def q_learning_meta_descending_sort_game() -> TrainingConfig:
  """Returns the Q-learning meta-policy config for DescendingSortGame."""
  return _make_card_deck_meta_q_learning_config(
      environment.EnvironmentName.DESCENDING_SORT_GAME
  )


@register
def q_learning_meta_triplet_matching_game() -> TrainingConfig:
  """Returns the Q-learning meta-policy config for TripletMatchingGame."""
  return _make_card_deck_meta_q_learning_config(
      environment.EnvironmentName.TRIPLET_MATCHING_GAME
  )


@register
def q_learning_meta_target_sum_game() -> TrainingConfig:
  """Returns the Q-learning meta-policy config for TargetSumGame."""
  return _make_card_deck_meta_q_learning_config(
      environment.EnvironmentName.TARGET_SUM_GAME
  )


@register
def q_learning_meta_high_card_maximizer_game() -> TrainingConfig:
  """Returns the Q-learning meta-policy config for HighCardMaximizerGame."""
  return _make_card_deck_meta_q_learning_config(
      environment.EnvironmentName.HIGH_CARD_MAXIMIZER_GAME
  )


@register
def q_learning_meta_low_card_minimalist_game() -> TrainingConfig:
  """Returns the Q-learning meta-policy config for LowCardMinimalistGame."""
  return _make_card_deck_meta_q_learning_config(
      environment.EnvironmentName.LOW_CARD_MINIMALIST_GAME
  )


@register
def q_learning_meta_even_odd_pattern_game() -> TrainingConfig:
  """Returns the Q-learning meta-policy config for EvenOddPatternGame."""
  return _make_card_deck_meta_q_learning_config(
      environment.EnvironmentName.EVEN_ODD_PATTERN_GAME
  )


@register
def q_learning_meta_sandwich_pair_game() -> TrainingConfig:
  """Returns the Q-learning meta-policy config for SandwichPairGame."""
  return _make_card_deck_meta_q_learning_config(
      environment.EnvironmentName.SANDWICH_PAIR_GAME
  )


@register
def q_learning_meta_rapid_clear_discard_game() -> TrainingConfig:
  """Returns the Q-learning meta-policy config for RapidClearDiscardGame."""
  return _make_card_deck_meta_q_learning_config(
      environment.EnvironmentName.RAPID_CLEAR_DISCARD_GAME
  )


@register
def q_learning_meta_full_house_triple_pair_game() -> TrainingConfig:
  """Returns the Q-learning meta-policy config for FullHouseTriplePairGame."""
  return _make_card_deck_meta_q_learning_config(
      environment.EnvironmentName.FULL_HOUSE_TRIPLE_PAIR_GAME
  )


@register
def test() -> TrainingConfig:
  """Returns the test config."""
  return TrainingConfig(
      model_config=policy_lib.ModelConfig(),
      training_interface_config=training_interface.TrainingInterfaceConfig(
          num_timesteps=1000,
          num_actors=2,
          batch_size=2,
      ),
      training_interface=TrainingInterfaceName.PPO,
      environment=environment.EnvironmentName.CARTPOLE,
      policy=PolicyName.MLP,
  )
