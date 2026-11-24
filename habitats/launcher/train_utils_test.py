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

"""Tests for training utilities."""

import dataclasses
import os

from absl.testing import absltest
from habitats.config import environment
from habitats.config import training
from habitats.launcher import train_utils
from habitats.policy import policy as policy_lib
from habitats.policy import tabular


class TrainUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = self.create_tempdir().full_path

  def test_run_training_trains_standard_policy(self):
    base_config = training.get_config("q_learning_ascending_sort_game")
    config = dataclasses.replace(
        base_config,
        training_interface_config=dataclasses.replace(
            base_config.training_interface_config,
            num_episodes=2,
            max_steps=5,
            num_eval_episodes=1,
            log_every=10,
            checkpoint_path=self.temp_dir,
        ),
    )

    output = train_utils.run_training(config)

    self.assertLen(output.episode_rewards, 2)

  def test_run_training_trains_meta_policy_with_sub_policy_paths(self):
    base_config = training.get_config("q_learning_meta_mini_card_deck")
    env = environment.get(base_config.environment)(0, False)
    sub_model_config = policy_lib.ModelConfig().apply_environment_spec(
        observations=env.observation_spec(),
        actions=env.action_spec(),
    )
    env.close()

    sub_paths = []
    for idx in range(2):
      sub_dir = os.path.join(self.temp_dir, f"sub_{idx}")
      sub_policy = tabular.TabularPolicy(sub_model_config)
      sub_policy.save(sub_dir)
      sub_paths.append(sub_dir)

    assert isinstance(base_config.model_config, policy_lib.MetaModelConfig)
    config = dataclasses.replace(
        base_config,
        model_config=base_config.model_config.with_sub_policy_paths(sub_paths),
        training_interface_config=dataclasses.replace(
            base_config.training_interface_config,
            num_episodes=2,
            max_steps=5,
            num_eval_episodes=1,
            log_every=10,
            checkpoint_path=os.path.join(self.temp_dir, "meta_out"),
        ),
    )

    output = train_utils.run_training(config)

    self.assertLen(output.episode_rewards, 2)

  def test_all_card_deck_game_configs_are_registered_with_5k_episodes(self):
    card_deck_configs = (
        "q_learning_ascending_sort_game",
        "q_learning_descending_sort_game",
        "q_learning_triplet_matching_game",
        "q_learning_target_sum_game",
        "q_learning_high_card_maximizer_game",
        "q_learning_low_card_minimalist_game",
        "q_learning_even_odd_pattern_game",
        "q_learning_sandwich_pair_game",
        "q_learning_rapid_clear_discard_game",
        "q_learning_full_house_triple_pair_game",
        "q_learning_meta_mini_card_deck",
        "q_learning_meta_ascending_sort_game",
        "q_learning_meta_descending_sort_game",
        "q_learning_meta_triplet_matching_game",
        "q_learning_meta_target_sum_game",
        "q_learning_meta_high_card_maximizer_game",
        "q_learning_meta_low_card_minimalist_game",
        "q_learning_meta_even_odd_pattern_game",
        "q_learning_meta_sandwich_pair_game",
        "q_learning_meta_rapid_clear_discard_game",
        "q_learning_meta_full_house_triple_pair_game",
    )
    for config_name in card_deck_configs:
      cfg = training.get_config(config_name)
      self.assertEqual(cfg.training_interface_config.num_episodes, 5000)
      self.assertEqual(cfg.training_interface_config.log_every, 10)
      self.assertEqual(cfg.training_interface_config.exploration_fraction, 0.8)


if __name__ == "__main__":
  absltest.main()
