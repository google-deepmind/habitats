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

"""Tests for training interface registry."""

from absl.testing import absltest
from habitats.training import ppo
from habitats.training import qlearning
from habitats.training import registry
from habitats.training import training_interface


class TrainingInterfaceRegistryTest(absltest.TestCase):

  def test_all_training_interface_names_are_registered(self):
    for name in training_interface.TrainingInterfaceName:
      self.assertIn(name, registry.TRAINING_INTERFACE_REGISTRY)
      self.assertTrue(
          issubclass(
              registry.get(name),
              training_interface.TrainingInterface,
          )
      )

  def test_registered_training_interface_classes(self):
    self.assertIs(
        registry.get(training_interface.TrainingInterfaceName.Q_LEARNING),
        qlearning.QLearningInterface,
    )
    self.assertIs(
        registry.get(training_interface.TrainingInterfaceName.PPO),
        ppo.PPOInterface,
    )


if __name__ == "__main__":
  absltest.main()
