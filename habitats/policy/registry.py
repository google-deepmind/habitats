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

"""Policy registry."""

from collections.abc import Callable

from habitats.lib import class_registry
from habitats.policy import neural
from habitats.policy import policy
from habitats.policy import tabular

PolicyName = policy.PolicyName

# Policies, built from a `ModelConfig`.

POLICY_REGISTRY: dict[
    PolicyName, Callable[[policy.ModelConfig], policy.Policy]
] = {}
register_policy, get_policy = class_registry.declare_registry(POLICY_REGISTRY)

register_policy(PolicyName.TABULAR)(tabular.TabularPolicy)
register_policy(PolicyName.MLP)(neural.MLPPolicy)

# Meta-policies, built from a `MetaModelConfig`.

META_POLICY_REGISTRY: dict[
    PolicyName, Callable[[policy.MetaModelConfig], policy.MetaPolicy]
] = {}
register_meta_policy, get_meta_policy = class_registry.declare_registry(
    META_POLICY_REGISTRY
)

register_meta_policy(PolicyName.TABULAR_META)(tabular.TabularMetaPolicy)


def is_meta_policy(name: PolicyName) -> bool:
  """Returns whether `name` is registered as a meta-policy."""
  return name in META_POLICY_REGISTRY


def create(
    name: PolicyName,
    config: policy.ModelConfig | policy.MetaModelConfig,
) -> policy.Policy:
  """Creates the policy registered under `name` from `config`."""
  if is_meta_policy(name):
    if not isinstance(config, policy.MetaModelConfig):
      raise TypeError(
          f"Meta-policy {name!r} requires a MetaModelConfig, got"
          f" {type(config).__name__}."
      )
    return get_meta_policy(name)(config)

  if not isinstance(config, policy.ModelConfig):
    raise TypeError(
        f"Policy {name!r} requires a ModelConfig, got {type(config).__name__}."
    )
  return get_policy(name)(config)
