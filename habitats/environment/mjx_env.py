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

"""Custom dm_env environment for MuJoCo XLA (MJX) via Brax."""

import functools
from typing import Any

from brax import envs
from brax.envs.wrappers import dm_env as brax_dm_env
import dm_env
from dm_env import specs
from habitats.environment import brax_collisions
from habitats.environment import brax_env
from habitats.environment import termination
import numpy as np

_BRAX_ENV_MAP: dict[str, str] = {
    "quadruped": "ant",
    "ant": "ant",
    "cheetah": "halfcheetah",
    "halfcheetah": "halfcheetah",
    "hopper": "hopper",
    "humanoid": "humanoid",
    "walker2d": "walker2d",
    "inverted_pendulum": "inverted_pendulum",
    "cartpole": "inverted_pendulum",
    "inverted_double_pendulum": "inverted_double_pendulum",
}

# Brax envs whose assets only let a few geoms (mostly the feet) touch the floor.
# Brax's own tasks hide this by ending episodes once the torso gets too low, but
# evolved rewards decide termination themselves, so a fallen agent would sink
# through the floor. Other envs keep Brax's settings: e.g. the inverted pendulum
# has no floor, and its cart would collide with the rail it slides on.
_BRAX_ENVS_NEEDING_FLOOR_COLLISIONS: frozenset[str] = frozenset(
    {"ant", "halfcheetah", "hopper", "humanoid", "walker2d"}
)

_OBSERVATION_KEYS: dict[str, list[str]] = {
    "cartpole": [
        "cart_position",
        "pole_angle",
        "cart_velocity",
        "pole_angular_velocity",
    ],
    "inverted_pendulum": [
        "cart_position",
        "pole_angle",
        "cart_velocity",
        "pole_angular_velocity",
    ],
    "cheetah": [
        "rootz",
        "rooty",
        "bthigh",
        "bshin",
        "bfoot",
        "fthigh",
        "fshin",
        "ffoot",
        "rootx_vel",
        "rootz_vel",
        "rooty_vel",
        "bthigh_vel",
        "bshin_vel",
        "bfoot_vel",
        "fthigh_vel",
        "fshin_vel",
        "ffoot_vel",
    ],
    "halfcheetah": [
        "rootz",
        "rooty",
        "bthigh",
        "bshin",
        "bfoot",
        "fthigh",
        "fshin",
        "ffoot",
        "rootx_vel",
        "rootz_vel",
        "rooty_vel",
        "bthigh_vel",
        "bshin_vel",
        "bfoot_vel",
        "fthigh_vel",
        "fshin_vel",
        "ffoot_vel",
    ],
    "hopper": [
        "rootz",
        "rooty",
        "thigh_joint",
        "leg_joint",
        "foot_joint",
        "rootx_vel",
        "rootz_vel",
        "rooty_vel",
        "thigh_vel",
        "leg_vel",
        "foot_vel",
    ],
    "quadruped": [
        "torso_z",
        "torso_orient_w",
        "torso_orient_x",
        "torso_orient_y",
        "torso_orient_z",
        "hip_1",
        "ankle_1",
        "hip_2",
        "ankle_2",
        "hip_3",
        "ankle_3",
        "hip_4",
        "ankle_4",
        "torso_vx",
        "torso_vy",
        "torso_vz",
        "torso_wx",
        "torso_wy",
        "torso_wz",
        "hip_1_vel",
        "ankle_1_vel",
        "hip_2_vel",
        "ankle_2_vel",
        "hip_3_vel",
        "ankle_3_vel",
        "hip_4_vel",
        "ankle_4_vel",
    ],
    "ant": [
        "torso_z",
        "torso_orient_w",
        "torso_orient_x",
        "torso_orient_y",
        "torso_orient_z",
        "hip_1",
        "ankle_1",
        "hip_2",
        "ankle_2",
        "hip_3",
        "ankle_3",
        "hip_4",
        "ankle_4",
        "torso_vx",
        "torso_vy",
        "torso_vz",
        "torso_wx",
        "torso_wy",
        "torso_wz",
        "hip_1_vel",
        "ankle_1_vel",
        "hip_2_vel",
        "ankle_2_vel",
        "hip_3_vel",
        "ankle_3_vel",
        "hip_4_vel",
        "ankle_4_vel",
    ],
    # Brax humanoid: only the first 45 entries (qpos[2:] and qvel) are named.
    # The remaining entries (com inertia, com velocity, actuator forces) are
    # decoded as generic `obs_{i}` keys. Joint order follows
    # brax/envs/assets/humanoid.xml.
    "humanoid": [
        # Positions (qpos[2:]).
        "torso_z",
        "torso_orient_w",
        "torso_orient_x",
        "torso_orient_y",
        "torso_orient_z",
        "abdomen_z",
        "abdomen_y",
        "abdomen_x",
        "right_hip_x",
        "right_hip_z",
        "right_hip_y",
        "right_knee",
        "left_hip_x",
        "left_hip_z",
        "left_hip_y",
        "left_knee",
        "right_shoulder1",
        "right_shoulder2",
        "right_elbow",
        "left_shoulder1",
        "left_shoulder2",
        "left_elbow",
        # Velocities (qvel).
        "torso_vx",
        "torso_vy",
        "torso_vz",
        "torso_wx",
        "torso_wy",
        "torso_wz",
        "abdomen_z_vel",
        "abdomen_y_vel",
        "abdomen_x_vel",
        "right_hip_x_vel",
        "right_hip_z_vel",
        "right_hip_y_vel",
        "right_knee_vel",
        "left_hip_x_vel",
        "left_hip_z_vel",
        "left_hip_y_vel",
        "left_knee_vel",
        "right_shoulder1_vel",
        "right_shoulder2_vel",
        "right_elbow_vel",
        "left_shoulder1_vel",
        "left_shoulder2_vel",
        "left_elbow_vel",
    ],
}


# Domains whose torso is attached via a free joint, exposing `torso_z`,
# `torso_orient_*`, `torso_v*` and `torso_w*` observation keys.
_FREE_JOINT_TORSO_DOMAINS: frozenset[str] = frozenset(
    {"quadruped", "ant", "humanoid"}
)


def _requires_domain(*domains: str):
  """Decorator that restricts a property to specific MuJoCo domains."""

  def decorator(func):
    @functools.wraps(func)
    def wrapper(self):
      if self.domain_name not in domains:
        raise AttributeError(
            f"'{func.__name__}' is only available for"
            f" {'/'.join(domains)} domains, not '{self.domain_name}'."
        )
      return func(self)

    return wrapper

  return decorator


def _make_brax_env(brax_env_name: str, backend: str) -> envs.PipelineEnv:
  """Creates a Brax env, with full-body floor collisions where needed."""
  env = envs.get_environment(brax_env_name, backend=backend)
  if not isinstance(env, envs.PipelineEnv):
    raise TypeError(f"Expected PipelineEnv, got {type(env)}")
  if brax_env_name in _BRAX_ENVS_NEEDING_FLOOR_COLLISIONS:
    # PipelineEnv reads `sys` on every reset/step, so swapping it here, before
    # anything is traced, changes the physics everywhere the env is used.
    env.sys = brax_collisions.enable_floor_collisions(env.sys)
  return env


class MJXEnv(
    dm_env.Environment, brax_env.BraxEnv, termination.ReplaceableTermination
):
  """A high-performance MuJoCo environment running on JAX/MJX."""

  metadata = {"render_modes": ["rgb_array"]}

  def __init__(
      self,
      domain_name: str = "quadruped",
      task_name: str = "run",
      seed: int = 0,
      backend: str = "mjx",
  ):
    """Initializes the MJX environment.

    Args:
      domain_name: MuJoCo domain name (e.g., 'quadruped', 'ant', 'cheetah',
        'hopper').
      task_name: Task name (kept for compatibility with dm_control API).
      seed: Random seed for JAX PRNG.
      backend: Physics backend ('mjx' for GPU/TPU-accelerated MuJoCo,
        'generalized' for pure JAX generalized coordinates).
    """
    del task_name
    self.domain_name = domain_name.lower()
    self._backend = backend

    brax_env_name = _BRAX_ENV_MAP.get(self.domain_name, self.domain_name)
    raw_env = _make_brax_env(brax_env_name, backend)

    self._raw_env = raw_env
    # Wrap in brax's standard dm_env adapter
    self._env = brax_dm_env.DmEnvWrapper(
        env=raw_env,
        seed=seed,
        backend=None,
    )
    self._last_obs: np.ndarray | None = None
    self._last_reward: float = 0.0

  def reset(self) -> dm_env.TimeStep:
    """Resets the environment."""
    timestep = self._env.reset()
    obs_np = np.asarray(timestep.observation, dtype=np.float32)
    self._last_obs = obs_np
    self._last_reward = 0.0
    return dm_env.TimeStep(
        step_type=timestep.step_type,
        reward=timestep.reward,
        discount=timestep.discount,
        observation=obs_np,
    )

  def step(self, action: Any) -> dm_env.TimeStep:
    """Steps the environment dynamics using JIT-compiled MJX."""
    action_arr = np.asarray(action, dtype=np.float32)
    timestep = self._env.step(action_arr)
    obs_np = np.asarray(timestep.observation, dtype=np.float32)
    reward_val = float(timestep.reward) if timestep.reward is not None else 0.0
    self._last_obs = obs_np
    self._last_reward = reward_val
    return dm_env.TimeStep(
        step_type=timestep.step_type,
        reward=reward_val,
        discount=timestep.discount,
        observation=obs_np,
    )

  def observation_spec(self) -> specs.BoundedArray:
    """Returns the continuous observation spec (float32)."""
    orig_spec = self._env.observation_spec()
    return specs.BoundedArray(
        shape=orig_spec.shape,
        dtype=np.float32,
        minimum=np.asarray(orig_spec.minimum, dtype=np.float32),
        maximum=np.asarray(orig_spec.maximum, dtype=np.float32),
        name="observation",
    )

  def action_spec(self) -> specs.BoundedArray:
    """Returns the continuous action spec (float32)."""
    orig_spec = self._env.action_spec()
    return specs.BoundedArray(
        shape=orig_spec.shape,
        dtype=np.float32,
        minimum=np.asarray(orig_spec.minimum, dtype=np.float32),
        maximum=np.asarray(orig_spec.maximum, dtype=np.float32),
        name="action",
    )

  def render(
      self, height: int = 240, width: int = 320, camera_id: int = 0
  ) -> np.ndarray:
    """Renders the current physics state."""
    del camera_id
    try:
      img = self._env.render()
      if img is not None:
        return np.asarray(img)
    except Exception:  # pylint: disable=broad-exception-caught
      pass
    return np.zeros((height, width, 3), dtype=np.uint8)

  def close(self) -> None:
    """Closes the environment."""
    pass

  def get_env(self) -> Any:
    """Returns the underlying Brax/MJX environment."""
    return self._env

  @property
  def brax_env(self) -> envs.PipelineEnv:
    """Returns the underlying Brax PipelineEnv."""
    return self._raw_env

  def to_brax_env(self) -> envs.PipelineEnv:
    """Returns the underlying Brax PipelineEnv for on-device JAX execution."""
    return self._raw_env

  def decode(self, state: np.ndarray | None = None) -> dict[str, float]:
    """Decodes the observation vector into named physics components.

    For environments with known observation structure (cartpole, cheetah,
    hopper), returns a dict with named keys. For other environments, returns
    generic indexed keys.

    Args:
      state: The observation array to decode. If None, uses the last observation
        from a step/reset call.

    Returns:
      A dictionary mapping observation component names to their float values.
    """
    if state is not None and isinstance(state, np.ndarray):
      obs = state
    elif self._last_obs is not None:
      obs = self._last_obs
    else:
      return {}

    keys = _OBSERVATION_KEYS.get(self.domain_name)
    obs_flat = obs.flatten()
    if keys is None:
      return {f"obs_{i}": float(v) for i, v in enumerate(obs_flat)}
    decoded = {k: float(obs_flat[i]) for i, k in enumerate(keys)}
    # Observations may contain more entries than named keys (e.g. humanoid
    # com inertia/velocity and actuator forces). Expose them generically.
    for i in range(len(keys), obs_flat.shape[0]):
      decoded[f"obs_{i}"] = float(obs_flat[i])
    return decoded

  def _get_obs_value(self, key: str) -> float:
    """Returns the value of a named observation component.

    Args:
      key: The observation key name (must exist in _OBSERVATION_KEYS for this
        domain).

    Returns:
      The float value of the observation component.

    Raises:
      RuntimeError: If no observation is available (env not yet stepped/reset).
      KeyError: If the key is not found in the observation keys.
    """
    if self._last_obs is None:
      raise RuntimeError("No observation available. Call reset() first.")
    keys = _OBSERVATION_KEYS.get(self.domain_name)
    if keys is None or key not in keys:
      raise KeyError(
          f"Observation key '{key}' not found for domain '{self.domain_name}'."
      )
    idx = keys.index(key)
    return float(self._last_obs.flatten()[idx])

  # --- Cartpole convenience properties ---

  @property
  @_requires_domain("cartpole", "inverted_pendulum")
  def cart_position(self) -> float:
    """Returns the horizontal position of the cart. Cartpole only."""
    return self._get_obs_value("cart_position")

  @property
  @_requires_domain("cartpole", "inverted_pendulum")
  def pole_angle(self) -> float:
    """Returns the pole angle in radians. 0.0 = upright. Cartpole only."""
    return self._get_obs_value("pole_angle")

  @property
  @_requires_domain("cartpole", "inverted_pendulum")
  def cart_velocity(self) -> float:
    """Returns the horizontal velocity of the cart. Cartpole only."""
    return self._get_obs_value("cart_velocity")

  @property
  @_requires_domain("cartpole", "inverted_pendulum")
  def pole_angular_velocity(self) -> float:
    """Returns the angular velocity of the pole. Cartpole only."""
    return self._get_obs_value("pole_angular_velocity")

  # --- Cheetah convenience properties ---

  @property
  @_requires_domain(
      "cheetah", "halfcheetah", "hopper", "quadruped", "ant", "humanoid"
  )
  def speed(self) -> float:
    """Returns the horizontal (forward) speed.

    Cheetah/Hopper/Quadruped/Humanoid.
    """
    if self.domain_name in _FREE_JOINT_TORSO_DOMAINS:
      return self._get_obs_value("torso_vx")
    return self._get_obs_value("rootx_vel")

  @property
  @_requires_domain("cheetah", "halfcheetah")
  def torso_height(self) -> float:
    """Returns the vertical position (height) of the torso. Cheetah only."""
    return self._get_obs_value("rootz")

  @property
  @_requires_domain("cheetah", "halfcheetah")
  def torso_angle(self) -> float:
    """Returns the orientation angle of the torso. Cheetah only."""
    return self._get_obs_value("rooty")

  # --- Hopper convenience properties ---

  @property
  @_requires_domain("hopper", "quadruped", "ant", "humanoid")
  def height(self) -> float:
    """Returns the height (z-coordinate) of the torso.

    Hopper/Quadruped/Humanoid.
    """
    if self.domain_name in _FREE_JOINT_TORSO_DOMAINS:
      return self._get_obs_value("torso_z")
    return self._get_obs_value("rootz")

  # --- Quadruped/Ant/Humanoid convenience properties ---

  @property
  @_requires_domain("quadruped", "ant", "humanoid")
  def lateral_speed(self) -> float:
    """Returns the lateral (y-axis) speed. Quadruped/Ant/Humanoid only."""
    return self._get_obs_value("torso_vy")

  @property
  @_requires_domain("quadruped", "ant", "humanoid")
  def torso_orientation(self) -> tuple[float, float, float, float]:
    """Returns the torso orientation quaternion (w, x, y, z).

    Quadruped/Humanoid.
    """
    return (
        self._get_obs_value("torso_orient_w"),
        self._get_obs_value("torso_orient_x"),
        self._get_obs_value("torso_orient_y"),
        self._get_obs_value("torso_orient_z"),
    )
