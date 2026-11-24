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

"""Unit tests for MJXEnv."""

from unittest import mock

from absl.testing import absltest
import dm_env
from dm_env import specs
from habitats.environment import mjx_env
from habitats.environment import termination
import numpy as np

# Map domain names to (obs_shape, action_shape).
_DOMAIN_SHAPES = {
    "cartpole": ((4,), (1,)),
    "cheetah": ((17,), (6,)),
    "hopper": ((11,), (3,)),
    "quadruped": ((27,), (8,)),
    "humanoid": ((244,), (17,)),
}


def _make_mock_dm_wrapper(obs_shape, action_shape):
  """Creates a mock dm_env wrapper that returns arrays of the given shape."""
  reset_obs = np.arange(obs_shape[0], dtype=np.float32) * 0.01
  step_obs = np.arange(obs_shape[0], dtype=np.float32) * 0.02 + 0.1

  wrapper = mock.MagicMock()
  wrapper.reset.return_value = dm_env.TimeStep(
      step_type=dm_env.StepType.FIRST,
      reward=None,
      discount=None,
      observation=reset_obs,
  )
  wrapper.step.return_value = dm_env.TimeStep(
      step_type=dm_env.StepType.MID,
      reward=0.5,
      discount=1.0,
      observation=step_obs,
  )
  wrapper.observation_spec.return_value = specs.BoundedArray(
      shape=obs_shape,
      dtype=np.float32,
      minimum=np.full(obs_shape, -np.inf, dtype=np.float32),
      maximum=np.full(obs_shape, np.inf, dtype=np.float32),
      name="observation",
  )
  wrapper.action_spec.return_value = specs.BoundedArray(
      shape=action_shape,
      dtype=np.float32,
      minimum=np.full(action_shape, -1.0, dtype=np.float32),
      maximum=np.full(action_shape, 1.0, dtype=np.float32),
      name="action",
  )
  wrapper.render.return_value = np.zeros((240, 320, 3), dtype=np.uint8)
  return wrapper


def _create_mjx_env(domain_name):
  """Creates an MJXEnv with mocked Brax internals for fast testing."""
  obs_shape, action_shape = _DOMAIN_SHAPES[domain_name]
  mock_dm_wrapper = _make_mock_dm_wrapper(obs_shape, action_shape)

  with (
      mock.patch.object(mjx_env, "envs") as mock_envs,
      mock.patch.object(mjx_env, "brax_dm_env") as mock_brax_dm,
      mock.patch.object(mjx_env, "brax_collisions"),
  ):
    # Make isinstance(raw_env, envs.PipelineEnv) pass.
    mock_envs.PipelineEnv = mock.MagicMock
    mock_envs.get_environment.return_value = mock.MagicMock()
    mock_brax_dm.DmEnvWrapper.return_value = mock_dm_wrapper

    env = mjx_env.MJXEnv(domain_name=domain_name)

  return env


class MJXEnvCartpoleTest(absltest.TestCase):
  """Tests for MJXEnv with cartpole domain."""

  def setUp(self):
    super().setUp()
    self.env = _create_mjx_env("cartpole")

  def test_init(self):
    """Tests initialization with continuous specs."""
    self.assertIsInstance(self.env.action_spec(), specs.BoundedArray)
    self.assertIsInstance(self.env.observation_spec(), specs.BoundedArray)
    self.assertEqual(self.env.action_spec().dtype, np.float32)
    self.assertEqual(self.env.observation_spec().dtype, np.float32)

  def test_evolved_reward_replaces_termination(self):
    """Tests only the evolved reward ends episodes of MuJoCo tasks."""
    self.assertTrue(termination.ignores_base_termination(self.env))

  def test_step_continuous(self):
    """Tests stepping returns valid timestep."""
    timestep = self.env.reset()
    self.assertEqual(timestep.step_type, dm_env.StepType.FIRST)
    self.assertIsInstance(timestep.observation, np.ndarray)
    self.assertEqual(timestep.observation.dtype, np.float32)

    action = np.zeros(self.env.action_spec().shape, dtype=np.float32)
    timestep = self.env.step(action)
    self.assertIsInstance(timestep, dm_env.TimeStep)
    self.assertIsInstance(timestep.observation, np.ndarray)
    self.assertIsInstance(timestep.reward, float)

  def test_decode_returns_named_dict(self):
    """Tests decode() returns named dict for cartpole."""
    timestep = self.env.reset()
    decoded = self.env.decode(timestep.observation)

    self.assertIsInstance(decoded, dict)
    expected_keys = {
        "cart_position",
        "pole_angle",
        "cart_velocity",
        "pole_angular_velocity",
    }
    self.assertEqual(set(decoded.keys()), expected_keys)
    for key, value in decoded.items():
      self.assertIsInstance(value, float, msg=f"Key '{key}' is not float")

  def test_decode_before_reset_returns_empty(self):
    """Tests decode() returns empty dict before any reset/step."""
    decoded = self.env.decode()
    self.assertEqual(decoded, {})

  def test_decode_after_step_reflects_new_state(self):
    """Tests decode() returns updated values after a step."""
    self.env.reset()
    decoded_before = self.env.decode()

    action = np.ones(self.env.action_spec().shape, dtype=np.float32)
    self.env.step(action)
    decoded_after = self.env.decode()

    self.assertIsInstance(decoded_after, dict)
    self.assertLen(decoded_after, 4)
    values_changed = any(
        decoded_before[k] != decoded_after[k] for k in decoded_before
    )
    self.assertTrue(values_changed, "Observation should change after a step")

  def test_decode_with_explicit_state(self):
    """Tests decode() with an explicitly passed state array."""
    self.env.reset()
    fake_state = np.array([1.0, 0.5, -0.3, 0.1], dtype=np.float32)
    decoded = self.env.decode(fake_state)

    self.assertEqual(decoded["cart_position"], 1.0)
    self.assertAlmostEqual(decoded["pole_angle"], 0.5)
    self.assertAlmostEqual(decoded["cart_velocity"], -0.3)
    self.assertAlmostEqual(decoded["pole_angular_velocity"], 0.1)

  def test_get_env_returns_wrapper(self):
    """Tests get_env() returns the underlying environment."""
    self.assertIsNotNone(self.env.get_env())

  def test_brax_env_property(self):
    """Tests brax_env property returns a value."""
    self.assertIsNotNone(self.env.brax_env)

  def test_to_brax_env(self):
    """Tests to_brax_env() returns a value."""
    self.assertIsNotNone(self.env.to_brax_env())

  def test_properties_return_floats(self):
    """Tests all cartpole properties return floats after reset."""
    self.env.reset()
    self.assertIsInstance(self.env.cart_position, float)
    self.assertIsInstance(self.env.pole_angle, float)
    self.assertIsInstance(self.env.cart_velocity, float)
    self.assertIsInstance(self.env.pole_angular_velocity, float)

  def test_properties_consistent_with_decode(self):
    """Tests that properties and decode() return consistent values."""
    self.env.reset()
    decoded = self.env.decode()
    self.assertAlmostEqual(
        self.env.cart_position, decoded["cart_position"], places=5
    )
    self.assertAlmostEqual(self.env.pole_angle, decoded["pole_angle"], places=5)
    self.assertAlmostEqual(
        self.env.cart_velocity, decoded["cart_velocity"], places=5
    )
    self.assertAlmostEqual(
        self.env.pole_angular_velocity,
        decoded["pole_angular_velocity"],
        places=5,
    )

  def test_cheetah_properties_raise(self):
    """Tests that cheetah properties raise AttributeError on cartpole."""
    self.env.reset()
    with self.assertRaises(AttributeError):
      _ = self.env.speed
    with self.assertRaises(AttributeError):
      _ = self.env.torso_height
    with self.assertRaises(AttributeError):
      _ = self.env.torso_angle

  def test_hopper_properties_raise(self):
    """Tests that hopper-only properties raise AttributeError on cartpole."""
    self.env.reset()
    with self.assertRaises(AttributeError):
      _ = self.env.height

  def test_domain_error_message_is_descriptive(self):
    """Tests that domain error names actual and expected domains."""
    self.env.reset()
    with self.assertRaises(AttributeError) as ctx:
      _ = self.env.height
    self.assertIn("hopper", str(ctx.exception))
    self.assertIn("cartpole", str(ctx.exception))

  def test_properties_raise_before_reset(self):
    """Tests that convenience properties raise before reset is called."""
    with self.assertRaises(RuntimeError):
      _ = self.env.cart_position

  def test_render_returns_array(self):
    """Tests that render returns a numpy array."""
    self.env.reset()
    frame = self.env.render()
    self.assertIsInstance(frame, np.ndarray)

  def test_close_does_not_raise(self):
    """Tests that close() can be called without errors."""
    self.env.close()


class MJXEnvCheetahTest(absltest.TestCase):
  """Tests for MJXEnv with cheetah domain."""

  def setUp(self):
    super().setUp()
    self.env = _create_mjx_env("cheetah")

  def test_decode_returns_named_dict(self):
    """Tests decode() returns named dict for cheetah."""
    timestep = self.env.reset()
    decoded = self.env.decode(timestep.observation)

    self.assertIsInstance(decoded, dict)
    self.assertLen(decoded, 17)
    self.assertIn("rootz", decoded)
    self.assertIn("rootx_vel", decoded)
    self.assertIn("ffoot_vel", decoded)

  def test_properties_return_floats(self):
    """Tests all cheetah properties return floats after reset."""
    self.env.reset()
    self.assertIsInstance(self.env.speed, float)
    self.assertIsInstance(self.env.torso_height, float)
    self.assertIsInstance(self.env.torso_angle, float)

  def test_cartpole_properties_raise(self):
    """Tests that cartpole properties raise AttributeError on cheetah."""
    self.env.reset()
    with self.assertRaises(AttributeError):
      _ = self.env.cart_position
    with self.assertRaises(AttributeError):
      _ = self.env.pole_angle
    with self.assertRaises(AttributeError):
      _ = self.env.cart_velocity
    with self.assertRaises(AttributeError):
      _ = self.env.pole_angular_velocity


class MJXEnvHopperTest(absltest.TestCase):
  """Tests for MJXEnv with hopper domain."""

  def setUp(self):
    super().setUp()
    self.env = _create_mjx_env("hopper")

  def test_decode_returns_named_dict(self):
    """Tests decode() returns named dict with 11 keys for hopper."""
    timestep = self.env.reset()
    decoded = self.env.decode(timestep.observation)

    self.assertIsInstance(decoded, dict)
    self.assertLen(decoded, 11)
    self.assertIn("rootz", decoded)
    self.assertIn("thigh_joint", decoded)
    self.assertIn("leg_joint", decoded)
    self.assertIn("foot_joint", decoded)
    self.assertIn("rootx_vel", decoded)

  def test_properties_return_floats(self):
    """Tests hopper properties return correct types after reset."""
    self.env.reset()
    self.assertIsInstance(self.env.height, float)
    self.assertIsInstance(self.env.speed, float)

  def test_cartpole_properties_raise(self):
    """Tests that cartpole properties raise AttributeError on hopper."""
    self.env.reset()
    with self.assertRaises(AttributeError):
      _ = self.env.cart_position
    with self.assertRaises(AttributeError):
      _ = self.env.pole_angle

  def test_cheetah_only_properties_raise(self):
    """Tests that cheetah-only properties raise AttributeError on hopper."""
    self.env.reset()
    with self.assertRaises(AttributeError):
      _ = self.env.torso_height
    with self.assertRaises(AttributeError):
      _ = self.env.torso_angle

  def test_speed_shared_with_cheetah(self):
    """Tests that speed property works on hopper (shared with cheetah)."""
    self.env.reset()
    self.assertIsInstance(self.env.speed, float)


class MJXEnvQuadrupedTest(absltest.TestCase):
  """Tests for MJXEnv with quadruped/ant domain."""

  def setUp(self):
    super().setUp()
    self.env = _create_mjx_env("quadruped")

  def test_decode_returns_named_dict(self):
    """Tests decode() returns named dict with 27 keys for quadruped."""
    timestep = self.env.reset()
    decoded = self.env.decode(timestep.observation)

    self.assertIsInstance(decoded, dict)
    self.assertLen(decoded, 27)
    self.assertIn("torso_z", decoded)
    self.assertIn("torso_orient_w", decoded)
    self.assertIn("hip_1", decoded)
    self.assertIn("ankle_4", decoded)
    self.assertIn("torso_vx", decoded)
    self.assertIn("torso_vy", decoded)
    self.assertIn("hip_4_vel", decoded)
    self.assertIn("ankle_4_vel", decoded)

  def test_properties_return_correct_types(self):
    """Tests quadruped properties return correct types after reset."""
    self.env.reset()
    self.assertIsInstance(self.env.speed, float)
    self.assertIsInstance(self.env.height, float)
    self.assertIsInstance(self.env.lateral_speed, float)
    self.assertIsInstance(self.env.torso_orientation, tuple)
    self.assertLen(self.env.torso_orientation, 4)
    for v in self.env.torso_orientation:
      self.assertIsInstance(v, float)

  def test_properties_consistent_with_decode(self):
    """Tests that properties and decode() return consistent values."""
    self.env.reset()
    decoded = self.env.decode()
    self.assertAlmostEqual(self.env.speed, decoded["torso_vx"], places=5)
    self.assertAlmostEqual(self.env.height, decoded["torso_z"], places=5)
    self.assertAlmostEqual(
        self.env.lateral_speed, decoded["torso_vy"], places=5
    )

  def test_speed_shared_with_cheetah_and_hopper(self):
    """Tests that speed property works on quadruped (shared)."""
    self.env.reset()
    self.assertIsInstance(self.env.speed, float)

  def test_cartpole_properties_raise(self):
    """Tests that cartpole properties raise AttributeError on quadruped."""
    self.env.reset()
    with self.assertRaises(AttributeError):
      _ = self.env.cart_position
    with self.assertRaises(AttributeError):
      _ = self.env.pole_angle

  def test_cheetah_only_properties_raise(self):
    """Tests that cheetah-only properties raise AttributeError on quadruped."""
    self.env.reset()
    with self.assertRaises(AttributeError):
      _ = self.env.torso_height
    with self.assertRaises(AttributeError):
      _ = self.env.torso_angle


class MJXEnvHumanoidTest(absltest.TestCase):
  """Tests for MJXEnv with humanoid domain."""

  def setUp(self):
    super().setUp()
    self.env = _create_mjx_env("humanoid")

  def test_decode_names_first_45_and_generic_rest(self):
    """Tests decode() names qpos/qvel entries and keeps trailing entries."""
    timestep = self.env.reset()
    decoded = self.env.decode(timestep.observation)

    self.assertLen(decoded, 244)
    for key in (
        "torso_z",
        "torso_orient_w",
        "abdomen_z",
        "right_knee",
        "left_elbow",
        "torso_vx",
        "torso_vy",
        "left_elbow_vel",
        "obs_45",
        "obs_243",
    ):
      self.assertIn(key, decoded)
    obs = timestep.observation
    self.assertAlmostEqual(decoded["torso_z"], obs[0], places=5)
    self.assertAlmostEqual(decoded["left_elbow"], obs[21], places=5)
    self.assertAlmostEqual(decoded["torso_vx"], obs[22], places=5)
    self.assertAlmostEqual(decoded["left_elbow_vel"], obs[44], places=5)
    self.assertAlmostEqual(decoded["obs_45"], obs[45], places=5)

  def test_properties_consistent_with_decode(self):
    """Tests humanoid properties match decode()."""
    self.env.reset()
    decoded = self.env.decode()
    self.assertAlmostEqual(self.env.speed, decoded["torso_vx"], places=5)
    self.assertAlmostEqual(self.env.height, decoded["torso_z"], places=5)
    self.assertAlmostEqual(
        self.env.lateral_speed, decoded["torso_vy"], places=5
    )
    self.assertLen(self.env.torso_orientation, 4)

  def test_other_domain_properties_raise(self):
    """Tests that cartpole/cheetah-only properties raise on humanoid."""
    self.env.reset()
    with self.assertRaises(AttributeError):
      _ = self.env.cart_position
    with self.assertRaises(AttributeError):
      _ = self.env.torso_height


if __name__ == "__main__":
  absltest.main()
