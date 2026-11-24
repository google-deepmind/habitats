Environment: MuJoCo Hopper Playground (Continuous, MJX/Brax backend)
Observation Space: jax array of shape (11,) representing the flattened physics state.
  Positions (indices 0-4):
    Index 0: rootz (float) — absolute vertical height of torso above ground in meters (neutral standing height is ~1.25m; torso touching floor is < 0.6m)
    Index 1: rooty (float) — torso pitch angle in radians (0.0 = upright vertical torso; POSITIVE = forward tilt / leaning forward; NEGATIVE = backward tilt / leaning backward)
    Index 2: thigh_joint (float) — thigh joint angle (range: [-2.62, 0.0] rad, strictly non-positive)
    Index 3: leg_joint (float) — leg joint angle (range: [-2.62, 0.0] rad, strictly non-positive)
    Index 4: foot_joint (float) — foot joint angle (range: [-0.785, 0.785] rad)
  Velocities (indices 5-10):
    Index 5: rootx_vel (float) — horizontal velocity (+ is forward, - is backward, typical hopping speeds 1.0 to 4.0 m/s)
    Index 6: rootz_vel (float) — vertical velocity (+ is jumping up, - is falling down)
    Index 7: rooty_vel (float) — torso pitch angular velocity (+ is forward angular velocity)
    Index 8: thigh_vel (float) — thigh joint velocity
    Index 9: leg_vel (float) — leg joint velocity
    Index 10: foot_vel (float) — foot joint velocity
  Note: horizontal position (rootx) is excluded from the observation to maintain translational invariance.

Action Space: Continuous(3) — jax array of shape (3,) in [-1.0, 1.0] representing torques applied to 3 joints:
  Index 0: thigh_joint torque
  Index 1: leg_joint torque
  Index 2: foot_joint torque

Function signature:
  def compute_reward(env, state, action, next_state, done, info, base_reward):
  - state, action, next_state: jax arrays (index directly, e.g., next_state[5] for forward speed)
  - done: jax scalar
  - base_reward: jax scalar
  - Return: MUST return a tuple `(reward, done)` of jnp scalars:
      - `reward`: jnp scalar representing the step reward.
      - `done`: jnp scalar (1.0 to terminate the episode early, 0.0 to continue). The environment ONLY terminates when your function returns done = 1.0 (or at the 1000-step time limit).
  - `jnp` and `jax` are already available. Use `jnp` for all math operations.

Playground dynamics and tips:
  - This is a PLAYGROUND. There is no predetermined goal.
  - The wrapper ONLY terminates when your reward function returns done = 1.0. The base environment's default termination conditions are bypassed.
  - You have full freedom to define when the episode should terminate.
Goal: Evolve a SIMPLE, interesting objective for the agent to explore in this playground (e.g., standing upright and balancing, hopping forward, hopping as high as possible, minimizing energy usage while standing, or achieving a target speed).
