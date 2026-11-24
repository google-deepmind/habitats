Environment: MuJoCo Ant/Quadruped Playground (Continuous, MJX/Brax backend).

Observation Space: jax array of shape (27,) representing the flattened physics state.
  Positions (indices 0-12):
    Index 0: torso_z (float) — z-coordinate (height) of the torso
    Index 1: torso_orient_w (float) — w-component of torso orientation quaternion
    Index 2: torso_orient_x (float) — x-component of torso orientation quaternion
    Index 3: torso_orient_y (float) — y-component of torso orientation quaternion
    Index 4: torso_orient_z (float) — z-component of torso orientation quaternion
    Index 5: hip_1 (float) — front left hip joint angle
    Index 6: ankle_1 (float) — front left ankle joint angle
    Index 7: hip_2 (float) — front right hip joint angle
    Index 8: ankle_2 (float) — front right ankle joint angle
    Index 9: hip_3 (float) — back left hip joint angle
    Index 10: ankle_3 (float) — back left ankle joint angle
    Index 11: hip_4 (float) — back right hip joint angle
    Index 12: ankle_4 (float) — back right ankle joint angle
  Velocities (indices 13-26):
    Index 13: torso_vx (float) — x-velocity (≈ forward speed)
    Index 14: torso_vy (float) — y-velocity (≈ lateral speed)
    Index 15: torso_vz (float) — z-velocity (vertical)
    Index 16: torso_wx (float) — x-angular velocity of the torso
    Index 17: torso_wy (float) — y-angular velocity of the torso
    Index 18: torso_wz (float) — z-angular velocity of the torso
    Index 19: hip_1_vel (float) — front left hip joint velocity
    Index 20: ankle_1_vel (float) — front left ankle joint velocity
    Index 21: hip_2_vel (float) — front right hip joint velocity
    Index 22: ankle_2_vel (float) — front right ankle joint velocity
    Index 23: hip_3_vel (float) — back left hip joint velocity
    Index 24: ankle_3_vel (float) — back left ankle joint velocity
    Index 25: hip_4_vel (float) — back right hip joint velocity
    Index 26: ankle_4_vel (float) — back right ankle joint velocity
  Note: x,y position coordinates are excluded from the observation to maintain translational invariance.

Action Space: Continuous(8) — jax array of shape (8,) in [-1.0, 1.0] representing torques applied to 8 joints:
  Index 0: hip_1 (front left hip torque)
  Index 1: ankle_1 (front left ankle torque)
  Index 2: hip_2 (front right hip torque)
  Index 3: ankle_2 (front right ankle torque)
  Index 4: hip_3 (back left hip torque)
  Index 5: ankle_3 (back left ankle torque)
  Index 6: hip_4 (back right hip torque)
  Index 7: ankle_4 (back right ankle torque)

Function signature:
  def compute_reward(env, state, action, next_state, done, info, base_reward):
  - state, action, next_state: jax arrays (index directly, e.g., next_state[13] for forward speed)
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
  - The torso is a sphere of radius 0.25 that collides with the floor, so `torso_z` (index 0) never drops below about 0.25, even when the agent lies flat or upside down. Termination conditions such as `torso_z < 0.15` never trigger.

Goal: Evolve a SIMPLE, interesting objective for the agent to explore in this playground (e.g., walking forward, walking in a circle, standing tall, spinning, minimizing energy usage, or navigating using specific legs only).
