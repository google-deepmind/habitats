Environment: MuJoCo Humanoid Playground (Continuous, MJX/Brax backend)
Observation Space: jax array of shape (244,) representing the flattened physics state. Only indices 0-44 are directly interpretable; prefer them when writing rewards.
  Positions (indices 0-21):
    Index 0: torso_z (float) — z-coordinate (height) of the torso. Starts at ≈1.4 when standing upright; ≈1.0 or lower usually means the humanoid has fallen.
    Index 1: torso_orient_w (float) — w-component of torso orientation quaternion (≈1.0 when upright and facing +x)
    Index 2: torso_orient_x (float) — x-component of torso orientation quaternion
    Index 3: torso_orient_y (float) — y-component of torso orientation quaternion
    Index 4: torso_orient_z (float) — z-component of torso orientation quaternion
    Index 5: abdomen_z (float) — abdomen twist angle around the vertical axis
    Index 6: abdomen_y (float) — abdomen forward/backward bend angle
    Index 7: abdomen_x (float) — pelvis side-to-side tilt angle
    Index 8: right_hip_x (float) — right hip abduction/adduction angle
    Index 9: right_hip_z (float) — right hip rotation angle
    Index 10: right_hip_y (float) — right hip flexion/extension angle (swings leg forward/backward)
    Index 11: right_knee (float) — right knee angle
    Index 12: left_hip_x (float) — left hip abduction/adduction angle
    Index 13: left_hip_z (float) — left hip rotation angle
    Index 14: left_hip_y (float) — left hip flexion/extension angle (swings leg forward/backward)
    Index 15: left_knee (float) — left knee angle
    Index 16: right_shoulder1 (float) — right shoulder angle (axis 1)
    Index 17: right_shoulder2 (float) — right shoulder angle (axis 2)
    Index 18: right_elbow (float) — right elbow angle
    Index 19: left_shoulder1 (float) — left shoulder angle (axis 1)
    Index 20: left_shoulder2 (float) — left shoulder angle (axis 2)
    Index 21: left_elbow (float) — left elbow angle
  Velocities (indices 22-44):
    Index 22: torso_vx (float) — x-velocity (≈ forward speed)
    Index 23: torso_vy (float) — y-velocity (≈ lateral speed)
    Index 24: torso_vz (float) — z-velocity (vertical)
    Index 25: torso_wx (float) — x-angular velocity of the torso (roll rate)
    Index 26: torso_wy (float) — y-angular velocity of the torso (pitch rate)
    Index 27: torso_wz (float) — z-angular velocity of the torso (yaw / spin rate)
    Index 28: abdomen_z_vel (float) — abdomen twist velocity
    Index 29: abdomen_y_vel (float) — abdomen bend velocity
    Index 30: abdomen_x_vel (float) — pelvis tilt velocity
    Index 31: right_hip_x_vel (float) — right hip abduction velocity
    Index 32: right_hip_z_vel (float) — right hip rotation velocity
    Index 33: right_hip_y_vel (float) — right hip flexion velocity
    Index 34: right_knee_vel (float) — right knee velocity
    Index 35: left_hip_x_vel (float) — left hip abduction velocity
    Index 36: left_hip_z_vel (float) — left hip rotation velocity
    Index 37: left_hip_y_vel (float) — left hip flexion velocity
    Index 38: left_knee_vel (float) — left knee velocity
    Index 39: right_shoulder1_vel (float) — right shoulder velocity (axis 1)
    Index 40: right_shoulder2_vel (float) — right shoulder velocity (axis 2)
    Index 41: right_elbow_vel (float) — right elbow velocity
    Index 42: left_shoulder1_vel (float) — left shoulder velocity (axis 1)
    Index 43: left_shoulder2_vel (float) — left shoulder velocity (axis 2)
    Index 44: left_elbow_vel (float) — left elbow velocity
  Additional features (indices 45-243, hard to interpret, avoid unless needed):
    Indices 45-154: center-of-mass-based inertia of the 11 bodies (10 values per body)
    Indices 155-220: center-of-mass-based velocities of the 11 bodies (6 values per body)
    Indices 221-243: actuator forces applied to the 23 degrees of freedom
  Note: x,y position coordinates are excluded from the observation to maintain translational invariance.

Action Space: Continuous(17) — jax array of shape (17,) in [-1.0, 1.0] representing torques applied to 17 joints:
  Index 0: abdomen_y (abdomen bend torque)
  Index 1: abdomen_z (abdomen twist torque)
  Index 2: abdomen_x (pelvis tilt torque)
  Index 3: right_hip_x (right hip abduction torque)
  Index 4: right_hip_z (right hip rotation torque)
  Index 5: right_hip_y (right hip flexion torque)
  Index 6: right_knee (right knee torque)
  Index 7: left_hip_x (left hip abduction torque)
  Index 8: left_hip_z (left hip rotation torque)
  Index 9: left_hip_y (left hip flexion torque)
  Index 10: left_knee (left knee torque)
  Index 11: right_shoulder1 (right shoulder torque, axis 1)
  Index 12: right_shoulder2 (right shoulder torque, axis 2)
  Index 13: right_elbow (right elbow torque)
  Index 14: left_shoulder1 (left shoulder torque, axis 1)
  Index 15: left_shoulder2 (left shoulder torque, axis 2)
  Index 16: left_elbow (left elbow torque)
  Note: the action order differs from the observation joint order for the abdomen (action 0 = abdomen_y, action 1 = abdomen_z).

Function signature:
  def compute_reward(env, state, action, next_state, done, info, base_reward):
  - state, action, next_state: jax arrays (index directly, e.g., next_state[22] for forward speed, next_state[0] for torso height)
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
  - The humanoid is unstable and falls quickly without control. Most behaviors require staying upright: consider a small alive/upright bonus (e.g., based on next_state[0]) and terminating when the torso height drops below ≈1.0.
  - Keep reward magnitudes moderate and prefer smooth terms (e.g., jnp.exp(-x**2), jnp.clip) over large sparse bonuses.

Goal: Evolve a SIMPLE, interesting objective for the agent to explore in this playground (e.g., standing still, walking or running forward, walking backward, side-stepping, turning in place, crouching, raising an arm, balancing on one leg, or minimizing energy usage).
