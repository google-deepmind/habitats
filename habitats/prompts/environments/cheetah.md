Environment: MuJoCo Cheetah Playground (Continuous, MJX/Brax backend)
Observation Space: jax array of shape (17,) representing the flattened physics state.
  Positions (indices 0-7):
    Index 0: rootz (float) — vertical displacement of torso from default standing pose (0.0 = neutral standing height ~0.7m world Z; negative values down to ~ -0.35m mean torso is touching/dragging on the ground; positive values > +0.1m mean airborne leaping)
    Index 1: rooty (float) — torso pitch angle in radians (0.0 = level horizontal torso; POSITIVE = pitch down / nose down / forward tilt; NEGATIVE = pitch up / nose up / backward tilt)
    Index 2: bthigh (float) — back thigh joint angle (range: [-0.52, 1.05] rad)
    Index 3: bshin (float) — back shin joint angle (range: [-0.785, 0.785] rad)
    Index 4: bfoot (float) — back foot joint angle (range: [-0.4, 0.785] rad)
    Index 5: fthigh (float) — front thigh joint angle (range: [-1.0, 0.7] rad)
    Index 6: fshin (float) — front shin joint angle (range: [-1.2, 0.87] rad)
    Index 7: ffoot (float) — front foot joint angle (range: [-0.5, 0.5] rad)
  Velocities (indices 8-16):
    Index 8: rootx_vel (float) — horizontal velocity (+ is forward run, - is backward run, typical running speeds 2.0 to 8.0 m/s)
    Index 9: rootz_vel (float) — vertical velocity (+ is leaping up, - is falling down)
    Index 10: rooty_vel (float) — torso pitch angular velocity (+ is pitching nose down, - is pitching nose up)
    Index 11: bthigh_vel (float) — back thigh joint velocity
    Index 12: bshin_vel (float) — back shin joint velocity
    Index 13: bfoot_vel (float) — back foot joint velocity
    Index 14: fthigh_vel (float) — front thigh joint velocity
    Index 15: fshin_vel (float) — front shin joint velocity
    Index 16: ffoot_vel (float) — front foot joint velocity
  Note: horizontal position (rootx) is excluded from the observation to maintain translational invariance.

Action Space: Continuous(6) — jax array of shape (6,) in [-1.0, 1.0] representing torques applied to 6 joints:
  Index 0: bthigh (back thigh torque)
  Index 1: bshin (back shin torque)
  Index 2: bfoot (back foot torque)
  Index 3: fthigh (front thigh torque)
  Index 4: fshin (front shin torque)
  Index 5: ffoot (front foot torque)

Function signature:
  def compute_reward(env, state, action, next_state, done, info, base_reward):
  - state, action, next_state: jax arrays (index directly, e.g., next_state[8] for forward speed)
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
Goal: Evolve a SIMPLE, interesting objective for the agent to explore in this playground (e.g., running forward at high speed, running backwards, maintaining a specific torso posture/tilt, bounding/leaping, or minimizing torque/energy usage).
