Environment: MuJoCo Cartpole Playground (Continuous, MJX/Brax backend)
Observation Space: jax array of shape (4,) representing the physics state:
  Index 0: cart_position (float) — horizontal position of the cart on the slider track (slider range: [-1.0, 1.0] meters, center = 0.0)
  Index 1: pole_angle (float) — vertical angle of the pole in radians (0.0 = upright vertical, starts near 0.0)
  Index 2: cart_velocity (float) — horizontal velocity of the cart in m/s
  Index 3: pole_angular_velocity (float) — angular velocity of the pole in rad/s

Action Space: Continuous(1) — jax array of shape (1,) in [-1.0, 1.0] representing horizontal force applied to the cart slider.

Function signature:
  def compute_reward(env, state, action, next_state, done, info, base_reward):
  - state, action, next_state: jax arrays (index directly, e.g., state[1] for pole angle)
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
Goal: Evolve a SIMPLE, interesting objective for the agent to explore in this playground (e.g., balancing the pole upright, swinging the pole, moving the cart to a specific position while balancing, or minimizing cart movement).
