Environment: Playing Card Playground (single-player)
State Space: Discrete(6561) representing the game state.
You can decode the state integer using `env.decode(state)` which returns a dict:
  {
    "hand": int (0-8, 0=empty),
    "table": list of 3 ints (0-8, 0=empty)
  }
Action Space: Discrete(9)
  0: Pass
  1: Draw (deck -> hand)
  2-4: Play hand to slot 0-2
  5-7: Take from slot 0-2 to hand
  8: Discard hand
This is a PLAYGROUND. There is no predetermined goal, winning condition, or built-in rules.
Goal: Evolve a SIMPLE, interesting objective (effectively creating a mini-game) for the agent to play in this playground (e.g., arranging cards on the table in patterns, matching values, hoarding cards, or cycling cards between hand and table).
