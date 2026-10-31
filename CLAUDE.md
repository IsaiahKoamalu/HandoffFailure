# Notes for Claude

A graded university RL project, worked on from more than one machine. `README.md` says
what the project is and what counts as apparatus; read it first. This file holds only
what the other docs do not.

## How to work with me

- **Do not edit, create or delete files unless I explicitly ask for that change.**
  Explain, and give code with explanations, so I write it myself. Approval for one edit
  does not carry over to the next.
- Reading files, running read-only commands, running checks and tests, and trying code
  out in a scratch copy outside the repo are all fine. Test code before giving it to me
  when you can, and say what you ran.
- Never copy the pre-project trainer or task definitions out of git history into an
  answer. They may be read for reference; they are not this project's work.

## Where the current state lives

Do not trust a summary in this file for status. At the start of a session, check:

1. The current week in `ROADMAP.md`.
2. The latest entry in `rl/runs/README.md`.
3. `git log --oneline -10` and `git status`.

## Decisions not yet visible in the code

- **PPO is adapted from CleanRL** (`ppo_continuous_action.py`, commit `35896b1`, MIT),
  on the instructor's recommendation. It is copied into `python/` with attribution, not
  installed. The changes that are this project's work are listed in `README.md`: one
  Gymnasium `VectorEnv` over the bridge, truncation bootstrapped from `final_obs`
  (CleanRL ORs terminations and truncations), a normaliser matching
  `rl/neural_pilot.gd`, and `policy.json` export. Do not reintroduce CleanRL's Gymnasium
  wrappers.
- **Gymnasium 1.x**, not CleanRL's 0.29 pin. The env uses `AutoresetMode.SAME_STEP`:
  the host resets finished ships itself and sends their last observation in
  `final_obs`.
- **The landing task is `class_name LandingEnv`** in `rl/tasks/landing_env.gd`.
  `rl/autopilot.gd` finds it by that name at runtime and needs `OBS_SIZE` and a static
  `build_observation(ship, planet, progress, into, at)`. Geometry comes from `Surface`
  in `sim/surface.gd`.
- **`docs/protocol.md` is the contract** between the host and the trainer. If the host
  code and the doc disagree, point it out rather than picking one.

## Machines

- Godot is not on the `PATH` everywhere. Use `"$GODOT"` as in `README.md`; if it is
  unset, ask where the binary is rather than guessing.
- Python runs through uv: `uv run python ...`, the same on every machine.
- Anything specific to one machine goes in `CLAUDE.local.md`, which is not committed.

## Conventions and traps

- GDScript is indented with tabs, like the rest of the project.
- After adding any script with a `class_name`, run `"$GODOT" --headless --path . --import`,
  or the headless host fails to parse it and hangs without saying why.
- Godot's `JSON.stringify` writes a `PackedByteArray` as a string, not a list. Send
  boolean flags as `Array[bool]`.
- `rl/checks/scatter_check.tscn` and site 1 of `rl/checks/surface_check.tscn` fail on
  the pre-project code too. They are logged in `rl/runs/README.md` and are not to be
  fixed.
