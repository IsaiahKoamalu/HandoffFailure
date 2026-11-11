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
  (CleanRL ORs terminations and truncations), a normalizer matching
  `rl/neural_pilot.gd`, and `policy.json` export. Do not reintroduce CleanRL's Gymnasium
  wrappers. The header of `python/ppo.py` carries CleanRL's MIT notice and the list of
  changes made so far; keep that list matching the diff from `4358005`, the unmodified
  copy.
- **The truncation fix keeps CleanRL's OR.** With same-step autoreset, `next_obs` after
  a truncation belongs to the next episode, so GAE must still cut the chain at every
  ending. The fix adds `gamma * V(final_obs)` to the truncated ships' training reward
  instead. Removing the OR is a different bug, not the fix. The step-by-step plan and
  the hand-worked test numbers are in `docs/TODO.md`.
- **Runs** go to `rl/runs/<exp>__<seed>__<unix time>/`, each with a `run.json` written
  before training starts: command, every argument, commit, and what the host built,
  `obs_names` included. `rl/runs/*/` is gitignored; a run that is reported is added
  deliberately.
- **`python/launch.py`** starts the headless host and connects to it, for both
  `ppo.py` and `env_test.py`. The port is fixed at 11008 in `rl/train/host.gd`, so there
  is no port argument.
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
- The project is Godot 4.7 (`project.godot`). An older engine opens it, and may
  rewrite `.godot/`, so check `"$GODOT" --version` before running anything new.
- On Windows `GODOT` is a user variable set with `setx`, which only reaches processes
  started afterwards. A shell that predates it sees it empty until VS Code restarts.
- Python runs through uv: `uv run python ...`, the same on every machine.
- Anything specific to one machine goes in `CLAUDE.local.md`, which is not committed.

## Conventions and traps

- GDScript is indented with tabs, like the rest of the project.
- After adding any script with a `class_name`, run `"$GODOT" --headless --path . --import`,
  or the headless host fails to parse it and hangs without saying why.
- Godot's `JSON.stringify` writes a `PackedByteArray` as a string, not a list. Send
  boolean flags as `Array[bool]`.
- `python/env_test.py` caps episodes at 600 ticks, too short for most ships to reach
  the ground, so its PASS does not show that crash detection works. Pass
  `--max-steps 600` to a short `ppo.py` check for the same reason in reverse: at
  27,000 ticks a 20,000-step run sees no episode end.
- `rl/checks/scatter_check.tscn` and site 1 of `rl/checks/surface_check.tscn` fail on
  the pre-project code too. They are logged in `rl/runs/README.md` and are not to be
  fixed.
