# Handoff failure between independently trained skills

A reinforcement learning project. One claim is being tested:

> **Mission outcome at a skill handoff is predicted by how far the delivered state
> falls outside the successor's training distribution.**

Two spacecraft control skills — orbit insertion, then descent and landing — are trained
separately, chained into one mission, and measured at the seam where one hands the ship
to the other. `ROADMAP.md` is the week-by-week plan. `proposal/main.typ` is the written
proposal.

## What is apparatus and what is the project

This matters for grading, so it is stated here rather than left to be inferred.

**The simulator predates this project and is fixed.** Everything under `sim/`: the
ship, the planets, the terrain, the orbital propagation, the station, the astronaut, the
HUD and the shaders. That includes `sim/surface.gd`, the surface geometry that the
autopilot, the checks and the landing task all share, recovered from the earlier
implementation. The check scenes under `rl/checks/` that verify the simulator are
pre-existing too. No feature work on any of it is in scope. A need to change the simulator is a signal
that a task definition is wrong, not that the simulator is.

Also apparatus, and also pre-existing: `rl/autopilot.gd`, the hand-written analytic
controller the ship flies on `[P]`, and `rl/neural_pilot.gd`, which runs a trained
network inside the engine. The analytic controller supplies the baselines, and it is the
predecessor at the second seam.

**Written for this project**, and not yet present in this repository:

| Component | Where it will live |
| --- | --- |
| PPO — clipped surrogate, GAE, actor-critic, observation normaliser | `python/` |
| Environment wrapper and the JSON-over-TCP bridge client | `python/` |
| Evaluation harness | `python/` |
| Task definition — orbit insertion | `rl/` |
| Task definition — descent and landing | `rl/` |
| Headless training host and scene | `rl/` |
| Mission runner, which carries the ship across the interface mismatch | `rl/` |
| Handoff measurement — the z-score logger | `python/` |

An earlier implementation of the trainer and the task definitions existed in this
repository's history and was removed at commit time to begin the rewrite. It is history,
not a starting point: it may be read, and it is not this project's work.

## Running the simulator

The engine is not on the `PATH` on either machine, so set `GODOT` to the binary once
per shell and the commands below work unchanged on both.

```bash
# macOS
export GODOT=/Applications/Godot.app/Contents/MacOS/Godot
# Windows, Git Bash
export GODOT=/c/path/to/Godot_v4.7.2-stable_win64_console.exe

"$GODOT" --headless --path . --import   # after adding any script with a class_name
"$GODOT" --path . sim/main.tscn
```

On macOS the binary is inside the `.app` bundle, at `Contents/MacOS/Godot`; opening the
bundle itself launches the editor and ignores the arguments. Use the `_console` build on
Windows, or the engine detaches and prints nothing.

A check scene runs the same way:

```bash
"$GODOT" --headless --fixed-fps 60 --path . res://rl/checks/terrain_report.tscn
```

The reimport is not optional. Godot resolves `class_name` through
`.godot/global_script_class_cache.cfg`, which only the editor builds. A headless run
against a stale cache fails to parse the new script, and because the failure happens
inside `_physics_process` the engine keeps running and never replies — the trainer
simply hangs, with nothing on stdout to say why.

With no trained policy installed the heads-up display reads `[P] autopilot untrained`
and the ship is flown by hand. That is the expected state of a fresh checkout. The
autopilot refuses to engage at all without a landing policy, including the analytic
transit and docking phases, so it has to be changed before it can supply the baselines.
See `rl/README.md`.

## Python environment

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

## Results

One directory per run under `rl/runs/`, holding the weights, the saved observation
record, the exact command and the log. Five seeds means five directories, not five
overwrites. `rl/runs/README.md` is the running record the paper is assembled from — a
week with no entry in it is a week that cannot be written up.
