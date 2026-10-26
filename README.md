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

**The simulator predates this project and is fixed.** `main.tscn`, `ship.gd`,
`planet.gd`, `planet_mesh.gd`, `station.gd`, `astronaut.gd`, `hud.gd`, the shaders, the
terrain, the orbital propagation, and the check scenes under `rl/` that verify them.
No feature work on any of it is in scope. A need to change the simulator is a signal
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

```bash
Godot --headless --path . --import   # after adding any script with a class_name
Godot --path . main.tscn
```

The reimport is not optional. Godot resolves `class_name` through
`.godot/global_script_class_cache.cfg`, which only the editor builds. A headless run
against a stale cache fails to parse the new script, and because the failure happens
inside `_physics_process` the engine keeps running and never replies — the trainer
simply hangs, with nothing on stdout to say why.

With no trained policy installed the heads-up display reads `[P] autopilot untrained`
and the ship flies on the analytic controller. That is the expected state of a fresh
checkout.

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
