# Runs

`landing/policy.json` is the policy the game loads (`NeuralPilot.DEFAULT_PATH`).
It is an **imitation policy**: four DAgger rounds against the hand-written
controller in `python/evaluate.py`, no PPO. It measured 92.0% clean landings on
the target continent, from uniformly random starts with orbital velocity and
tumbling, against 93.0% for the expert it was cloned from.

Reproduce with `--warm-start 7000 --dagger-rounds 4 --clone-epochs 14
--hidden 128 --steps 1`. `--steps 1` gives zero PPO iterations, so what gets
saved is the imitation policy; no `progress.csv` sits beside it because there
are no iterations to log.

The other directories are older evidence for `../landing-log.html`. **Their
landing rates are not comparable to the number above**: they were measured
before the environment could tell land from water, on planets that were 90-97%
ocean, and a splashdown was scored as a landing. See the first entry under
"Things that were true and inconvenient" in `../README.md`.

| Directory | What it is | Landing rate (as then measured) |
| --- | --- | --- |
| `clone/` | 3,000 ticks of behaviour cloning, no PPO | 39.7% |
| `scratch-ppo/` | PPO from random init, 5.0M steps | 4.0% |
| `finetune/` | clone + critic warm-up + PPO, 3.0M steps | 3.0% |

`.gdignore` keeps Godot from importing the CSVs as resources.
