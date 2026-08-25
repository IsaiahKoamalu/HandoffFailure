# Runs

`landing/policy.json` is the policy the game loads (`NeuralPilot.DEFAULT_PATH`).
It is currently the **behaviour clone**, which measured best: 39.7% clean
landings against 4.0% for PPO from scratch and 3.0% after fine-tuning.

The other directories are kept as evidence for `../landing-log.html`, each with
the `progress.csv` its own run produced:

| Directory | What it is | Landing rate |
| --- | --- | --- |
| `clone/` | 3,000 ticks of behaviour cloning, no PPO | 39.7% |
| `scratch-ppo/` | PPO from random init, 5.0M steps | 4.0% |
| `finetune/` | clone + critic warm-up + PPO, 3.0M steps | 3.0% |

`.gdignore` keeps Godot from importing the CSVs as resources.
