# Runs

`landing/policy.json` is the policy the game loads (`NeuralPilot.DEFAULT_PATH`).
It is a **20-input, 64-wide network trained for 600,000 steps at one decision
per six physics ticks** (`"repeat": 6` in the file; `ShipAutopilot` reads that
field and holds actions to match). It carries no `task` field, as every
policy here predates the multi-task split; `NeuralPilot` reads those as landers,
which they all are. Verified end to end: 12/12 landed on the
target continent through the real `main.tscn` ship, from starting angles up to
3.03 radians — very nearly antipodal — at 0.79-1.50 units/s with alignment
1.000, and GDScript inference agrees with numpy to 6.22e-07.

It is **not the best policy in this directory**. `vnorm/policy_best.json`
measured 98.0% at the 27,000-tick limit against this one's 92.0%. Swapping them
is in `../../TODO.md` and wants a deliberate re-verification, not a file copy.

## Reading the landing rates below

Two environment changes make older numbers incomparable, and both were large:

- **Before water** (`clone/`, `scratch-ppo/`, `finetune/`): the planets were
  90-97% ocean and a splashdown scored as a landing. See the first entries under
  "Things that were true and inconvenient" in `../README.md`.
- **Before the time limit was rebased** (everything trained at `max_steps`
  9,000): roughly 6.5% of episodes were cut off in level flight. The
  hand-written controller scores 92.5% under that limit and 99.0% without it,
  having changed not at all.

Rates below are as measured at the time, under the environment of the time.

| Directory | What it is | Landing rate |
| --- | --- | --- |
| `clone/` | 3,000 ticks of behaviour cloning, no PPO | 39.7% |
| `scratch-ppo/` | PPO from random init, 5.0M steps | 4.0% |
| `finetune/` | clone + critic warm-up + PPO, 3.0M steps | 3.0% |
| `landing20/` | 600k steps, 20-input observation | 90.0% |
| `fuel40/` | 900k steps against a 40-second fuel budget | 75.0% |
| `repeat6/` | first run at 10 Hz decisions and sigma 0.08 | 92.0% |
| `repeat6b/` | as above plus `--critic-warmup 15` | 92.0% |
| `vnorm/` | as above plus normalised value targets | 92.0% at 9k, **98.0%** at 27k |
| `base27k/` | retrained at the rebased 27,000-tick limit | 92.0% |
| `fuel120/` | trained against a 120-second fuel budget | 92.0% |

`repeat6` through `fuel120` are the runs behind the results table in
`../README.md`. Their headline value is the *comparison*, not any single rate:
three different critics all landing at exactly 92.0% is what exposed the time
limit as the thing being measured.

`base27k` *was* the anomaly worth chasing — trained on the correct horizon and
worse for it than `vnorm`, which trained on the wrong one. It does not
reproduce. Re-run in one environment with one seed and one recipe, with
`--max-steps` the only difference and both scored at 27,000 over 200 episodes,
the correct horizon wins: 94.5% trained at 27,000 against 92.0% trained at
9,000. The original pair straddled the water change and the ground fixes, so it
compared two environments as much as two horizons — which is the hazard the
warning at the top of this table exists for, and it caught us anyway.

`.gdignore` keeps Godot from importing the CSVs as resources.
