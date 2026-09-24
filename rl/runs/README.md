# Runs

`landing/policy.json` is the policy the game loads (`NeuralPilot.DEFAULT_PATH`).
It is **`base27k-refit/policy_best.json`**, a 20-input, 64-wide network trained
for 581,632 steps at one decision per six physics ticks (`"repeat": 6` in the
file; `ShipAutopilot` reads that field and holds actions to match), on the
27,000-tick horizon it is scored at.

It was **chosen**, which the one before it was not — that one was overwritten by
a training run mid-session. The predecessor is kept as `landing-prev/` so the
comparison below can be repeated.

| 400 episodes, seed 7, `--repeat 6` | landing-prev | **shipped** |
| --- | --- | --- |
| landed | 88.5% | **92.5%** |
| mean return | 116.66 | **128.01** |
| crashed | 2.5% | **1.0%** |
| ditched | 4.5% | **2.0%** |
| timeout | 4.5% | 4.5% |
| touchdown, of the landings | 1.23 u/s | **0.96 u/s** |

Better on every axis but truncation, and it touches down gentler rather than
buying the extra landings with harder arrivals. An independent 200-episode run
at the default seed showed the same 4.0-point margin (94.5% against 90.5%),
which is what makes a 4-point difference on 400 episodes worth acting on: one
sample of it is about 1.9 sigma, two agreeing samples are not a coincidence.

Verified before installing, not after: `res://rl/autoland_check.tscn` 12/12 on
the real `main.tscn` ship with alignment 1.000 throughout, and GDScript
inference agrees with numpy to 4.81e-07. Both checks can now be pointed at a
file (`-- --policy=<path>`); neither could before, which is why this item sat
open — `verify_pilot` passed the path to the numpy side only and compared every
candidate against the installed policy, reporting a 0.91 disagreement that was
pure artefact.

The old note that `vnorm/policy_best.json` is better, at 98.0% against 92.0%,
does not survive scrutiny: those two numbers were measured in different
environments, before the water change and the ground fixes. Under today's
environment the predecessor scores 88.5%, not 92.0%. Nothing in the table below
is comparable to anything measured now.

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
| `base27k-refit/` | the base27k recipe re-run today, scored at 27,000 | **92.5%** |
| `landing-prev/` | what `landing/` held before, re-scored today | 88.5% |

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
