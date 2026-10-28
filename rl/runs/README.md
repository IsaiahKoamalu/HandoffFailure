# Runs

One directory per run: weights, the saved observation mean and variance, the exact
command, the log. Nothing here is overwritten — a policy chosen deliberately beats a
policy left wherever the last training run happened to put it, and keeping the
predecessor is what makes a comparison repeatable a month later.

This file is the running record. Every Friday, what was measured and what failed goes
in here, and the paper is assembled from it in the final week.

## Task definition freeze

Both task definitions freeze once E1 is measured — success criteria, reward terms,
episode length. **Freeze date: not yet.** Record it here when it happens. Any change
after that date is reported as its own before/after comparison, never as a silent
correction.

## Reference numbers from the earlier implementation

Different code, and two of these stand at capture tolerances since changed. They are
sanity checks for the rewrite, not results.

| Task | Rate | Conditions |
| --- | --- | --- |
| Descent and landing | 92.5% landed | 400 episodes, seed 7, `--repeat 6`, 27,000 ticks |
| Orbit insertion, learned | 91.0% captured | 200 episodes, seed 7, widened tolerances |
| Orbit insertion, analytic | 66.5% captured | same episodes |

## Known failures in the apparatus

Measured on Oct 27, 2026, after the simulator moved into `sim/`. Both also fail
identically on `37563f7`, the last commit before the rewrite began, so neither was
caused by the move. Neither is in the two-skill chain's path, and simulator fixes are
out of scope.

| Check | Result |
| --- | --- |
| `checks/scatter_check.tscn` | Fails on all five worlds: 0 boulders in either streamed patch |
| `checks/surface_check.tscn` | Site 1 drifts 0.0518 m while parked; the other four drift about 0.023 m and pass |

`checks/cruise_check.tscn` passes, and `checks/terrain_report.tscn` runs.
`autoland_check`, `journey_check` and `dock_check` stop at "no landing task" until the
landing task exists.

## Measurements
