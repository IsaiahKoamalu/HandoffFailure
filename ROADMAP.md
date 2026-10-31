# Roadmap — handoff failure between independently trained skills

Oct 26 → Dec 13, 2026. Seven weeks, one claim to test:

> **Mission outcome at a skill handoff is predicted by how far the delivered state
> falls outside the successor's training distribution.**

Everything below exists to produce that number, or to stop me fooling myself about it.
The week headings match the timeline table in `proposal/main.typ`; if a date slips here,
it slips there too.

---

## Ground rules

**Fixed apparatus.** Everything under `sim/`: the ship, the planets, the terrain, the
orbital propagation, and `sim/surface.gd`, the shared surface geometry. No feature work on the simulator is in scope. If I think I need a
simulator change, that is a signal the task definition is wrong, not the sim.

**Adapted, not written.** PPO is CleanRL's `ppo_continuous_action.py`, on the
instructor's recommendation, copied into `python/` with attribution. The changes to it
— vector environment, truncation bootstrap, normaliser, policy export — are this
project's work; the rest of the file is not, and is not claimed.

**Written for this project.** The environment wrapper and bridge client, both task
definitions, the mission runner, the z-score logger, the evaluation harness. The earlier
trainer and task definitions in git history are a reference implementation from before
the project — they are allowed to be read, not copied. Citing them as this project's
work is not.

**Freeze points.** Once E1 is measured, the two task definitions are frozen: success
criteria, reward terms, episode length. Any change after that date is reported as its
own before/after comparison, never as a silent correction. Write the freeze date into
`rl/runs/README.md` when it happens.

**Definition of done**, for every row below: the number exists, it is reproducible from
a command in the log, and the command is in the repo.

**Weekly rhythm.** Monday: pick the week's one target. Friday: write what happened into
`rl/runs/README.md`, including what failed. The paper is assembled from that file in
week 7, so a week with no entry is a week that cannot be written up.

---

## Critical path

Only three things can sink this:

1. **PPO that is subtly wrong.** Using CleanRL narrows this to the changes made to it —
   the truncation bootstrap and the normaliser most of all. Mitigated by the unit tests
   and by reproducing a known answer in week 1 (landing at ≈92%) before trusting the
   result on anything new.
2. **The mission runner.** It is the only genuinely new component. It is scheduled in
   week 3 and nothing about it is familiar.
3. **No spread in the handoff states.** If orbit insertion always delivers nearly the
   same orbit, there is no variation to correlate with outcome and E3 has nothing to
   measure. **This is the real risk and it is checked in week 2, not week 3.**

Everything else is schedule padding.

---

## Week of Oct 26 — the trainer, and proof that it works

Goal: a trainer I can trust, demonstrated against an answer I already know.

- [ ] Bridge client and environment wrapper. JSON over TCP, 32 ships, `--repeat 6`. The
      wrapper is a Gymnasium 1.x `VectorEnv` with same-step autoreset, so the host's
      own resets and its `final_obs` map onto it directly.
- [ ] Smoke test that runs in under a minute: the socket answers, a random policy
      crashes nearly every episode, throughput is in the expected range.
- [ ] PPO: copy CleanRL's `ppo_continuous_action.py` into `python/` with attribution,
      replace its environment setup with the wrapper, and replace its episode logging
      with outcome counts.
- [ ] Observation normaliser that matches `rl/neural_pilot.gd` (std ≤ 1e-3 counts as
      constant, clamp to ±10), frozen at evaluation, with mean and std saved next to the
      weights in `policy.json`. **That saved record is the measurement instrument for
      E3, so build it now, not in week 3.**
- [ ] Unit test — truncation. A time-limit cutoff is not a terminal state; the value
      estimate must still be bootstrapped. CleanRL ORs terminations and truncations
      together, so this tests the change that fixes it. Work one episode by hand and
      assert against it. Getting this backwards previously produced an advantage of
      +45.835 where the correct value was −4.015.
- [ ] Unit test — reward scale. Assert at startup that a maximum-length episode cannot
      cost more in per-step charges than the smallest failure penalty. Otherwise quitting
      on purpose is the cheapest move. Previously: 0.02/tick × 18,000 = 360, against a
      capture bonus of 150.
- [ ] Explained variance logged every iteration. A critic that is not tracking returns
      shows up here before it shows up in the success rate.
- [ ] Descent-and-landing task definition, trained to reproduce ≈92% at a 27,000-tick
      horizon.

**Done when** landing reproduces at its reference rate and both unit tests pass.
Reference: 92.5% landed over 400 episodes at seed 7 with `--repeat 6`.

**Traps.** Run `--headless --import` after adding any script with a `class_name`, or the
headless trainer parses nothing, never replies, and simply hangs. Freeze the normaliser
at evaluation time — normaliser drift alone moves the score, which was confirmed by
reproducing it with the learning rate set to zero. Do not keep CleanRL's Gymnasium
wrappers: its observation normaliser divides by `sqrt(var + 1e-8)`, which turns a
constant channel into huge values where `neural_pilot.gd` reads it as unit scale, and its
reward normaliser changes the reward scale the unit test asserts on. CleanRL pins
`gymnasium==0.29.1`; this project uses 1.x, which only touches the environment setup and
logging that are being replaced anyway.

---

## Week of Nov 2 — the second skill, and E1

Goal: both skills measured on their own, and the spread question answered early.

- [ ] Orbit-insertion task definition. Success is in the orbital elements, not position:
      periapsis clear of terrain, apoapsis inside the capture shell, eccentricity under
      threshold, radius within tolerance, held long enough that a trajectory merely
      crossing the boundary does not count.
- [ ] Put the capture tolerances back where they belong (±15%, e 0.1, 2s hold). The
      fourfold widening was a discovery experiment, not a decision.
- [ ] Both analytic controllers in place as baselines.
- [ ] **E1**: each skill alone, 200 episodes × 5 seeds, mean and spread. Episodes are
      paired across controllers — the same starting situations for each, so a gap cannot
      come from one drawing easier cases. Exploration noise off at evaluation.
- [ ] **Freeze both task definitions. Record the date.**
- [ ] **Spread check (do this before anything else this week finishes).** Log the state
      orbit insertion leaves the ship in across ~100 runs and look at the variation per
      channel. If the delivered states are nearly identical, E3 is dead on arrival —
      widen the insertion spawn distribution until there is spread, and say so in the
      write-up as a deliberate design choice.

**Done when** both skills have five-seed success rates and the delivered-state spread is
known to be non-degenerate.

**Traps.** Do not budget time for warm-starting insertion from the analytic law. It was
tried: a 65.6% law clones to 0.0%, a smoothed 39.6% law to 1.0%. Train from scratch.
Also watch fuel weight — at 40, a full tank cost less than the shaping that burning it
earned, so the agent emptied the tank chasing radius error. 120 worked.

---

## Week of Nov 9 — the mission runner, E2 and E3

Goal: the chain runs end to end and the central measurement exists.

- [ ] Mission runner across the interface mismatch: hold ship state through the
      transition, swap the observation builder and the active policy, reconfigure the
      ship (auto-stabiliser off → on, fuel budget → unlimited, 18,000 → 27,000 tick
      limit), and decide when to hand over. Both of those ship settings are already
      runtime-settable, so this needs no simulator change.
- [ ] Verify the swap in isolation before measuring anything: run insertion to its
      success condition, hand over, and confirm the landing skill's first observation is
      built correctly. A mismatched channel order would look exactly like the
      hypothesis being true.
- [ ] **E2**: chained success rate against the product of the individual rates.
      90% × 90% predicts ≈81%; well below that means the handoff is costing something.
- [ ] **E3**: at the handoff, express each of the landing skill's input channels as the
      number of typical variations it sits from that skill's usual value, using the
      record saved during its training. Summarise two ways — largest single departure,
      and count of channels past three variations — and record both against whether the
      mission succeeded.
- [ ] First look at the correlation. Strange handoffs failing more often is the claim.

**Done when** E2 and E3 are both measured at five seeds on the learned → learned seam.

**Traps.** Check that numpy inference and the in-engine GDScript forward pass agree to
about 1e-6 before believing any in-engine result; previously they agreed to 4.81e-07.
Make sure a policy path passed to a verification reaches *both* sides of it — passing it
only to the numpy side silently compares every candidate against the installed policy.

---

## Week of Nov 16 — progress report

Goal: hand in the report, spend the rest of the week on debt.

- [ ] Progress report, assembled from `rl/runs/README.md`.
- [ ] Re-run anything that is single-seed and shouldn't be.
- [ ] Figures for E2 and E3 drafted now, while the data is fresh.

**Done when** the report is submitted.

---

## Week of Nov 23 — the second seam

Short week; Thanksgiving is Nov 26. Keep the scope small on purpose.

- [ ] Replicate E3 at the analytic-transit → orbit-insertion boundary, where the
      predecessor is hand-written rather than learned.
- [ ] Compare the two seams. If unusual handoffs predict failure at both, the
      relationship is a property of composition rather than of one particular skill
      pair. This is replication at almost no cost — the measurement code already exists.

**Done when** the same measurement exists at both seams.

---

## Week of Nov 30 — the three repairs

Goal: find out which fix actually helps. Cheapest first; stop when the week ends.

- [ ] **Wider training distribution.** Retrain the successor over a broader range of
      starting states so the handoff is no longer unusual. Cheapest to implement,
      slowest to run — start it Monday.
- [ ] **Train on what the predecessor actually produces.** Run insertion, collect the
      states it delivers, continue training the successor from those.
- [ ] **Learned handover timing.** Replace the fixed handover criterion with a small
      policy that chooses the moment. Most interesting, least likely to finish — if one
      repair gets cut, it is this one.
- [ ] Each repair scored on the same paired episodes as E2, so the comparison is clean.

**Done when** at least two repairs are measured against the E2 baseline.

---

## Week of Dec 7 — analysis, paper, buffer

- [ ] Final figures.
- [ ] Paper written from `rl/runs/README.md`, including the before/after of any
      post-freeze change.
- [ ] All three controls reported: frozen-network re-run of any surprising curve, the
      analytic controller on the held-out world as a baseline-drop check, and the two
      unit tests.
- [ ] Repo tidy: one command per reported number, `rl/runs/README.md` current.

**Dec 13** — code final, presentation ready.

---

## If I fall behind, cut in this order

1. The learned handover policy (third repair).
2. The second repair, leaving domain randomisation as the only one.
3. The second seam.
4. The held-out-world generalisation check.

**Never cut**: E1, E2, E3, the five seeds, the frozen task definitions, or the three
controls. E3 is the contribution; everything else is support or extension.

---

## Bookkeeping

One directory per run under `rl/runs/`, holding the weights, the saved observation
record, the exact command and the log. Five seeds means five directories, not five
overwrites — a policy chosen deliberately beats a policy left wherever the last training
run put it, and keeping the predecessor is what makes a comparison repeatable later.

Reference numbers from the pre-project implementation, useful as sanity checks and
nothing more (different code, and two of them stand at tolerances since changed):

| Task | Rate | Conditions |
| --- | --- | --- |
| Descent and landing | 92.5% landed | 400 episodes, seed 7, `--repeat 6`, 27,000 ticks |
| Orbit insertion (learned) | 91.0% captured | 200 episodes, seed 7, widened tolerances |
| Orbit insertion (analytic) | 66.5% captured | same episodes |

## Deliverables

| What | When |
| --- | --- |
| Proposal | submitted |
| Progress report | week of Nov 16 |
| Paper | week of Dec 7 |
| Code and presentation | Dec 13 |
