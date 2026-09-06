# TODO

Ordered roughly by how much they would change what we believe, not by effort.

## Reinforcement learning

- [ ] **Chase the `base27k` anomaly.** A policy trained at `max_steps` 9,000 and
      evaluated at 27,000 scores 98.0%. The same recipe *trained* at 27,000
      scores 92.0%, with real crashes and ditches instead of timeouts. Training
      on the correct horizon should not be worse. Until this is understood, any
      conclusion about what PPO can reach here is provisional. First thing to
      check: episodes and terminal events per 600k steps at each limit — the
      guess is that a 3x longer clock starves the run of landings to learn from.
- [ ] **Decide what ships.** `rl/runs/landing/policy.json` was overwritten by a
      training run mid-session rather than chosen. It passes 12/12 in game, but
      `vnorm/policy_best.json` measured 98.0% against its 92.0%. Re-verify the
      candidate in game (`res://rl/autoland_check.tscn`) and `verify_pilot.py`
      before swapping, and record *why* in `rl/runs/README.md`.
- [ ] **Stop `policy_best.json` selecting on noise.** It takes the best of ~73
      rolling-window landing rates that carry ~1.9% binomial noise each, so it
      overstates by two or three points. Either evaluate candidates properly
      before saving, or average over several windows.
- [ ] **Find out why fine-tuning erodes after iteration ~20.** Both the critic
      fix and warm-start noise fix raised the peak without removing the decay;
      runs give back ten points over the back half. Suspects: learning-rate
      schedule, `--ent-coef` at 0.003, sustained off-policy drift.
- [ ] **Make RL beat the expert, or say plainly that it does not.** It has tied
      or lost in every configuration measured, including under a fuel budget,
      which was supposed to be the case where a learned policy wins. If a fair
      attempt keeps landing on parity, that is a finding and the README should
      lead with it rather than treating it as unfinished business.

- [ ] **Train a docking policy, and see whether it beats the baseline.** The
      task, the station and the harness all exist; nothing has been trained. The
      hand-written controller docks 76.7% of the time against 0.0% for no input,
      and its failures are almost all spawns behind the station that it flies
      around badly. That is a beatable number and a legible failure mode, which
      makes this the fairest test yet of the question above.

## Environment

- [ ] **Give fuel a reason to be spent well.** The 120s budget binds only on
      far-side crossings, so most episodes never feel it. A budget that binds
      everywhere, or a genuinely fuel-scarce variant, would make the constraint
      shape the whole trajectory instead of just the tail.
- [ ] **Land on ground the ship can sit on.** The task rewards arriving gently
      on the right continent; it says nothing about slope. A ship parked on a
      slope creeps (see below), so "landed" and "landed somewhere useful" are
      not yet the same thing. `Planet` already samples terrain, and
      `rl/terrain_report.tscn` already measures slope.
- [ ] **Hold out a planet.** All training and all evaluation happen on the same
      three terrain seeds, so nothing distinguishes a policy that generalises
      from one that memorised. Train on two, score on the third.
- [ ] **Touchdown is analytic, never collided.** Episodes end when altitude
      reaches zero against `Planet.get_surface_radius_at`. The agent has never
      touched a collision mesh, and is never scored on staying upright after it
      stops.

- [ ] **Docking is blind for a human pilot.** The port is on the stern, so the
      ship backs in and the pilot is looking the other way. The autopilot does
      not care; a person does. A rear camera, a port-relative HUD cue, or a
      free-look that snaps aft would each solve it.
- [ ] **`ppo.py --task docking --warm-start` would clone the wrong expert.** The
      warm start calls `baseline_controller` unconditionally, which is the
      *landing* law reading a 20-float observation. Docking observations are 22
      floats. Nothing has hit this because docking has never been trained.

- [ ] **Scattered boulders have no collision.** They are sized to knee height so
      that walking through them does not read as a bug, which is a workaround
      rather than a fix. A single convex shape per instance on the few nearest
      the client would cover it without paying for several hundred.
- [ ] **Surveying a site does nothing but tick a counter.** The sites exist,
      are targetable and remember being visited, but finding all five on a world
      earns nothing. Fuel, a nav upgrade, or simply a log entry describing what
      was found would each close the loop.
- [ ] **A landmark can be placed on ground the ship cannot land beside.** The
      slope test rejects steep sites, but nothing checks that there is a flat
      approach, and on Ember a single ridge is enough to hide one from twelve
      metres away.
- [ ] **The ship's fuel budget is enabled and there is exactly one place to
      refill it.** 900 seconds is generous and the station tops it up, but a
      pilot who runs dry on the far side of Thule has no recourse at all. Either
      a second station, a surface depot, or a slow ambient regeneration.
- [ ] **The shipped policy has never seen fuel vary.** It trains with unlimited
      tanks, so observation 19 is constant at 1.0 for the whole run and the
      normaliser's standard deviation for it is a floor. `CONSTANT_STD` stops
      that from exploding, but the honest fix is to train with a fuel budget so
      the channel means something. See the note in `rl/README.md`.

## Physics

- [ ] **A parked ship sits on a knife edge.** Parking needs relative speed under
      `Ship._landing_speed` (0.2), and the ship's own residual drift sits right
      around that value — so it flickers in and out of parked, and any nudge
      that pushes it over stays pushed. On a slope it never parks at all and
      creeps at ~0.13 units/s indefinitely. Raising the threshold treats the
      symptom; the drift is the thing to fix.

## Documentation

- [ ] **`rl/landing-log.html` is two rounds stale.** It describes the
      water-avoidance task: no continent targeting, no DAgger, no thrust fix,
      and a scoreboard (99.3% / 97.0%) for a task that no longer exists. It is
      also published, so updating it means redeploying to the same URL.
