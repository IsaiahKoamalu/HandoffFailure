# TODO

Ordered roughly by how much they would change what we believe, not by effort.

## Reinforcement learning

- [x] **Chase the `base27k` anomaly.** It does not reproduce. Re-run as a
      controlled pair -- one environment, one seed, one recipe, `--max-steps`
      the only difference, both scored at 27,000 over 200 episodes -- training
      on the correct horizon wins: **94.5% trained at 27,000 against 92.0%
      trained at 9,000** (finals 93.5% and 90.5%). The original 98.0% vs 92.0%
      was measured across two different environments, before the ground fixes
      and the landmarks, so it was never like for like. One seed each, so the
      2.5-point margin is suggestive rather than settled -- but the *inversion*
      is the point, and nothing here needs explaining any more.
- [x] **Decide what ships.** Done. `rl/runs/landing/policy.json` is now
      `base27k-refit/policy_best.json`, chosen rather than left where a training
      run happened to put it, and the predecessor is kept as `landing-prev/` so
      the comparison can be repeated. It wins on every axis but truncation --
      92.5% against 88.5% over 400 paired episodes, fewer crashes and ditches,
      and a gentler touchdown -- with the same 4.0-point margin in an
      independent 200-episode sample. Verified before installing: 12/12 in game,
      numpy and GDScript agree to 4.81e-07. Written up in `rl/runs/README.md`.

      The `vnorm` 98.0% that this item was waiting on is not a real comparison:
      it and the 92.0% were measured in different environments. The predecessor
      scores 88.5% today.

- [ ] **Neither in-game check could test an uninstalled policy, and that is
      fixed but worth remembering.** `verify_pilot.py` passed `--policy-path` to
      the numpy side only, so the engine loaded its own default and every
      candidate was compared against the shipped one -- a candidate that is fine
      reports a 0.91 disagreement. `autoland_check` had no policy argument at
      all. Both now take `-- --policy=<path>`. The general shape is worth
      watching for: a verification tool that silently verifies the wrong thing
      passes and fails for reasons that have nothing to do with what was asked.
- [ ] **Stop `policy_best.json` selecting on noise.** It takes the best of ~73
      rolling-window landing rates that carry ~1.9% binomial noise each, so it
      overstates by two or three points. Measured on docking, where the run was
      long enough to take the best of ~360 windows: it claimed 91.0% and those
      weights score 75.0% over 200 episodes. Sixteen points, not two or three.
      Either evaluate candidates properly before saving, or average over several
      windows.
- [ ] **Find out why fine-tuning erodes after iteration ~20.** Both the critic
      fix and warm-start noise fix raised the peak without removing the decay;
      runs give back ten points over the back half. Suspects: learning-rate
      schedule, `--ent-coef` at 0.003, sustained off-policy drift. Not the
      observation normaliser: that was the obvious suspect after docking, and
      freezing it makes landing worse rather than steadier (94.5% live, 91.0%
      frozen). The fresh 27,000-tick run above did not erode at all -- it ends
      at its peak -- so the next thing to establish is whether the decay is even
      still there, or was a property of the pre-rebase 9,000-tick environment.
- [ ] **Make RL beat the expert, or say plainly that it does not.** It has tied
      or lost in every configuration measured, including under a fuel budget,
      which was supposed to be the case where a learned policy wins. If a fair
      attempt keeps landing on parity, that is a finding and the README should
      lead with it rather than treating it as unfinished business.

- [x] **Train a docking policy, and see whether it beats the baseline.** Done.
      At `collide_penalty` 40 it lost by 9 points, 75.0% to 84.1%, giving all of
      it back in hull strikes because the reward paid twice for closing fast and
      charged 40 once for hitting the structure. At 150 -- derived from that
      trade, not tuned -- it ties: **81.0% against 82.3% over 300 paired
      episodes**, about 0.4 sigma, while colliding *less* than the hand-written
      law (8.3% against 10.7%) and docking in 42% of the decisions. So RL
      reaches parity here and wins decisively on time. See `rl/README.md`.

- [x] **Re-measure landing now that `--freeze-norm` exists.** Done, and the
      worry was wrong. Landing is not harmed by the drifting normaliser and is
      made *worse* by freezing it: 94.5% live against 91.0% frozen, same seed
      and recipe. Nor does the live run collapse the way docking did -- it ends
      at its peak. The difference is how far PPO's state distribution moves from
      the clone's: docking's episodes stretch from 342 decisions to 1,500 and
      sit far from the port, landing's cloned and trained trajectories look
      much alike, so the running mean barely moves. `--freeze-norm` is for tasks
      where those two distributions diverge, not a general fix, and it should
      stay off by default.

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
- [x] **Hold out a planet.** Done, and the answer is that it generalises.
      `--holdout=` on ppo.py and `--planets=` on evaluate.py select which worlds
      a run may use, so training and scoring can be pointed at different terrain.
      Trained with Planet3 held out, 300 paired episodes each:

          policy    95.0% trained -> 75.0% held out  (-20.0)
          baseline  95.7% trained -> 76.8% held out  (-18.9)

      The baseline reads landness and cannot memorise anything, and it drops by
      the same amount, so the gap is Planet3 being hard rather than the policy
      having learned three maps. Their failure modes are opposites, which is the
      more interesting half: on the hard world the baseline flies on looking for
      ground until the clock runs out (23.7% timeout, 0.3% ditched) and the
      policy commits and lands in water (24.0% ditched, 0% timeout).
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

- [ ] **The dock controller cannot hold a standoff.** It is a velocity
      controller with no position damping, so at an axial setpoint the sign of
      its command flips either side of the mark and it hunts. On a normal
      approach this never shows, because the ship crosses the standoff moving
      and commits before it settles there. It shows immediately if the transit
      leg is made to hand over at the mouth of the corridor: the ship then holds
      at the standoff, the commands saturate, and it sits there swinging between
      two units a second in and two out until it drifts off. That is why the
      handover is still at twice the corridor length and not at the corridor
      itself, which would otherwise be the tidier arrangement -- transit already
      aims there. A derivative term on the axial error, or a position hold that
      is not built out of a velocity loop, would fix it.

## Physics

- [x] **A parked ship sits on a knife edge.** Parking needed relative speed
      under `Ship._landing_speed` (0.2) *every tick*, and a ship sitting still
      on its struts reads a steady 0.19 — because parking pins the position and
      then stops updating the velocity that produced it, so `linear_velocity`
      goes on accumulating gravity against a hull that is not moving. One tick
      over the line un-parked it, and un-parked the mouse flies the hull instead
      of panning the camera: the pilot looked around the landing site and the
      ship went with them. Entering park is now a measurement (under
      `_park_speed`, 0.75, held for 12 ticks) and staying parked is a decision
      only the pilot reverses. `rl/surface_check.tscn` holds the line.
- [ ] **The residual drift itself is unexplained.** A ship at rest reads 0.19
      units/s against the surface and it does not decay. Parking no longer cares,
      but something is still wrong in either the surface-velocity model or the
      resting contact, and the next thing that reads that number will be misled
      by it the same way.

## Documentation

- [ ] **`rl/landing-log.html` is two rounds stale.** It describes the
      water-avoidance task: no continent targeting, no DAgger, no thrust fix,
      and a scoreboard (99.3% / 97.0%) for a task that no longer exists. It is
      also published, so updating it means redeploying to the same URL.
