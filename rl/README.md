# rl/

Everything the learning system needs on the engine side, plus the scenes that verify
the simulator still behaves.

## Pre-existing, and treated as fixed

| File | What it is |
| --- | --- |
| `autopilot.gd` | The hand-written analytic controller. Flies transit, then descent, on `[P]`. Supplies the baselines, and is the predecessor at the second seam. |
| `neural_pilot.gd` | Forward pass of a trained network in GDScript. Loads `runs/landing/policy.json`, and degrades to "untrained" when there is none. |
| `checks/autoland_check.tscn` | Lands repeatedly under a policy and reports the rate. |
| `checks/cruise_check.tscn` | Checks the flight assist can hold a velocity between bodies. |
| `checks/journey_check.tscn` | Flies whole journeys, planet to planet. |
| `checks/dock_check.tscn` | Docks at a station, checks the capture is smooth, walks the astronaut around it. |
| `checks/surface_check.tscn` | Lands at five sites and checks the ship parks and stays parked. |
| `checks/scatter_check.tscn` | Checks the boulders return to the same places when terrain streams. |
| `checks/view_check.tscn` | Renders one frame from the pilot's seat. |
| `checks/portrait_check.tscn` | Photographs every world, from orbit and from the ground. |
| `checks/terrain_report.tscn` | Prints what the terrain generator produced per body. |

## To be written

A spectator — a free camera that watches a training run and names each outcome —
was removed with the rest of the trainer. It is in history at `4a714be` as
`rl/spectator.gd`, and the camera and overlay work; only the outcome codes and the
handle on the environment need re-pointing once the task definitions exist. Lift it
back rather than writing one.

The two task definitions, the headless training host and its scene, and the mission
runner that carries the ship across the interface mismatch between the skills. The two
skills do not share an interface: they read different numbers of inputs, disagree about
whether the auto-stabiliser is on, and assume different limits on how long a run may
last. Reconciling that mid-flight is the mission runner's job, and both of those ship
settings are already settable at runtime, so none of it needs a simulator change.

## What the autopilot expects of the landing task

`autopilot.gd` finds the landing task at runtime by its class name, `LandingEnv`, rather
than referring to it directly, so the game compiles on a checkout where the task has not
been written yet. The new task must keep that name and provide `OBS_SIZE` and a static
`build_observation(ship, planet, progress, into, at)`. Without it the HUD reads
"no landing task". Geometry (altitude, up, land under the ship, bearing to the target
continent) comes from `Surface` in `sim/surface.gd`, and the task should use it too
rather than computing any of it a second way.

The autopilot also refuses to engage unless a landing policy is installed. That blocks
the analytic transit and docking phases too, which the baselines need. Changing it is a
change to apparatus and is logged in `runs/README.md` when it happens.

## Verify before believing

Check that the engine's forward pass and the trainer's agree to about 1e-6 before
trusting any in-engine number, and make sure a policy path passed to a check reaches
**both** sides of it. Passing it to only one side silently compares every candidate
against whatever policy is installed, which once produced a 0.91 disagreement that was
pure artefact.
