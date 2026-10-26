# rl/

Everything the learning system needs on the engine side, plus the scenes that verify
the simulator still behaves.

## Pre-existing, and treated as fixed

| File | What it is |
| --- | --- |
| `autopilot.gd` | The hand-written analytic controller. Flies transit, then descent, on `[P]`. Supplies the baselines, and is the predecessor at the second seam. |
| `neural_pilot.gd` | Forward pass of a trained network in GDScript. Loads `runs/landing/policy.json`, and degrades to "untrained" when there is none. |
| `spectator.gd` | Free camera for watching a run. |
| `autoland_check.tscn` | Lands repeatedly under a policy and reports the rate. |
| `cruise_check.tscn` | Checks the flight assist can hold a velocity between bodies. |
| `journey_check.tscn` | Flies whole journeys, planet to planet. |
| `dock_check.tscn` | Docks at a station, checks the capture is smooth, walks the astronaut around it. |
| `surface_check.tscn` | Lands at five sites and checks the ship parks and stays parked. |
| `scatter_check.tscn` | Checks the boulders return to the same places when terrain streams. |
| `view_check.tscn` | Renders one frame from the pilot's seat. |
| `portrait_check.tscn` | Photographs every world, from orbit and from the ground. |
| `terrain_report.tscn` | Prints what the terrain generator produced per body. |

## To be written

The two task definitions, the headless training host and its scene, and the mission
runner that carries the ship across the interface mismatch between the skills. The two
skills do not share an interface: they read different numbers of inputs, disagree about
whether the auto-stabiliser is on, and assume different limits on how long a run may
last. Reconciling that mid-flight is the mission runner's job, and both of those ship
settings are already settable at runtime, so none of it needs a simulator change.

## Verify before believing

Check that the engine's forward pass and the trainer's agree to about 1e-6 before
trusting any in-engine number, and make sure a policy path passed to a check reaches
**both** sides of it. Passing it to only one side silently compares every candidate
against whatever policy is installed, which once produced a 0.91 disagreement that was
pure artefact.
