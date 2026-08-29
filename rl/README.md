# Learned landing autopilot

A PPO agent that flies the ship down to a planet surface, trained against the
real physics and then run inside the game.

Nothing about the ship is reimplemented for training. The agent writes to
`Ship.command_thrust` / `Ship.command_turn`, the same seam the mouse and
keyboard feed, and everything downstream — gravity, flight assist, buoyancy,
engine spool, auto-levelling — is the shipping code. A policy trained against a
simplified copy of the ship would be a policy that cannot fly the real one.

```
python/ppo.py  <--- JSON over TCP, one exchange per physics tick --->  rl/train.tscn
   torch                                                         32 ships, 1 solar system
```

## Layout

| File | What it is |
| --- | --- |
| `rl/landing_env.gd` | The task: spawns, observations, reward, termination |
| `rl/train_host.gd` | Headless host — loads `main.tscn`, strips the player, serves the socket |
| `rl/train.tscn` | The scene the trainer runs |
| `rl/neural_pilot.gd` | MLP forward pass in GDScript, loads `policy.json` |
| `rl/autopilot.gd` | Flies the player's ship with a trained policy (`[P]` in the cockpit) |
| `python/godot_env.py` | Client for the socket protocol, Gym-shaped |
| `python/ppo.py` | The trainer |
| `python/evaluate.py` | Scores a controller against two baselines |
| `python/smoke_test.py` | Checks the bridge before you spend an hour on a run |

## Setup

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

`numpy` and `torch` are the only dependencies. CPU is the right choice — the
network is 64 units wide and the bottleneck is the physics, not the gradients.

### Reimport after adding a script with a `class_name`

Godot resolves `class_name` through `.godot/global_script_class_cache.cfg`,
which is built by the editor. A headless run against a project whose cache
predates a new script fails to parse it, and because the failure happens inside
`_physics_process` the engine keeps running without ever replying — the trainer
just hangs. So after adding or renaming any `class_name`:

```bash
Godot_v4.7.2-stable_win64_console.exe --headless --path . --import
```

Opening the project in the editor once does the same thing.

## Running it

Check the bridge first. This takes under a minute and catches almost everything:

```bash
cd python
python smoke_test.py --godot /path/to/Godot_v4.7.2-stable_win64.exe --project ..
```

Expect roughly 8,000 steps/second with 32 agents, and a random policy that
crashes essentially every episode.

Train:

```bash
python ppo.py --godot /path/to/Godot.exe --project .. \
    --agents 32 --rollout 256 --steps 5000000 --out ../rl/runs/landing
```

Weights land in `rl/runs/landing/policy.json`, a per-iteration log in
`progress.csv`.

### Watching it learn

Add `--render` to open a window on the run instead of flying blind:

```bash
python ppo.py --godot /path/to/Godot.exe --project ..     --agents 16 --warm-start 3000 --steps 1000000 --render --render-rate 180
```

You get a chase camera that follows whichever agent is nearest the ground, a
live readout of the trainer's own numbers, and a bar per agent in the batch.
Keys: `[` and `]` pick an agent, `A` returns to auto-follow, `,` and `.` change
speed, `H` hides the overlay.

`--render-rate` is physics ticks per simulated second: 60 is real time, 180 is
the default, and `,`/`.` move it between 60 and 960 while the run is going. The
step stays 1/60 of a simulated second whatever it is set to, so **this changes
the speed of playback and not the physics** — a rendered run and a headless run
from the same seed produce the same trajectories and the same gradients. It
costs throughput and nothing else: about 2,700 steps/second against 5,700
headless on this machine.

Worth doing at least once. A learning curve is a lossy summary of a policy —
"mean return 61.6" does not tell you that the ship is arriving level but braking
late, or that it has found a hover and is farming the shaping term, or that a
third of the batch is tumbling. The batch strip is what makes the hover obvious:
a policy that has learned to land and one that has learned to wait look the same
through the chase camera for the first ten seconds and completely different as
sixteen altitude bars that stop going down.

Score the result against both baselines:

```bash
python evaluate.py --godot /path/to/Godot.exe --project .. --controller zero
python evaluate.py --godot /path/to/Godot.exe --project .. --controller baseline
python evaluate.py --godot /path/to/Godot.exe --project .. --controller policy
```

Then fly it: get in the ship, get near a planet, press `[P]`.

## The task

**Observation** (19 floats, `LandingEnv.build_observation`) — everything is
relative to the surface being landed on and expressed in the ship's own frame,
so one policy generalises across planets, orbits and headings:

| | |
| --- | --- |
| `0` | altitude / surface radius |
| `1` | `tanh(altitude / 0.05R)` — a sharper channel for the last few metres |
| `2–4` | velocity relative to the surface, in ship-local axes |
| `5–7` | local up, in ship-local axes (this is the attitude error) |
| `8–9` | vertical and horizontal speed relative to the surface |
| `10` | local gravity, 1.0 at the surface |
| `11` | engine spool |
| `12` | fraction of the episode elapsed |
| `13` | submerged |
| `14` | landness: >0 over dry ground, <0 over sea, 0 at the coastline |
| `15–17` | great-circle heading to the target continent, ship-local axes |
| `18` | angle to the target continent, over pi |

Channels 14–18 are what make the task learnable at all. Without them an ocean
and a plain are identical in every other channel, so no policy over them can
beat a coin flip on where to come down. `15–17` is deliberately in the ship's
frame — the same frame `command_thrust` is in — so the policy can map it onto a
translation without first learning the rotation.

The heading is **global**, and that is the point. It replaced a local gradient
of the land field, which could only walk to the *nearest* coast and went to
exactly zero in open ocean — pointing nowhere at the moment a direction was most
needed. A bearing to a known point is defined everywhere on the sphere,
including the antipode.

**Action** (6 floats in [-1, 1]) — three translation axes and three rotation
axes. Exactly the control set the player has.

**Reward** — potential-based shaping on altitude, surface-relative speed,
belly-down alignment and being over land, plus a graded touchdown bonus up to
120. Touching down at sea, or going under, is a flat -30 and no bonus at all.
An episode ends at touchdown, on submersion, on leaving the sphere of
influence, or after 1,500 ticks.

**Outcomes** — `landed`, `crashed`, `ditched`, `missed`, `escaped`, `timeout`.
Ditching and missing are tracked apart from crashing because they are different
mistakes: the ship can arrive perfectly and still have arrived in the sea, or on
the wrong continent. Averaging those into "crashed" hides which half of the job
is failing.

**Spawns** — deliberately enormous, because the autopilot is meant to be
engageable from wherever the player happens to be:

| | |
| --- | --- |
| altitude | 0.02 to 1.20 surface radii |
| tangential velocity | up to 0.85 of local circular orbital speed |
| attitude | 30% fully random, tumbling at 0.6 rad/s |
| position | half over the target continent, half anywhere at all |

Half on-target rather than none, because a purely uniform spawn makes the
terminal landing rare: most episodes would be spent crossing, and the few ticks
that actually decide a touchdown would be swamped.

## Things that were true and inconvenient

**The landing rate was measuring splashdowns.** This one invalidated a whole
round of results, so it goes first.

The planets were configured with almost no dry land — 2.7%, 9.8% and 4.5% of
each sphere — so a uniformly placed agent came down in the sea essentially
every episode. The environment had no notion of water, and water is *soft*:
`_water_drag` is 3.0 against the flight assist's 1.6, plus buoyancy. A ship
that hits the ocean at terminal velocity is braked to well under the 2.0
units/s touchdown gate before it reaches the seabed, and was scored a clean
landing for it.

So the reported 62.3% baseline and 39.7% behaviour clone were, overwhelmingly,
ships falling in the sea. Measured after the fact with the current outcome
classifier, the old controller on the old planets ends **85.6% of episodes in
water**. The controller was not solving the task; the metric was not asking
for it.

The lesson is not about water. It is that a success metric assembled from
conditions the environment happens to expose (`speed`, `align`) will quietly
answer a different question than the one intended, and it looks exactly like
success while it does.

**The ship could not stop.** Thrust used to be 10.0 units/s² against a surface
gravity of 9.81 — a thrust-to-weight ratio of **1.02**. That is not merely
tight, it is unflyable, and the reason is a detail of the flight assist:
`_process_flight_assist` deliberately leaves the axis you are thrusting along
undamped, so a ship braking its own descent gets thrust minus gravity and
nothing else. That is 0.19 units/s², which needs 31 seconds and ninety units
of altitude to wash off a six-units-a-second fall — more than the ship ever
spawns with. A controlled touchdown on land was not hard, it was unreachable,
and the only reason anything ever appeared to land was the ocean.

Thrust is now 18.0, a ratio of about 1.83. With that the hand-written
controller lands 99.3% of the time at 1.04 units/s. Flight assist still does
real work — it caps free fall near `g / 1.6 = 6.1` units/s, which is three
times the touchdown gate — but the ship is no longer relying on it to stop.

**A gain tuned for a ship that could not stop.** After the thrust change the
baseline still landed 0% of the time, at 3.11 units/s. Its proportional gain
was 15, so a three-units-a-second overshoot asked for only half thrust — which
was academic when there was no thrust to give and was pure slack afterwards.
Raising it to 45 and tightening the flare took it from 0% to 99.3% with no
other change. Retune the controller when you change the plant.

**A pass/fail bonus is a cliff.** The first reward gave +100 for a clean
touchdown and nothing otherwise. Every near miss scored the same as a
forty-units-per-second smash, so there was no gradient back toward a landing.
Scoring speed and alignment continuously fixed it.

**Shaping has to be scaled to the thing the task turns on.** Observations are
normalised by orbital speed at the surface (~42 units/s), and reusing that scale
in the reward made the entire difference between a 6 units/s crash and a perfect
touchdown worth about 0.7 reward. Rescaling to a landing-relevant 10 units/s was
the difference between a policy that ignored speed and one that did not.

**Penalising crashes made the policy refuse to land.** With an impact-scaled
penalty, hovering for the full episode cost about -1.8 and a bad landing cost
-3, so the policy correctly learned to stall until the clock ran out. The
penalty is now flat and small, and time is charged at a rate that makes running
out the clock worse than any touchdown.

**Alignment was gated but barely shaped.** Belly-down alignment gates the
terminal bonus, but at a shaping weight of 0.3 it was worth at most 3 reward
across a whole episode against a 120 bonus. The policy learned to arrive quickly
and roughly sideways: ~30 return, 3% clean landings. Raising the weight to 1.5
roughly doubled the landing rate within the first 400k steps.

## Results

300 episodes each, on the retuned planets, policies at the distribution mean.

200 episodes each. The task is now **reach the planet's largest continent from
anywhere and land on it**, so a landing counts only on the target landmass.

| Controller | Landing rate | Ditched | Missed | Timeout | Touchdown |
| --- | --- | --- | --- | --- | --- |
| Hand-written baseline | **93.0%** | 0.5% | 0.0% | 6.5% | 1.00 |
| Behaviour clone | 83.0% | 2.0% | 0.0% | 14.5% | 1.23 |
| Clone + 4 DAgger rounds | **92.0%** | 0.5% | 0.0% | 7.0% | 1.00 |

In-game, through `ShipAutopilot` on the real `main.tscn` ship, engaging from
uniformly random positions with orbital velocity and half of them tumbling:
**12/12 landed** on the target continent, 1.09-1.38 units/s, alignment 1.000,
from starting angles up to 2.40 radians — most of the way round the planet.

**Cloning alone plateaus; DAgger fixes it.** Plain behaviour cloning stuck at
83% however much data or capacity it was given — at one point action MSE
improved 2.5x with *no* change in landing rate, and the failures merely moved
from ditching to hovering. That is compounding error, not underfitting: fitting
the expert's actions on the expert's own states teaches nothing about the states
the learner's small errors actually lead to.

Four rounds of DAgger (Ross, Gordon & Bagnell 2011) — the learner flies, the
expert labels what the learner reaches — took it to 92.0%, within a point of the
expert. Reproduce with `--warm-start 7000 --dagger-rounds 4 --clone-epochs 14
--hidden 128 --steps 1`; `--steps 1` gives zero PPO iterations, so what is saved
is the imitation policy.

**PPO still makes things worse**, and that has not changed. On the earlier,
narrower task six hundred thousand steps of fine-tuning took a 97% clone to
2.5%, with the return falling too — PPO destroying a good initialisation rather
than exploiting the objective. The shipped `policy.json` is the imitation
policy.

**One expert change was made purely for the learner's benefit.** The
descend-or-cruise decision started as a hard boolean, which works fine as a
controller and clones badly: a network fitted by least squares cannot reproduce
a step, so it learns a smeared version and half-commits. Making the expert
smooth — a continuous `ready` blend instead of a branch — cost the controller
about four points and cut clone MSE by 2.5x. **If a policy is going to be
imitated, the demonstrator's own discontinuities are a design decision.**

## Terrain

`rl/terrain_report.tscn` measures the shape of every planet, because "too spiky"
and "mostly ocean" should be numbers:

```bash
Godot.exe --headless --fixed-fps 60 --path . res://rl/terrain_report.tscn
```

It reports land fraction, the largest single connected landmass, relief, and
ground slope in degrees over an eight-unit baseline — roughly a ship length,
which is the scale that decides whether struts can sit level.

| | land | biggest continent | slope mean | slope p99 |
| --- | --- | --- | --- | --- |
| Planet1 | 41.1% | 30.1% | 5.8 deg | 17.7 deg |
| Planet2 | 43.4% | 23.1% | 4.8 deg | 14.1 deg |
| Planet3 | 37.4% | 12.3% | 7.1 deg | 27.3 deg |

`continent_frequency` reads as *features per planet*, so the 100.0 it was set to
meant a hundred continents across a sphere — which is not terrain, it is static,
and it measured 51 and 40 degrees of mean slope. At 0.95-1.5 the same noise
makes continents, and the mean slope drops to 5-7 degrees.

## Known limits

- **Touchdown is analytic, not collided.** Terrain collision is streamed as
  patches under a registered client, which is far too expensive for hundreds of
  thousands of steps. The surface is analytic anyway —
  `Planet.get_surface_radius_at` is the function the collision mesh is built
  from — so an episode ends when altitude reaches zero. The two agree to within
  mesh resolution, but the agent has never touched a collider.
- **The policy has never seen the ground under a landing leg.** It is scored on
  arriving gently and level over a point, not on staying upright afterwards.
- **Observation `12` makes the policy time-dependent**, which is a wart. The
  autopilot feeds it ticks-since-engaged over the same horizon so at least it is
  consistent with training.
- Agent ships ignore each other (collision layer 9) and share one solar system
  and one clock.
- **Land-seeking is a local gradient walk.** Observations 15-17 are the gradient
  of the continent field at the ship, so an agent walks uphill to the *nearest*
  coast and cannot know a better one lies behind it. Where the field goes flat,
  mid-ocean, the signal is zero and the honest behaviour is to hover until the
  clock runs out. This is why `spawn_land_fraction` exists rather than being a
  uniform spawn: the task is "do not come down in the sea", not "cross one".
