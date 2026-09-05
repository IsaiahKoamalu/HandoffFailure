# Learned landing autopilot

A PPO agent that flies the ship down to a planet surface, trained against the
real physics and then run inside the game.

Nothing about the ship is reimplemented for training. The agent writes to
`Ship.command_thrust` / `Ship.command_turn`, the same seam the mouse and
keyboard feed, and everything downstream — gravity, flight assist, buoyancy,
engine spool, auto-levelling — is the shipping code. A policy trained against a
simplified copy of the ship would be a policy that cannot fly the real one.

```
python/ppo.py  <--- JSON over TCP, one exchange per decision --->  rl/train.tscn
   torch                                                    32 ships, 1 solar system
```

## Layout

| File | What it is |
| --- | --- |
| `rl/ship_task_env.gd` | Base for every control task: the agent pool, the clock, the action seam |
| `rl/landing_env.gd` | The landing task: spawns, observations, reward, termination |
| `rl/docking_env.gd` | The docking task, on the same base |
| `station.gd` | An orbiting structure with a port. Not a Planet — see the file |
| `rl/train_host.gd` | Headless host — loads `main.tscn`, strips the player, serves the socket |
| `rl/train.tscn` | The scene the trainer runs |
| `rl/neural_pilot.gd` | MLP forward pass in GDScript, loads `policy.json` |
| `rl/autopilot.gd` | The autopilot stack: analytic transit, then the policy for descent (`[P]`) |
| `python/godot_env.py` | Client for the socket protocol, Gym-shaped |
| `python/ppo.py` | The trainer |
| `python/evaluate.py` | Scores a controller against two baselines |
| `python/smoke_test.py` | Checks the bridge before you spend an hour on a run |
| `rl/cruise_check.tscn` | Checks that the assist can hold a velocity between bodies |
| `rl/journey_check.tscn` | Flies whole journeys end to end, planet to planet |

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

Expect roughly 2,300 transitions/second with 16 agents at one decision per
tick, and a random policy that crashes essentially every episode. With
`--repeat 6` that falls to about 470 transitions/second, but each one covers six
physics ticks, so the engine is doing slightly more work per second, not less.

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
costs throughput and nothing else.

Worth doing at least once. A learning curve is a lossy summary of a policy —
"mean return 61.6" does not tell you that the ship is arriving level but braking
late, or that it has found a hover and is farming the shaping term, or that a
third of the batch is tumbling. The batch strip is what makes the hover obvious:
a policy that has learned to land and one that has learned to wait look the same
through the chase camera for the first ten seconds and completely different as
sixteen altitude bars that stop going down.

### One host, several tasks

`ShipTaskEnv` owns what the tasks share — a pool of ships wired for external
control, the episode clock, the shaping baseline, and `apply_action`, which is
identical everywhere because every task flies the same six commands. A task
supplies a *reference target* and a *terminal condition*: a planet surface for
landing, a station's port for docking, and later an orbit for a transfer. Pick
one with `--task` (default `landing`); the registry is `_make_env` in
`train_host.gd`, one line per task.

The engine **describes the task on connect**, before any data flows:

```json
{"type": "hello", "task": "landing", "obs_size": 20, "act_size": 6,
 "num_agents": 32, "repeat": 6, "max_steps": 27000,
 "outcomes": {"1": "landed", "2": "crashed", ...}}
```

There are two tasks now — `landing` and `docking` — and `ppo.py` builds its
network to whichever width the engine reports, taking its log labels from those
names. Both used to be constants in `godot_env.py` — fine with one task, and
silently wrong with two: the network would be allocated at the wrong shape and
the logs would label outcomes belonging to a task nobody was running. The
handshake also cross-checks `num_agents` and `repeat`, which are otherwise told
to each side separately and desynchronise the protocol a few frames later with
an unhelpful message.

`policy.json` records `"task"` alongside `"repeat"`, and `NeuralPilot` refuses a
policy trained for another one. A mismatched policy does not fail on its own —
it produces six numbers in range either way and flies something it has never
seen. Policies predating the field are read as landers, which they all were.

### The docking task

Second on `ShipTaskEnv`, and the first where contact is the point rather than
something to avoid. Landing ends when a distance to an analytic surface reaches
zero; docking ends when two *oriented* faces meet, at a speed and an alignment
that would survive a latch.

**Observation** (22 floats, `DockingEnv.build_observation`) — port-to-port
offset, relative velocity, the port's outward normal, alignment, and the
corridor decomposition (`axial`, how far in front of the port face; `lateral`,
how far off the centreline), all in the ship's own frame. Plus spin, throttle,
progress, fuel, and local gravity — the station orbits a planet and flight
assist is trimming against *that planet*, not against the station, so the policy
is flying inside a field it did not choose and should be able to see it.

**Outcomes** — `docked`, `collided` (hit the hull), `botched` (reached the port
face out of tolerance), `drifted`, `timeout`. Botching and colliding are tracked
apart because they are different mistakes: one is nearly right and the other is
arriving at the wrong part of the structure.

**Scored against the hand-written baseline**, which is the point of having one:

| Controller | Docked | Mean return |
| --- | --- | --- |
| no input (assist only) | 0.0% | −226.9 |
| hand-written baseline | **76.7%** | +59.7 |

The baseline's remaining failures are the spawns that start behind the station;
it has no route-around, only a push back out in front of the port face.

```bash
python evaluate.py --godot ... --project .. --task docking --controller baseline
python ppo.py --godot ... --project .. --task docking --out ../rl/runs/docking
```

#### Four things the baseline had to learn the hard way

All four were found by measuring, and all four are properties of the *plant*
rather than of the controller — a learned policy meets every one of them.

**Flight assist trims against the wrong body.** Near a station the dominant body
is the planet it orbits, so the assist holds station over the *ground*, not over
the station. That is a standing disturbance of about a unit a second, and a
proportional law settles where its command cancels it. It is why the lateral
error will not go to zero and why the closing gain has to be high.

**Centring and closing must not compete.** The first version blended the aim
point between the port and a standoff. The two commands fought, and the lateral
error settled at exactly whatever threshold the blend used — 0.41 units for one
value, 0.76 for another — with the ship parked there holding a perfect corridor
until the clock ran out. Correcting lateral at full authority *always* and
throttling only the approach has no such fixed point.

**A standoff must not track the ship.** Aiming at `max(axial, safe)` meant the
target followed the ship outward, so nothing pulled it back: it drifted from 20
units to 38 over two minutes while the controller commanded nothing about it.

**Proportional laws stop short.** Everything above tapers to nothing on arrival,
so the approach parked at a range of 2.4 units — alignment −1.00, closing 0.41 —
four tenths short of a 2.0-unit gate. A constant creep over the last few metres
closes it and cannot wind up the way an integral term would.

### Decision rate

`--repeat N` holds each action for N physics ticks and exchanges once per hold,
summing the rewards in between. The ship is *simulated* at 60 Hz throughout;
only the rate at which it is *steered* changes.

This is not a performance knob, it is the discount horizon. At gamma 0.997 the
agent sees `1/(1-gamma) = 333` decisions ahead. At one decision per tick that is
5.6 seconds against episodes running 18 to 54 — so the +120 landing bonus
reaches the start of an episode discounted to `120 * 0.997^2000 = 0.3` reward,
while shaping is worth ~39 over the same episode and is paid densely and early.
The agent was never weighing 39 against 120; it was weighing 39 against 0.3, and
could not see the landing at all. `--repeat 6` puts a whole episode inside the
horizon.

The cost is close to nothing: the hand-written controller lands 92.5% of the
time deciding at 10 Hz against 93.0% at 60 Hz. It is a proportional law, and
proportional laws do not need sixty corrections a second.

A policy records what it was trained at (`"repeat"` in `policy.json`) and
`ShipAutopilot` holds actions to match, so a 10 Hz policy is not silently driven
at 60 Hz in game.

### Warm starts

Cloning leaves a near-deterministic policy and PPO then samples around it with
whatever `log_std` holds — by default `exp(-1) = 0.37`, against actions bounded
to [-1, 1]. That is noise of the same order as the commands themselves, which
does not explore *around* a precision landing so much as replace it, and the
first update then learns from the wreckage. `--warm-std 0.08` is the fix.
`--critic-warmup 15` helps too, for the same underlying reason: a cloned actor
arrives competent and the critic arrives random.

Score the result against both baselines:

```bash
python evaluate.py --godot /path/to/Godot.exe --project .. --controller zero
python evaluate.py --godot /path/to/Godot.exe --project .. --controller baseline
python evaluate.py --godot /path/to/Godot.exe --project .. --controller policy
```

Then fly it: get in the ship, pick a destination with `[Tab]`, press `[P]`.
It crosses the system on analytic guidance and hands over to the policy once
inside the destination's influence; the HUD names the phase.

## The task

**Observation** (20 floats, `LandingEnv.build_observation`) — everything is
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
| `19` | fuel remaining as a fraction of the tank, 1.0 when unlimited |

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
influence, or after `max_steps` ticks — 27,000, which is 450 simulated seconds.
That number is not a detail; see **The time limit was setting the score** below.

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

**The time limit was setting the score.** `max_steps` was 9,000 ticks — 150
seconds. Tripling it took the hand-written controller from **92.5% to 99.0%**
and its timeouts from 6.5% to zero. Every one of those thirteen episodes became
a landing and not one became a crash or a ditch: they were far-side starts that
needed longer than 150 seconds to cross the planet, cut off in level flight.

Three separate PPO runs had all plateaued at exactly 92%, which read as a
control ceiling and was in fact the clock. Nobody was failing to land, they were
failing to arrive before the buzzer, and roughly seven points of the benchmark
were unreachable for reasons that had nothing to do with control. Any comparison
between controllers measured under that limit was partly a comparison of who ran
out of clock.

The general form: **an environment constant that truncates episodes is part of
the metric, not part of the scenery.** If a controller's failures are dominated
by one outcome class, vary the constant that produces it before concluding
anything about the controller.

**Value clipping was freezing the critic.** PPO's value loss never converged —
54, then 20, then back to 43, for 600k steps. The cause was one line:

```gdscript
clipped_value = b_val + (new_value - b_val).clamp(-args.clip, args.clip)
```

`args.clip` is the *policy ratio* clip, 0.2. Applied as an absolute bound on a
value prediction in return units running to ~130 with errors around 50, it is a
leash 250 times shorter than the error — and `clamp` has no gradient once
saturated, so any time the critic wanted to move more than 0.2 it received
nothing at all. The policy was being handed advantages computed from a critic
that could not learn.

`ValueNorm` normalises the value targets so the critic works in units of 1,
where 0.2 is a sensible step and the clip does the job it was written for. Value
loss went from 43.29 to **0.01** — the fraction of return variance it fails to
explain, so it now accounts for about 99% of it.

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

200 episodes each, policies at the distribution mean, decisions at 10 Hz
(`--repeat 6`). The task is **reach the planet's largest continent from anywhere
and land on it**, so a landing counts only on the target landmass.

The time limit is reported alongside every number, because it turned out to be
part of the measurement rather than part of the setup:

| Controller | 9,000 ticks | 27,000 ticks | 27,000 + 120s fuel |
| --- | --- | --- | --- |
| Hand-written baseline | 92.5% | **99.0%** | **92.5%** |
| Best PPO policy | 92.0% | 98.0% | 92.0% |

At the old limit both controllers lose ~6.5% of episodes to the clock and
nothing else. At 27,000 the expert's failures fall to one crash and one ditch in
200. Under a fuel budget both give back about six points.

**PPO no longer destroys a good initialisation — but it has never beaten the
expert.** It ties or loses in every configuration above. What changed is the
failure mode, not the ranking:

| | before | after |
| --- | --- | --- |
| PPO on a clone | 97% -> **2.5%** | 80% -> **92.5%** |
| return vs landing rate | moved *opposite* | correlate **+0.41** |

The second row is the one that matters. Training return climbing while landings
collapsed was the signature of a reward the agent could not see the important
half of; with the horizon fixed, the two move together and the gradient points
somewhere worth going. That is the difference between a broken setup and a
working one that has not yet earned its keep.

**Beware the peak.** A run's best iteration is not its score. The landing-rate
plateau has mean 92.0% and sd 1.6%, against binomial noise of 1.9% on a
200-episode window — so the spread is entirely sampling error, and taking the
maximum over 73 such windows reliably overstates by two or three points. One run
peaked at 94.0% and evaluated deterministically at 92.0%. `policy_best.json`
selects on that noisy statistic, so it means "luckiest window", not "best
policy"; always re-measure with `evaluate.py`.

**Fuel buys economy, not reliability.** Trained against a 120-second budget, the
policy lands on 22.4s median fuel against the expert's 25.6s and 55.6s worst
against 61.0s — a real ~12% efficiency gain, and the one dimension where
learning beats the controller. It pays for it with more ditches (7.5% vs 6.0%).
Since no reward term mentions fuel, that trade was not asked for; it emerged.
Worth knowing, not worth calling a win.

**Cloning alone plateaus; DAgger fixes it.** Plain behaviour cloning stuck at
83% however much data or capacity it was given — at one point action MSE
improved 2.5x with *no* change in landing rate, and the failures merely moved
from ditching to hovering. That is compounding error, not underfitting: fitting
the expert's actions on the expert's own states teaches nothing about the states
the learner's small errors actually lead to. Four rounds of DAgger (Ross, Gordon
& Bagnell 2011) — the learner flies, the expert labels what the learner reaches
— closed it to within a point of the expert.

**One expert change was made purely for the learner's benefit.** The
descend-or-cruise decision started as a hard boolean, which works fine as a
controller and clones badly: a network fitted by least squares cannot reproduce
a step, so it learns a smeared version and half-commits. Making the expert
smooth — a continuous `ready` blend instead of a branch — cost the controller
about four points and cut clone MSE by 2.5x. **If a policy is going to be
imitated, the demonstrator's own discontinuities are a design decision.**

**Training on the correct horizon made things worse, and I cannot explain it.**
A policy trained at 9,000 ticks and merely *evaluated* at 27,000 scores 98.0%,
despite reading an episode-progress input that is three times out of calibration.
The same recipe trained at 27,000 scores 92.0%, with real crashes and ditches
rather than timeouts. The likeliest story is that a 3x longer clock removes the
pressure to be efficient and yields fewer terminal events per 600k steps, so
less landing signal — but that is a guess, and it is the most anomalous result
here. See `TODO.md`.

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
- **Reinforcement learning has not yet paid for itself.** The best PPO policy
  matches the hand-written controller and does not beat it, in any configuration
  measured. What ships is competitive, not superior, and the honest description
  of the pipeline is "imitation, with RL that no longer hurts".
- **`policy_best.json` selects on a noisy statistic** — the best of ~73
  rolling-window landing rates, each with about 1.9% binomial noise. It
  overstates by two or three points. Treat it as a checkpoint to evaluate, never
  as a result.
- **Fine-tuning peaks around iteration 20 and then erodes**, giving back ten
  points over the rest of a 600k-step run. Fixing the critic raised the peak but
  did not remove the erosion.
