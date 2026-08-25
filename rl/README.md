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
`progress.csv`. Score the result against both baselines:

```bash
python evaluate.py --godot /path/to/Godot.exe --project .. --controller zero
python evaluate.py --godot /path/to/Godot.exe --project .. --controller baseline
python evaluate.py --godot /path/to/Godot.exe --project .. --controller policy
```

Then fly it: get in the ship, get near a planet, press `[P]`.

## The task

**Observation** (14 floats, `LandingEnv.build_observation`) — everything is
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

**Action** (6 floats in [-1, 1]) — three translation axes and three rotation
axes. Exactly the control set the player has.

**Reward** — potential-based shaping on altitude, surface-relative speed and
belly-down alignment, plus a graded touchdown bonus up to 120. An episode ends
at touchdown, on leaving the sphere of influence, or after 1,500 ticks.

## Things that were true and inconvenient

**The ship cannot land without flight assist.** Thrust is 10.0 units/s² and
surface gravity is 9.81 — a thrust-to-weight ratio of **1.02**. Pointing
straight up at full power decelerates the ship at 0.19 units/s², so killing 30
units/s of descent would take 158 seconds and about 2,400 units of altitude,
more than a planet radius. The assist's drag term (`-drift * 1.6 * mass`) is
what actually lands this ship; it caps free fall near `g / 1.6 = 6.1` units/s.
Training with assist off, which was the original plan, is training on a task
with no solution. It is left on, and the policy's job is to descend under
control and brake below 2.0 before contact — still three times under what the
assist gives you for free.

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
