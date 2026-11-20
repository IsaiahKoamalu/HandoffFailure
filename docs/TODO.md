## Step 12: the real landing task, and why it does not train yet

Step 11 (policy export and the engine check) is finished and committed at `2e5c823`.

**Status.** Everything below was built and tested in a scratch copy of the repo between
Nov 11 and Nov 19. It runs: it parses, the smoke test passes, the reward-scale test
passes, and training runs end to end with an engine-verified export. **It has not yet
learned to land.** Four training runs all converged on hovering until the time limit.
The findings are in "What training showed", and the next experiment is in 12g.

Nothing here is in the repo's code yet. This file is the only copy.

---

### What training showed

All runs: 32 ships, `--repeat 6`, 3 million steps, 27,000-tick limit (4,500 decisions).

| Run | Spawn | γ | Timeout | Result |
| --- | --- | --- | --- | --- |
| 1 | full (0.2–1.0 R, up to 80% orbital speed) | 0.99 | truncated | 0 landings. After ~150k steps, ~90% of episodes time out: it hovers. |
| 2 | full | 0.999 | truncated | Same. 0 landings, ~90% timeouts. |
| 3 | easy (0.05–0.2 R, no orbital speed) | 0.999 | truncated | ~1% landed in the first 370k steps, then hovered: 81–100% timeouts. |
| 4 | easy | 0.999 | **failure, −100**; crash and ditch softened to −50 | Last 500k steps: 84% timeout, 12% crash, **0 landed**. |

What each result rules in or out:
- **γ = 0.99 is too short-sighted for this task.** Its effective horizon is 1/(1−γ) = 100
  decisions = 10 s. A landing 500 decisions away is worth 100 × 0.99⁵⁰⁰ ≈ 0.7 today.
  Use **γ = 0.999** (horizon ~1,000 decisions). It was necessary but not sufficient.
- **Exploration is not the main problem.** Run 3 landed sometimes and still converged on
  hovering.
- **Making the timeout a failure did not help (run 4).** The −100 arrives 4,500 decisions
  after the start, and 100 × 0.999⁴⁵⁰⁰ ≈ 1 today. Discounting hides any penalty at the
  far end of a long hover, while a crash costs −50 immediately. The change is principled
  and is kept (see 12b), but it doesn't address the cause.
- **The likely cause is shaping that is far too weak near the ground.** It's scaled per
  planet radius and per orbital speed. From the easy start (9–36 units up), descending
  all the way earns at most ~2 points of altitude shaping and slowing to touchdown speed
  ~0.5, against a crash cost of 50. Nothing in the reward rewards getting better at
  descending once the agent has learned that the ground hurts. **12g tests a stronger
  potential.**

Speed:
- **The code runs at about 4,000–4,500 steps/s,** about 13 minutes per 3 million steps.
  Godot's physics is ~90% of each iteration, and the PPO update takes 0.2 s.
- **Runs 1 and 2 took 4 and 9 hours** because the Mac slept or throttled (on battery, lid
  closed). `caffeinate -i` doesn't stop sleep from closing the lid. Long runs belong on
  the server.

---

### 12a. `rl/tasks/landing_env.gd`: the whole file

This is the version used in run 4: full spawn distribution, timeout as failure, crash and
ditch at −50.

```gdscript
extends RefCounted
class_name LandingEnv

## Descent and landing: from anywhere between a fifth of a radius and a full
## radius above a planet, possibly carrying most of an orbit's sideways speed, to
## a slow, upright touchdown on dry ground on that planet's largest continent.
##
## The task owns everything task-specific: where an episode starts, what the
## policy sees, what it is rewarded for, and what ends it. The host owns the
## socket and the clock.
##
## Every constant below is part of the task definition and freezes with it once
## E1 is measured. Changing one afterwards is reported as its own comparison.

const TASK_NAME := "landing"

# --- Outcomes ----------------------------------------------------------------

const RUNNING: int = 0
const LANDED: int = 1
## Touched down too fast or too tilted.
const CRASHED: int = 2
## Came down in water. A separate outcome from CRASHED because it is a separate
## mistake: the flying can be fine and the place wrong.
const DITCHED: int = 3
## Landed cleanly on dry ground, but not on the target continent.
const MISSED: int = 4
const ESCAPED: int = 5
const TIMEOUT: int = 6
const OUTCOMES := {
	RUNNING: "running",
	LANDED: "landed",
	CRASHED: "crashed",
	DITCHED: "ditched",
	MISSED: "missed",
	ESCAPED: "escaped",
	TIMEOUT: "timeout",
}
const SUCCESS: int = LANDED

# --- Spawn distribution ------------------------------------------------------
# This is the landing skill's training distribution, and so the thing E3
# measures handoffs against. It has to cover what orbit insertion delivers.

const SPAWN_ALT_MIN: float = 0.2 # planet radii
const SPAWN_ALT_MAX: float = 1.0
## Sideways speed at spawn, as a fraction of circular orbital speed at that height.
const SPAWN_ORBIT_MAX: float = 0.8
## Downward speed at spawn, units/s.
const SPAWN_SINK_MAX: float = 5.0
## Largest tilt away from belly-down at spawn.
const SPAWN_TILT_MAX: float = deg_to_rad(30.0)

# --- Touchdown and termination ----------------------------------------------

const TOUCHDOWN_SPEED: float = 2.0 # units/s relative to the ground
## Cosine of the largest tilt that still counts as upright, about 25 degrees.
const UPRIGHT_MIN: float = 0.9
const ESCAPE_ALT: float = 4.0 # planet radii

# --- Reward ------------------------------------------------------------------
# Terminal rewards, a small charge per physics tick, and potential-based
# shaping (Ng, Harada and Russell 1999). The host applies the shaping as
# gamma * potential(after) - potential(before) per decision, with the potential
# of an ending taken as zero, so the shaping cannot change the optimal policy.

const LANDED_BONUS: float = 100.0
const CRASHED_PENALTY: float = 50.0
const DITCHED_PENALTY: float = 50.0
const ESCAPED_PENALTY: float = 100.0
const MISSED_PENALTY: float = 20.0
## Running out the clock is a failure, as bad as escaping: the policy sees the
## clock in its `progress` input, so the time limit is part of its world and
## ending there is a real ending (Pardo et al. 2018, Time Limits in RL).
const TIMEOUT_PENALTY: float = 100.0
const TIME_COST: float = 0.0005 # per physics tick

const POTENTIAL_ALTITUDE: float = 10.0 # per planet radius
const POTENTIAL_SPEED: float = 10.0 # per circular orbital speed
const POTENTIAL_TILT: float = 5.0 # per unit of (1 - upright)
const POTENTIAL_SITE: float = 20.0 # from the antipode to the target

# --- Observation -------------------------------------------------------------

const OBS_SIZE: int = 18
const OBS_NAMES: Array[String] = [
	"altitude", "altitude_near",
	"velocity_right", "velocity_up", "velocity_back",
	"up_right", "up_up", "up_back",
	"spin_pitch", "spin_yaw", "spin_roll",
	"land_below",
	"site_right", "site_up", "site_back", "site_distance",
	"throttle", "progress",
]
## Altitude at which altitude_near is halfway to saturating, units.
const NEAR_SCALE: float = 20.0
## Angular speed that reads as 1.0, rad/s.
const SPIN_SCALE: float = 2.0

var ships: Array[Ship] = []
var targets: Array[Planet] = []
var _landable: Array[Planet] = []
var _rng: RandomNumberGenerator


func _init(world: SolarSystem, agent_ships: Array[Ship], rng: RandomNumberGenerator) -> void:
	assert(OBS_NAMES.size() == OBS_SIZE, "OBS_NAMES and OBS_SIZE disagree")
	ships = agent_ships
	_rng = rng
	for planet in world.Planets:
		if planet.largest_landmass_center != Vector3.ZERO:
			_landable.append(planet)
	targets.resize(ships.size())


## The largest total a maximum-length episode can be charged per tick.
static func max_episode_charge(max_steps: int) -> float:
	return TIME_COST * max_steps


## The mildest way to fail. Per-tick charges must stay below it, or running out
## the clock is a cheaper way to fail than trying and missing.
static func smallest_failure_penalty() -> float:
	return minf(minf(minf(CRASHED_PENALTY, DITCHED_PENALTY), minf(ESCAPED_PENALTY, MISSED_PENALTY)), TIMEOUT_PENALTY)


## Start ship `i` on a new episode.
func reset(i: int) -> void:
	var ship := ships[i]
	var planet := _landable[_rng.randi_range(0, _landable.size() - 1)]
	targets[i] = planet

	var up := Vector3(_rng.randfn(), _rng.randfn(), _rng.randfn()).normalized()
	var radius := planet.surface_radius * (1.0 + _rng.randf_range(SPAWN_ALT_MIN, SPAWN_ALT_MAX))
	var position := planet.global_position + up * radius

	# A random direction along the ground, for both heading and sideways motion.
	var east := up.cross(Vector3.UP)
	if east.length_squared() < 1e-6:
		east = up.cross(Vector3.RIGHT)
	east = east.normalized()
	var north := up.cross(east)
	var angle := _rng.randf_range(0.0, TAU)
	var along := east * cos(angle) + north * sin(angle)

	var orbital_speed := sqrt(planet.standard_gravitational_parameter / radius)
	var sideways := along * orbital_speed * _rng.randf_range(0.0, SPAWN_ORBIT_MAX)
	var sink := -up * _rng.randf_range(0.0, SPAWN_SINK_MAX)

	var heading := _rng.randf_range(0.0, TAU)
	var facing := east * cos(heading) + north * sin(heading)
	var basis := Basis.looking_at(facing, up)
	var tilt_axis := (east * cos(angle + 1.0) + north * sin(angle + 1.0)).normalized()
	basis = Basis(tilt_axis, _rng.randf_range(0.0, SPAWN_TILT_MAX)) * basis

	ship.global_transform = Transform3D(basis, position)
	ship.linear_velocity = planet.get_surface_velocity(position) + sideways + sink
	ship.angular_velocity = Vector3.ZERO
	ship.throttle = 0.0
	ship.command_thrust = Vector3.ZERO
	ship.command_turn = Vector3.ZERO
	ship.refuel()


## Score ship `i` for one physics tick. Returns [reward, outcome]; RUNNING means
## the episode goes on. Shaping is not included: the host adds it per decision.
func evaluate(i: int) -> Array:
	var ship := ships[i]
	var planet := targets[i]
	var frame := Surface.surface_frame(ship.global_position, planet)
	var altitude: float = frame[0]
	var up: Vector3 = frame[1]

	if altitude > ESCAPE_ALT * planet.surface_radius:
		return [-ESCAPED_PENALTY, ESCAPED]
	if ship.is_submerged:
		return [-DITCHED_PENALTY, DITCHED]
	if altitude > 0.0:
		return [-TIME_COST, RUNNING]

	if not Surface.is_over_land(ship.global_position, planet):
		return [-DITCHED_PENALTY, DITCHED]
	var relative := ship.linear_velocity - planet.get_surface_velocity(ship.global_position)
	if relative.length() > TOUCHDOWN_SPEED or ship.global_basis.y.dot(up) < UPRIGHT_MIN:
		return [-CRASHED_PENALTY, CRASHED]
	if not Surface.is_on_target_site(ship.global_position, planet):
		return [-MISSED_PENALTY, MISSED]
	return [LANDED_BONUS, LANDED]


## How close ship `i` is to a landing, as a number that rises toward zero.
func potential(i: int) -> float:
	var ship := ships[i]
	var planet := targets[i]
	var frame := Surface.surface_frame(ship.global_position, planet)
	var up: Vector3 = frame[1]
	var altitude := clampf(float(frame[0]) / planet.surface_radius, 0.0, ESCAPE_ALT)
	var relative := ship.linear_velocity - planet.get_surface_velocity(ship.global_position)
	var speed := relative.length() / Surface.speed_scale(planet)
	var tilt := 1.0 - ship.global_basis.y.dot(up)
	var site: float = Surface.site_frame(ship.global_position, planet)[0] / PI
	return -(POTENTIAL_ALTITUDE * altitude + POTENTIAL_SPEED * speed
		+ POTENTIAL_TILT * tilt + POTENTIAL_SITE * site)


func observe(i: int, progress: float, into: PackedFloat32Array, at: int) -> void:
	build_observation(ships[i], targets[i], progress, into, at)


## The one definition of what the policy sees. Static because the in-game
## autopilot builds it too, and the two must never disagree. Everything is in
## the ship's own frame, so one policy works on every planet and heading.
static func build_observation(
	ship: Ship, planet: Planet, progress: float, into: PackedFloat32Array, at: int
) -> void:
	var frame := Surface.surface_frame(ship.global_position, planet)
	var altitude: float = frame[0]
	var up: Vector3 = frame[1]
	var to_ship := ship.global_basis.inverse()
	var relative := ship.linear_velocity - planet.get_surface_velocity(ship.global_position)
	var velocity := to_ship * relative / Surface.speed_scale(planet)
	var up_local := to_ship * up
	var spin := to_ship * ship.angular_velocity / SPIN_SCALE

	var site := Surface.site_frame(ship.global_position, planet)
	var toward: Vector3 = site[1]
	if toward.length_squared() > 1e-12:
		toward = (to_ship * toward).normalized()

	into[at + 0] = clampf(altitude / planet.surface_radius, -0.5, ESCAPE_ALT)
	into[at + 1] = tanh(altitude / NEAR_SCALE)
	into[at + 2] = velocity.x
	into[at + 3] = velocity.y
	into[at + 4] = velocity.z
	into[at + 5] = up_local.x
	into[at + 6] = up_local.y
	into[at + 7] = up_local.z
	into[at + 8] = spin.x
	into[at + 9] = spin.y
	into[at + 10] = spin.z
	into[at + 11] = tanh(4.0 * Surface.land_signal_at(ship.global_position, planet))
	into[at + 12] = toward.x
	into[at + 13] = toward.y
	into[at + 14] = toward.z
	into[at + 15] = float(site[0]) / PI
	into[at + 16] = ship.throttle
	into[at + 17] = progress
```

Design notes:
- **Six outcomes, not two.** "Crashed", "ditched" and "missed" are different mistakes:
  bad flying, the wrong surface, the wrong continent. Separating them makes the logs
  readable and lets E3 ask which kind of failure a strange handoff causes.
- **The spawn is the training distribution E3 measures against.** It has to cover what
  orbit insertion delivers, which is why it spawns up to 80% of orbital speed sideways
  rather than starting gently above the pad.
- **Touchdown is altitude ≤ 0 on the analytic surface,** as before. Training ships have
  no terrain colliders, and `Surface.surface_frame` uses the same height function the
  terrain mesh is built from. `ship.is_submerged` catches water before the analytic
  ground does.
- **`potential` is ≤ 0 and rises toward 0 at a landing.** The host turns it into shaping
  (12b). It's scaled per radius and per orbital speed, which is what 12g changes.
- **The observation is entirely in the ship's frame and normalized by planet-sized
  units,** so one policy works on every planet and heading. `progress` is what makes the
  time limit observable, and so a legitimate ending.
- **`observe` now takes `progress`,** because the host owns the step counter.
- **`max_episode_charge` and `smallest_failure_penalty`** exist for the reward-scale
  check (12b, 12d).

---

### 12b. `rl/train/host.gd`: changes

**New variables**, with the others:
```gdscript
## The trainer's discount per decision. Shaping uses it, so it must match.
var _gamma: float = 0.99
```
```gdscript
## Per ship: the task's potential when the decision in progress began.
var _phi_start := PackedFloat32Array()
```

**In `_hello`, after `_max_steps = ...`:** read `gamma`, then run the reward-scale check
**before building anything**, so a refused `hello` leaves the host ready for another:
```gdscript
	_gamma = float(message.get("gamma", _gamma))

	# Before building anything: a run whose per-tick charges can outweigh its
	# mildest failure would teach the agent to quit, so it does not start.
	var charge := LandingEnv.max_episode_charge(_max_steps)
	var penalty := LandingEnv.smallest_failure_penalty()
	if charge >= penalty:
		_send({"type": "error", "reason": "reward scale: a %d-tick episode can be charged %.2f, not less than the smallest failure penalty %.2f"
			% [_max_steps, charge, penalty]})
		return
```
Add `_phi_start.resize(count)` with the other resizes. Add to the `hello` reply:
```gdscript
		"gamma": _gamma,
		"max_episode_charge": charge,
		"smallest_failure_penalty": penalty,
```

**In `_step`, at the end of the loop that sets each ship's commands:**
```gdscript
		_phi_start[i] = _task.potential(i)
```

**In `_tick`, the timeout branch becomes a real ending:**
```gdscript
		if outcome == LandingEnv.RUNNING and _steps[i] >= _max_steps:
			# The task observes the clock, so its time limit is a real ending.
			_terminated[i] = true
			_reward[i] -= LandingEnv.TIMEOUT_PENALTY
			outcome = LandingEnv.TIMEOUT
```
**Then, where the episode ends, before `_outcome[i] = outcome`:** close the shaping, and
pass `progress` to `observe`:
```gdscript
		# Close the episode's shaping: a real ending has potential zero; a time
		# limit that does not end anything would keep its own.
		var phi_end := _gamma * _task.potential(i) if _truncated[i] else 0.0
		_reward[i] += phi_end - _phi_start[i]
```
```gdscript
		_task.observe(i, _progress(i), final, 0)
```

**In `_finish_step`, before the loop that resets finished ships:**
```gdscript
	for i in ships.size():
		if not _done[i]:
			_reward[i] += _gamma * _task.potential(i) - _phi_start[i]
```

**In `_observe_all`:** `_task.observe(i, _progress(i), row, 0)`. Add the helper:
```gdscript
## How far ship `i` is through its episode, 0 at the start and 1 at the time limit.
func _progress(i: int) -> float:
	return float(_steps[i]) / float(_max_steps)
```

Why it's built this way:
- **Shaping is applied per decision, as γΦ(after) − Φ(before),** the exact form Ng et al.
  prove can't change the optimal policy. It's done in the host rather than the task
  because only the host knows where decisions begin and end. The γ has to be the
  trainer's γ, which is why it now travels in `hello`.
- **The potential of an ending is 0.** That's what the proof requires.
- **The logged episode returns include shaping, undiscounted.** In runs 1–3, timeout
  episodes logged +400 to +900. That isn't a bug: with Φ < 0 and γ < 1, each surviving
  decision gains (1−γ)|Φ|, and that sum telescopes away once discounted. **Judge runs by
  the outcome rates, not by `episodic_return`.**
- **The `_truncated` branch of `phi_end` is now never taken for landing,** because the
  timeout is terminated. It stays correct for any future task that doesn't observe the
  clock.

---

### 12c. Python changes

**`python/godot_env.py`:** add `gamma=0.99` to `__init__` (before `port`), send
`"gamma": gamma` in `hello`, and add `("gamma", gamma)` to the tuple of echoed values
that are checked.

**`python/launch.py`:** `connect(host, num_envs, seed, repeat, max_steps, gamma=0.99,
timeout=30.0)`, passing `gamma` on to `GodotVectorEnv(...)`.

**`python/ppo.py`:**
- Pass `gamma=args.gamma` to `connect(...)`.
- Change the default `gamma` in `Args` to **0.999** (all useful runs used `--gamma 0.999`).
- Add `flush=True` to the `iter` print, or a log file shows nothing until the run ends.

---

### 12d. `python/tests/test_reward_scale.py`

```python
"""Unit test for the reward scale: quitting must never be the cheapest move.

A maximum-length episode must not be able to cost more in per-tick charges than
the smallest failure penalty, or an agent learns that running out the clock is a
cheaper way to fail than trying. The earlier implementation got this wrong:
0.02 per tick over 18,000 ticks is 360, against a capture bonus of 150.

The check lives in the host, which refuses a hello whose episode length breaks
it. This test asks the real host twice: once with an episode length that must be
refused, and once with the training length, which must be accepted.

    uv run python python/tests/test_reward_scale.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bridge import Bridge
from launch import launch_host

TRAINING_MAX_STEPS = 27000
ABSURD_MAX_STEPS = 10**8


def connect_bridge(host, timeout=30.0):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return Bridge()
        except ConnectionRefusedError:
            if host.poll() is not None or time.monotonic() > deadline:
                raise
            time.sleep(0.2)


def main():
    host = launch_host()
    try:
        bridge = connect_bridge(host)

        # The host checks before building the world, so a refused hello leaves it
        # ready for another.
        try:
            bridge.request({"type": "hello", "seed": 7, "ships": 1, "max_steps": ABSURD_MAX_STEPS})
        except RuntimeError as error:
            assert "reward scale" in str(error), error
        else:
            raise AssertionError(f"a {ABSURD_MAX_STEPS}-tick episode was accepted")

        hello = bridge.request({"type": "hello", "seed": 7, "ships": 1, "max_steps": TRAINING_MAX_STEPS})
        charge, penalty = hello["max_episode_charge"], hello["smallest_failure_penalty"]
        assert charge < penalty, (charge, penalty)
        print(f"{TRAINING_MAX_STEPS}-tick episode: at most {charge:.2f} in charges, "
              f"smallest failure penalty {penalty:.2f}")

        bridge.request({"type": "close"})
        bridge.close()
        host.wait(timeout=10)
    finally:
        if host.poll() is None:
            host.kill()
    print("PASS")


if __name__ == "__main__":
    main()
```
- **It tests both directions:** a 100-million-tick episode must be refused, and the real
  27,000-tick one accepted. A check that never refused anything would pass the second
  half alone.
- **It talks to the real host through `Bridge`,** because the check lives in GDScript.
- Tested result: `27000-tick episode: at most 13.50 in charges, smallest failure penalty
  20.00`, then `PASS`.

---

### 12e. Docs that change with it

- **`docs/protocol.md`:** `hello` gains `gamma` in the request, and `gamma`,
  `max_episode_charge` and `smallest_failure_penalty` in the reply. A refused `hello`
  ("reward scale: …") belongs in the Errors section. The `terminated`/`truncated` section
  should say that a task which observes its clock may end at the time limit as
  *terminated*, which landing now does.
- **`ROADMAP.md`:** the truncation item says "A time-limit cutoff is not a terminal
  state". That stays true in general; add that landing observes its clock and treats the
  limit as a real ending, and why. Tick the reward-scale test.
- **`rl/runs/README.md`:** record the four runs above as the first Measurements entry.
  The hovering result is a real finding for the progress report.

---

### 12f. Running it

```bash
"$GODOT" --headless --path . --import        # LandingEnv changed
uv run python python/tests/test_reward_scale.py
uv run python python/env_test.py
uv run python python/ppo.py --total-timesteps 3000000 --gamma 0.999 --exp-name landing
```
Expected smoke test (tested with the full spawn): random actions mostly crash and ditch,
landing under 1%. The full training run reproduces run 1/2: hovering.

For a quick experiment, the **easy spawn** used in runs 3 and 4 is three constants:
`SPAWN_ALT_MIN = 0.05`, `SPAWN_ALT_MAX = 0.2`, `SPAWN_ORBIT_MAX = 0.0`. Change them back
before any run that counts.

---

### 12g. Next experiment (not yet run): stronger shaping near the ground

Replace the altitude and speed terms of `potential` so they are sized like the terminal
rewards and steep where a landing is decided. Everything else stays as in run 4 (easy
spawn, γ = 0.999, timeout as failure).

```gdscript
const POTENTIAL_ALTITUDE: float = 50.0
const POTENTIAL_SPEED: float = 50.0
## Altitude and speed at which their terms are about three-quarters spent.
const POTENTIAL_ALTITUDE_SCALE: float = 50.0 # units
const POTENTIAL_SPEED_SCALE: float = 10.0 # units/s
```
```gdscript
	var altitude := maxf(float(frame[0]), 0.0)
	var speed := relative.length()
	return -(POTENTIAL_ALTITUDE * tanh(altitude / POTENTIAL_ALTITUDE_SCALE)
		+ POTENTIAL_SPEED * tanh(speed / POTENTIAL_SPEED_SCALE)
		+ POTENTIAL_TILT * tilt + POTENTIAL_SITE * site)
```
- **Why:** from 36 units up, descending to the ground now earns up to ~30 points instead
  of ~2, and slowing from 10 units/s to 2 earns ~28 instead of ~0.5. The signal for
  "getting better at descending" becomes comparable to the crash cost.
- **Still potential-based,** so the best policy is unchanged. Only the learning signal
  is.
- **What to look for:** a landing rate that climbs after the first few hundred thousand
  steps instead of collapsing into timeouts. If it learns from the easy spawn, repeat at
  full difficulty. If it doesn't, the next candidates are an entropy bonus (`--ent-coef
  0.01`) and a curriculum that starts easy and widens.
- Run it on the server, inside `tmux`.
