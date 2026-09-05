extends ShipTaskEnv
class_name DockingEnv

## A vectorised docking environment wrapped around the real ship.
##
## The second task on `ShipTaskEnv`, and the first one where contact is the
## point rather than something to be avoided. Landing ends when a distance to an
## analytic surface reaches zero; docking ends when two *oriented* faces meet, at
## a speed and an alignment that would survive a latch.
##
## What that changes, and what it does not:
##
##   * **Not the control seam.** Same six commands, same ship, same dynamics.
##     `apply_action` is the base class's and is not overridden.
##   * **Not the shaping form.** Potential-based, same as landing, for the same
##     reason (Ng, Harada & Russell) — it can speed learning up but cannot bend
##     the agent toward a different task.
##   * **The reference frame.** Landing is expressed against a surface that is
##     everywhere below you. Docking is expressed against a single point with a
##     direction attached, so *where* the ship is relative to the corridor
##     matters as much as how far away it is. A ship one unit from the port and
##     approaching from behind the station has not nearly docked.
##
## Flight assist is left on, as in landing. Near a station the dominant body is
## usually the planet the station orbits, so the assist trims drift relative to
## that planet's surface — which is not the station's frame. That mismatch is
## real and the policy has to fly through it, exactly as a player would.

const OBS_SIZE: int = 25

# Episode outcomes. RUNNING is the base class's; these number upward from it.
const DOCKED: int = 1
## Hit the hull. A separate outcome from a bad dock because it is a separate
## mistake: arriving at the wrong part of the station at all.
const COLLIDED: int = 2
## Reached the port face out of tolerance — too fast, too crooked, or too far
## off centre. The flying was nearly right, which is worth telling apart from
## flying into the side of the structure.
const BOTCHED: int = 3
## Left the neighbourhood entirely.
const DRIFTED: int = 4
const TIMEOUT: int = 5

## Speed scale for observations and shaping, in units per second. A docking
## approach happens at single digits, so the orbital speed scale that normalises
## the landing observation — about 42 for these planets — would compress the
## entire useful range into a fifth of one channel.
const SPEED_SCALE: float = 20.0
## Rotation rate scale, radians per second.
const RATE_SCALE: float = 1.5

## Ships stepped in parallel share one station. They are independent rollouts,
## so the hull is put on the agents' own transparent layer for the duration:
## thirty-two ships queueing for one airlock is not the task. The env only ever
## runs headless under train_host, which has already removed the player, so
## nothing that can see the station is affected.
const AGENT_LAYER: int = 1 << 9

@export_category("Docking tolerances")
## Port-to-port distance that counts as contact.
@export var dock_radius: float = 2.0
## Speed of the ship's port relative to the station's port at contact.
@export var dock_speed: float = 1.0
## Dot product of the two port normals. They face each other, so a perfect
## alignment is -1 and the gate is a number close to it.
@export var dock_align: float = -0.9
## How far off the centreline contact may be.
@export var dock_lateral: float = 1.0
## How closely the two "up" axes must agree, as a dot product. Kept in step with
## Ship.dock_roll: the astronaut walks through the join, and a ship latched
## upside down relative to the deck would have gravity reverse in the doorway.
@export var dock_roll: float = 0.9

@export_category("Spawns")
## Spawn shell around the port, as multiples of the station's approach range.
@export var spawn_range_min: float = 0.3
@export var spawn_range_max: float = 2.5
## Fraction of episodes that start inside the approach corridor rather than
## anywhere on the sphere around the port.
##
## The same argument as LandingEnv.spawn_on_site_fraction: a purely uniform
## spawn makes the terminal few seconds rare, and those are the seconds that
## decide the outcome. Half and half keeps both halves of the job represented —
## finding the corridor, and flying it.
@export_range(0.0, 1.0, 0.05) var spawn_corridor_fraction: float = 0.5
## Fraction of episodes that start from a completely random attitude, tumbling.
@export_range(0.0, 1.0, 0.05) var spawn_tumble_fraction: float = 0.3
## Largest speed at spawn relative to the station, in units per second.
@export var spawn_speed: float = 12.0
## Episode ends as a failure past this many approach ranges from the port.
@export var drift_range: float = 6.0

@export_category("Reward")
## Weight on closing the port-to-port range, in the potential.
##
## These four are sized against `dock_bonus`, not chosen for feel. The per-step
## reward is 10x the change in the potential, so a whole approach from the top
## of the spawn shell is worth `10 * (P_end - P_start)` -- and the first version
## of this file weighted them 8/12/4/1, which made that swing about 1,000
## against a 120 bonus. A random policy scored -770 and the terminal event was
## a rounding error on the shaping.
##
## At these weights an approach is worth roughly 50, against the 120 for
## arriving cleanly: the shaping tells the agent which way to go and the bonus
## is still what it is going there for. That is the same ratio landing settled
## on -- about 39 of shaping against 120 -- and for the same reason.
@export var range_weight: float = 1.0
## Weight on being on the centreline rather than merely close. Larger than
## range_weight on purpose: a ship that closes without lining up arrives at the
## hull, and the shaping has to prefer the corridor to the straight line.
@export var lateral_weight: float = 1.5
## Weight on facing the port.
@export var align_weight: float = 0.6
## Weight on rolling the ship's up onto the station's. Half the facing weight:
## it is a real gate but a cheap one to satisfy, and over-weighting it makes the
## policy roll before it has finished pointing.
@export var roll_weight: float = 0.3
## Weight on not spinning. A ship that arrives perfectly while tumbling cannot
## latch, and without this the policy has no reason to stop the rotation it was
## spawned with.
@export var spin_weight: float = 0.3
## Charged for hitting the hull.
@export var collide_penalty: float = 40.0
## Charged for reaching the port face out of tolerance. Smaller than a hull
## strike, because it is a better outcome and the ordering is what the policy
## learns from.
@export var botch_penalty: float = 10.0
## Paid for a clean dock, graded by how clean.
@export var dock_bonus: float = 120.0

var targets: Array[Station] = []

var _stations: Array[Station] = []


# --- Description -------------------------------------------------------------

func obs_size() -> int:
	return OBS_SIZE


func task_name() -> String:
	return "docking"


## 150 seconds. A docking approach is not a landing: the ship starts within a
## few hundred units of the port rather than a planet's radius away, and the
## base class's 27,000 would leave an agent that never commits scoring better
## than one that tries. Sized against the worst spawn -- 2.5 approach ranges out
## and on the wrong side of the structure -- with room, not against the median.
func default_max_steps() -> int:
	return 9000


func outcome_names() -> Dictionary:
	return {
		DOCKED: "docked",
		COLLIDED: "collided",
		BOTCHED: "botched",
		DRIFTED: "drifted",
		TIMEOUT: "timeout",
	}


func _bind_world() -> bool:
	for station in SolarSystem.Instance.Stations:
		_stations.append(station)
		# See AGENT_LAYER: the agents share one station and must not queue.
		station.collision_layer = AGENT_LAYER
	if _stations.is_empty():
		push_error("No Station in main.tscn; nothing to dock with.")
		return false
	targets.resize(num_agents)
	return true


# --- Episode control ---------------------------------------------------------

func reset_agent(i: int) -> void:
	var ship: Ship = ships[i]
	var station: Station = _stations[_rng.randi_range(0, _stations.size() - 1)]
	targets[i] = station

	var port: Vector3 = station.port_position()
	var normal: Vector3 = station.port_normal()
	var range_out: float = station.approach_range * _rng.randf_range(
		spawn_range_min, spawn_range_max
	)

	var direction: Vector3
	if _rng.randf() < spawn_corridor_fraction:
		# Inside the corridor: along the port normal, with a small cone of
		# scatter so the policy still has to correct rather than being handed a
		# perfect line every time.
		direction = (normal + Vector3(
			_rng.randfn(), _rng.randfn(), _rng.randfn()
		) * 0.18).normalized()
	else:
		# Anywhere on the sphere, including behind the station, where the ship
		# has to fly around the structure to reach the port at all.
		direction = Vector3(_rng.randfn(), _rng.randfn(), _rng.randfn()).normalized()

	var pos: Vector3 = port + direction * range_out
	# A spawn that starts inside the hull is not a hard case, it is a broken
	# one: the episode would end as a collision on tick one having been given no
	# chance. Push it out along the spawn direction until it is clear.
	var guard: int = 0
	while station.is_inside_hull(pos, 6.0) and guard < 32:
		pos += direction * 6.0
		guard += 1

	# Attitude: roughly facing the port, so most episodes start with a job to
	# finish rather than a job to discover...
	var toward: Vector3 = (port - pos).normalized()
	var up: Vector3 = Vector3.UP
	if absf(toward.dot(up)) > 0.95:
		up = Vector3.RIGHT
	var start := Basis.looking_at(toward, up)
	start = start.rotated(start.x, _rng.randf_range(-0.4, 0.4))
	start = start.rotated(start.y, _rng.randf_range(-0.4, 0.4))
	# ...and sometimes not facing it at all. A player engaging a docking
	# autopilot has no obligation to be pointed the right way first.
	if _rng.randf() < spawn_tumble_fraction:
		start = Basis(
			Vector3(_rng.randfn(), _rng.randfn(), _rng.randfn()).normalized(),
			_rng.randf_range(-PI, PI)
		) * start

	ship.global_transform = Transform3D(start.orthonormalized(), pos)
	# Velocity relative to the station, which is orbiting: an agent that had to
	# carry the station's orbital velocity as a bias term would be learning the
	# wrong thing.
	ship.linear_velocity = station.constant_linear_velocity + Vector3(
		_rng.randfn(), _rng.randfn(), _rng.randfn()
	).normalized() * spawn_speed * _rng.randf() * _rng.randf()
	ship.angular_velocity = Vector3(
		_rng.randfn(), _rng.randfn(), _rng.randfn()
	) * (0.6 if _rng.randf() < spawn_tumble_fraction else 0.0)
	ship.throttle = 0.0
	ship.command_thrust = Vector3.ZERO
	ship.command_turn = Vector3.ZERO
	ship.refuel()

	_steps[i] = 0
	_potential[i] = _compute_potential(i)


# --- Geometry ----------------------------------------------------------------

## Everything the task is scored and observed on, from one place.
##
## Returns [offset, distance, axial, lateral, alignment, relative velocity].
## `offset` runs from the ship's port to the station's port, so it points the
## way the ship has to go. `axial` is how far out along the corridor the ship
## is — positive is in front of the port, negative is *behind the port face*,
## which means inside or past the station. `lateral` is how far off the
## centreline.
##
## Static and world-space for the same reason LandingEnv.surface_frame is: an
## in-game docking autopilot has to compute exactly this, and a second nearly
## identical copy is how a policy quietly ends up flying a different task than
## the one it was trained on.
static func dock_frame(ship: Ship, station: Station) -> Array:
	var port: Vector3 = station.port_position()
	var normal: Vector3 = station.port_normal()
	var offset: Vector3 = port - ship.port_position()
	var distance: float = offset.length()
	# Along the outward normal from the port to the ship. The sign matters: a
	# ship level with the port face has nowhere left to go but into it.
	var axial: float = -offset.dot(normal)
	var lateral: float = (offset + normal * axial).length()
	var alignment: float = ship.port_normal().dot(normal)
	var relative: Vector3 = ship.port_velocity() - station.port_velocity()
	return [offset, distance, axial, lateral, alignment, relative]


# --- Observation -------------------------------------------------------------

func observe(i: int, into: PackedFloat32Array, at: int) -> void:
	build_observation(ships[i], targets[i], episode_progress(i), into, at)


## The one and only definition of what the policy sees.
##
## Both callers matter: the trainer builds it for every agent every tick, and an
## in-game docking autopilot builds it for the player's ship. If the two ever
## disagreed the policy would still run, still look busy, and quietly fly a
## different task than the one it was trained on — so there is exactly one copy.
static func build_observation(
	ship: Ship, station: Station, progress: float, into: PackedFloat32Array, at: int
) -> void:
	var frame := dock_frame(ship, station)
	var offset: Vector3 = frame[0]
	var distance: float = frame[1]
	var axial: float = frame[2]
	var lateral: float = frame[3]
	var alignment: float = frame[4]
	var relative: Vector3 = frame[5]

	var scale: float = maxf(station.approach_range, 1.0)
	var inverse: Basis = ship.global_basis.inverse()
	# Everything is expressed in the ship's own frame — the frame
	# command_thrust is in — so the policy can map a direction onto a
	# translation without first learning the rotation.
	var offset_local: Vector3 = inverse * offset
	var velocity_local: Vector3 = inverse * relative
	var normal_local: Vector3 = inverse * station.port_normal()
	var spin_local: Vector3 = inverse * ship.angular_velocity

	into[at + 0] = clampf(distance / scale, 0.0, 8.0)
	# A second, sharper range channel. distance/scale is nearly flat over the
	# last couple of metres, which is exactly where the tolerances are.
	into[at + 1] = tanh(distance / (0.12 * scale))
	into[at + 2] = clampf(offset_local.x / scale, -8.0, 8.0)
	into[at + 3] = clampf(offset_local.y / scale, -8.0, 8.0)
	into[at + 4] = clampf(offset_local.z / scale, -8.0, 8.0)
	into[at + 5] = velocity_local.x / SPEED_SCALE
	into[at + 6] = velocity_local.y / SPEED_SCALE
	into[at + 7] = velocity_local.z / SPEED_SCALE
	# Closing speed along the corridor, signed: positive is approaching.
	into[at + 8] = -relative.dot(station.port_normal()) / SPEED_SCALE
	into[at + 9] = normal_local.x
	into[at + 10] = normal_local.y
	into[at + 11] = normal_local.z
	# The attitude error, as one number. -1 is faces meeting.
	into[at + 12] = alignment
	# Where in the corridor: how far in front of the port face, and how far off
	# the centreline. Negative axial means the ship is level with or behind the
	# face, which is the failure this channel exists to make visible.
	into[at + 13] = clampf(axial / scale, -8.0, 8.0)
	into[at + 14] = clampf(lateral / scale, 0.0, 8.0)
	into[at + 15] = clampf(spin_local.x / RATE_SCALE, -4.0, 4.0)
	into[at + 16] = clampf(spin_local.y / RATE_SCALE, -4.0, 4.0)
	into[at + 17] = clampf(spin_local.z / RATE_SCALE, -4.0, 4.0)
	into[at + 18] = ship.throttle
	into[at + 19] = progress
	into[at + 20] = ship.fuel_fraction()
	# Local gravity, because the station orbits a planet and the assist is
	# trimming against that planet rather than against the station. The policy
	# is flying inside a field it did not choose and should be able to see it.
	into[at + 21] = clampf(
		SolarSystem.Instance.get_acceleration_at_position(
			ship.global_position
		).length() / 10.0, 0.0, 2.0)
	# The station's up, in the ship's frame. Ports are keyed, so pointing at the
	# hole is not enough -- the ship has to arrive the same way up as the deck,
	# and without these three channels which way that is cannot be seen at all.
	var up_local: Vector3 = inverse * station.global_basis.y
	into[at + 22] = up_local.x
	into[at + 23] = up_local.y
	into[at + 24] = up_local.z


# --- Reward ------------------------------------------------------------------

## Range, centreline, facing and spin. See ShipTaskEnv._compute_potential for
## why this is a potential rather than a bonus.
func _compute_potential(i: int) -> float:
	var ship: Ship = ships[i]
	var station: Station = targets[i]
	var frame := dock_frame(ship, station)
	var distance: float = frame[1]
	var lateral: float = frame[3]
	var alignment: float = frame[4]
	var scale: float = maxf(station.approach_range, 1.0)

	# Alignment runs -1 (faces meeting) to +1 (back turned), so `1 + alignment`
	# is an error that is zero when correct — the same shape as every other
	# term here, which is what keeps the weights comparable.
	# Roll error is 1 - dot for the same reason facing error is 1 + dot: both
	# are zero when correct, which is what keeps the weights comparable.
	var roll: float = ship.global_basis.y.dot(station.global_basis.y)
	return -range_weight * (distance / scale) \
		- lateral_weight * (lateral / scale) \
		- align_weight * (1.0 + alignment) \
		- roll_weight * (1.0 - roll) \
		- spin_weight * (ship.angular_velocity.length() / RATE_SCALE)


func step_result(i: int, action: PackedFloat32Array, at: int) -> Array:
	var ship: Ship = ships[i]
	var station: Station = targets[i]
	_steps[i] += 1

	var potential: float = _compute_potential(i)
	var reward: float = 10.0 * (potential - _potential[i])
	_potential[i] = potential

	# Small costs, so the policy does not learn to hold station forever or to
	# chatter the controls. Both are free under shaping alone.
	var effort: float = 0.0
	for k in ACT_SIZE:
		effort += action[at + k] * action[at + k]
	reward -= 0.001 * effort
	# Time is charged at a rate that makes running out the clock clearly worse
	# than any contact at all. Landing learned this the hard way: charge too
	# little and the best available policy is to hover just short of the goal
	# and never commit.
	reward -= 0.02

	var frame := dock_frame(ship, station)
	var distance: float = frame[1]
	var lateral: float = frame[3]
	var alignment: float = frame[4]
	var relative: Vector3 = frame[5]
	var speed: float = relative.length()

	last_outcome[i] = RUNNING
	last_impact[i] = speed

	# The hull, before the port: a ship that has reached the structure through
	# the side has collided whatever its port is doing.
	if station.is_inside_hull(ship.global_position, 2.5):
		reward -= collide_penalty
		last_outcome[i] = COLLIDED
		return [reward, true, false]

	if distance <= dock_radius:
		var clean: bool = speed <= dock_speed \
			and alignment <= dock_align \
			and lateral <= dock_lateral \
			and ship.global_basis.y.dot(station.global_basis.y) >= dock_roll
		# Graded, not pass/fail. A touch just outside the gate and a ten-metre-
		# a-second arrival are not the same mistake, and scoring them the same
		# leaves nothing to follow back to a dock. Landing paid for this lesson
		# already; see "A pass/fail bonus is a cliff" in rl/README.md.
		var speed_score: float = clampf(1.0 - speed / (dock_speed * 3.0), 0.0, 1.0)
		var align_score: float = clampf(
			(-alignment - 0.3) / (-dock_align - 0.3), 0.0, 1.0
		)
		var lateral_score: float = clampf(
			1.0 - lateral / (dock_lateral * 3.0), 0.0, 1.0
		)
		reward += dock_bonus * speed_score * align_score * lateral_score
		if not clean:
			reward -= botch_penalty
		last_outcome[i] = DOCKED if clean else BOTCHED
		return [reward, true, false]

	if distance > drift_range * station.approach_range:
		reward -= 50.0
		last_outcome[i] = DRIFTED
		return [reward, true, false]

	if _steps[i] >= max_steps:
		last_outcome[i] = TIMEOUT
		return [reward, false, true]

	return [reward, false, false]
