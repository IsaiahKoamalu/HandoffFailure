extends ShipTaskEnv
class_name TransferEnv

## Orbit insertion: arrive at a planet and be in orbit around it, on a budget.
##
## The third task on `ShipTaskEnv`, and the first one the existing analytic
## guidance is not already good at. Landing and docking both had a hand-written
## law worth cloning; `ShipAutopilot`'s TRANSIT phase flies to a body by pure
## pursuit -- point at it, close at a brake-limited speed, stop -- which arrives
## *at* a planet rather than *in orbit around* it, and pays for the whole trip
## with the engines. That is the thing a learned policy has room to beat.
##
## What is being asked for:
##
##   * **A bound orbit, not a position.** Success is a state of the two-body
##     problem -- periapsis clear of the terrain, apoapsis inside the capture
##     shell, eccentricity low enough to be a parking orbit rather than a
##     plunge. Being at the right altitude with the wrong velocity is a fall.
##   * **On a fuel budget that binds.** Fuel is the point here, which is why
##     this task sets its own default rather than inheriting unlimited tanks.
##     A policy that reaches orbit having spent everything has not solved the
##     job it was given; see the fuel note in `rl/README.md`.
##
## Flight assist is **off**, unlike landing and docking. The assist cancels
## velocity relative to the dominant planet, which is precisely the velocity an
## orbit is made of: with it on, the only stable state is hovering and the task
## has no solution. This is the first task where the pilot aid and the objective
## are in direct conflict, and the env says so rather than quietly fighting it.

const OBS_SIZE: int = 26

const CAPTURED: int = 1
## Hit the surface, or dipped below it. Periapsis inside the planet is a
## trajectory that ends in the ground whether or not it has got there yet, but
## this is charged only on actual contact -- an agent is allowed to correct a
## bad periapsis on the way down.
const IMPACTED: int = 2
## Left the neighbourhood: past the capture shell and still climbing, with no
## fuel to come back on.
const ESCAPED: int = 3
## Ran the tanks dry without reaching orbit. Told apart from a timeout because
## it is a different mistake -- one is spending badly, the other is dithering.
const STRANDED: int = 4
const TIMEOUT: int = 5

## Speed scale for the observation, in units per second. Circular orbit speed
## around these planets runs 30-60, so this keeps the channel in range without
## flattening the difference between orbiting and falling.
const SPEED_SCALE: float = 60.0
const RATE_SCALE: float = 1.5

@export_category("Target orbit")
## Orbit radius to aim for, as a multiple of the destination's surface radius.
##
## 1.5, which is a narrow band to hit: the capture shell is 2.0 surface radii
## (SolarSystem.CAPTURE_RADIUS_FACTOR), so there is exactly one sensible
## parking orbit between the terrain and the edge of the planet's influence.
@export var target_radius_factor: float = 1.5
## How far off that radius still counts, as a fraction of it.
@export var radius_tolerance: float = 0.40
## Largest eccentricity that counts as a parking orbit. 0.1 is nearly circular;
## anything much above it is an ellipse that will graze the terrain or the shell
## on a later pass, and "in orbit" should not mean "in orbit for now".
@export var max_eccentricity: float = 0.35
## Periapsis must clear the surface by this fraction of the surface radius.
@export var periapsis_margin: float = 0.05
## Ticks the orbit must be held before it counts. An instantaneous read of the
## elements can be satisfied in passing by a trajectory that is merely crossing
## the band; holding it means the burn actually finished.
@export var hold_ticks: int = 30

@export_category("Spawns")
## Where the ship starts, as a multiple of the destination's surface radius.
## Outside the capture shell, so every episode contains an approach and not just
## a circularisation burn.
@export var spawn_range_min: float = 2.5
@export var spawn_range_max: float = 6.0
## Speed at spawn relative to the destination, as a fraction of the circular
## orbit speed at the spawn radius. A ship dropped at rest would be a pure
## free-fall problem; a ship already moving has to cancel what it has.
@export var spawn_speed_min: float = 0.0
@export var spawn_speed_max: float = 1.2

@export_category("Reward")
## Weight on closing to the target orbit radius.
@export var radius_weight: float = 1.0
## Weight on circularising. Larger than radius_weight because the radius is the
## easy half -- a ship can be at the right distance while on a trajectory that
## leaves immediately, and the shaping has to prefer the orbit to the altitude.
@export var shape_weight: float = 1.5
## Weight on being bound at all, measured as specific orbital energy. This is
## what separates an approach from a flyby and it is the first thing the agent
## has to learn.
@export var energy_weight: float = 1.0
## Charged for hitting the planet.
@export var impact_penalty: float = 150.0
## Charged for leaving, or for running dry short of orbit. Smaller than an
## impact: both are failures, but flying into the ground is the one that would
## end a mission with the ship in pieces.
@export var escape_penalty: float = 100.0
@export var stranded_penalty: float = 100.0
## Paid for a held orbit, graded by how close to the asked-for one it is.
@export var capture_bonus: float = 150.0
## Charged per unit of fuel burned, as a fraction of the tank. The budget is the
## constraint; this is what makes spending it visible in the gradient rather
## than only at the moment the tanks run dry.
##
## 120, and it has to be read against the shaping. Closing from a spawn to the
## target orbit is worth about 48 in shaping, so at 40 the whole tank was
## cheaper than the progress it could buy: the agent burned everything every
## episode, chasing radius and eccentricity error until it ran dry. Measured
## three times, at two time charges and two capture tolerances -- stranding
## climbed past 70% of episodes by iteration 60 with captures at zero every
## time. At 120 the tank costs more than the shaping can pay for, so fuel is
## only worth spending on something that ends in a capture.
@export var fuel_weight: float = 120.0

var targets: Array[Planet] = []

var _destinations: Array[Planet] = []
var _held: PackedInt32Array = PackedInt32Array()
var _last_fuel: PackedFloat32Array = PackedFloat32Array()


# --- Description -------------------------------------------------------------

func obs_size() -> int:
	return OBS_SIZE


func task_name() -> String:
	return "transfer"


## 300 seconds. An orbit at 1.5 surface radii takes 40-90 seconds to go round,
## so this is a handful of orbits: long enough to approach, insert and hold,
## short enough that running out the clock is clearly worse than trying.
func default_max_steps() -> int:
	return 18000


## Seconds of full thrust, when the caller does not say. Unlike the other two
## tasks this one defaults to a real budget, because a transfer with unlimited
## fuel is not a transfer -- pure pursuit solves it and there is nothing to
## learn. Sized so the analytic law can just about do the job and no more.
func default_fuel_seconds() -> float:
	return 90.0


func outcome_names() -> Dictionary:
	return {
		CAPTURED: "captured",
		IMPACTED: "impacted",
		ESCAPED: "escaped",
		STRANDED: "stranded",
		TIMEOUT: "timeout",
	}


func _bind_world() -> bool:
	for planet in SolarSystem.Instance.Planets:
		if world_allowed(planet.name):
			_destinations.append(planet)
	if _destinations.is_empty():
		push_error("No planet allowed by --planets/--holdout; nowhere to go.")
		return false

	# See the note at the top: the assist holds the ship still relative to the
	# planet, and an orbit is the opposite of still. Forced rather than exposed,
	# because a run configured with it on would not fail, it would quietly have
	# no solution and look like a training problem.
	if flight_assist:
		print("[transfer] flight assist forced off; it cancels orbital velocity")
		flight_assist = false

	if fuel_seconds <= 0.0:
		fuel_seconds = default_fuel_seconds()
	print("[transfer] %d worlds, %.0fs of fuel" % [_destinations.size(), fuel_seconds])

	targets.resize(num_agents)
	_held.resize(num_agents)
	_last_fuel.resize(num_agents)
	return true


# --- Orbital elements --------------------------------------------------------

## Semi-major axis, eccentricity, periapsis and apoapsis of the ship's orbit
## about `planet`, plus specific energy, from position and velocity alone.
##
## Static and world-space for the same reason LandingEnv.surface_frame and
## DockingEnv.dock_frame are: an in-game transfer autopilot needs exactly this,
## and a second nearly identical copy is how a policy ends up flying a different
## task than the one it was trained on.
##
## An unbound orbit has energy >= 0 and no apoapsis. That case is returned with
## apoapsis INF rather than a negative number, so callers comparing against the
## capture shell get the right answer without special-casing the hyperbola.
static func orbit_frame(ship: Ship, planet: Planet) -> Array:
	var mu: float = planet.standard_gravitational_parameter
	var offset: Vector3 = ship.global_position - planet.global_position
	# The planet is on rails and carries its own orbital velocity; what matters
	# here is the ship's motion *relative to it*.
	var velocity: Vector3 = ship.linear_velocity - planet.constant_linear_velocity
	var r: float = maxf(offset.length(), 0.001)
	var v: float = velocity.length()

	var energy: float = 0.5 * v * v - mu / r
	var momentum: Vector3 = offset.cross(velocity)
	var eccentricity_vector: Vector3 = velocity.cross(momentum) / mu - offset / r
	var e: float = eccentricity_vector.length()

	var a: float = INF
	var periapsis: float = 0.0
	var apoapsis: float = INF
	if energy < -0.000001:
		a = -mu / (2.0 * energy)
		periapsis = a * (1.0 - e)
		apoapsis = a * (1.0 + e)
	else:
		# Unbound. Periapsis is still defined and still the thing that decides
		# whether this trajectory ends in the ground.
		var h: float = momentum.length()
		periapsis = (h * h / mu) / (1.0 + maxf(e, 0.0))

	return [offset, velocity, r, a, e, periapsis, apoapsis, energy]


# --- Episode control ---------------------------------------------------------

func reset_agent(i: int) -> void:
	var ship: Ship = ships[i]
	var planet: Planet = _destinations[_rng.randi_range(0, _destinations.size() - 1)]
	targets[i] = planet

	var range_out: float = planet.surface_radius * _rng.randf_range(
		spawn_range_min, spawn_range_max
	)
	var direction := Vector3(_rng.randfn(), _rng.randfn(), _rng.randfn()).normalized()
	var pos: Vector3 = planet.global_position + direction * range_out

	# Speed as a fraction of what a circular orbit here would need, in a
	# direction that is neither aimed at the planet nor perfectly tangential:
	# an episode that started on a circular orbit would already be finished, and
	# one aimed straight in is a lob rather than a transfer.
	var circular: float = sqrt(planet.standard_gravitational_parameter / range_out)
	var speed: float = circular * _rng.randf_range(spawn_speed_min, spawn_speed_max)
	var tangent: Vector3 = direction.cross(
		Vector3(_rng.randfn(), _rng.randfn(), _rng.randfn())
	).normalized()
	var toward: Vector3 = -direction
	var mix: float = _rng.randf_range(0.0, 1.0)
	var heading: Vector3 = (tangent * (1.0 - mix) + toward * mix).normalized()

	ship.global_transform = Transform3D(
		Basis.looking_at(heading, Vector3.UP).orthonormalized(), pos
	)
	ship.linear_velocity = planet.constant_linear_velocity + heading * speed
	ship.angular_velocity = Vector3.ZERO
	ship.throttle = 0.0
	ship.command_thrust = Vector3.ZERO
	ship.command_turn = Vector3.ZERO
	ship.refuel()

	_steps[i] = 0
	_held[i] = 0
	_last_fuel[i] = ship.fuel_fraction()
	_potential[i] = _compute_potential(i)


# --- Observation -------------------------------------------------------------

func observe(i: int, into: PackedFloat32Array, at: int) -> void:
	build_observation(ships[i], targets[i], episode_progress(i), into, at)


## The one and only definition of what the policy sees.
static func build_observation(
	ship: Ship, planet: Planet, progress: float, into: PackedFloat32Array, at: int
) -> void:
	var frame := orbit_frame(ship, planet)
	var offset: Vector3 = frame[0]
	var velocity: Vector3 = frame[1]
	var r: float = frame[2]
	var e: float = frame[4]
	var periapsis: float = frame[5]
	var apoapsis: float = frame[6]
	var energy: float = frame[7]

	var scale: float = maxf(planet.surface_radius, 1.0)
	var inverse: Basis = ship.global_basis.inverse()
	var up: Vector3 = offset / maxf(offset.length(), 0.001)

	# Everything in the ship's own frame, the frame command_thrust is in, so a
	# direction maps onto a translation without first learning the rotation.
	var offset_local: Vector3 = inverse * offset
	var velocity_local: Vector3 = inverse * velocity
	var up_local: Vector3 = inverse * up
	var spin_local: Vector3 = inverse * ship.angular_velocity

	# Radial and tangential speed, which is what an orbit is actually made of:
	# circular means the radial component is zero and the tangential one matches
	# sqrt(mu/r). Split out rather than left for the network to infer, because
	# the difference between them is the whole task.
	var radial: float = velocity.dot(up)
	var tangential: float = (velocity - up * radial).length()
	var circular: float = sqrt(planet.standard_gravitational_parameter / maxf(r, 0.001))

	into[at + 0] = clampf(r / scale, 0.0, 10.0)
	into[at + 1] = clampf(offset_local.x / scale, -10.0, 10.0)
	into[at + 2] = clampf(offset_local.y / scale, -10.0, 10.0)
	into[at + 3] = clampf(offset_local.z / scale, -10.0, 10.0)
	into[at + 4] = velocity_local.x / SPEED_SCALE
	into[at + 5] = velocity_local.y / SPEED_SCALE
	into[at + 6] = velocity_local.z / SPEED_SCALE
	into[at + 7] = up_local.x
	into[at + 8] = up_local.y
	into[at + 9] = up_local.z
	into[at + 10] = radial / SPEED_SCALE
	into[at + 11] = tangential / SPEED_SCALE
	# Speed as a fraction of circular here. 1.0 is orbital, below is falling,
	# above is climbing away -- one number that says which of the three.
	into[at + 12] = clampf(velocity.length() / maxf(circular, 0.001), 0.0, 4.0)
	into[at + 13] = clampf(e, 0.0, 4.0)
	into[at + 14] = clampf(periapsis / scale, 0.0, 10.0)
	into[at + 15] = clampf(apoapsis / scale, 0.0, 10.0) if is_finite(apoapsis) else 10.0
	# Bound or not, as a sign rather than a magnitude: the energy itself spans
	# orders of magnitude across these planets and this is the part that matters.
	into[at + 16] = 1.0 if energy < 0.0 else -1.0
	# How far the target orbit is from here, signed, in surface radii.
	into[at + 17] = clampf(
		(planet.surface_radius * 1.5 - r) / scale, -10.0, 10.0)
	into[at + 18] = clampf(spin_local.x / RATE_SCALE, -4.0, 4.0)
	into[at + 19] = clampf(spin_local.y / RATE_SCALE, -4.0, 4.0)
	into[at + 20] = clampf(spin_local.z / RATE_SCALE, -4.0, 4.0)
	into[at + 21] = ship.throttle
	into[at + 22] = progress
	into[at + 23] = ship.fuel_fraction()
	# Local gravity, which varies by a factor of ten across these worlds. A
	# policy that could not see it would have to guess which planet it was at.
	into[at + 24] = clampf(
		planet.get_gravity_magnitude_at_position(ship.global_position) / 10.0,
		0.0, 4.0)
	into[at + 25] = clampf(planet.surface_radius / 200.0, 0.0, 2.0)


# --- Reward ------------------------------------------------------------------

## Radius error, circularity and boundness, as one potential.
func _compute_potential(i: int) -> float:
	var ship: Ship = ships[i]
	var planet: Planet = targets[i]
	var frame := orbit_frame(ship, planet)
	var r: float = frame[2]
	var e: float = frame[4]
	var energy: float = frame[7]

	var scale: float = maxf(planet.surface_radius, 1.0)
	var wanted: float = planet.surface_radius * target_radius_factor
	var radius_error: float = absf(r - wanted) / scale

	# Boundness as a smooth quantity rather than the flag the observation
	# carries. Energy for a circular orbit at the target radius is the natural
	# zero -- being far above it is a flyby, far below is a plunge.
	var reference: float = -planet.standard_gravitational_parameter / (2.0 * wanted)
	var energy_error: float = clampf(
		absf(energy - reference) / absf(reference), 0.0, 4.0)

	return -radius_weight * radius_error \
		- shape_weight * clampf(e, 0.0, 4.0) \
		- energy_weight * energy_error


func step_result(i: int, action: PackedFloat32Array, at: int) -> Array:
	var ship: Ship = ships[i]
	var planet: Planet = targets[i]
	_steps[i] += 1

	var potential: float = _compute_potential(i)
	var reward: float = 10.0 * (potential - _potential[i])
	_potential[i] = potential

	# Fuel, charged as it is spent rather than only when it runs out. Without
	# this the budget is invisible until the moment it binds, and a policy
	# cannot learn to be frugal from a single terminal event.
	var fuel: float = ship.fuel_fraction()
	reward -= fuel_weight * maxf(_last_fuel[i] - fuel, 0.0)
	_last_fuel[i] = fuel

	var effort: float = 0.0
	for k in ACT_SIZE:
		effort += action[at + k] * action[at + k]
	reward -= 0.001 * effort
	# A quarter of what landing and docking charge, because this episode is
	# twice as long and the two numbers have to be read together. At 0.02 a
	# tick the clock costs 360 over a full episode -- more than twice the
	# capture bonus, and more than any single failure -- so the cheapest way to
	# stop paying it was to end the episode early and eat a penalty. Measured:
	# deliberate stranding climbed 24% -> 39.5% -> 67% of episodes while
	# captures stayed at zero, which is the agent correctly answering the
	# question it was asked.
	#
	# At 0.005 the clock costs 90, and the failure penalties above are larger
	# than that, so running out of time is now better than arranging to fail and
	# capturing is better than either.
	reward -= 0.005

	var frame := orbit_frame(ship, planet)
	var r: float = frame[2]
	var e: float = frame[4]
	var periapsis: float = frame[5]
	var apoapsis: float = frame[6]

	last_outcome[i] = RUNNING
	last_impact[i] = (ship.linear_velocity - planet.constant_linear_velocity).length()

	var surface: float = planet.get_surface_radius_at(
		(planet.global_basis.inverse() * (
			ship.global_position - planet.global_position
		)).normalized()
	)
	if r <= surface:
		reward -= impact_penalty
		last_outcome[i] = IMPACTED
		return [reward, true, false]

	var shell: float = planet.surface_radius * SolarSystem.CAPTURE_RADIUS_FACTOR

	# In the band, nearly circular, and clear of the ground at the low point.
	var wanted: float = planet.surface_radius * target_radius_factor
	var in_band: bool = absf(r - wanted) <= wanted * radius_tolerance
	var clear: bool = periapsis >= planet.surface_radius * (1.0 + periapsis_margin)
	var bound: bool = is_finite(apoapsis) and apoapsis <= shell
	if in_band and clear and bound and e <= max_eccentricity:
		_held[i] += 1
		if _held[i] >= hold_ticks:
			# Graded, not pass/fail, for the reason landing and docking both
			# had to learn: a touch inside the gate and a textbook circular
			# orbit are not the same result, and scoring them alike leaves
			# nothing to follow. Fuel left is part of the grade because it is
			# the constraint the task exists to exercise.
			var radius_score: float = clampf(
				1.0 - absf(r - wanted) / (wanted * radius_tolerance), 0.0, 1.0)
			var circle_score: float = clampf(
				1.0 - e / max_eccentricity, 0.0, 1.0)
			reward += capture_bonus * (
				0.5 + 0.3 * radius_score + 0.2 * circle_score
			) * (0.7 + 0.3 * fuel)
			last_outcome[i] = CAPTURED
			return [reward, true, false]
	else:
		_held[i] = 0

	# Gone, and nothing left to come back with.
	if r > shell * 2.0 and not is_finite(apoapsis):
		reward -= escape_penalty
		last_outcome[i] = ESCAPED
		return [reward, true, false]

	if ship.has_fuel_limit and fuel <= 0.0:
		reward -= stranded_penalty
		last_outcome[i] = STRANDED
		return [reward, true, false]

	if _steps[i] >= max_steps:
		last_outcome[i] = TIMEOUT
		return [reward, false, true]

	return [reward, false, false]
