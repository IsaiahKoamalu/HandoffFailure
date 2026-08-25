extends Node3D
class_name LandingEnv

## A vectorised powered-descent environment wrapped around the real ship.
##
## The agent flies `Ship` through `Ship.command_thrust` / `command_turn` — the
## same seam a human's mouse and keyboard feed. Nothing about the dynamics is
## reimplemented here: gravity, drag, buoyancy, engine spool and auto-levelling
## are all the shipping code. That is deliberate. A policy trained against a
## simplified copy of the ship would be a policy that cannot fly the real one.
##
## Two things are simulated rather than collided:
##
##   * Ground. Terrain collision is streamed as patches under a client, which is
##     far too expensive to run for a few hundred thousand training steps. The
##     surface is analytic anyway — `Planet.get_surface_radius_at` is the same
##     function the collision mesh is built from — so an episode ends when
##     altitude reaches zero rather than when a collider is touched.
##   * Flight assist is left ON, and that is not a shortcut — it is forced by
##     the ship's own numbers. Thrust is 10.0 units/s^2 and surface gravity is
##     9.81, a thrust-to-weight ratio of 1.02, so a ship pointing straight up at
##     full power decelerates at 0.19 units/s^2. Killing 30 units/s of descent
##     would take 158 seconds and about 2,400 units of altitude, which is more
##     than a planet radius. Without assist this task has no solution at all.
##     The assist's drag term (-drift * 1.6 * mass) is what actually lands this
##     ship: it caps free fall at roughly g / 1.6 = 6.1 units/s. That is still
##     three times the touchdown limit, so the policy has real work to do —
##     descend under control, hold the belly down, and brake to under 2.0 before
##     contact. It just cannot do it by thrust alone.
##
## Many ships share one solar system and one clock. They are independent bodies
## under analytic gravity, so this costs nothing in fidelity and buys a whole
## vectorised batch per engine tick.

const OBS_SIZE: int = 14
const ACT_SIZE: int = 6

# Episode outcomes, reported to the trainer for logging.
const RUNNING: int = 0
const LANDED: int = 1
const CRASHED: int = 2
const ESCAPED: int = 3
const TIMEOUT: int = 4

## Ships stepped in parallel. Each has its own episode and its own target.
@export var num_agents: int = 16
## Leave the ship's own drift assist enabled. See the note at the top: with it
## off the ship's thrust-to-weight ratio makes landing impossible, so this is
## really a switch for experiments, not a tuning knob.
@export var flight_assist: bool = true
## Physics ticks before an episode is cut short. At 60 Hz, 1500 is 25 seconds.
## It has to be comfortably longer than a damped descent from the top of the
## spawn shell, or 'ran out of time' and 'refused to land' become the same
## observation and the logs stop meaning anything: assist caps free fall near
## 6 units/s, so the 63-unit ceiling below is about 10 seconds of falling.
@export var max_steps: int = 1500

## Touchdown counts as a landing below this speed relative to the surface.
## Deliberately looser than Ship._landing_speed (0.2), which is the threshold
## for *parking* an already-stationary ship, not for surviving contact.
@export var touchdown_speed: float = 2.0
## ...and with the belly at least this aligned with the ground, as a dot product.
@export var touchdown_align: float = 0.9

## Reference speed for reward shaping, in world units per second.
##
## Deliberately NOT the orbital speed scale used to normalise observations:
## that is about 42 for these planets, which would make the entire difference
## between a 6 units/s crash and a nought units/s touchdown worth roughly 0.7
## reward. Shaping has to be scaled to the quantity the task turns on.
@export var reward_speed_scale: float = 10.0
## Weight on belly-down alignment in the shaping potential.
##
## Started at 0.3, which was far too low to matter: alignment gates the
## terminal bonus outright, but over a whole episode the shaping term was
## worth at most 3 reward against a 120 bonus. The policy learned to arrive
## quickly and roughly level-ish, scoring about 30 and landing cleanly 3% of
## the time. A gate deserves shaping in proportion to the gate.
@export var align_weight: float = 1.5

## Spawn shell, as multiples of the target's surface radius above the ground.
@export var spawn_alt_min: float = 0.10
@export var spawn_alt_max: float = 0.35
## Episode ends as a failure once the ship is this many radii above the ground.
@export var escape_alt: float = 1.5

var ships: Array[Ship] = []
var targets: Array[Planet] = []

## Last episode outcome per agent, for the trainer's logs.
var last_outcome: PackedInt32Array = PackedInt32Array()
## Speed relative to the surface at the moment of the last touchdown.
var last_impact: PackedFloat32Array = PackedFloat32Array()

var _steps: PackedInt32Array = PackedInt32Array()
var _potential: PackedFloat32Array = PackedFloat32Array()
var _landable: Array[Planet] = []
var _rng := RandomNumberGenerator.new()


func setup(ship_scene: PackedScene, planets: Array[Planet], seed_value: int) -> void:
	_rng.seed = seed_value
	_landable = planets
	_steps.resize(num_agents)
	_potential.resize(num_agents)
	last_outcome.resize(num_agents)
	last_impact.resize(num_agents)
	targets.resize(num_agents)

	for i in num_agents:
		var ship: Ship = ship_scene.instantiate()
		# Both are read by Ship._ready, so they have to be set before the ship
		# enters the tree.
		ship.headless_agent = true
		ship.external_control = true
		ship.name = "Agent%d" % i
		add_child(ship)
		# Agents fly through each other. They are separate rollouts that happen
		# to share a world; a collision between two of them is not a real event.
		ship.collision_layer = 1 << 9
		ship.collision_mask = ship.collision_mask & ~(1 << 9)
		var ramp := ship.get_node_or_null("Ramp") as PhysicsBody3D
		if ramp != null:
			ramp.collision_layer = 1 << 9
			ramp.collision_mask = ramp.collision_mask & ~(1 << 9)
		ship.flight_assist = flight_assist
		ships.append(ship)
		reset_agent(i)


# --- Episode control ---------------------------------------------------------

func reset_agent(i: int) -> void:
	var ship: Ship = ships[i]
	var planet: Planet = _landable[_rng.randi_range(0, _landable.size() - 1)]
	targets[i] = planet

	# A normalised Gaussian is uniform on the sphere. Drawing spherical angles
	# instead would bunch the spawns at the poles.
	var dir := Vector3(_rng.randfn(), _rng.randfn(), _rng.randfn()).normalized()
	var ground: float = planet.get_surface_radius_at(dir)
	var alt: float = planet.surface_radius * _rng.randf_range(spawn_alt_min, spawn_alt_max)
	var up := dir

	# Any basis with `up` as its +Y, so the ship starts roughly belly-down but
	# with a random heading — the policy cannot memorise a single approach.
	var side := up.cross(Vector3(_rng.randfn(), _rng.randfn(), _rng.randfn()))
	if side.length_squared() < 1e-6:
		side = up.cross(Vector3.RIGHT)
	side = side.normalized()
	var forward := side.cross(up)
	var start := Basis(side, up, forward).orthonormalized()
	# Tip it off level, so attitude has to be corrected and not just descended.
	start = start.rotated(side, _rng.randf_range(-0.35, 0.35))
	start = start.rotated(forward, _rng.randf_range(-0.35, 0.35))

	var pos: Vector3 = planet.global_position + up * (ground + alt)
	ship.global_transform = Transform3D(start, pos)

	# Velocity is set relative to the surface, which is both orbiting and
	# spinning. An agent that had to carry the planet's orbital velocity as a
	# bias term would be learning the wrong thing.
	var drift: Vector3 = side * _rng.randf_range(-6.0, 6.0) + forward * _rng.randf_range(-6.0, 6.0)
	ship.linear_velocity = planet.get_surface_velocity(pos) + drift + up * _rng.randf_range(-6.0, 2.0)
	ship.angular_velocity = Vector3.ZERO
	ship.throttle = 0.0
	ship.command_thrust = Vector3.ZERO
	ship.command_turn = Vector3.ZERO

	_steps[i] = 0
	_potential[i] = _compute_potential(i)


func reset_all() -> void:
	for i in num_agents:
		reset_agent(i)


# --- Geometry ----------------------------------------------------------------

## Altitude above the analytic terrain surface, the local up direction, and the
## distance from the planet's centre. Returned together because all three come
## from the same offset and every caller wants at least two of them.
##
## Static, and taking a world position rather than an agent index, because the
## autopilot needs exactly this and must not compute it a second, subtly
## different way. See build_observation.
static func surface_frame(world_pos: Vector3, planet: Planet) -> Array:
	var offset: Vector3 = world_pos - planet.global_position
	var distance: float = offset.length()
	if distance < 0.001:
		return [0.0, Vector3.UP, 0.001]
	var up: Vector3 = offset / distance
	# get_surface_radius_at wants a direction in the planet's own frame: the
	# noise is sampled in body space, and the planet spins.
	var local_dir: Vector3 = (planet.global_basis.inverse() * up).normalized()
	return [distance - planet.get_surface_radius_at(local_dir), up, distance]


func _altitude_and_up(i: int) -> Array:
	return surface_frame(ships[i].global_position, targets[i])


## Speed scale for a planet: the circular orbital speed at its surface. A
## physical scale rather than a hand-picked constant is what keeps observations
## in the same range across planets of very different sizes.
static func speed_scale(planet: Planet) -> float:
	return maxf(sqrt(planet.standard_gravitational_parameter / planet.surface_radius), 1.0)


# --- Observation -------------------------------------------------------------

func observe(i: int, into: PackedFloat32Array, at: int) -> void:
	build_observation(ships[i], targets[i], float(_steps[i]) / float(max_steps), into, at)


## The one and only definition of what the policy sees.
##
## Both callers matter: the trainer builds it for every agent every tick, and
## the in-game autopilot builds it for the player's ship. If these two ever
## disagreed the policy would still run, still look busy, and quietly fly a
## different task than the one it was trained on — so there is exactly one copy.
static func build_observation(
	ship: Ship, planet: Planet, progress: float, into: PackedFloat32Array, at: int
) -> void:
	var geometry := surface_frame(ship.global_position, planet)
	var alt: float = geometry[0]
	var up: Vector3 = geometry[1]
	var distance: float = geometry[2]

	var radius: float = planet.surface_radius
	var v_scale: float = speed_scale(planet)

	# Everything the policy sees is relative to the surface it is landing on and
	# expressed in the ship's own frame, so one policy generalises across
	# planets, orbits and headings.
	var relative: Vector3 = ship.linear_velocity - planet.get_surface_velocity(ship.global_position)
	var inverse: Basis = ship.global_basis.inverse()
	var v_local: Vector3 = inverse * relative
	var up_local: Vector3 = inverse * up
	var v_up: float = relative.dot(up)
	var v_horizontal: float = (relative - up * v_up).length()

	into[at + 0] = clampf(alt / radius, -0.5, 4.0)
	# A second, sharper altitude channel: alt/radius is nearly flat over the
	# last few metres, which is exactly where the policy needs resolution.
	into[at + 1] = tanh(alt / (0.05 * radius))
	into[at + 2] = v_local.x / v_scale
	into[at + 3] = v_local.y / v_scale
	into[at + 4] = v_local.z / v_scale
	into[at + 5] = up_local.x
	into[at + 6] = up_local.y
	into[at + 7] = up_local.z
	into[at + 8] = v_up / v_scale
	into[at + 9] = v_horizontal / v_scale
	# Local gravity, normalised so the surface reads 1.0.
	into[at + 10] = clampf(pow(radius / maxf(distance, 1.0), 2.0), 0.0, 1.5)
	into[at + 11] = ship.throttle
	into[at + 12] = progress
	into[at + 13] = 1.0 if ship.is_submerged else 0.0


# --- Reward ------------------------------------------------------------------

## Potential-based shaping. The per-step reward is the *change* in this, which
## (Ng, Harada & Russell, 1999) leaves the optimal policy unchanged — shaping of
## this form can only speed learning up, it cannot bend the agent toward a
## different task. A raw "closer is better" bonus has no such guarantee.
func _compute_potential(i: int) -> float:
	var ship: Ship = ships[i]
	var planet: Planet = targets[i]
	var geometry := _altitude_and_up(i)
	var alt: float = geometry[0]
	var up: Vector3 = geometry[1]
	var relative: Vector3 = ship.linear_velocity - planet.get_surface_velocity(ship.global_position)
	var align: float = ship.global_basis.y.dot(up)
	return -(alt / planet.surface_radius) \
		- (relative.length() / reward_speed_scale) \
		- align_weight * (1.0 - align)


## Read the result of a tick that has already been simulated, and decide whether
## the episode is over. Returns [reward, terminated, truncated].
func step_result(i: int, action: PackedFloat32Array, at: int) -> Array:
	var ship: Ship = ships[i]
	var planet: Planet = targets[i]
	_steps[i] += 1

	var potential: float = _compute_potential(i)
	var reward: float = 10.0 * (potential - _potential[i])
	_potential[i] = potential

	# Small costs, so the policy does not learn to hover forever or to chatter
	# the controls. Both are free under shaping alone.
	var effort: float = 0.0
	for k in ACT_SIZE:
		effort += action[at + k] * action[at + k]
	reward -= 0.001 * effort
	# Time is charged at a rate that makes running out the clock clearly worse
	# than any touchdown at all. An earlier version charged a tenth of this,
	# and the policy promptly learned to hover just off the deck for the whole
	# episode: stalling scored about -1.8 while a bad landing scored -3, so
	# refusing to land was, correctly, the best policy available to it.
	reward -= 0.02

	var geometry := _altitude_and_up(i)
	var alt: float = geometry[0]
	var up: Vector3 = geometry[1]
	var relative: Vector3 = ship.linear_velocity - planet.get_surface_velocity(ship.global_position)
	var speed: float = relative.length()
	var align: float = ship.global_basis.y.dot(up)

	last_outcome[i] = RUNNING
	last_impact[i] = speed

	if alt <= 0.0:
		# A pass/fail bonus is a cliff, and a policy cannot climb a cliff: a
		# touchdown just outside the gate scores exactly the same as one at
		# forty units a second, so there is nothing to follow back to a
		# landing. Scoring both conditions continuously turns 'not quite a
		# landing' into a gradient that points at one. The pass/fail test
		# survives, but only to label the outcome for the logs.
		var speed_score: float = clampf(1.0 - speed / (touchdown_speed * 3.0), 0.0, 1.0)
		var align_score: float = clampf((align - 0.3) / (touchdown_align - 0.3), 0.0, 1.0)
		reward += 120.0 * speed_score * align_score
		# Flat, and small. A penalty that scales with impact speed sounds right
		# but it is what made hovering pay: it let the worst touchdowns score
		# far below simply never arriving. The graded bonus above already ranks
		# touchdowns against each other, so this only has to keep a smash from
		# tying with a clean landing.
		if not (speed <= touchdown_speed and align >= touchdown_align):
			reward -= 5.0
		last_outcome[i] = LANDED \
			if (speed <= touchdown_speed and align >= touchdown_align) \
			else CRASHED
		return [reward, true, false]

	if alt > escape_alt * planet.surface_radius:
		reward -= 50.0
		last_outcome[i] = ESCAPED
		return [reward, true, false]

	if _steps[i] >= max_steps:
		last_outcome[i] = TIMEOUT
		return [reward, false, true]

	return [reward, false, false]


func apply_action(i: int, action: PackedFloat32Array, at: int) -> void:
	var ship: Ship = ships[i]
	ship.command_thrust = Vector3(action[at + 0], action[at + 1], action[at + 2])
	ship.command_turn = Vector3(action[at + 3], action[at + 4], action[at + 5])
