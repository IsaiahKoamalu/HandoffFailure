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
##   * Flight assist is left ON. Thrust is 18.0 units/s^2 against a surface
##     gravity of 9.81, a thrust-to-weight ratio of about 1.83, so the ship can
##     now arrest its own descent — it could not at the old 10.0, where the
##     assist was doing all of the braking and the only landings that happened
##     were in the sea. The assist still matters: it caps free fall near
##     g / 1.6 = 6.1 units/s, three times the touchdown limit, and it bleeds off
##     tangential velocity, which is what makes an orbital arrival recoverable
##     inside an episode at all.
##
## Many ships share one solar system and one clock. They are independent bodies
## under analytic gravity, so this costs nothing in fidelity and buys a whole
## vectorised batch per engine tick.

const OBS_SIZE: int = 20
const ACT_SIZE: int = 6

# Episode outcomes, reported to the trainer for logging.
const RUNNING: int = 0
const LANDED: int = 1
const CRASHED: int = 2
const ESCAPED: int = 3
const TIMEOUT: int = 4
## Touched down at sea, or went under. A separate outcome from CRASHED
## because it is a separate mistake: the ship can arrive perfectly and
## still have arrived somewhere useless.
const DITCHED: int = 5
## Landed cleanly on dry ground, but on the wrong landmass. A real outcome
## rather than a variety of failure: the flying was right and only the
## navigation was wrong, and the logs are much easier to read when those two
## are not averaged together.
const MISSED: int = 6

## Angular step for the land gradient when standing on a coastline, in radians.
## Continent noise runs at roughly one feature per planet, so this reads the
## slope of a shoreline rather than the noise on top of it.
const LAND_STEP_NEAR: float = 0.08
## ...and the step used far from any coast. A fixed small step is useless in the
## middle of an ocean: the field is flat there, the difference between two
## samples 0.08 radians apart is nothing, and the observation points nowhere at
## exactly the moment the agent most needs a direction. Widening the stencil
## with distance from the shoreline turns it into a coarse search that can still
## see a continent forty degrees away.
const LAND_STEP_FAR: float = 0.75

## Ships stepped in parallel. Each has its own episode and its own target.
@export var num_agents: int = 16
## Leave the ship's own drift assist enabled. See the note at the top: with it
## off the ship's thrust-to-weight ratio makes landing impossible, so this is
## really a switch for experiments, not a tuning knob.
@export var flight_assist: bool = true
## Give each agent its exhaust plume. Costs nothing when nobody is watching
## and is off by default; see rl/spectator.gd.
@export var visible_agents: bool = false
## Physics ticks before an episode is cut short. At 60 Hz, 1500 is 25 seconds.
## It has to be comfortably longer than a damped descent from the top of the
## spawn shell, or 'ran out of time' and 'refused to land' become the same
## observation and the logs stop meaning anything: assist caps free fall near
## 6 units/s, so the 63-unit ceiling below is about 10 seconds of falling.
## At 60 Hz this is 450 seconds. It has to cover the worst case the autopilot
## is now expected to survive: falling in from a radius above the surface, then
## crossing to a coast. A limit sized for the old narrow shell turned every
## long approach into a timeout and taught the policy that distance was fatal.
##
## It was 9000, and that number was quietly setting the score. Measured on the
## hand-written controller, tripling it took landings from 92.5% to 99.0% and
## timeouts from 6.5% to zero -- every single one of those thirteen episodes
## converted to a landing, and not one to a crash or a ditch. They were far-side
## starts that needed longer than 150 seconds to cross the planet, cut off
## mid-flight. Three separate PPO runs had all plateaued at "92%", which looked
## like a control ceiling and was in fact the clock: nobody was failing to land,
## they were failing to arrive before the buzzer.
@export var max_steps: int = 27000

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
##
## Deliberately enormous compared to the 0.10-0.35 it started at. The autopilot
## is meant to be engageable from wherever the player happens to be, and a
## policy only ever shown a narrow shell is extrapolating the moment it is asked
## from anywhere else — it does not fail loudly, it just flies badly. The bottom
## of the range is close enough to the ground to be a nearly-finished landing;
## the top is more than a planet radius out.
@export var spawn_alt_min: float = 0.02
@export var spawn_alt_max: float = 1.20
## Episode ends as a failure once the ship is this many radii above the ground.
## Has to clear spawn_alt_max with room, or a high spawn is scored as an escape
## before it has done anything wrong.
@export var escape_alt: float = 3.0

## Largest tangential velocity at spawn, as a fraction of the local circular
## orbital speed. At 1.0 the ship is genuinely in orbit rather than falling.
##
## Flight assist reads orbital velocity as drift and cancels it, which is what
## makes this survivable: the ship bleeds tangential speed at 1.6/s and can
## come down. It is also why the ship cannot coast in an orbit with assist on,
## a tradeoff the game had already made.
@export_range(0.0, 1.2, 0.05) var spawn_orbit_fraction: float = 0.85
## Fraction of episodes that start from a completely random attitude, tumbling.
@export_range(0.0, 1.0, 0.05) var spawn_tumble_fraction: float = 0.3

## Fraction of episodes that start over dry land.
##
## Fraction of episodes that start directly above the target continent.
##
## The rest start anywhere at all, which is the task as stated: the autopilot
## has to be engageable from wherever the player is. But a purely uniform spawn
## makes the terminal landing rare — most episodes would be spent crossing, and
## the few ticks that actually decide a touchdown would be swamped. Half and
## half keeps both halves of the job represented.
@export_range(0.0, 1.0, 0.05) var spawn_on_site_fraction: float = 0.5
## Weight on being over land in the shaping potential.
@export var land_weight: float = 1.0
## Charged for touching down at sea, or going under. Comfortably larger than
## the -5 for a hard landing: ditching gently is still ditching.
@export var ditch_penalty: float = 30.0
## Charged for landing on dry ground that is not the target continent.
## Smaller than a ditch, because it is a better outcome than a ditch and the
## ordering of the penalties is what the policy actually learns from.
@export var missed_site_penalty: float = 15.0
## Weight on closing the angle to the target continent, in the potential.
##
## This is the term that does the navigating. It has to be large enough to
## be worth crossing an ocean for: at 25, coming half way round the planet is
## worth about 39 reward, against 120 for the landing itself.
@export var site_weight: float = 25.0

## Seconds of full thrust each agent starts an episode with. Zero is unlimited.
##
## No reward term goes with this, deliberately. Fuel is a constraint, not a
## preference: running dry is punished already, by the crash or the ditch that
## follows. Paying a bonus for leftover fuel would invent a second objective and
## make it impossible to say afterwards whether the policy learned to land
## economically or merely learned to hoard.
@export var fuel_seconds: float = 0.0

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
		ship.fuel_seconds = fuel_seconds
		ship.headless_agent = true
		ship.agent_visuals = visible_agents
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
	#
	# This direction is in the planet's own frame, because that is what the
	# terrain functions read; it is turned into a world direction below. The
	# two differ by however far the planet has spun.
	var local_dir := Vector3(_rng.randfn(), _rng.randfn(), _rng.randfn()).normalized()
	if _rng.randf() < spawn_on_site_fraction:
		# Rejection sampling onto the target continent. It covers 12-30% of
		# these planets, so this usually lands within a handful of draws; the
		# bound stops a planet configured into a waterworld from hanging the
		# trainer, and falling through it just gives an ordinary spawn.
		for _attempt in 64:
			if planet.is_on_largest_landmass(local_dir):
				break
			local_dir = Vector3(_rng.randfn(), _rng.randfn(), _rng.randfn()).normalized()

	var ground: float = planet.get_surface_radius_at(local_dir)
	var alt: float = planet.surface_radius * _rng.randf_range(spawn_alt_min, spawn_alt_max)
	var dir := (planet.global_basis * local_dir).normalized()
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

	# ...and sometimes throw it away entirely. A player engaging the autopilot
	# has no obligation to be the right way up first, and a policy that has only
	# ever started belly-down has no idea what to do about it.
	if _rng.randf() < spawn_tumble_fraction:
		start = Basis(
			Vector3(_rng.randfn(), _rng.randfn(), _rng.randfn()).normalized(),
			_rng.randf_range(-PI, PI)
		) * start

	var pos: Vector3 = planet.global_position + up * (ground + alt)
	ship.global_transform = Transform3D(start, pos)

	# Velocity is set relative to the surface, which is both orbiting and
	# spinning. An agent that had to carry the planet's orbital velocity as a
	# bias term would be learning the wrong thing.
	# Tangential velocity scaled to the local circular orbital speed, so a spawn
	# high up can be a genuine orbit rather than a hover. Squaring the draw keeps
	# most episodes slow while still producing the occasional full orbital
	# arrival — the hard case should be represented, not dominant.
	var circular: float = sqrt(
		planet.standard_gravitational_parameter / maxf(ground + alt, 1.0)
	)
	var roll: float = _rng.randf()
	var tangential: float = circular * spawn_orbit_fraction * roll * roll
	var heading: float = _rng.randf_range(-PI, PI)
	var drift: Vector3 = (side * cos(heading) + forward * sin(heading)) * tangential
	drift += side * _rng.randf_range(-6.0, 6.0) + forward * _rng.randf_range(-6.0, 6.0)

	ship.linear_velocity = planet.get_surface_velocity(pos) + drift + up * _rng.randf_range(-8.0, 2.0)
	# A tumble to go with the attitude, on the same episodes.
	ship.angular_velocity = Vector3(
		_rng.randfn(), _rng.randfn(), _rng.randfn()
	) * (0.6 if _rng.randf() < spawn_tumble_fraction else 0.0)
	ship.throttle = 0.0
	ship.command_thrust = Vector3.ZERO
	ship.command_turn = Vector3.ZERO
	ship.refuel()

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


## Where the ground under a world position sits relative to sea level, and
## which way the nearest land lies.
##
## Returns [signal, gradient]: `signal` is positive over land and negative
## over water, `gradient` is a world-space tangent vector pointing the way
## the land signal rises — toward the coast from at sea, and inland from a
## beach. Its length is the strength of that slope, so it goes to nothing in
## the middle of a featureless ocean, which is the honest answer there.
##
## Static and world-space, for the same reason surface_frame is: the
## in-game autopilot has to compute exactly this, and a second nearly
## identical copy is how a policy quietly ends up flying a different task
## than the one it was trained on.
static func land_frame(world_pos: Vector3, planet: Planet) -> Array:
	var offset: Vector3 = world_pos - planet.global_position
	if offset.length_squared() < 1e-6:
		return [0.0, Vector3.ZERO]
	var up: Vector3 = offset.normalized()
	# The noise is sampled in the planet's own frame, and the planet spins.
	var local_dir: Vector3 = (planet.global_basis.inverse() * up).normalized()
	var here: float = planet.get_land_signal(local_dir)

	var east: Vector3 = local_dir.cross(Vector3.UP)
	if east.length_squared() < 1e-8:
		east = local_dir.cross(Vector3.RIGHT)
	east = east.normalized()
	var north: Vector3 = east.cross(local_dir).normalized()

	# Central differences: symmetric, so a constant slope reads the same from
	# either side and the coastline does not drift with sampling direction.
	#
	# The width adapts to how far from a coast we are. |here| is small near the
	# shoreline and large deep inland or far out to sea, so this reads fine
	# detail exactly where the decision is delicate and sweeps wide where the
	# only question is which way the nearest landmass lies.
	var step: float = clampf(
		LAND_STEP_NEAR + absf(here) * 2.0, LAND_STEP_NEAR, LAND_STEP_FAR
	)
	var d_east: float = \
		planet.get_land_signal((local_dir + east * step).normalized()) \
		- planet.get_land_signal((local_dir - east * step).normalized())
	var d_north: float = \
		planet.get_land_signal((local_dir + north * step).normalized()) \
		- planet.get_land_signal((local_dir - north * step).normalized())

	var gradient: Vector3 = planet.global_basis * (east * d_east + north * d_north)
	return [here, gradient]


## Where the target continent lies from a world position.
##
## Returns [angle, tangent]: `angle` is the great-circle angle in radians to
## the middle of the planet's largest landmass, and `tangent` is a world-space
## unit vector pointing along the ground toward it.
##
## This is a *global* bearing, and that is the whole reason it exists. The
## land-signal gradient it replaces was local: it walked to the nearest coast
## and had no way to know a bigger continent lay behind it, and out in open
## ocean it pointed nowhere at all. A bearing to a known point is defined
## everywhere on the sphere, including the antipode.
static func site_frame(world_pos: Vector3, planet: Planet) -> Array:
	var offset: Vector3 = world_pos - planet.global_position
	if offset.length_squared() < 1e-6 or planet.largest_landmass_center == Vector3.ZERO:
		return [0.0, Vector3.ZERO]
	var up: Vector3 = offset.normalized()
	var local_dir: Vector3 = (planet.global_basis.inverse() * up).normalized()
	var target: Vector3 = planet.largest_landmass_center
	var angle: float = acos(clampf(local_dir.dot(target), -1.0, 1.0))

	# Component of the target direction perpendicular to where we are: the
	# heading along the great circle joining them. Degenerate when standing on
	# the target or exactly opposite it, and in both cases a heading is either
	# meaningless or arbitrary.
	var tangent_local: Vector3 = target - local_dir * local_dir.dot(target)
	if tangent_local.length_squared() < 1e-8:
		return [angle, Vector3.ZERO]
	var tangent: Vector3 = (planet.global_basis * tangent_local.normalized()).normalized()
	return [angle, tangent]


## True when the ground under a world position belongs to the target continent.
static func is_on_target_site(world_pos: Vector3, planet: Planet) -> bool:
	var offset: Vector3 = world_pos - planet.global_position
	if offset.length_squared() < 1e-6:
		return false
	var local_dir: Vector3 = (planet.global_basis.inverse() * offset.normalized()).normalized()
	return planet.is_on_largest_landmass(local_dir)


## Just the land signal at a world position, without the gradient.
##
## One noise lookup where land_frame costs five. Both the potential and the
## terminal test want the scalar only, and they run for every agent every tick.
static func land_signal_at(world_pos: Vector3, planet: Planet) -> float:
	var offset: Vector3 = world_pos - planet.global_position
	if offset.length_squared() < 1e-6:
		return 0.0
	var up: Vector3 = offset.normalized()
	var local_dir: Vector3 = (planet.global_basis.inverse() * up).normalized()
	return planet.get_land_signal(local_dir)


## True when the ground under a world position is dry.
static func is_over_land(world_pos: Vector3, planet: Planet) -> bool:
	return land_signal_at(world_pos, planet) > 0.0


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

	# Is the ground directly below dry? Saturating, because the useful
	# distinction is coast-versus-not and deep ocean does not need telling apart
	# from deeper ocean. This is the local check that decides the last few
	# seconds of a descent.
	into[at + 14] = tanh(land_signal_at(ship.global_position, planet) * 4.0)

	# ...and where the target continent is, which decides everything before
	# that. Without these four the task is unobservable: every ocean looks like
	# every other from the remaining channels, and no policy over them could do
	# better than guessing which way to go.
	var site := site_frame(ship.global_position, planet)
	# Heading to the continent, in the ship's own frame — the same frame
	# command_thrust is expressed in, so the policy can map this onto a
	# translation directly instead of learning the rotation first.
	var toward: Vector3 = site[1]
	if toward.length_squared() > 1e-12:
		toward = (inverse * toward).normalized()
	else:
		toward = Vector3.ZERO
	into[at + 15] = toward.x
	into[at + 16] = toward.y
	into[at + 17] = toward.z
	# How far, as a fraction of the farthest it could possibly be. 0 is standing
	# on the target, 1 is the antipode.
	into[at + 18] = float(site[0]) / PI
	# Fuel left, as a fraction. Reads 1.0 on a ship with unlimited tanks, so a
	# policy trained without a budget sees a constant here and one trained with
	# a budget sees the thing it has to ration.
	into[at + 19] = ship.fuel_fraction()


# --- Reward ------------------------------------------------------------------

## Re-take the shaping baseline without paying out for the change.
##
## An agent that ends its episode part way through a held action coasts to the
## next decision boundary before its new episode starts being scored. That coast
## still moves the ship, and without this the movement would be charged to the
## first action of the new episode, which did not cause it.
func rebaseline(i: int) -> void:
	_potential[i] = _compute_potential(i)


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
	# Two site terms, and they do different jobs.
	#
	# The angle to the target continent is the navigation signal: it falls
	# steadily all the way across an ocean, so progress is rewarded on every
	# tick of a long crossing rather than only on arrival. Local landness is the
	# fine correction that keeps a descent off a lake or a bay once the ship is
	# already over the right continent.
	#
	# Both are potential-based, so neither can invent a preference on its own —
	# they telescope away across an episode. It is the terminal penalties that
	# make the wrong site a mistake; these only make it discoverable earlier
	# than the last tick.
	var landness: float = tanh(land_signal_at(ship.global_position, planet) * 4.0)
	var angle: float = float(site_frame(ship.global_position, planet)[0])
	return -(alt / planet.surface_radius) \
		- (relative.length() / reward_speed_scale) \
		- align_weight * (1.0 - align) \
		+ land_weight * landness \
		- site_weight * (angle / PI)


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

	# Going under ends it wherever it happens. The ship floats and the sea has
	# no collider, so without this an agent can sink to the seabed and be
	# scored on how gently it got there.
	if ship.is_submerged:
		reward -= ditch_penalty
		last_outcome[i] = DITCHED
		return [reward, true, false]

	if alt <= 0.0:
		if not is_over_land(ship.global_position, planet):
			# Reached the seabed. Arriving gently does not redeem it, so there
			# is no graded bonus here at all — a soft ditch and a hard one are
			# the same failure, and grading them would hand the policy a
			# gradient pointing at a tidier way to get it wrong.
			reward -= ditch_penalty
			last_outcome[i] = DITCHED
			return [reward, true, false]

		if not is_on_target_site(ship.global_position, planet):
			# Dry ground, wrong continent. Penalised, but less than the sea, and
			# still no bonus: the task is to reach a particular place, and a
			# tidy arrival somewhere else does not partly accomplish it.
			reward -= missed_site_penalty
			last_outcome[i] = MISSED
			return [reward, true, false]

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
