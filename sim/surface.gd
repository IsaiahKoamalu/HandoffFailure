extends RefCounted
class_name Surface

## Where a world position sits relative to a planet's ground: altitude, local
## up, whether the ground below is dry, and which way the target continent lies.
##
## These are questions about the planet, not about any task, so they live with
## the simulator. Everything that flies near a surface asks them — the
## autopilot, the checks, and the landing task — and every one of them must get
## the same answer. A second, nearly identical copy is how a policy quietly ends
## up flying a different task than the one it was trained on.
##
## Recovered from the pre-project ShipTaskEnv and LandingEnv, where they sat as
## static helpers on the training environment; the code is unchanged. Pre-existing
## apparatus, not this project's work.

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


## Altitude above the analytic terrain surface, the local up direction, and the
## distance from the planet's centre. Returned together because all three come
## from the same offset and every caller wants at least two of them.
##
## Static, and taking a world position rather than an agent index, because the
## in-game autopilot needs exactly this and must not compute it a second,
## subtly different way.
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


## Speed scale for a planet: the circular orbital speed at its surface. A
## physical scale rather than a hand-picked constant is what keeps observations
## in the same range across planets of very different sizes.
static func speed_scale(planet: Planet) -> float:
	return maxf(sqrt(planet.standard_gravitational_parameter / planet.surface_radius), 1.0)


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
