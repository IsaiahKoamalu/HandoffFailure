extends StaticBody3D
class_name Planet

## A body on a fixed circular orbit around an optional parent.
##
## Orbits are evaluated analytically from the simulation clock rather than
## integrated tick by tick. Nothing ever perturbs these orbits — `init()` places
## each body on an exactly circular path — so the closed form is both cheaper
## and strictly more accurate than integrating it. It also means position is a
## pure function of time, which is what lets SolarSystem run at arbitrary
## `time_scale` and reproduce the same system state after a reload.

@export_category("Orbit")
@export var _orbital_parent: Planet
@export var _orbital_radius: float = 0.0
## Starting angle of the orbit, in degrees.
@export_range(0.0, 360.0, 0.1, "or_greater", "or_less") var _orbital_angle: float = 0.0

@export_category("Gravity")
@export var surface_radius: float = 1.0
@export var _surface_gravity: float = 9.81

@export_category("Rotation")
@export var _day_length: float = 0.0
@export var _tidally_locked: bool = false

@export_category("Terrain")
@export var terrain_enabled: bool = false
## Peak height as a fraction of surface_radius. 0.05 is Earth-like; past 0.15
## the horizon starts to look wrong.
@export var terrain_amplitude: float = 0.06
@export var terrain_seed: int = 0
## Vertices per cube-face edge. Triangle count is 6*(n-1)^2*2, so this gets
## expensive fast: 64 is ~47k triangles, 128 is ~190k.
@export_range(4, 256, 1) var mesh_resolution: int = 64
## Because noise is sampled on the unit sphere, frequency reads as "features
## per planet". Around 1.0 gives a handful of continents.
@export var continent_frequency: float = 1.1
@export var detail_frequency: float = 4.0
## Raise to flood the planet, lower to dry it out.
@export_range(-0.5, 0.5, 0.01) var sea_level_bias: float = 0.15

@export_category("Terrain collision")
## Angular half-width of the streamed collision patch, in radians. Expressing
## it as an angle rather than a distance means it scales with the planet.
@export_range(0.05, 1.0, 0.01) var patch_half_angle: float = 0.35
@export_range(4, 128, 1) var patch_resolution: int = 48

var standard_gravitational_parameter: float = 0.0

var _continents: FastNoiseLite
var _detail: FastNoiseLite
var _mountains: FastNoiseLite
var _terrain_collider: CollisionShape3D
## Centre of the live collision patch, as a planet-local direction.
var _patch_centre: Vector3 = Vector3.ZERO

## Signed orbital rate in rad/s. Negative because the orbit runs in the
## direction `get_orbital_velocity` reports, which is clockwise about UP.
var _mean_motion: float = 0.0
var _phase: float = 0.0
## Signed spin rate about the body's own up axis, in rad/s.
var _spin_rate: float = 0.0
## Orientation at t = 0, so an authored axial tilt survives the spin.
var _rest_basis: Basis = Basis.IDENTITY


func _ready() -> void:
	# Given surface gravity g at radius R: g = GM / R^2  =>  GM = g * R^2
	standard_gravitational_parameter = _surface_gravity * surface_radius * surface_radius

	# Must run before anything can ask for a height.
	_setup_noise()
	if terrain_enabled:
		_build_terrain()


# --- Terrain ----------------------------------------------------------------

func _setup_noise() -> void:
	_continents = FastNoiseLite.new()
	_continents.seed = terrain_seed
	_continents.frequency = continent_frequency
	_continents.fractal_type = FastNoiseLite.FRACTAL_FBM
	_continents.fractal_octaves = 3

	_detail = FastNoiseLite.new()
	_detail.seed = terrain_seed + 1
	_detail.frequency = detail_frequency
	_detail.fractal_type = FastNoiseLite.FRACTAL_FBM
	_detail.fractal_octaves = 5
	_detail.fractal_gain = 0.45

	_mountains = FastNoiseLite.new()
	_mountains.seed = terrain_seed + 2
	_mountains.frequency = detail_frequency * 0.5
	_mountains.fractal_type = FastNoiseLite.FRACTAL_RIDGED
	_mountains.fractal_octaves = 4


func _build_terrain() -> void:
	for child in get_children():
		if child is MeshInstance3D:
			(child as MeshInstance3D).mesh = PlanetMesh.build(self, mesh_resolution)

	# The authored SphereShape3D is deliberately left in place, shrunk just
	# inside the lowest possible terrain. Because get_terrain_height never
	# returns less than zero, that sphere can never poke through the surface —
	# so it works as a sea-level backstop, catching anything that reaches the
	# planet before a collision patch has been streamed in for it.
	for child in get_children():
		if child is CollisionShape3D:
			var collider := child as CollisionShape3D
			if collider.shape is SphereShape3D:
				# Duplicate first: the scene shares one SphereShape3D across
				# every planet, so resizing it in place would resize them all.
				var sphere: SphereShape3D = collider.shape.duplicate()
				sphere.radius = surface_radius * 0.995
				collider.shape = sphere

	_terrain_collider = CollisionShape3D.new()
	_terrain_collider.name = "TerrainPatch"
	add_child(_terrain_collider)


## Radial offset from `surface_radius` for a planet-LOCAL direction, which must
## be normalized. Never negative: basins sit flat at the reference radius
## rather than being carved below it.
##
## Pure function of direction and seed, so it is safe to call from any thread
## and gives the same planet on every run.
func get_terrain_height(dir: Vector3) -> float:
	if not terrain_enabled:
		return 0.0

	# Positive is land, negative is basin.
	var mask: float = _continents.get_noise_3dv(dir) + sea_level_bias
	if mask <= 0.0:
		return 0.0
	mask = minf(mask, 1.0)

	# Ease the coastline in so land does not begin at a cliff edge.
	var shore: float = smoothstep(0.0, 0.25, mask)

	var rolling: float = _detail.get_noise_3dv(dir) * 0.5 + 0.5
	var ridges: float = _mountains.get_noise_3dv(dir) * 0.5 + 0.5

	# Squaring the ridges keeps foothills low and peaks sharp.
	var h: float = rolling * 0.35 + ridges * ridges * mask * 0.85
	return h * shore * terrain_amplitude * surface_radius


## Radius of the terrain surface along a planet-local direction.
func get_surface_radius_at(dir: Vector3) -> float:
	return surface_radius + get_terrain_height(dir)


## Guarantee real collision exists under a world-space point. Cheap enough to
## call every tick: it only rebuilds once the point nears the patch edge.
func update_collision_for(world_pos: Vector3) -> void:
	if not terrain_enabled or _terrain_collider == null:
		return

	# to_local, not a world-space offset: the patch is stored under the
	# planet's own transform, and terrain spins with the planet.
	var dir := to_local(world_pos).normalized()
	if dir.is_zero_approx():
		return

	# Rebuild once the point has drifted a third of the way to the patch edge.
	if _patch_centre != Vector3.ZERO and dir.dot(_patch_centre) > cos(patch_half_angle * 0.33):
		return

	_patch_centre = dir
	_terrain_collider.shape = PlanetMesh.build_patch(
		self, dir, patch_half_angle, patch_resolution)


# Call this after all planets are ready.
# Make sure to call init() on a parent's planet before its children.
func init() -> void:
	_rest_basis = global_basis.orthonormalized()

	if _orbital_parent != null:
		_phase = deg_to_rad(_orbital_angle)
		var period := get_orbital_period()
		_mean_motion = 0.0 if period == 0.0 else -TAU / period

	if _tidally_locked:
		_day_length = get_orbital_period()
	_spin_rate = 0.0 if _day_length == 0.0 else -TAU / _day_length
	constant_angular_velocity = _spin_rate * _rest_basis.y

	update_orbit(0.0)


## Place the body where it belongs at `time` seconds of simulation clock.
## SolarSystem drives this in parent-first order, so reading the parent's
## `global_position` here always sees the parent already moved to `time`.
func update_orbit(time: float) -> void:
	if _orbital_parent != null:
		var angle := _phase + _mean_motion * time
		var radial := Vector3(-sin(angle), 0.0, -cos(angle))
		# d/dangle of `radial`, scaled by the rate the angle is being swept.
		var tangent := Vector3(-cos(angle), 0.0, sin(angle))

		global_position = _orbital_parent.global_position + radial * _orbital_radius
		constant_linear_velocity = _orbital_parent.constant_linear_velocity \
			+ tangent * (_mean_motion * _orbital_radius)

	global_basis = _rest_basis * Basis(Vector3.UP, _spin_rate * time)


# Acceleration due to this planet’s gravity at a world-space point
func get_acceleration_at_position(world_pos: Vector3) -> Vector3:
	var d := world_pos - global_position
	return -d.normalized() * standard_gravitational_parameter / d.length_squared()


# Magnitude of the above, without the normalize. Used by hot paths that only
# need to rank planets against each other.
func get_gravity_magnitude_at_position(world_pos: Vector3) -> float:
	return standard_gravitational_parameter / global_position.distance_squared_to(world_pos)


# Circular orbital velocity (direction is perpendicular to radius in the UP-plane)
func get_orbital_velocity(world_pos: Vector3) -> Vector3:
	var d := world_pos - global_position
	var speed := sqrt(standard_gravitational_parameter / d.length())
	var direction := d.normalized().cross(Vector3.UP)
	return direction * speed


# Orbital period around the parent at _orbital_radius
func get_orbital_period() -> float:
	if _orbital_parent == null:
		return 0.0
	return 2.0 * PI * sqrt(pow(_orbital_radius, 3) / _orbital_parent.standard_gravitational_parameter)


# World-space velocity of this planet’s surface at a given point,
# i.e. orbital motion plus the spin of the surface underneath that point.
func get_surface_velocity(world_pos: Vector3) -> Vector3:
	return constant_linear_velocity + constant_angular_velocity.cross(world_pos - global_position)


# Linear velocity relative to this planet’s surface at a given position/velocity
func get_relative_velocity_to_surface(world_pos: Vector3, linear_velocity: Vector3) -> Vector3:
	return linear_velocity - get_surface_velocity(world_pos)


# Number of ancestors in the orbital chain
# e.g., Sun=0, Earth=1 (Sun), Moon=2 (Earth->Sun)
func how_many_parents() -> int:
	if _orbital_parent == null:
		return 0
	return _orbital_parent.how_many_parents() + 1
