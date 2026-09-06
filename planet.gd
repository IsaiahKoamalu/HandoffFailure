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

## Name shown to the player. The node is called Planet2 and always will be --
## its name is a NodePath in three other places and a key in the save file --
## but nobody is flying to "Planet2".
@export var display_name: String = ""

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
## Resolution of the stand-in mesh used once the planet is far enough away that
## the detail cannot be resolved.
@export_range(4, 128, 1) var far_resolution: int = 64
## The detail mesh is drawn within this many planet radii of the camera; past
## that the stand-in takes over. At 5 radii a planet subtends about 23 degrees.
@export_range(2.0, 30.0, 0.5) var detail_range_factor: float = 5.0
## Because noise is sampled on the unit sphere, frequency reads as "features
## per planet". Around 1.0 gives a handful of continents.
@export var continent_frequency: float = 1.1
@export var detail_frequency: float = 4.0
## Raise to flood the planet, lower to dry it out.
@export_range(-0.5, 0.5, 0.01) var sea_level_bias: float = 0.15
## Weight of the rolling fBm layer. Raise for broad, soft relief.
@export_range(0.0, 1.5, 0.01) var detail_weight: float = 0.35
## Weight of the ridged layer. Raise for sharp alpine ranges, drop for hills.
@export_range(0.0, 1.5, 0.01) var ridge_weight: float = 0.85
## How far inland the ground takes to rise from sea level, as a fraction of the
## continent signal. Wide leaves a broad apron of near-zero height around every
## landmass that the colour ramp paints in beach sand, so the continents read as
## pale from orbit; narrow gives them a defined edge.
##
## It changes the height field, so it is per-planet rather than a global tweak:
## the landing task is measured against Kestrel's ground and a world the trained
## policy has never seen is not the world it was scored on.
@export_range(0.02, 0.5, 0.01) var shore_easing: float = 0.25

@export_category("Appearance")
## Hypsometric ramp, deepest seabed to highest peak. The single biggest lever
## on how distinct one planet looks from another.
@export var color_deep: Color = Color(0.06, 0.13, 0.24)
@export var color_shallow: Color = Color(0.16, 0.35, 0.46)
@export var color_shore: Color = Color(0.64, 0.58, 0.42)
@export var color_low: Color = Color(0.28, 0.38, 0.23)
@export var color_high: Color = Color(0.42, 0.33, 0.25)
@export var color_peak: Color = Color(0.88, 0.89, 0.90)
## Colour of exposed rock on anything too steep to hold soil. The terrain
## shader blends to this by slope, which is what stops a planet reading as a
## painted dome the moment you get close enough to see a hillside.
@export var color_rock: Color = Color(0.33, 0.30, 0.27)
## Size of the close-range surface texture, in cycles per world unit. The mesh
## has no UVs, so this is fed to a triplanar sample rather than a mapping.
@export_range(0.02, 2.0, 0.01) var detail_texture_scale: float = 0.35
## Ground below sea level that emits its own light. Zero strength is every
## ordinary world; a volcanic one turns it up and its basins glow.
@export var lava_color: Color = Color(1.0, 0.35, 0.05)
@export_range(0.0, 8.0, 0.1) var lava_strength: float = 0.0

@export_category("Atmosphere")
## A shell of added light around the planet. Off for airless bodies, which is
## most of what makes a moon look like a moon next to a world.
@export var atmosphere_enabled: bool = false
@export var atmosphere_color: Color = Color(0.35, 0.55, 1.0)
## Colour the limb takes where the light is grazing. Sunrise, essentially.
@export var atmosphere_sunset_color: Color = Color(1.0, 0.55, 0.28)
## Shell height as a fraction of surface_radius.
@export_range(0.005, 0.3, 0.001) var atmosphere_height: float = 0.1
@export_range(0.0, 4.0, 0.05) var atmosphere_density: float = 1.0

@export_category("Clouds")
@export var clouds_enabled: bool = false
@export var cloud_color: Color = Color(1.0, 1.0, 1.0)
## How much of the sky is covered. 0 is clear, 1 is solid overcast.
@export_range(0.0, 1.0, 0.01) var cloud_coverage: float = 0.45
## Cloud deck height as a fraction of surface_radius.
##
## Has to clear the terrain by more than head height. At 0.018 on a 180-unit
## world the deck sits three metres up, which looks correct from orbit and puts
## an astronaut standing on a hill above the weather.
@export_range(0.01, 0.3, 0.001) var cloud_height: float = 0.06
## Drift relative to the ground, in degrees per second. Small: weather that
## visibly races across a landscape reads as a spinning texture, not weather.
@export_range(0.0, 5.0, 0.01) var cloud_drift: float = 0.35

@export_category("Ocean")
@export var water_enabled: bool = true
## How far basins are carved below sea level, as a fraction of
## terrain_amplitude. Zero leaves seabeds flat at sea level, which makes the
## ocean surface z-fight with them.
@export_range(0.0, 1.0, 0.01) var ocean_depth: float = 0.35
## The ocean is a smooth sphere, so this can stay low until you add waves.
@export_range(4, 256, 1) var water_resolution: int = 96
@export var water_color: Color = Color(0.05, 0.25, 0.45, 0.75)
## Colour looking straight down into deep water. Alpha is used: deep water is
## opaque, and a single translucent colour everywhere reads as coloured glass.
@export var water_deep_color: Color = Color(0.02, 0.10, 0.22, 0.94)

@export_category("Terrain collision")
## Half-width of the streamed collision patch, in world units. You need a fixed
## amount of ground underfoot regardless of how big the planet is, so this is a
## distance — but on a small body that distance would wrap most of the way
## round, so it is capped by an angle.
@export var patch_extent: float = 30.0
@export_range(0.05, 1.0, 0.01) var patch_max_angle: float = 0.35
@export_range(4, 128, 1) var patch_resolution: int = 48

@export_category("Surface scatter")
## Boulders strewn across the ground near whoever is standing on it.
##
## The terrain mesh cannot carry features at this size -- one vertex spans
## several metres -- so from a hundred metres up a planet is a smooth dome with
## nothing in it to judge height, speed or scale against. These are that
## missing scale reference, and the difference between ground and a backdrop.
##
## Streamed on the same patch the collision uses, and rebuilt when it recentres,
## so the cost is a fixed few hundred instances per client rather than something
## that grows with the size of the world.
@export var scatter_enabled: bool = false
@export_range(0, 2000, 10) var scatter_count: int = 420
## Boulder size in world units, smallest to largest.
##
## Kept to roughly knee height. These have no collision -- several hundred
## concave shapes per client is not a cost worth paying for scenery -- and a
## rock you can see over is one you do not expect to stop you. Sized waist-high
## and up they read as obstacles, and walking through an obstacle looks broken
## in a way walking over rubble does not.
@export var scatter_size: Vector2 = Vector2(0.3, 1.2)
## Tint. Left black, the rock colour is used, which is usually what is wanted.
@export var scatter_color: Color = Color(0, 0, 0, 0)
## Steepest ground that still collects loose rock, as a slope dot product.
## Boulders on a cliff face look stuck on, because they are.
@export_range(0.0, 1.0, 0.01) var scatter_max_slope: float = 0.55

@export_category("Landmarks")
## Sites on the surface worth flying to: a beacon, a wreck, a mineral outcrop.
##
## Distinct terrain and weather give a world a character; they do not give a
## pilot a reason to prefer one landing site over another. These do, and they
## are what the navigation panel can be pointed at once you are in orbit.
@export_range(0, 12, 1) var landmark_count: int = 0

var standard_gravitational_parameter: float = 0.0

var _continents: FastNoiseLite
var _detail: FastNoiseLite
var _mountains: FastNoiseLite
## Built off-thread by generate_terrain_mesh, attached by apply_terrain.
var _built_mesh: ArrayMesh = null
var _far_mesh: ArrayMesh = null
## The cloud shell, or null on a body with no weather.
var _clouds: Node3D = null
## The one boulder mesh every scattered rock on this planet instances.
var _boulder: ArrayMesh = null
## Surface sites, in the order they were placed.
var landmarks: Array[Landmark] = []
## One streamed patch per terrain client, keyed by the client node.
##
## Deliberately not a single shared patch: two clients standing apart on the
## same planet — a parked ship and an astronaut who has walked off — would each
## recentre it every frame, so both spend alternate frames with no ground under
## them and fall through the world.
var _patches: Dictionary = {}


## One client's streamed collision patch and the state of its in-flight build.
class Patch extends RefCounted:
	var collider: CollisionShape3D
	## Boulders for this client, or null on a planet with none.
	var scatter: MultiMeshInstance3D = null
	## Centre the scatter was last populated for, which lags `centre` by one
	## rebuild -- the collision shape is adopted off a worker thread and the
	## scatter is filled in on the main thread when that lands.
	var scatter_centre: Vector3 = Vector3.ZERO
	## Centre of the live shape, as a planet-local direction.
	var centre: Vector3 = Vector3.ZERO
	var pending_centre: Vector3 = Vector3.ZERO
	## In-flight build. -1 when idle.
	var task: int = -1
	## Written by the worker thread, read by the main thread only after
	## is_task_completed reports done — that check is the memory barrier.
	var result: ConcavePolygonShape3D = null

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
	_find_largest_landmass()


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


## Generate the render mesh. Pure maths over this planet's own noise — no
## scene access — so SolarSystem runs every planet's copy concurrently.
func generate_terrain_mesh() -> void:
	if terrain_enabled:
		_built_mesh = PlanetMesh.build(self, mesh_resolution)
		_far_mesh = PlanetMesh.build(self, far_resolution)


## Attach what generate_terrain_mesh produced and build the rest. Main thread
## only: this adds child nodes.
func apply_terrain() -> void:
	if not terrain_enabled:
		return
	_build_terrain()


## True when there is no renderer to draw for -- a training run, or any of the
## headless checks. Atmospheres, clouds and the noise textures the terrain
## shader samples all cost real CPU to build and nothing at all to look at.
static func _drawing() -> bool:
	return DisplayServer.get_name() != "headless"


## Noise for the shader to sample. Seamless, because a tiling artefact on a
## planet-sized surface is a grid you can see from orbit.
static func _noise_texture(seed_value: int, frequency: float, octaves: int) -> NoiseTexture2D:
	var noise := FastNoiseLite.new()
	noise.seed = seed_value
	noise.frequency = frequency
	noise.fractal_type = FastNoiseLite.FRACTAL_FBM
	noise.fractal_octaves = octaves

	var texture := NoiseTexture2D.new()
	texture.width = 256
	texture.height = 256
	texture.seamless = true
	texture.noise = noise
	return texture


func _build_terrain() -> void:
	var mat: Material = _terrain_material()

	# Distance culling. Every planet used to draw its full mesh at all times, so
	# a body 59 pixels tall on screen cost the same 303k triangles as the one
	# under your feet.
	var switch_at: float = surface_radius * detail_range_factor

	for child in get_children():
		if child is MeshInstance3D:
			var mi := child as MeshInstance3D
			mi.mesh = _built_mesh
			mi.material_override = mat
			# Hard switch, no fade margin. VISIBILITY_RANGE_FADE_SELF would be
			# smoother, but the margin is applied on one side of the threshold
			# and getting that side wrong leaves a band where neither mesh
			# draws. Worth trying with the viewport open; not worth guessing.
			mi.visibility_range_end = switch_at
			mi.visibility_range_fade_mode = GeometryInstance3D.VISIBILITY_RANGE_FADE_DISABLED

	# The authored SphereShape3D is deliberately left in place, shrunk just
	# inside the lowest possible terrain. Sized against get_min_terrain_height
	# it can never poke through the surface, so it works as a backstop that
	# catches anything reaching the planet before a collision patch has been
	# streamed in for it. Sizing it against surface_radius instead would leave
	# an invisible sphere floating above every seabed.
	for child in get_children():
		if child is CollisionShape3D:
			var collider := child as CollisionShape3D
			if collider.shape is SphereShape3D:
				# Duplicate first: the scene shares one SphereShape3D across
				# every planet, so resizing it in place would resize them all.
				var sphere: SphereShape3D = collider.shape.duplicate()
				sphere.radius = (surface_radius + get_min_terrain_height()) * 0.995
				collider.shape = sphere

	# Patch colliders are created per client, on demand.

	var far := MeshInstance3D.new()
	far.name = "FarMesh"
	far.mesh = _far_mesh
	far.material_override = mat
	far.visibility_range_begin = switch_at
	far.visibility_range_fade_mode = GeometryInstance3D.VISIBILITY_RANGE_FADE_DISABLED
	add_child(far)

	if water_enabled:
		_build_water()
	if _drawing():
		if atmosphere_enabled:
			_build_atmosphere()
		if clouds_enabled:
			_build_clouds()
	# Landmarks are built even without a renderer: they are collidable geometry
	# and a navigation target, not decoration, and the checks that fly to them
	# run headless.
	if landmark_count > 0:
		_build_landmarks()


## Material for the terrain mesh.
##
## Vertex colours carry the hypsometric ramp either way -- without something
## reading them the planet renders in Godot's default white. The shader adds
## what the ramp alone cannot: close-range texture and bare rock on slopes.
func _terrain_material() -> Material:
	if not _drawing():
		var plain := StandardMaterial3D.new()
		plain.vertex_color_use_as_albedo = true
		plain.roughness = 0.95
		plain.metallic = 0.0
		return plain

	var mat := ShaderMaterial.new()
	mat.shader = load("res://shaders/terrain.gdshader")
	# Frequencies are per texture, and the shader scales the sample point, so
	# these two only have to be broadband -- the scale uniforms do the choosing.
	mat.set_shader_parameter("detail_noise", _noise_texture(terrain_seed + 11, 0.04, 4))
	mat.set_shader_parameter("macro_noise", _noise_texture(terrain_seed + 12, 0.01, 3))
	mat.set_shader_parameter("detail_scale", detail_texture_scale)
	# Macro variation is sized against the planet, so a small moon and a large
	# world both get roughly the same number of patches across their faces.
	mat.set_shader_parameter("macro_scale", 0.9 / maxf(surface_radius, 1.0))
	mat.set_shader_parameter("rock_color", color_rock)
	mat.set_shader_parameter("emission_color", lava_color)
	mat.set_shader_parameter("emission_strength", lava_strength)
	# Glow starts at sea level and reaches full strength at the deepest the
	# basins go, which is the same floor the ocean would have sat on.
	mat.set_shader_parameter("emission_radius", surface_radius)
	mat.set_shader_parameter("emission_depth",
		maxf(absf(get_min_terrain_height()), 0.001))
	return mat


func _build_water() -> void:
	var ocean := MeshInstance3D.new()
	ocean.name = "Ocean"
	ocean.mesh = PlanetMesh.build(null, water_resolution, surface_radius)

	if _drawing():
		var mat := ShaderMaterial.new()
		mat.shader = load("res://shaders/ocean.gdshader")
		mat.set_shader_parameter("shallow_color", water_color)
		mat.set_shader_parameter("deep_color", water_deep_color)
		ocean.material_override = mat
	else:
		var plain := StandardMaterial3D.new()
		plain.albedo_color = water_color
		plain.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		plain.cull_mode = BaseMaterial3D.CULL_DISABLED
		ocean.material_override = plain

	add_child(ocean)


## The glow around the limb. A sphere shell a few percent larger than the
## planet, drawn as added light.
##
## Deliberately outside the terrain rather than a post-process: it has to sit
## correctly in front of and behind other bodies, and a screen-space halo does
## not survive a moon passing between the camera and the planet.
func _build_atmosphere() -> void:
	var shell := MeshInstance3D.new()
	shell.name = "Atmosphere"
	var sphere := SphereMesh.new()
	var r: float = surface_radius * (1.0 + atmosphere_height)
	sphere.radius = r
	sphere.height = r * 2.0
	sphere.radial_segments = 64
	sphere.rings = 32
	shell.mesh = sphere
	shell.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF

	var mat := ShaderMaterial.new()
	mat.shader = load("res://shaders/atmosphere.gdshader")
	mat.set_shader_parameter("atmosphere_color", atmosphere_color)
	mat.set_shader_parameter("sunset_color", atmosphere_sunset_color)
	mat.set_shader_parameter("density", atmosphere_density)
	# The star is at the origin of the system and never moves, so where it is
	# can be written once instead of pushed to the GPU every frame.
	mat.set_shader_parameter("sun_position", Vector3.ZERO)
	shell.material_override = mat
	add_child(shell)


## The cloud deck. A second shell, spun slowly against the ground by
## `update_orbit` so weather is not painted onto the terrain.
func _build_clouds() -> void:
	var deck := MeshInstance3D.new()
	deck.name = "Clouds"
	var sphere := SphereMesh.new()
	var r: float = surface_radius * (1.0 + cloud_height)
	sphere.radius = r
	sphere.height = r * 2.0
	sphere.radial_segments = 96
	sphere.rings = 48
	deck.mesh = sphere
	deck.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF

	var mat := ShaderMaterial.new()
	mat.shader = load("res://shaders/clouds.gdshader")
	# Three octaves, not the terrain's four: a fourth adds a speckle at cloud
	# scale that reads as television static rather than weather.
	mat.set_shader_parameter("cloud_noise", _noise_texture(terrain_seed + 31, 0.015, 3))
	mat.set_shader_parameter("cloud_color", cloud_color)
	mat.set_shader_parameter("coverage", cloud_coverage)
	# Sized against the planet, so coverage means the same thing on any body.
	mat.set_shader_parameter("noise_scale", 1.3 / maxf(surface_radius, 1.0))
	mat.set_shader_parameter("softness", 0.3)
	mat.set_shader_parameter("opacity", 0.62)
	deck.material_override = mat
	add_child(deck)
	_clouds = deck


## Signed "landness" for a planet-LOCAL direction: above zero is dry ground,
## below zero is seabed, and zero is exactly the coastline.
##
## This is the quantity `get_terrain_height` branches on, exposed on its own
## because its *sign* decides land or sea and its *gradient* points toward the
## nearest coast. The detail and ridge layers do neither: they are what makes a
## continent bumpy, not what makes it a continent. Sampling this alone is also a
## single noise lookup rather than three, which matters when a batch of agents
## asks for a gradient every physics tick.
##
## Smooth across the shoreline, so it can be followed. A boolean land test
## cannot be: it is flat on both sides and tells a searcher nothing about which
## way to go.
func get_land_signal(dir: Vector3) -> float:
	if not terrain_enabled:
		return 1.0
	return _continents.get_noise_3dv(dir) + sea_level_bias


## True where there is dry ground rather than sea, for a planet-LOCAL direction.
func is_land_at(dir: Vector3) -> bool:
	return get_land_signal(dir) > 0.0


## Radial offset from `surface_radius` for a planet-LOCAL direction, which must
## be normalized. Negative inside basins, which is what gives the ocean
## somewhere to sit; `get_min_terrain_height` is the floor.
##
## Pure function of direction and seed, so it is safe to call from any thread
## and gives the same planet on every run.
func get_terrain_height(dir: Vector3) -> float:
	if not terrain_enabled:
		return 0.0

	# Positive is land, negative is basin.
	var mask: float = get_land_signal(dir)
	if mask <= 0.0:
		# Shelve smoothly from the shoreline down to the deep, so the seabed
		# does not drop off a cliff the moment it crosses sea level.
		return -smoothstep(0.0, 0.35, -mask) * ocean_depth * terrain_amplitude * surface_radius
	mask = minf(mask, 1.0)

	# Ease the coastline in so land does not begin at a cliff edge.
	var shore: float = smoothstep(0.0, shore_easing, mask)

	var rolling: float = _detail.get_noise_3dv(dir) * 0.5 + 0.5
	var ridges: float = _mountains.get_noise_3dv(dir) * 0.5 + 0.5

	# Squaring the ridges keeps foothills low and peaks sharp.
	var h: float = rolling * detail_weight + ridges * ridges * mask * ridge_weight
	return h * shore * terrain_amplitude * surface_radius


## Radius of the terrain surface along a planet-local direction.
func get_surface_radius_at(dir: Vector3) -> float:
	return surface_radius + get_terrain_height(dir)


## Angular half-width of a collision patch, in radians.
func _patch_angle() -> float:
	return minf(patch_max_angle, patch_extent / surface_radius)


## Ceiling on terrain height. Not reached in practice — the layers rarely peak
## together — so the colour ramp is scaled against a fraction of it.
func get_max_terrain_height() -> float:
	if not terrain_enabled:
		return 0.0
	return (detail_weight + ridge_weight) * terrain_amplitude * surface_radius


## Colour for a given terrain height, used as vertex colour on the mesh.
func get_terrain_color(h: float) -> Color:
	if h < 0.0:
		var deep := absf(get_min_terrain_height())
		var t := clampf(-h / maxf(deep, 0.0001), 0.0, 1.0)
		return color_shallow.lerp(color_deep, sqrt(t))

	# Measured peaks land near two thirds of the ceiling, so scale to that or
	# the snow line never appears.
	var t2 := clampf(h / maxf(get_max_terrain_height() * 0.66, 0.0001), 0.0, 1.0)
	# The bands below are where the snow line, the treeline and the beach sit.
	# They were retuned once already: with ridge_weight raised to give the
	# worlds real mountains, a peak band starting at 0.6 put snow over most of
	# every continent and the temperate world came out white.
	# The shore band is deliberately narrow. Most of a continent sits low, so a
	# wide one paints whole interiors in beach sand -- measured, it turned a
	# temperate world into a pale desert with a green fringe.
	if t2 < 0.05:
		return color_shore.lerp(color_low, t2 / 0.05)
	if t2 < 0.55:
		return color_low.lerp(color_high, (t2 - 0.05) / 0.5)
	if t2 < 0.84:
		return color_high
	return color_high.lerp(color_peak, (t2 - 0.84) / 0.16)


## Deepest the seabed can ever go, relative to surface_radius. Always <= 0.
## The sea-level backstop collider is sized against this.
func get_min_terrain_height() -> float:
	if not terrain_enabled:
		return 0.0
	return -ocean_depth * terrain_amplitude * surface_radius



# --- The largest continent ---------------------------------------------------

## Directions sampled when locating the largest landmass. A continent is a big
## target, so this only has to be fine enough to tell one apart from another --
## and the flood fill below is quadratic, so doubling it costs four times as
## much for no gain in the answer.
const LANDMASS_SAMPLES: int = 1200

## Unit direction, in this planet's own frame, of the middle of the biggest
## connected landmass. Vector3.ZERO on a planet with no land at all.
var largest_landmass_center: Vector3 = Vector3.ZERO
## Angular radius of that landmass in radians -- the farthest any of its points
## sits from the centre. Useful as a scale, not as a membership test: continents
## are not discs, which is what `is_on_largest_landmass` is for.
var largest_landmass_extent: float = 0.0
## Fraction of the sphere it covers.
var largest_landmass_share: float = 0.0

## Its sampled points, kept so membership can be tested exactly rather than by
## pretending the landmass is a circle.
var _landmass_points: PackedVector3Array = PackedVector3Array()
## Angular spacing of the sample grid, which sets how close a direction has to
## be to a sampled point to count as the same landmass.
var _landmass_reach: float = 0.0


## Locate the biggest connected stretch of dry land, once, at startup.
##
## The agent is asked to land on a specific continent, which means something has
## to know where that continent is. Nothing else in the project does: terrain is
## a pure function of direction, so "biggest landmass" is not represented
## anywhere until it is searched for. This runs a flood fill over an even
## sampling of the sphere and keeps the largest connected component.
##
## Costs a few hundred milliseconds per planet at load and never runs again.
func _find_largest_landmass() -> void:
	if not terrain_enabled:
		return

	var directions := _fibonacci_sphere(LANDMASS_SAMPLES)
	var land := PackedInt32Array()
	land.resize(LANDMASS_SAMPLES)
	for i in LANDMASS_SAMPLES:
		land[i] = 1 if get_land_signal(directions[i]) > 0.0 else 0

	# N points spread over a sphere of area 4*PI leave each with 4*PI/N, hence
	# this spacing; 1.5x of it reaches the immediate ring of neighbours without
	# stepping across a one-point channel of sea.
	_landmass_reach = 1.5 * sqrt(4.0 * PI / float(LANDMASS_SAMPLES))
	var reach_squared: float = _landmass_reach * _landmass_reach

	var seen := PackedInt32Array()
	seen.resize(LANDMASS_SAMPLES)
	var best: PackedInt32Array = PackedInt32Array()

	for start in LANDMASS_SAMPLES:
		if seen[start] == 1 or land[start] == 0:
			continue
		var component := PackedInt32Array()
		var stack: Array[int] = [start]
		seen[start] = 1
		while not stack.is_empty():
			var current: int = stack.pop_back()
			component.append(current)
			for other in LANDMASS_SAMPLES:
				if seen[other] == 1 or land[other] == 0:
					continue
				if directions[current].distance_squared_to(directions[other]) <= reach_squared:
					seen[other] = 1
					stack.append(other)
		if component.size() > best.size():
			best = component

	if best.is_empty():
		push_warning("%s has terrain enabled but no dry land to land on." % name)
		return

	var centre := Vector3.ZERO
	_landmass_points.resize(best.size())
	for k in best.size():
		var dir: Vector3 = directions[best[k]]
		_landmass_points[k] = dir
		centre += dir

	# Summing unit vectors and renormalising gives the spherical centroid. For a
	# landmass wrapped most of the way round a planet the sum can cancel to
	# nothing; falling back to any member keeps the target well defined.
	largest_landmass_center = centre.normalized() if centre.length_squared() > 1e-6 \
		else _landmass_points[0]
	largest_landmass_share = float(best.size()) / float(LANDMASS_SAMPLES)

	var extent: float = 0.0
	for dir in _landmass_points:
		extent = maxf(extent, acos(clampf(dir.dot(largest_landmass_center), -1.0, 1.0)))
	largest_landmass_extent = extent


## True when a planet-LOCAL direction is dry ground belonging to the biggest
## landmass, rather than to some other island.
##
## Tested against the sampled points rather than against a circle around the
## centre, because a continent is not a disc and a circular test would both
## accept sea off its coast and reject its own peninsulas.
func is_on_largest_landmass(dir: Vector3) -> bool:
	if _landmass_points.is_empty() or get_land_signal(dir) <= 0.0:
		return false
	var reach_squared: float = _landmass_reach * _landmass_reach
	for point in _landmass_points:
		if dir.distance_squared_to(point) <= reach_squared:
			return true
	return false


## Angle in radians between a planet-LOCAL direction and the target continent.
func angle_to_largest_landmass(dir: Vector3) -> float:
	if largest_landmass_center == Vector3.ZERO:
		return 0.0
	return acos(clampf(dir.dot(largest_landmass_center), -1.0, 1.0))


## Evenly spaced points on a sphere, via the golden angle. Deterministic, so
## every run and every client agrees on where the continent is.
static func _fibonacci_sphere(count: int) -> Array[Vector3]:
	var points: Array[Vector3] = []
	points.resize(count)
	var golden: float = PI * (3.0 - sqrt(5.0))
	for i in count:
		var y: float = 1.0 - 2.0 * (float(i) + 0.5) / float(count)
		var radius: float = sqrt(maxf(1.0 - y * y, 0.0))
		var theta: float = golden * float(i)
		points[i] = Vector3(cos(theta) * radius, y, sin(theta) * radius)
	return points


# --- Ocean ------------------------------------------------------------------

## True when a world-space point is below sea level.
##
## The ocean carries no collision shape at all: it is a perfect sphere of known
## radius, so submersion is one comparison rather than a physics query. That is
## also what lets the ship fly into the sea instead of bouncing off it.
func is_underwater(world_pos: Vector3) -> bool:
	if not (terrain_enabled and water_enabled):
		return false
	return global_position.distance_squared_to(world_pos) < surface_radius * surface_radius


## Depth below sea level at a world-space point. Negative above the surface.
func get_depth_at(world_pos: Vector3) -> float:
	return surface_radius - global_position.distance_to(world_pos)


## Guarantee real collision exists under a world-space point for `client`.
## Cheap enough to call every tick: it only rebuilds once the client nears the
## edge of its own patch, and the rebuild happens on a worker thread.
##
## The old shape stays live until the new one is ready, so there is never a
## frame without collision under the client.
func update_collision_for(client: Node3D, world_pos: Vector3) -> void:
	if not terrain_enabled:
		return

	# to_local, not a world-space offset: patches are stored under the planet's
	# own transform, and terrain spins with the planet.
	var dir := to_local(world_pos).normalized()
	if dir.is_zero_approx():
		return

	var patch := _patch_for(client)
	_adopt_finished(patch)
	if patch.task != -1:
		return

	# Rebuild once the client has drifted a third of the way to the patch edge.
	if patch.centre != Vector3.ZERO and dir.dot(patch.centre) > cos(_patch_angle() * 0.33):
		return

	# build_patch reads only this planet's noise and scalars and touches no
	# scene state, so it is safe to run concurrently.
	patch.pending_centre = dir
	patch.task = WorkerThreadPool.add_task(_build_threaded.bind(patch))


## Build a client's patch synchronously. Used when something needs ground under
## it this instant and cannot wait a frame — spawning onto a surface.
func build_collision_now(client: Node3D, world_pos: Vector3) -> void:
	if not terrain_enabled:
		return
	var dir := to_local(world_pos).normalized()
	if dir.is_zero_approx():
		return
	var patch := _patch_for(client)
	patch.centre = dir
	patch.collider.shape = PlanetMesh.build_patch(
		self, dir, _patch_angle(), patch_resolution)
	_populate_scatter(patch)


## Drop a client's patch once it is no longer on this planet, so colliders do
## not accumulate across every body the ship visits.
func release_client(client: Node3D) -> void:
	var patch: Patch = _patches.get(client)
	if patch == null:
		return
	if patch.task != -1:
		WorkerThreadPool.wait_for_task_completion(patch.task)
	patch.collider.queue_free()
	if patch.scatter != null:
		patch.scatter.queue_free()
	_patches.erase(client)


func _patch_for(client: Node3D) -> Patch:
	var patch: Patch = _patches.get(client)
	if patch != null:
		return patch

	patch = Patch.new()
	patch.collider = CollisionShape3D.new()
	patch.collider.name = "TerrainPatch_" + client.name
	add_child(patch.collider)
	_patches[client] = patch
	return patch


func _build_threaded(patch: Patch) -> void:
	patch.result = PlanetMesh.build_patch(
		self, patch.pending_centre, _patch_angle(), patch_resolution)


func _adopt_finished(patch: Patch) -> void:
	if patch.task == -1 or not WorkerThreadPool.is_task_completed(patch.task):
		return
	# Required even after is_task_completed: it releases the task's resources.
	WorkerThreadPool.wait_for_task_completion(patch.task)
	patch.task = -1
	if patch.result != null:
		patch.collider.shape = patch.result
		patch.centre = patch.pending_centre
		patch.result = null
		_populate_scatter(patch)


## Place the surface sites, at positions that are a pure function of the seed.
##
## Spread by rejection rather than by construction: candidates that are at sea,
## on a slope too steep to stand a mast on, or too close to a site already
## placed are thrown away and the next hash is tried. Simple, and it cannot
## produce the clustered-at-a-pole result that sampling angles uniformly does.
func _build_landmarks() -> void:
	var placed: Array[Vector3] = []
	var attempt: int = 0
	while landmarks.size() < landmark_count and attempt < 600:
		var index: int = attempt
		attempt += 1
		# Two hashes into a direction, rejection-sampled onto the sphere so the
		# distribution is uniform over area rather than over latitude.
		var dir := Vector3(
			_hash01(index, 0, 101) * 2.0 - 1.0,
			_hash01(index, 0, 102) * 2.0 - 1.0,
			_hash01(index, 0, 103) * 2.0 - 1.0)
		if dir.length_squared() < 0.05 or dir.length_squared() > 1.0:
			continue
		dir = dir.normalized()

		if not is_land_at(dir):
			continue
		var too_close: bool = false
		for other in placed:
			if other.dot(dir) > cos(0.55):
				too_close = true
				break
		if too_close:
			continue
		if _slope_at(dir) > 0.35:
			continue

		var site := Landmark.new()
		site.kind = ([Landmark.Kind.BEACON, Landmark.Kind.WRECK,
			Landmark.Kind.OUTCROP][landmarks.size() % 3]) as Landmark.Kind
		site.planet = self
		site.site_id = "%d-%d" % [terrain_seed, landmarks.size() + 1]
		site.name = "Landmark%d" % (landmarks.size() + 1)
		# Standing on the ground, with its own up along the local vertical.
		var up: Vector3 = dir
		var side: Vector3 = up.cross(Vector3.UP)
		if side.length_squared() < 0.001:
			side = up.cross(Vector3.RIGHT)
		site.transform = Transform3D(
			Basis.looking_at(side.normalized(), up),
			dir * get_surface_radius_at(dir))
		add_child(site)
		landmarks.append(site)
		placed.append(dir)


## Steepness of the ground at a planet-local direction, as a gradient. Sampled
## from the height field either side of the point rather than read off the mesh,
## which is cheaper and cannot disagree with where the collision will be.
func _slope_at(dir: Vector3) -> float:
	var axis_a: Vector3 = dir.cross(Vector3.UP)
	if axis_a.length_squared() < 0.001:
		axis_a = dir.cross(Vector3.RIGHT)
	axis_a = axis_a.normalized()
	var axis_b: Vector3 = dir.cross(axis_a)
	var probe: float = 0.01
	var here: float = get_surface_radius_at(dir)
	var da: float = get_surface_radius_at((dir + axis_a * probe).normalized())
	var db: float = get_surface_radius_at((dir + axis_b * probe).normalized())
	var run: float = probe * surface_radius
	return Vector2(da - here, db - here).length() / maxf(run, 0.001)


## Fill a patch's boulder field for wherever it has just been recentred.
##
## Placement is a hash of the boulder's own grid cell rather than a running
## random sequence, so a patch rebuilt after the client walks away and comes
## back puts the same rocks in the same places. A sequence would reshuffle the
## landscape every time it streamed, which is worse than having no rocks: the
## ground would visibly change behind you.
func _populate_scatter(patch: Patch) -> void:
	if not scatter_enabled or not _drawing() or scatter_count <= 0:
		return
	if patch.centre.is_zero_approx():
		return
	if patch.scatter != null and patch.scatter_centre.dot(patch.centre) > 0.9999:
		return

	if patch.scatter == null:
		patch.scatter = MultiMeshInstance3D.new()
		patch.scatter.name = "Scatter_" + patch.collider.name
		patch.scatter.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		var multi := MultiMesh.new()
		multi.transform_format = MultiMesh.TRANSFORM_3D
		multi.mesh = _boulder_mesh()
		patch.scatter.multimesh = multi
		add_child(patch.scatter)

	var up: Vector3 = patch.centre.normalized()
	var axis_a: Vector3 = up.cross(Vector3.UP)
	if axis_a.length_squared() < 0.001:
		axis_a = up.cross(Vector3.RIGHT)
	axis_a = axis_a.normalized()
	var axis_b: Vector3 = up.cross(axis_a)

	# Sample inside the patch the collision covers and no further: a boulder
	# past its edge stands on ground that has no collision under it, and an
	# astronaut walking out to it drops through the world to reach it.
	var reach: float = _patch_angle() * 0.92
	var multimesh: MultiMesh = patch.scatter.multimesh
	var placed: int = 0
	var transforms: Array[Transform3D] = []
	transforms.resize(scatter_count)

	# A grid of cells with one jittered boulder each, rather than free random
	# points: free points clump, and a clump of boulders reads as a pile
	# somebody left rather than as a scattered field.
	var side: int = maxi(1, int(ceil(sqrt(float(scatter_count)))))
	for index in scatter_count:
		var cell_x: int = index % side
		var cell_y: int = index / side
		var jitter_x: float = _hash01(cell_x, cell_y, 1)
		var jitter_y: float = _hash01(cell_x, cell_y, 2)
		var u: float = ((cell_x + jitter_x) / side * 2.0 - 1.0) * reach
		var v: float = ((cell_y + jitter_y) / side * 2.0 - 1.0) * reach
		var dir: Vector3 = (up + axis_a * u + axis_b * v).normalized()

		if not is_land_at(dir):
			continue
		# Slope from the height field directly, sampled either side of the
		# point. Cheaper and more honest than reading it back off the mesh.
		var probe: float = _patch_angle() * 0.02
		var here: float = get_surface_radius_at(dir)
		var da: float = get_surface_radius_at((dir + axis_a * probe).normalized())
		var db: float = get_surface_radius_at((dir + axis_b * probe).normalized())
		var run: float = probe * surface_radius
		var slope: Vector2 = Vector2(da - here, db - here) / maxf(run, 0.001)
		if slope.length() > scatter_max_slope:
			continue

		var scale: float = lerpf(scatter_size.x, scatter_size.y,
			_hash01(cell_x, cell_y, 3))
		var spin: float = _hash01(cell_x, cell_y, 4) * TAU
		# Sunk slightly, so they sit in the ground instead of resting on it.
		var at: Vector3 = dir * (here - scale * 0.28)
		var basis := Basis.looking_at(axis_a, dir).rotated(dir, spin).scaled(
			Vector3(scale, scale, scale))
		transforms[placed] = Transform3D(basis, at)
		placed += 1

	if OS.is_stdout_verbose():
		print("scatter %s: %d of %d placed" % [name, placed, scatter_count])
	multimesh.instance_count = placed
	for i in placed:
		multimesh.set_instance_transform(i, transforms[i])
	patch.scatter_centre = patch.centre


## Deterministic [0, 1) from three integers. Placement has to be a function of
## where a boulder is, not of the order it was generated in.
func _hash01(x: int, y: int, salt: int) -> float:
	var h: int = (x * 73856093) ^ (y * 19349663) ^ (salt * 83492791) ^ (terrain_seed * 2654435761)
	h = h & 0x7fffffff
	h = (h ^ (h >> 13)) * 1274126177
	h = h & 0x7fffffff
	return float(h % 1000003) / 1000003.0


## One lumpy rock, shared by every boulder on this planet. A sphere pushed
## around by the planet's own noise: distinct enough at the size these are
## drawn, and it costs one mesh rather than an art pipeline.
func _boulder_mesh() -> ArrayMesh:
	if _boulder != null:
		return _boulder
	var source := SphereMesh.new()
	source.radial_segments = 10
	source.rings = 6
	source.radius = 0.5
	source.height = 1.0

	var lump := FastNoiseLite.new()
	lump.seed = terrain_seed + 71
	lump.frequency = 1.4

	var arrays: Array = source.get_mesh_arrays()
	var verts: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
	for i in verts.size():
		var v: Vector3 = verts[i]
		var n: Vector3 = v.normalized()
		verts[i] = v * (1.0 + lump.get_noise_3dv(n * 3.0) * 0.45)
	arrays[Mesh.ARRAY_VERTEX] = verts
	# The normals no longer match the vertices they were generated for, and a
	# faceted rock is the look wanted anyway.
	arrays[Mesh.ARRAY_NORMAL] = null
	arrays[Mesh.ARRAY_TANGENT] = null

	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)

	var mat := StandardMaterial3D.new()
	mat.albedo_color = scatter_color if scatter_color.a > 0.0 else color_rock
	mat.roughness = 0.94
	mat.metallic = 0.0
	mesh.surface_set_material(0, mat)
	_boulder = mesh
	return _boulder


# Never leave a worker running against a freed planet.
func _exit_tree() -> void:
	for patch: Patch in _patches.values():
		if patch.task != -1:
			WorkerThreadPool.wait_for_task_completion(patch.task)
			patch.task = -1
			patch.result = null


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

	# Clouds are a child, so they already have the planet's spin. This is the
	# extra drift *relative to the ground* -- without it the deck is painted on
	# and the weather never changes.
	if _clouds != null:
		_clouds.rotation = Vector3(0.0, deg_to_rad(cloud_drift) * time, 0.0)


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
## What to call this body on screen. Falls back to the node name, so a body
## with no authored name is still identifiable rather than blank.
func label() -> String:
	return display_name if display_name != "" else name


func how_many_parents() -> int:
	if _orbital_parent == null:
		return 0
	return _orbital_parent.how_many_parents() + 1
