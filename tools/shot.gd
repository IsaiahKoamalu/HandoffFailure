extends Node3D

## Render framed screenshots of the world, so visual work can be looked at
## rather than argued about.
##
##     Godot --path . res://tools/shot.tscn -- --shot=orbit --out=orbit.png
##
## Not headless: headless has no renderer and this is entirely about what is on
## screen. Writes the file and quits.
##
## Shots are named viewpoints rather than free camera parameters on the command
## line. A named shot is reproducible across a change -- which is the entire
## point of having it -- and a hand-typed camera position is not.

const MAIN_SCENE := "res://main.tscn"

## Shots that frame a specific planet take it by index into SolarSystem.Planets.
## -1 means the first one with terrain, which skips the star -- the star is a
## Planet too, and index 0 is therefore a picture of the sun.
var _shot: String = "orbit"
var _out: String = "shot.png"
var _planet_index: int = -1
## Extra seconds of simulation before the shutter, for anything that settles.
var _settle: float = 0.0
## Child nodes of the framed planet to hide before the shutter, by name. For
## answering "which of these three shells is washing the picture out", which is
## not a question you can answer by looking at the finished frame.
var _hide: PackedStringArray = PackedStringArray()

var _world: Node
var _camera: Camera3D


func _ready() -> void:
	for argument in OS.get_cmdline_user_args():
		var pair := argument.split("=", true, 1)
		if pair.size() != 2:
			continue
		match pair[0]:
			"--shot": _shot = pair[1]
			"--out": _out = pair[1]
			"--planet": _planet_index = int(pair[1])
			"--settle": _settle = float(pair[1])
			"--hide": _hide = pair[1].split(",")

	_world = load(MAIN_SCENE).instantiate()
	add_child(_world)

	# The HUD is the subject of its own shots and clutter in every other one.
	if _shot != "hud":
		var hud: Node = _world.get_node_or_null("HUD")
		if hud != null:
			_world.remove_child(hud)
			hud.queue_free()

	# Every other camera in the world stands down first. The ship and the
	# astronaut each carry one, and whichever of them is current renders the
	# frame no matter what this one is doing -- which is how a shot reporting a
	# camera seven metres above the ground came out looking like low orbit.
	for other in _all_cameras(_world):
		other.current = false

	_camera = Camera3D.new()
	_camera.near = 0.05
	_camera.far = 40000.0
	add_child(_camera)
	_camera.current = true

	await _run()


func _all_cameras(node: Node) -> Array[Camera3D]:
	var found: Array[Camera3D] = []
	if node is Camera3D:
		found.append(node)
	for child in node.get_children():
		found.append_array(_all_cameras(child))
	return found


func _planet() -> Planet:
	var planets: Array[Planet] = SolarSystem.Instance.Planets
	if _planet_index >= 0:
		return planets[mini(_planet_index, planets.size() - 1)]
	for candidate in planets:
		if candidate.terrain_enabled:
			return candidate
	return planets[0]


## Point the camera at a target from a given position.
func _look(from: Vector3, at: Vector3, up: Vector3 = Vector3.UP) -> void:
	var forward: Vector3 = (at - from).normalized()
	if absf(forward.dot(up)) > 0.999:
		up = Vector3.RIGHT
	_camera.global_transform = Transform3D(Basis.looking_at(forward, up), from)


## A direction on the planet's surface that is dry land *and* in daylight.
##
## Both halves are needed. Land, or half the surface shots come out looking at
## open water; daylight, or they come out black -- the first version of this
## picked the best land on the planet and it was on the night side.
func _land_direction(planet: Planet) -> Vector3:
	var sun: Node3D = _world.get_node("Sun")
	var toward_sun: Vector3 = planet.to_local(sun.global_position).normalized()
	var best: Vector3 = toward_sun
	var best_score: float = -INF
	var rng := RandomNumberGenerator.new()
	rng.seed = 12345
	for _i in 800:
		var dir := Vector3(rng.randfn(), rng.randfn(), rng.randfn()).normalized()
		var lit: float = dir.dot(toward_sun)
		if lit < 0.35:
			continue
		# Aimed at ground that is comfortably inland rather than at the highest
		# land on the planet. Maximising the land signal finds the middle of the
		# largest continent, which is also its summit, so every surface shot
		# came back as a snowfield and said nothing about the rest of the world.
		var inland: float = planet.get_land_signal(dir)
		var score: float = -absf(inland - 0.22) + lit * 0.25
		if score > best_score:
			best_score = score
			best = dir
	return best


func _run() -> void:
	# Before anything is framed. Planets are placed by update_orbit, which runs
	# on the physics tick, so until one has passed every body is still at its
	# authored editor position -- Kestrel by nearly a thousand units. Framing
	# against that put the ship on a surface the planet then left, and the shot
	# came out as a dark hull against a distant world.
	for _i in 3:
		await get_tree().physics_frame

	var planet: Planet = _planet()
	for hidden in _hide:
		var node: Node3D = planet.get_node_or_null(NodePath(hidden))
		if node != null:
			node.visible = false
		else:
			print("no node '%s' on %s" % [hidden, planet.name])
	var sun: Node3D = _world.get_node("Sun")

	match _shot:
		"orbit":
			# Lit from behind the camera's shoulder rather than straight on: a
			# head-on light flattens every bit of relief the terrain has.
			var toward_sun: Vector3 = (sun.global_position - planet.global_position).normalized()
			var side: Vector3 = toward_sun.cross(Vector3.UP).normalized()
			var out: Vector3 = (toward_sun * 0.8 + side * 0.5 + Vector3.UP * 0.25).normalized()
			_look(planet.global_position + out * planet.surface_radius * 2.6,
				planet.global_position)
		"surface":
			# Standing on the largest landmass, looking along the horizon, so
			# the shot shows terrain and sky rather than a map from above.
			var dir: Vector3 = _land_direction(planet)
			var ground: float = planet.get_surface_radius_at(dir)
			var eye: Vector3 = planet.to_global(dir * (ground + 2.0))
			var north: Vector3 = dir.cross(Vector3.UP).cross(dir).normalized()
			_look(eye, planet.to_global(
				(dir * 0.985 + north * 0.17).normalized() * ground), dir)
		"low":
			# Low pass, a few hundred metres up and looking ahead and down.
			var dir2: Vector3 = _land_direction(planet)
			var ground2: float = planet.get_surface_radius_at(dir2)
			var along: Vector3 = dir2.cross(Vector3.UP).cross(dir2).normalized()
			var eye2: Vector3 = planet.to_global(
				dir2 * (ground2 + planet.surface_radius * 0.06))
			_look(eye2, planet.to_global(
				(dir2 + along * 0.5).normalized() * ground2), dir2)
		"system":
			# The whole system from above and outside, for scale and orbits.
			_look(sun.global_position + Vector3(2600.0, 2400.0, 2600.0),
				sun.global_position)
		"ship", "station", "cockpit", "hud":
			await _stage_hardware(planet)
		_:
			push_error("unknown shot '%s'" % _shot)

	_report(planet)
	await _shutter()


## Shots with the ship or station in them have to put the hardware somewhere
## first, which needs the world to have ticked at least once.
func _stage_hardware(planet: Planet) -> void:
	var ship: Ship = _world.get_node("Ship")
	var station: Station = null
	for candidate in SolarSystem.Instance.Stations:
		station = candidate
		break

	match _shot:
		"ship":
			var dir: Vector3 = _land_direction(planet)
			var ground: float = planet.get_surface_radius_at(dir)
			var up: Vector3 = (planet.global_basis * dir).normalized()
			var along: Vector3 = up.cross(Vector3.UP).normalized()
			# Framed off the transform being assigned, not off the body read
			# back afterwards. A RigidBody3D does not move until the physics
			# step, so reading its position here aims the camera at wherever it
			# used to be -- which put the ship out of frame and the shot a
			# hundred units up, looking at a horizon.
			var place := Transform3D(
				Basis.looking_at(along, up),
				planet.to_global(dir * (ground + 1.6)))
			ship.global_transform = place
			ship.linear_velocity = planet.get_orbital_velocity(place.origin)
			ship.freeze = true
			# up is the planet's local up, not the world's. A surface shot framed
			# against world up comes out rolled by however far the site is from
			# the pole, and a horizon running down the side of the frame reads
			# as a shot from orbit rather than as one from the ground.
			_look(place.origin + up * 2.4 - along * 8.0
				+ along.cross(up) * 6.0, place.origin + up * 0.8, up)
		"station":
			_look(station.global_position + station.global_basis
				* Vector3(26.0, 12.0, -34.0), station.global_position)
		"cockpit", "hud":
			var out: Vector3 = Vector3(0.0, 0.3, 1.0).normalized()
			var here: Vector3 = planet.global_position + out * planet.surface_radius * 2.2
			ship.global_transform = Transform3D(
				Basis.looking_at(planet.global_position - here, Vector3.UP), here)
			ship.linear_velocity = planet.get_orbital_velocity(here)
			ship.board()
			_camera.current = false


## What the frame actually contains, printed rather than inferred from looking
## at it. Framing bugs in a 3D scene look exactly like content bugs.
func _report(planet: Planet) -> void:
	var camera: Vector3 = _camera.global_position
	print("planet %s at %.0f,%.0f,%.0f  r=%.0f" % [planet.name,
		planet.global_position.x, planet.global_position.y,
		planet.global_position.z, planet.surface_radius])
	print("camera %.0f,%.0f,%.0f  %.1f from planet centre (%.2f radii), %.1f above ground"
		% [camera.x, camera.y, camera.z,
			camera.distance_to(planet.global_position),
			camera.distance_to(planet.global_position) / planet.surface_radius,
			camera.distance_to(planet.global_position) - planet.surface_radius])
	var current: Camera3D = get_viewport().get_camera_3d()
	print("rendering from %s" % ("the shot camera" if current == _camera
		else ("'%s'" % current.name if current != null else "NO CAMERA")))
	var ship: Ship = _world.get_node_or_null("Ship")
	if ship != null:
		print("ship   %.1f from camera, %.1f above ground" % [
			ship.global_position.distance_to(camera),
			ship.global_position.distance_to(planet.global_position)
				- planet.surface_radius])


func _shutter() -> void:
	# Terrain streams in on worker threads and lights take a frame to settle.
	var ticks: int = 4 + int(_settle * 60.0)
	for _i in ticks:
		await get_tree().physics_frame
	await get_tree().process_frame
	await RenderingServer.frame_post_draw
	var image := get_viewport().get_texture().get_image()
	image.save_png(_out)
	print("wrote %s (%dx%d)  shot=%s" % [_out, image.get_width(), image.get_height(), _shot])
	get_tree().quit()
