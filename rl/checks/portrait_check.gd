extends Node3D

## Photograph every world in the system, so a change to how a planet looks can
## be looked at rather than argued about.
##
##     Godot --path . res://rl/checks/portrait_check.tscn -- --out=/tmp/shots --dist=2.5
##
## Not headless: headless has no renderer, and this is entirely about what is on
## screen. Not through the cockpit either, the way view_check is — the hull glass
## tints and blooms over everything, which is the right test for the hull and
## the wrong one for a planet.
##
## The camera is placed three-quarters lit: straight down the sun line gives a
## flat disc with no terminator, and from behind gives a silhouette. Both hide
## exactly what a change to shading is meant to alter.

const MAIN_SCENE := "res://sim/main.tscn"

var _out: String = "."
## Distance from the surface, in planet radii.
var _dist: float = 2.0
## Render from ground level instead: standing on the largest continent.
var _surface: bool = false
var _suffix: String = ""


func _ready() -> void:
	for argument in OS.get_cmdline_user_args():
		var pair := argument.split("=", true, 1)
		if pair.size() != 2:
			continue
		match pair[0]:
			"--out": _out = pair[1]
			"--dist": _dist = float(pair[1])
			"--surface": _surface = int(pair[1]) != 0
			"--suffix": _suffix = pair[1]

	var world: Node = load(MAIN_SCENE).instantiate()
	add_child(world)
	# The player is not in shot, and the HUD would be painted over every frame.
	for node_name in ["HUD", "Ship", "Astronaut"]:
		var node: Node = world.get_node_or_null(node_name)
		if node != null:
			world.remove_child(node)
			node.queue_free()

	var camera := Camera3D.new()
	camera.far = 100000.0
	add_child(camera)
	camera.current = true

	await get_tree().process_frame
	for planet in SolarSystem.Instance.Planets:
		if planet.name == "Sun":
			continue
		_aim(camera, planet)
		# Two frames plus the draw: the first settles the transform, the second
		# is what the shaders see.
		await get_tree().process_frame
		await get_tree().process_frame
		await RenderingServer.frame_post_draw
		var path: String = "%s/%s%s.png" % [_out, planet.name.to_lower(), _suffix]
		get_viewport().get_texture().get_image().save_png(path)
		print("wrote %s" % path)
	get_tree().quit()


func _aim(camera: Camera3D, planet: Planet) -> void:
	# The star sits at the origin, so this is the direction the light comes from.
	var sunward: Vector3 = (Vector3.ZERO - planet.global_position).normalized()
	var side: Vector3 = sunward.cross(Vector3.UP).normalized()
	if side.length_squared() < 1e-6:
		side = sunward.cross(Vector3.RIGHT).normalized()

	if _surface:
		# Standing on the target continent, looking along the ground. Whatever
		# the terrain does close up is a different question from what it does
		# from orbit, and a change can improve one while ruining the other.
		var local: Vector3 = planet.largest_landmass_center
		if local == Vector3.ZERO:
			local = Vector3.UP
		var up: Vector3 = (planet.global_basis * local).normalized()
		var here: Vector3 = planet.global_position \
			+ up * (planet.get_surface_radius_at(local) + 2.0)
		var along: Vector3 = up.cross(side).normalized()
		camera.global_transform = Transform3D(
			Basis.looking_at(along - up * 0.12, up), here)
		return

	# Three-quarters lit, and raised so the pole is not edge on. Toward the star,
	# not away from it: the first version had this sign wrong and photographed
	# five night sides.
	var out: Vector3 = (sunward * 0.55 + side * 0.75 + Vector3.UP * 0.36).normalized()
	var here: Vector3 = planet.global_position + out * (planet.surface_radius * (1.0 + _dist))
	camera.global_transform = Transform3D(
		Basis.looking_at(planet.global_position - here, Vector3.UP), here)
