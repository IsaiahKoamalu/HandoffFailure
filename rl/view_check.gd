extends Node3D

## Render one frame from the pilot's seat and save it, so a change to the hull
## can be looked at rather than argued about.
##
##     Godot --path . res://rl/view_check.tscn -- --out=/tmp/cockpit.png
##
## Not headless: headless has no renderer and this is entirely about what is on
## screen. It quits itself after writing the file.

const MAIN_SCENE := "res://main.tscn"

var _out: String = "cockpit.png"
## Hide the nose panels, for a reference frame of the view without them.
var _bare: bool = false


func _ready() -> void:
	for argument in OS.get_cmdline_user_args():
		var pair := argument.split("=", true, 1)
		if pair.size() == 2 and pair[0] == "--out":
			_out = pair[1]
		elif pair.size() == 2 and pair[0] == "--bare":
			_bare = int(pair[1]) != 0

	var world: Node = load(MAIN_SCENE).instantiate()
	add_child(world)
	var ship: Ship = world.get_node("Ship")
	var planet: Planet = null
	for candidate in SolarSystem.Instance.Planets:
		if candidate.terrain_enabled:
			planet = candidate
			break

	# Park it well off a planet and point at it: a view with something in it
	# says more about an obstruction than a view of empty space, and far enough
	# out that the planet is an object rather than a wall of terrain.
	var out: Vector3 = Vector3(0.0, 0.35, 1.0).normalized()
	var here: Vector3 = planet.global_position + out * (planet.surface_radius * 3.0)
	ship.global_transform = Transform3D(
		Basis.looking_at(planet.global_position - here, Vector3.UP), here)
	ship.linear_velocity = planet.constant_linear_velocity
	ship.board()

	# Optionally strip the nose panels, so the same frame can be rendered with
	# and without them and the difference is the obstruction and nothing else.
	if _bare:
		for node_name in ["NoseHatch", "NoseCheekLeft", "NoseCheekRight", "Windshield"]:
			var panel: Node3D = ship.get_node_or_null(node_name)
			if panel != null:
				panel.visible = false

	await get_tree().process_frame
	await get_tree().process_frame
	await RenderingServer.frame_post_draw
	var image := get_viewport().get_texture().get_image()
	image.save_png(_out)
	print("wrote %s (%dx%d)" % [_out, image.get_width(), image.get_height()])
	get_tree().quit()
