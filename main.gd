extends Node3D
class_name SolarSystem

# How far out a planet still counts as the body you are standing on / falling
# toward, as a multiple of its surface radius.
const CAPTURE_RADIUS_FACTOR: float = 2.0

static var Instance: SolarSystem
var Planets: Array[Planet] = []

func _ready() -> void:
	Instance = self

	# Collect children that are Planet instances
	Planets.clear()
	for child in get_children():
		if child is Planet:
			Planets.append(child)

	# Order planets by how deep they are in the hierarchy (fewest parents first)
	Planets.sort_custom(func(a: Planet, b: Planet) -> bool:
		return a.how_many_parents() < b.how_many_parents()
	)

	# Initialize in parent-first order
	for planet in Planets:
		planet.init()
		

# Total acceleration due to every planet's gravity at a world-space point.
func get_acceleration_at_position(world_pos: Vector3) -> Vector3:
	var total := Vector3.ZERO
	for planet in Planets:
		total += planet.get_acceleration_at_position(world_pos)
	return total


# The planet whose gravity dominates at a world-space point, or null when the
# point is not inside any planet's sphere of influence. Used to decide which
# frame of reference a body should be moving in.
func get_dominant_planet(world_pos: Vector3) -> Planet:
	var dominant: Planet = null
	var strongest := 0.0
	for planet in Planets:
		if planet.global_position.distance_to(world_pos) >= _capture_radius(planet):
			continue
		var magnitude := planet.get_acceleration_at_position(world_pos).length()
		if magnitude > strongest:
			dominant = planet
			strongest = magnitude
	return dominant


func _capture_radius(planet: Planet) -> float:
	return CAPTURE_RADIUS_FACTOR * planet.surface_radius


# Release the mouse so the user can click UI, regardless of which body they are
# currently controlling. Lives here rather than on the ship or the astronaut so
# it keeps working across a swap.
func _input(event: InputEvent) -> void:
	if event.is_action_pressed("ui_cancel"):
		if Input.get_mouse_mode() != Input.MOUSE_MODE_VISIBLE:
			Input.set_mouse_mode(Input.MOUSE_MODE_VISIBLE)
		else:
			Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)


# Button to exit the sim
func _on_button_quit_pressed() -> void:
	get_tree().quit()
