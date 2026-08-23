extends Node3D
class_name SolarSystem

# How far out a planet still counts as the body you are standing on / falling
# toward, as a multiple of its surface radius.
const CAPTURE_RADIUS_FACTOR: float = 2.0

## Seconds of simulation clock elapsed. Planet positions are a pure function of
## this, so scaling it fast-forwards the system exactly rather than approximately.
@export var time_scale: float = 1.0

static var Instance: SolarSystem
var Planets: Array[Planet] = []
var time: float = 0.0

## Bodies that need real ground beneath them. The astronaut is the obvious one,
## but the ship's _ground_cast and ramp probe raycast against terrain too, and
## both fail silently against ground that was never generated.
var _terrain_clients: Array[Node3D] = []
## Which planet each client was on last tick, so its patch can be released.
var _client_planet: Dictionary = {}

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

	# Planet meshes are pure functions of their own seeds, so build them all at
	# once instead of one after another. Attaching them touches the scene tree,
	# so that part stays on the main thread.
	if Planets.size() > 0:
		var task := WorkerThreadPool.add_group_task(_generate_planet_mesh, Planets.size())
		WorkerThreadPool.wait_for_group_task_completion(task)
	for planet in Planets:
		planet.apply_terrain()


func _generate_planet_mesh(index: int) -> void:
	Planets[index].generate_terrain_mesh()


# Planets are driven from here rather than from their own _physics_process so
# they always advance in parent-first order: a moon reads its planet's position
# for the current tick, not last tick's.
func _physics_process(delta: float) -> void:
	time += delta * time_scale
	for planet in Planets:
		planet.update_orbit(time)

	for client in _terrain_clients:
		var planet := get_dominant_planet(client.global_position)
		# Hand the old planet's patch back when a client moves on, so colliders
		# do not pile up on every body the ship has ever visited.
		var previous: Planet = _client_planet.get(client)
		if previous != planet and previous != null:
			previous.release_client(client)
		_client_planet[client] = planet
		if planet != null:
			planet.update_collision_for(client, client.global_position)


## Ask SolarSystem to keep terrain collision streamed in under this node.
func register_terrain_client(node: Node3D) -> void:
	if not _terrain_clients.has(node):
		_terrain_clients.append(node)


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
		var capture := _capture_radius(planet)
		if planet.global_position.distance_squared_to(world_pos) >= capture * capture:
			continue
		var magnitude := planet.get_gravity_magnitude_at_position(world_pos)
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
