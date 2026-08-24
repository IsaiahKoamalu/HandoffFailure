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
## Body the HUD is navigating toward. Null until the player picks one.
var target: Planet = null

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
## Step to the next body, wrapping through "no target" so it can be cleared.
func cycle_target() -> void:
	if Planets.is_empty():
		return
	var i := Planets.find(target)
	if i < 0:
		target = Planets[0]
	elif i == Planets.size() - 1:
		target = null
	else:
		target = Planets[i + 1]


## Distance from a world point to the target's surface, and how fast it is
## closing. Positive closing speed means approaching. Returns [distance, speed]
## in world units, or an empty array when nothing is targeted.
func get_target_vector(world_pos: Vector3, world_velocity: Vector3) -> Array:
	if target == null:
		return []
	var to_target := target.global_position - world_pos
	var range_to_surface: float = to_target.length() - target.surface_radius
	# The target is moving too, so closing speed is relative.
	var relative := world_velocity - target.constant_linear_velocity
	return [range_to_surface, relative.dot(to_target.normalized())]


# --- Save / load -------------------------------------------------------------

const SAVE_PATH := "user://quicksave.dat"

## A save is small because of two earlier decisions: orbits are analytic, so the
## entire solar system is recoverable from a single float, and terrain is a pure
## function of a seed, so none of it is serialised at all. What is left is where
## the player left their ship and their body.
func save_game() -> bool:
	var ship: Ship = null
	var astronaut: Astronaut = null
	for client in _terrain_clients:
		if client is Ship:
			ship = client
		elif client is Astronaut:
			astronaut = client
	if ship == null or astronaut == null:
		return false

	var data := {
		"version": 1,
		"time": time,
		"target": target.name if target != null else "",
		"ship_transform": ship.global_transform,
		"ship_linear_velocity": ship.linear_velocity,
		"ship_angular_velocity": ship.angular_velocity,
		"ship_piloted": ship.is_piloted,
		"ship_ramp_open": ship.is_ramp_open,
		"ship_headlight": ship.is_headlight_on,
		"ship_cabin_light": ship.is_cabin_switch,
		"astronaut_transform": astronaut.global_transform,
		"astronaut_velocity": astronaut.get_world_velocity(),
		"astronaut_active": astronaut.is_active,
		"astronaut_lamp": astronaut.is_lamp_on,
	}

	var file := FileAccess.open(SAVE_PATH, FileAccess.WRITE)
	if file == null:
		push_warning("Could not write save: %s" % error_string(FileAccess.get_open_error()))
		return false
	file.store_var(data)
	return true


func load_game() -> bool:
	if not FileAccess.file_exists(SAVE_PATH):
		return false
	var file := FileAccess.open(SAVE_PATH, FileAccess.READ)
	if file == null:
		return false
	var data = file.get_var()
	if typeof(data) != TYPE_DICTIONARY or data.get("version", 0) != 1:
		push_warning("Save file is not readable by this version.")
		return false

	var ship: Ship = null
	var astronaut: Astronaut = null
	for client in _terrain_clients:
		if client is Ship:
			ship = client
		elif client is Astronaut:
			astronaut = client
	if ship == null or astronaut == null:
		return false

	# Wind the whole system back first, so every body is where it was before
	# anything is placed relative to it.
	time = data["time"]
	for planet in Planets:
		planet.update_orbit(time)

	target = null
	for planet in Planets:
		if planet.name == data["target"]:
			target = planet

	ship.global_transform = data["ship_transform"]
	ship.linear_velocity = data["ship_linear_velocity"]
	ship.angular_velocity = data["ship_angular_velocity"]
	ship.is_ramp_open = data["ship_ramp_open"]
	ship.is_headlight_on = data["ship_headlight"]
	ship.is_cabin_switch = data["ship_cabin_light"]
	ship.apply_light_state()

	astronaut.set_active(data["astronaut_active"])
	astronaut.is_lamp_on = data["astronaut_lamp"]
	astronaut.apply_lamp_state()
	astronaut.restore(data["astronaut_transform"], data["astronaut_velocity"])
	ship.set_piloted(data["ship_piloted"])
	return true


func _input(event: InputEvent) -> void:
	if event.is_action_pressed("quick_save"):
		print("Saved." if save_game() else "Save failed.")
		return

	if event.is_action_pressed("quick_load"):
		print("Loaded." if load_game() else "No save to load.")
		return

	if event.is_action_pressed("cycle_target"):
		cycle_target()
		return

	if event.is_action_pressed("ui_cancel"):
		if Input.get_mouse_mode() != Input.MOUSE_MODE_VISIBLE:
			Input.set_mouse_mode(Input.MOUSE_MODE_VISIBLE)
		else:
			Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)


# Button to exit the sim
func _on_button_quit_pressed() -> void:
	get_tree().quit()
