extends CharacterBody3D
class_name Astronaut

## An on-foot character that walks the surface of a Planet or the cabin of a Ship.
##
## Planets orbit at speeds far higher than walking pace, so the astronaut is
## reparented to whichever body it is currently riding — a planet's surface, or
## the ship's cabin. Inside that parent, `velocity` is a *frame-relative*
## velocity: standing still means `velocity == Vector3.ZERO`, and riding the
## planet's orbit and spin (or the ship's flight) falls out of the parent
## transform for free. In open space the parent is the SolarSystem and
## `velocity` is plain world-space motion.
##
## Because of that, the scene disables Godot's moving-platform handling
## (`platform_floor_layers = 0`). A Planet is a StaticBody3D carrying a
## `constant_linear_velocity`, so Godot would otherwise add the planet's orbital
## velocity on top of the motion the reparenting already provides, and the
## double count throws the astronaut off the surface as soon as they walk.

@export_category("Nodes")
@export var _camera: Camera3D
@export var _camera_pivot: Node3D
@export var _ship: Ship
@export var _interact_probe: Area3D
## Helmet lamp. Lives under the camera pivot, so it lights whatever you look at.
@export var _headlamp: SpotLight3D

@export_category("Settings")
@export var _mouse_sensitivity: float = 0.3
@export var _walk_speed: float = 5.0
@export var _jump_speed: float = 6.0
@export var _auto_orient_speed: float = 4.0
## Cabin "grav plating" pull, along the ship's own down axis. Keeps walking
## around inside predictable no matter how the ship is oriented or whether it
## is parked on a planet or under thrust in deep space.
@export var _cabin_gravity: float = 9.81

@export_category("Swimming")
## Upward acceleration underwater as a multiple of local gravity. Above 1.0 and
## the astronaut drifts toward the surface; below 1.0 they sink to the seabed.
@export var _buoyancy: float = 1.08
@export var _swim_speed: float = 3.0
## How hard water pulls velocity toward the swim input. Higher is stickier, and
## also sets the terminal rise rate against buoyancy.
@export var _water_drag: float = 2.5

var is_active: bool = false
var is_lamp_on: bool = true
## Below sea level on the planet currently being ridden.
var is_submerged: bool = false

## The body this astronaut is parented to and moving relative to: a Planet, the
## Ship, or null when adrift in open space.
var _frame_body: Node3D = null
## Up axis and gravity for the current frame. Resolved once per tick in
## `_update_frame_field` because every consumer wants both and the open-space
## case costs a full pass over the planet list.
var _up: Vector3 = Vector3.UP
var _gravity: Vector3 = Vector3.ZERO
var _mouse_delta: Vector2 = Vector2.ZERO
var _camera_x_rotation: float = 0.0

@onready var _collision: CollisionShape3D = $CollisionShape3D


func _ready() -> void:
	set_active(false)
	_register_terrain.call_deferred()


# SolarSystem.Instance is assigned in Main's _ready, which Godot runs after its
# children's, so this cannot be called directly from _ready.
func _register_terrain() -> void:
	SolarSystem.Instance.register_terrain_client(self)


func _physics_process(delta: float) -> void:
	_update_frame(_pick_frame())
	_update_frame_field()
	up_direction = _up

	if is_active:
		_process_look_inputs()
	_align_to_up(_up, delta)

	if is_submerged:
		_process_swimming(delta)
	elif is_on_floor():
		# On the ground the astronaut fully controls its frame-relative
		# velocity, so releasing the keys means riding the frame exactly.
		velocity = _get_walk_velocity(_up)
		if is_active and Input.is_action_just_pressed("jump"):
			velocity += _up * _jump_speed
	else:
		velocity += _gravity * delta

	move_and_slide()
	_mouse_delta = Vector2.ZERO


func _unhandled_input(event: InputEvent) -> void:
	if not is_active:
		return

	if event.is_action_pressed("toggle_light"):
		get_viewport().set_input_as_handled()
		toggle_lamp()
		return

	if not event.is_action_pressed("interact"):
		return
	var target := get_interactable()
	if target == null:
		return
	# Consume it: boarding flips control to the ship, which handles the very same
	# interact event further along the propagation chain and would read it as an
	# instruction to get straight back out again.
	get_viewport().set_input_as_handled()
	target.activate(self)


func _input(e: InputEvent) -> void:
	if not is_active:
		return
	if e is InputEventMouseMotion and Input.get_mouse_mode() == Input.MOUSE_MODE_CAPTURED:
		var mm: InputEventMouseMotion = e
		_mouse_delta += mm.relative


# --- Interaction ------------------------------------------------------------

## Nearest enabled Interactable in range, or null.
func get_interactable() -> Interactable:
	var best: Interactable = null
	var best_distance := INF
	for area in _interact_probe.get_overlapping_areas():
		var candidate := area as Interactable
		if candidate == null or not candidate.enabled:
			continue
		var distance := candidate.global_position.distance_squared_to(global_position)
		if distance < best_distance:
			best = candidate
			best_distance = distance
	return best


## Switch the helmet lamp. Bound to the toggle_light action rather than an
## Interactable: it is carried, not walked up to.
func toggle_lamp() -> void:
	is_lamp_on = not is_lamp_on
	_headlamp.visible = is_lamp_on


func get_interaction_prompt() -> String:
	var target := get_interactable()
	return "[E] " + target.prompt if target != null else ""


# --- Frame of reference -----------------------------------------------------

## Whatever the astronaut is riding right now. The ship wins over the planet it
## is parked on, so walking up the ramp hands off cleanly at the doorway.
func _pick_frame() -> Node3D:
	if _ship != null and _ship.contains_point(global_position):
		return _ship
	return SolarSystem.Instance.get_dominant_planet(global_position)


# Reparent to the new frame, converting `velocity` out of the old frame into
# world space and then into the new one as we cross over.
func _update_frame(frame: Node3D) -> void:
	if frame == _frame_body:
		return

	velocity += _frame_velocity_at(_frame_body)
	_frame_body = frame

	var new_parent: Node = frame if frame != null else SolarSystem.Instance
	if get_parent() != new_parent:
		reparent(new_parent, true)

	velocity -= _frame_velocity_at(frame)


# World-space velocity of a frame at the astronaut's position. Planet and Ship
# both expose get_surface_velocity, but they share no base class, so dispatch here.
func _frame_velocity_at(frame: Node3D) -> Vector3:
	if frame is Planet:
		return (frame as Planet).get_surface_velocity(global_position)
	if frame is Ship:
		return (frame as Ship).get_surface_velocity(global_position)
	return Vector3.ZERO


# Resolve `_up` and `_gravity` together for the current frame.
#
# Inside a planet's frame only that planet's gravity is felt: the frame is
# itself in free fall, so the parent's orbital acceleration cancels out.
func _update_frame_field() -> void:
	if _frame_body is Ship:
		_up = _frame_body.global_basis.y
		_gravity = -_up * _cabin_gravity
		is_submerged = false
		return

	if _frame_body is Planet:
		# Gravity here is just -up scaled by GM/d^2, so deriving it from `_up`
		# saves normalizing the same offset twice.
		var planet := _frame_body as Planet
		var offset := global_position - planet.global_position
		_up = offset.normalized()
		_gravity = -_up * planet.standard_gravitational_parameter / offset.length_squared()
		is_submerged = planet.is_underwater(global_position)
		return

	_gravity = SolarSystem.Instance.get_acceleration_at_position(global_position)
	_up = -_gravity.normalized() if _gravity != Vector3.ZERO else global_basis.y
	is_submerged = false


# --- Movement ---------------------------------------------------------------

# Underwater the astronaut is neutrally buoyant rather than in free fall, and
# drag replaces the ground contact that normally gives full control of velocity.
func _process_swimming(delta: float) -> void:
	# Buoyancy opposes gravity, so only the remainder is felt.
	velocity += _gravity * (1.0 - _buoyancy) * delta

	# Drag pulls velocity toward the swim input, which both stops the astronaut
	# coasting forever and caps the buoyant rise at a terminal speed.
	velocity = velocity.lerp(_get_swim_velocity(), clampf(_water_drag * delta, 0.0, 1.0))


## Swim input in world space. Unlike walking this is deliberately not flattened
## onto the tangent plane: underwater you go where you are looking.
func _get_swim_velocity() -> Vector3:
	if not is_active:
		return Vector3.ZERO

	var forward: Vector3 = -_camera.global_transform.basis.z
	var right: Vector3 = _camera.global_transform.basis.x

	var wish := Vector3.ZERO
	if Input.is_action_pressed("move_forward"):
		wish += forward
	if Input.is_action_pressed("move_backward"):
		wish -= forward
	if Input.is_action_pressed("move_right"):
		wish += right
	if Input.is_action_pressed("move_left"):
		wish -= right
	if Input.is_action_pressed("move_up"):
		wish += _up
	if Input.is_action_pressed("move_down"):
		wish -= _up

	if wish.is_zero_approx():
		return Vector3.ZERO
	return wish.normalized() * _swim_speed


func _get_walk_velocity(up: Vector3) -> Vector3:
	if not is_active:
		return Vector3.ZERO

	var forward: Vector3 = -global_transform.basis.z
	var right: Vector3 = global_transform.basis.x

	var wish := Vector3.ZERO
	if Input.is_action_pressed("move_forward"):
		wish += forward
	if Input.is_action_pressed("move_backward"):
		wish -= forward
	if Input.is_action_pressed("move_right"):
		wish += right
	if Input.is_action_pressed("move_left"):
		wish -= right

	# Flatten onto the tangent plane so walking follows the curve of the planet.
	wish -= wish.project(up)
	if wish.is_zero_approx():
		return Vector3.ZERO
	return wish.normalized() * _walk_speed


func _process_look_inputs() -> void:
	var dx: float = _mouse_delta.y * _mouse_sensitivity
	var dy: float = -_mouse_delta.x * _mouse_sensitivity

	rotate_object_local(Vector3.UP, deg_to_rad(dy))
	if _camera_x_rotation + dx > -90.0 and _camera_x_rotation + dx < 90.0:
		_camera_pivot.rotate_x(deg_to_rad(-dx))
		_camera_x_rotation += dx


# Keep the astronaut's feet pointed at the floor, preserving current heading.
func _align_to_up(up: Vector3, delta: float) -> void:
	var current_q: Quaternion = global_basis.get_rotation_quaternion()
	var target_q: Quaternion = (Quaternion(global_basis.y, up) * current_q).normalized()

	if is_on_floor():
		global_basis = Basis(target_q)
	else:
		global_basis = Basis(current_q.slerp(target_q, _auto_orient_speed * delta))


# --- Boarding ---------------------------------------------------------------

## Drop the astronaut at a world transform with zero frame-relative velocity,
## so they step out of the seat matching the ship's motion exactly.
func place_at(spawn: Transform3D) -> void:
	global_transform = spawn
	_frame_body = null
	_update_frame(_pick_frame())
	# Patch streaming is asynchronous, so a fresh spawn would otherwise spend a
	# frame or two with no ground under it and drop through to the backstop.
	if _frame_body is Planet:
		(_frame_body as Planet).build_collision_now(self, global_position)
	velocity = Vector3.ZERO
	_camera_x_rotation = 0.0
	_camera_pivot.rotation = Vector3.ZERO


func set_active(active: bool) -> void:
	is_active = active
	visible = active
	_camera.current = active
	process_mode = Node.PROCESS_MODE_INHERIT if active else Node.PROCESS_MODE_DISABLED
	_collision.set_deferred("disabled", not active)
	if active:
		Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)
