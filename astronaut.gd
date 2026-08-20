extends CharacterBody3D
class_name Astronaut

## An on-foot character that walks the surface of a Planet.
##
## Planets orbit at speeds far higher than walking pace, so the astronaut is
## reparented to whichever planet currently dominates its gravity. Inside that
## parent, `velocity` is a *surface-relative* velocity: standing still means
## `velocity == Vector3.ZERO`, and riding the planet's orbit and spin falls out
## of the parent transform for free. Leaving a planet's sphere of influence
## reparents back to the SolarSystem and converts the velocity to world space.
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

@export_category("Settings")
@export var _mouse_sensitivity: float = 0.3
@export var _walk_speed: float = 5.0
@export var _jump_speed: float = 6.0
@export var _auto_orient_speed: float = 4.0

var is_active: bool = false

## The planet this astronaut is currently parented to, or null when in space.
var _frame_planet: Planet = null
var _mouse_delta: Vector2 = Vector2.ZERO
var _camera_x_rotation: float = 0.0

@onready var _collision: CollisionShape3D = $CollisionShape3D


func _ready() -> void:
	set_active(false)


func _physics_process(delta: float) -> void:
	_update_frame(SolarSystem.Instance.get_dominant_planet(global_position))

	var up := _get_up_direction()
	up_direction = up

	if is_active:
		_process_look_inputs()
	_align_to_up(up, delta)

	if is_on_floor():
		# On the ground the astronaut fully controls its surface-relative
		# velocity, so releasing the keys means riding the planet exactly.
		velocity = _get_walk_velocity(up)
		if is_active and Input.is_action_just_pressed("jump"):
			velocity += up * _jump_speed
	else:
		velocity += _get_gravity() * delta

	move_and_slide()
	_mouse_delta = Vector2.ZERO


func _unhandled_input(event: InputEvent) -> void:
	if is_active and event.is_action_pressed("interact") and _ship.can_be_boarded(global_position):
		_ship.board()


func _input(e: InputEvent) -> void:
	if not is_active:
		return
	if e is InputEventMouseMotion and Input.get_mouse_mode() == Input.MOUSE_MODE_CAPTURED:
		var mm: InputEventMouseMotion = e
		_mouse_delta += mm.relative


# --- Frame of reference -----------------------------------------------------

# Reparent to whichever planet owns this astronaut, converting `velocity`
# between world space and that planet's rotating surface frame as we cross over.
func _update_frame(planet: Planet) -> void:
	if planet == _frame_planet:
		return

	# Back out of the old frame into world space.
	if _frame_planet != null:
		velocity += _frame_planet.get_surface_velocity(global_position)

	_frame_planet = planet

	var new_parent: Node = planet if planet != null else SolarSystem.Instance
	if get_parent() != new_parent:
		reparent(new_parent, true)

	# ...and into the new one.
	if planet != null:
		velocity -= planet.get_surface_velocity(global_position)


# Inside a planet's frame only that planet's gravity is felt: the frame is
# itself in free fall, so the parent's orbital acceleration cancels out.
func _get_gravity() -> Vector3:
	if _frame_planet != null:
		return _frame_planet.get_acceleration_at_position(global_position)
	return SolarSystem.Instance.get_acceleration_at_position(global_position)


func _get_up_direction() -> Vector3:
	if _frame_planet != null:
		return (global_position - _frame_planet.global_position).normalized()
	var gravity := _get_gravity()
	return -gravity.normalized() if gravity != Vector3.ZERO else global_transform.basis.y


# --- Movement ---------------------------------------------------------------

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


# Keep the astronaut's feet pointed at the planet, preserving current heading.
func _align_to_up(up: Vector3, delta: float) -> void:
	var current_q: Quaternion = global_basis.get_rotation_quaternion()
	var target_q: Quaternion = (Quaternion(global_basis.y, up) * current_q).normalized()

	if is_on_floor():
		global_basis = Basis(target_q)
	else:
		global_basis = Basis(current_q.slerp(target_q, _auto_orient_speed * delta))


# --- Boarding ---------------------------------------------------------------

## Drop the astronaut at a world transform with zero surface-relative velocity,
## so they step out matching the parked ship's motion instead of being flung off.
func place_at(spawn: Transform3D) -> void:
	global_transform = spawn
	_frame_planet = null
	_update_frame(SolarSystem.Instance.get_dominant_planet(spawn.origin))
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
