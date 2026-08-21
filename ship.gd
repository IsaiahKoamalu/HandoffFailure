extends RigidBody3D
class_name Ship

## A small landable ship with a walk-in cabin and a hinged rear ramp.
##
## The cabin is a frame of reference for the Astronaut in the same way a Planet
## is: while inside, the astronaut is reparented here and rides the ship's
## motion for free. See `contains_point` and `get_surface_velocity`.

@export_category("Nodes")
@export var _camera: Camera3D
@export var _camera_pivot: Node3D
@export var _ground_cast: RayCast3D
## Where the astronaut stands when they get out of the pilot seat.
@export var _exit_point: Node3D
@export var _astronaut: Astronaut
## Hinge node for the rear ramp; rotated about its local X axis.
@export var _ramp: AnimatableBody3D
@export var _cockpit: Interactable

@export_category("Cabin")
## Walkable volume in ship-local space, used to decide when the astronaut is
## aboard. Kept as a box rather than an Area3D so the test is deterministic and
## does not depend on physics-signal ordering.
@export var _interior_center: Vector3 = Vector3(0.0, 1.2, 0.0)
@export var _interior_extents: Vector3 = Vector3(1.5, 1.2, 2.5)

@export_category("Ramp")
## Steepest the ramp will go if it never finds ground. Keep at or below the
## astronaut's floor_max_angle or the lowered ramp becomes an unclimbable wall.
@export var _ramp_open_degrees: float = 45.0
@export var _ramp_closed_degrees: float = -90.0
@export var _ramp_speed: float = 55.0
## Distance from the hinge to the lip, used to probe for ground.
@export var _ramp_length: float = 3.2

@export_category("Settings")
@export var _mouse_sensitivity: float = 0.3
## Thrust as an acceleration, so handling does not change with the ship's mass.
@export var _thrust: float = 10.0
@export var _auto_orient_speed: float = 2.0
## Speed relative to the ground below which a landed ship parks itself.
@export var _landing_speed: float = 0.2

## Gravity of the dominant planet, as a force. Zero when in open space.
var closest_force: Vector3 = Vector3.ZERO
var is_piloted: bool = false
var is_ramp_open: bool = false

var _surface_position: Vector3 = Vector3.ZERO
var _is_parked: bool = false
var _ramp_degrees: float = 0.0
var _mouse_delta: Vector2 = Vector2.ZERO
var _camera_x_rotation: float = 0.0


func _ready() -> void:
	# The ramp is a body in its own right so it can swing independently; without
	# this the hull and the ramp shove each other apart where they meet.
	add_collision_exception_with(_ramp)
	_ramp_degrees = _ramp_closed_degrees
	_ramp.rotation.x = deg_to_rad(_ramp_degrees)
	set_piloted(true)


func _physics_process(delta: float) -> void:
	var solar_system := SolarSystem.Instance
	apply_central_force(solar_system.get_acceleration_at_position(global_position) * mass)

	var dominant: Planet = solar_system.get_dominant_planet(global_position)
	closest_force = Vector3.ZERO
	if dominant != null:
		closest_force = dominant.get_acceleration_at_position(global_position) * mass

	var movement := _get_movement_input() if is_piloted else Vector3.ZERO
	if is_piloted:
		_process_look_inputs()

	_process_parking(movement)

	if movement != Vector3.ZERO:
		apply_central_force(_thrust * mass * movement.normalized())

	_process_auto_orientation(delta)
	_process_ramp(delta)

	_mouse_delta = Vector2.ZERO


func _unhandled_input(event: InputEvent) -> void:
	if not is_piloted:
		return
	if event.is_action_pressed("interact"):
		get_viewport().set_input_as_handled()
		disembark()
	elif event.is_action_pressed("toggle_ramp"):
		get_viewport().set_input_as_handled()
		toggle_ramp()


# --- Cabin ------------------------------------------------------------------

## True when a world-space point is inside the walkable cabin volume.
func contains_point(world_pos: Vector3) -> bool:
	var local := to_local(world_pos) - _interior_center
	return absf(local.x) <= _interior_extents.x \
		and absf(local.y) <= _interior_extents.y \
		and absf(local.z) <= _interior_extents.z


## World-space velocity of the cabin at a point, matching Planet's signature so
## the astronaut can convert frames without caring which kind of body it rides.
func get_surface_velocity(world_pos: Vector3) -> Vector3:
	return linear_velocity + angular_velocity.cross(world_pos - global_position)


# --- Ramp -------------------------------------------------------------------

func toggle_ramp() -> void:
	is_ramp_open = not is_ramp_open


func _process_ramp(delta: float) -> void:
	# A ramp lowered to a fixed angle only meets the ground on a flat world. These
	# planets are small enough that the surface curves away noticeably over the
	# ramp's own length, so it keeps lowering until it finds ground and rests
	# there, exactly like a real cargo ramp.
	if is_ramp_open and _ramp_degrees > _ramp_closed_degrees and _ramp_lip_on_ground():
		return

	var target := _ramp_open_degrees if is_ramp_open else _ramp_closed_degrees
	if is_equal_approx(_ramp_degrees, target):
		return
	_ramp_degrees = move_toward(_ramp_degrees, target, _ramp_speed * delta)
	_ramp.rotation.x = deg_to_rad(_ramp_degrees)


func _ramp_lip_on_ground() -> bool:
	var lip: Vector3 = _ramp.global_transform * Vector3(0.0, 0.0, _ramp_length)
	var up: Vector3 = global_basis.y
	var query := PhysicsRayQueryParameters3D.create(lip + up * 0.1, lip - up * 0.15)
	query.exclude = [get_rid(), _ramp.get_rid()]
	return not get_world_3d().direct_space_state.intersect_ray(query).is_empty()


# --- Boarding ---------------------------------------------------------------

func board() -> void:
	_astronaut.set_active(false)
	set_piloted(true)


## Leave the pilot seat and stand up in the cabin. Allowed in flight as well as
## on the ground: the cabin has its own gravity, so it is safe to walk around.
func disembark() -> void:
	set_piloted(false)
	_astronaut.place_at(_exit_point.global_transform)
	_astronaut.set_active(true)


func set_piloted(piloted: bool) -> void:
	is_piloted = piloted
	_camera.current = piloted
	_cockpit.enabled = not piloted
	if piloted:
		Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)


func _on_cockpit_activated(_who: Node3D) -> void:
	board()


func _on_ramp_control_activated(_who: Node3D) -> void:
	toggle_ramp()


# --- Movement ---------------------------------------------------------------

func _get_movement_input() -> Vector3:
	var movement := Vector3.ZERO

	var forward: Vector3 = -global_transform.basis.z
	var left: Vector3 = -global_transform.basis.x
	var up: Vector3 = global_transform.basis.y

	if Input.is_action_pressed("move_forward"):
		movement += forward
	if Input.is_action_pressed("move_backward"):
		movement -= forward
	if Input.is_action_pressed("move_left"):
		movement += left
	if Input.is_action_pressed("move_right"):
		movement -= left
	if Input.is_action_pressed("move_up"):
		movement += up
	if Input.is_action_pressed("move_down"):
		movement -= up

	return movement


# A parked ship is pinned to surface-local coordinates so it rides the planet's
# orbit and spin instead of drifting away while the astronaut is outside.
func _process_parking(movement: Vector3) -> void:
	var ground: Planet = get_ground()
	var should_park: bool = (movement == Vector3.ZERO) \
		and (ground != null) \
		and (ground.get_relative_velocity_to_surface(global_position, linear_velocity).length() < _landing_speed)

	if should_park:
		if _is_parked:
			global_position = ground.to_global(_surface_position)
		else:
			_surface_position = ground.to_local(global_position)
			_is_parked = true
	else:
		_is_parked = false


func _process_look_inputs() -> void:
	var dx: float = _mouse_delta.y * _mouse_sensitivity
	var dy: float = -_mouse_delta.x * _mouse_sensitivity

	if Input.is_action_pressed("rotate"):
		rotate(_camera_pivot.global_transform.basis.z, deg_to_rad(dy))
	else:
		rotate_object_local(Vector3.UP, deg_to_rad(dy))
		if _camera_x_rotation + dx > -90.0 and _camera_x_rotation + dx < 90.0:
			_camera_pivot.rotate_x(deg_to_rad(-dx))
			_camera_x_rotation += dx


func _process_auto_orientation(delta: float) -> void:
	angular_velocity = Vector3.ZERO
	angular_damp = 10.0

	if closest_force == Vector3.ZERO:
		var dx: float = lerp(0.0, -_camera_x_rotation, _auto_orient_speed * delta)
		_camera_x_rotation += dx
		_camera_pivot.rotate_x(deg_to_rad(-dx))
		rotate(_camera_pivot.global_transform.basis.x, deg_to_rad(dx))
		return

	var up_direction: Vector3 = -closest_force.normalized()
	var current_q: Quaternion = global_transform.basis.get_rotation_quaternion()
	var target_q: Quaternion = Quaternion(global_transform.basis.y, up_direction) * current_q

	var ground: Planet = get_ground()
	if ground != null:
		angular_velocity = ground.constant_angular_velocity.project(up_direction)
		angular_damp = 0.0
		global_rotation = target_q.normalized().get_euler()
	else:
		var rot: Quaternion = current_q.slerp(target_q.normalized(), _auto_orient_speed * delta)
		global_rotation = rot.get_euler()


func _input(e: InputEvent) -> void:
	if e is InputEventMouseMotion and Input.get_mouse_mode() == Input.MOUSE_MODE_CAPTURED:
		var mm: InputEventMouseMotion = e
		_mouse_delta += mm.relative


# --- Helpers ---------------------------------------------------------------

## Landed on an actual Planet. Deliberately not just `_ground_cast.is_colliding()`:
## the ray can hit the astronaut or the ramp, and the orientation code needs a
## Planet or nothing at all.
func is_grounded() -> bool:
	return get_ground() != null


func get_ground() -> Planet:
	return _ground_cast.get_collider() as Planet
