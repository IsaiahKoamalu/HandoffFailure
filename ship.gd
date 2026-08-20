extends RigidBody3D
class_name Ship

@export_category("Nodes")
@export var _camera: Camera3D
@export var _camera_pivot: Node3D
@export var _ground_cast: RayCast3D
## Where the astronaut is placed when stepping out.
@export var _exit_point: Node3D
@export var _astronaut: Astronaut

@export_category("Settings")
@export var _mouse_sensitivity: float = 0.3
@export var _thrust: float = 10.0
@export var _auto_orient_speed: float = 2.0
## Speed relative to the ground below which a landed ship parks itself.
@export var _landing_speed: float = 0.2

## Gravity of the dominant planet, as a force. Zero when in open space.
var closest_force: Vector3 = Vector3.ZERO
var is_piloted: bool = false

var _surface_position: Vector3 = Vector3.ZERO
var _is_parked: bool = false
var _mouse_delta: Vector2 = Vector2.ZERO
var _camera_x_rotation: float = 0.0


func _ready() -> void:
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
		apply_central_force(_thrust * movement.normalized())

	_process_auto_orientation(delta)

	_mouse_delta = Vector2.ZERO


func _unhandled_input(event: InputEvent) -> void:
	if is_piloted and event.is_action_pressed("interact"):
		disembark()


# --- Boarding ---------------------------------------------------------------

## True when the ship is parked and the astronaut is close enough to climb in.
func can_be_boarded(world_pos: Vector3) -> bool:
	return _is_parked and _exit_point.global_position.distance_to(world_pos) < 4.0


func board() -> void:
	_astronaut.set_active(false)
	set_piloted(true)


func disembark() -> void:
	if not _is_parked:
		return
	set_piloted(false)
	_astronaut.place_at(_exit_point.global_transform)
	_astronaut.set_active(true)


func set_piloted(piloted: bool) -> void:
	is_piloted = piloted
	_camera.current = piloted
	if piloted:
		Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)


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
	var should_park: bool = (movement == Vector3.ZERO)         and (ground != null)         and (ground.get_relative_velocity_to_surface(global_position, linear_velocity).length() < _landing_speed)

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

	if is_grounded():
		var ground: Planet = get_ground()
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

func is_grounded() -> bool:
	return _ground_cast.is_colliding()


func get_ground() -> Planet:
	return _ground_cast.get_collider() as Planet
