extends RigidBody3D
class_name Player

@export_category("Nodes")
@export var _camera: Camera3D
@export var _camera_pivot: Node3D
@export var _ground_cast: RayCast3D

var _closest_force: Vector3 = Vector3.ZERO
var _surface_position: Vector3 = Vector3.ZERO
var _is_stuck_to_surface: bool = false

@export_category("Settings")
@export var _mouse_sensitivity: float = 0.3
var _mouse_delta: Vector2 = Vector2.ZERO
var _camera_x_rotation: float = 0.0

@export var _thrust: float = 1.0
@export var _auto_orient_speed: float = 0.5
@export var _jump_impulse: float = 5.0

var _in_map: bool = false


func _ready() -> void:
	show_map(false)


func _process(_delta: float) -> void:
	if Input.is_action_just_released("map"):
		show_map(!_in_map)


func _physics_process(delta: float) -> void:
	var closest_force_magnitude: float = 0.0
	_closest_force = Vector3.ZERO

	for planet: Planet in SolarSystem.Instance.Planets:
		var force: Vector3 = planet.get_acceleration_at_position(global_position) * mass
		apply_central_force(force)

		if planet.global_position.distance_to(global_position) < 2.0 * planet.surface_radius:
			var magnitude: float = force.length()
			if magnitude > closest_force_magnitude:
				_closest_force = force
				closest_force_magnitude = magnitude

	if not _in_map:
		_process_movement_inputs(delta)
		_process_look_inputs(delta)

	_process_auto_orientation(delta)

	_mouse_delta = Vector2.ZERO


func _process_movement_inputs(_delta: float) -> void:
	var movement: Vector3 = Vector3.ZERO

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

	var ground: Planet = _get_ground()
	var should_stick: bool = (movement == Vector3.ZERO) \
		and (ground != null) \
		and (ground.get_relative_velocity_to_surface(global_position, linear_velocity).length() < 0.2)

	if should_stick:
		if _is_stuck_to_surface:
			global_position = ground.to_global(_surface_position)
		else:
			_surface_position = ground.to_local(global_position)
			_is_stuck_to_surface = true
	else:
		_is_stuck_to_surface = false

	if movement != Vector3.ZERO:
		apply_central_force(_thrust * movement.normalized())

	if _is_grounded() and Input.is_action_just_released("jump"):
		apply_central_impulse(_jump_impulse * up)


func _process_look_inputs(_delta: float) -> void:
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

	var in_zero_g: bool = (_closest_force == Vector3.ZERO)

	if in_zero_g:
		var dx: float = lerp(0.0, -_camera_x_rotation, _auto_orient_speed * delta)
		_camera_x_rotation += dx
		_camera_pivot.rotate_x(deg_to_rad(-dx))
		rotate(_camera_pivot.global_transform.basis.x, deg_to_rad(dx))
	else:
		var up_direction: Vector3 = -_closest_force.normalized()
		var current_q: Quaternion = global_transform.basis.get_rotation_quaternion()
		var target_q: Quaternion = Quaternion(global_transform.basis.y, up_direction) * current_q

		if _is_grounded():
			var ground: Planet = _get_ground()
			angular_velocity = ground.constant_angular_velocity.project(up_direction)
			angular_damp = 0.0
			global_rotation = target_q.normalized().get_euler()
		else:
			var rot: Quaternion = current_q.slerp(target_q.normalized(), _auto_orient_speed * delta)
			global_rotation = rot.get_euler()


func _input(e: InputEvent) -> void:
	if e is InputEventMouseMotion:
		var mm: InputEventMouseMotion = e
		_mouse_delta += mm.relative


func show_map(in_map: bool) -> void:
	_in_map = in_map
	if _in_map:
		Input.set_mouse_mode(Input.MOUSE_MODE_VISIBLE)
		_camera.current = false
	else:
		Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)
		_camera.current = true


# --- Helpers ---------------------------------------------------------------

func _is_grounded() -> bool:
	return _ground_cast.is_colliding()

func _get_ground() -> Planet:
	return _ground_cast.get_collider() as Planet
