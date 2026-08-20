extends StaticBody3D
class_name Planet

@export_category("Orbit")
@export var _orbital_parent: Planet
@export var _orbital_radius: float = 0.0
## Starting angle of the orbit, in degrees.
@export_range(0.0, 360.0, 0.1, "or_greater", "or_less") var _orbital_angle: float = 0.0

@export_category("Gravity")
@export var surface_radius: float = 1.0
@export var _surface_gravity: float = 9.81

@export_category("Rotation")
@export var _day_length: float = 0.0
@export var _tidally_locked: bool = false

var standard_gravitational_parameter: float = 0.0
var acceleration_by_gravity: Vector3 = Vector3.ZERO

func _ready() -> void:
	# Given surface gravity g at radius R: g = GM / R^2  =>  GM = g * R^2
	standard_gravitational_parameter = _surface_gravity * surface_radius * surface_radius


# Call this after all planets are ready.
# Make sure to call init() on a parent's planet before its children.
func init() -> void:
	if _orbital_parent != null:
		global_position = _orbital_parent.global_position \
			+ (Vector3.FORWARD * _orbital_radius).rotated(Vector3.UP, deg_to_rad(_orbital_angle))
		constant_linear_velocity = _orbital_parent.constant_linear_velocity \
			+ _orbital_parent.get_orbital_velocity(global_position)

	if _tidally_locked:
		_day_length = get_orbital_period()

	var spin_rate := 0.0 if _day_length == 0.0 else (2.0 * PI / _day_length)
	constant_angular_velocity = -spin_rate * Vector3.UP

	print("Initialized ", name)


# Acceleration due to this planet’s gravity at a world-space point
func get_acceleration_at_position(world_pos: Vector3) -> Vector3:
	var d := world_pos - global_position
	return -d.normalized() * standard_gravitational_parameter / d.length_squared()


# Circular orbital velocity (direction is perpendicular to radius in the UP-plane)
func get_orbital_velocity(world_pos: Vector3) -> Vector3:
	var d := world_pos - global_position
	var speed := sqrt(standard_gravitational_parameter / d.length())
	var direction := d.normalized().cross(Vector3.UP)
	return direction * speed


# Orbital period around the parent at _orbital_radius
func get_orbital_period() -> float:
	if _orbital_parent == null:
		return 0.0
	return 2.0 * PI * sqrt(pow(_orbital_radius, 3) / _orbital_parent.standard_gravitational_parameter)


# World-space velocity of this planet’s surface at a given point,
# i.e. orbital motion plus the spin of the surface underneath that point.
func get_surface_velocity(world_pos: Vector3) -> Vector3:
	return constant_linear_velocity + constant_angular_velocity.cross(world_pos - global_position)


# Linear velocity relative to this planet’s surface at a given position/velocity
func get_relative_velocity_to_surface(world_pos: Vector3, linear_velocity: Vector3) -> Vector3:
	return linear_velocity - get_surface_velocity(world_pos)


func _physics_process(delta: float) -> void:
	if _orbital_parent != null:
		acceleration_by_gravity = _orbital_parent.get_acceleration_at_position(global_position) \
			+ _orbital_parent.acceleration_by_gravity
		constant_linear_velocity += acceleration_by_gravity * delta
		global_position += constant_linear_velocity * delta

	global_rotation += constant_angular_velocity * delta


# Number of ancestors in the orbital chain
# e.g., Sun=0, Earth=1 (Sun), Moon=2 (Earth->Sun)
func how_many_parents() -> int:
	if _orbital_parent == null:
		return 0
	return _orbital_parent.how_many_parents() + 1
