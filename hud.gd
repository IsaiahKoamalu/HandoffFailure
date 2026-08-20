extends CanvasLayer
class_name Hud

@export var _ship: Ship
@export var _astronaut: Astronaut

@export_category("Labels")
@export var _label_mode: Label
@export var _label_lin_x: Label
@export var _label_lin_y: Label
@export var _label_lin_z: Label
@export var _label_hint: Label


func _process(_delta: float) -> void:
	var on_foot := _astronaut.is_active
	var velocity: Vector3 = _astronaut.velocity if on_foot else _ship.linear_velocity

	_label_mode.text = "Surface-relative velocity" if on_foot else "Linear velocity"
	_label_lin_x.text = "X: " + str(snapped(velocity.x, 0.01))
	_label_lin_y.text = "Y: " + str(snapped(velocity.y, 0.01))
	_label_lin_z.text = "Z: " + str(snapped(velocity.z, 0.01))

	if on_foot:
		_label_hint.text = "[E] Board ship" if _ship.can_be_boarded(_astronaut.global_position) else ""
	else:
		_label_hint.text = "[E] Step outside" if _ship.is_grounded() else ""
