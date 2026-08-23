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

	var submerged: bool = _astronaut.is_submerged if on_foot else _ship.is_submerged
	var mode := "Frame-relative velocity" if on_foot else "Linear velocity"
	if submerged:
		mode += "   ·   underwater"
	_label_mode.text = mode
	_label_lin_x.text = "X: " + str(snapped(velocity.x, 0.01))
	_label_lin_y.text = "Y: " + str(snapped(velocity.y, 0.01))
	_label_lin_z.text = "Z: " + str(snapped(velocity.z, 0.01))

	if on_foot:
		var lamp := "[L] lamp %s" % ("off" if _astronaut.is_lamp_on else "on")
		var prompt := _astronaut.get_interaction_prompt()
		if prompt == "":
			_label_hint.text = lamp
		else:
			_label_hint.text = prompt + "    " + lamp
	else:
		var ramp := "close" if _ship.is_ramp_open else "open"
		var beam := "off" if _ship.is_headlight_on else "on"
		_label_hint.text = "[E] Leave seat    [F] %s ramp    [L] headlight %s" % [ramp, beam]
