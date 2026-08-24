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
@export var _label_nav: Label


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

	_label_nav.text = _navigation_text(on_foot)

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
		var assist := "on" if _ship.flight_assist else "OFF"
		_label_hint.text = "[E] Leave seat    [F] %s ramp    [L] headlight %s    [Z] assist %s    thr %3d%%" % [
			ramp, beam, assist, roundi(_ship.throttle * 100.0)]


# --- Navigation -------------------------------------------------------------

## Where you are, where you are going, and how fast you are getting there.
## Without this the planets are 1,200 to 3,600 units apart and moving, with
## nothing on screen to aim at.
func _navigation_text(on_foot: bool) -> String:
	var system := SolarSystem.Instance
	var body: Node3D = _astronaut if on_foot else _ship
	# The astronaut's `velocity` is frame-relative; navigation needs world.
	var world_velocity: Vector3 = _astronaut.get_world_velocity() if on_foot else _ship.linear_velocity

	var lines: Array[String] = []

	var here := system.get_dominant_planet(body.global_position)
	if here == null:
		lines.append("DEEP SPACE")
	else:
		var local := here.to_local(body.global_position)
		var altitude: float = local.length() - here.get_surface_radius_at(local.normalized())
		lines.append("%s      alt %s" % [here.name, _distance(altitude)])

	if system.target == null:
		lines.append("no target        [Tab]")
	else:
		var v := system.get_target_vector(body.global_position, world_velocity)
		var range_to: float = v[0]
		var closing: float = v[1]
		lines.append("→ %s   %s" % [system.target.name, _distance(range_to)])
		if closing > 0.5:
			lines.append("closing %s/s    eta %s" % [_distance(closing), _duration(range_to / closing)])
		elif closing < -0.5:
			lines.append("opening %s/s" % _distance(-closing))
		else:
			lines.append("holding range")

	return "
".join(lines)


func _distance(d: float) -> String:
	if absf(d) >= 1000.0:
		return "%.2f km" % (d / 1000.0)
	return "%.1f m" % d


func _duration(seconds: float) -> String:
	if seconds >= 3600.0:
		return "%.1f h" % (seconds / 3600.0)
	if seconds >= 60.0:
		return "%d:%02d" % [int(seconds) / 60, int(seconds) % 60]
	return "%.0f s" % seconds
