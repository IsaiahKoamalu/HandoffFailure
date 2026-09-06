extends CanvasLayer
class_name Hud

## The instrument panel.
##
## Drawn rather than assembled out of Labels. The readouts that matter here are
## bars, a reticle and a velocity vector projected onto the screen, and none of
## those are text -- laying the whole thing out in one _draw keeps the pieces in
## proportion to each other and to the window, which a pile of Control nodes at
## fixed offsets does not.
##
## Everything is read from the ship, the astronaut and the solar system every
## frame. The HUD owns no state of its own except which mode it is in, and that
## is derived too.

@export var _ship: Ship
@export var _astronaut: Astronaut

## Kept so the scene's existing NodePaths still resolve. The labels themselves
## are hidden on the first frame: this draws its own.
@export_category("Legacy labels")
@export var _label_mode: Label
@export var _label_lin_x: Label
@export var _label_lin_y: Label
@export var _label_lin_z: Label
@export var _label_hint: Label
@export var _label_nav: Label

const PANEL_FILL := Color(0.04, 0.06, 0.09, 0.72)
const PANEL_EDGE := Color(0.45, 0.72, 0.95, 0.55)
const TEXT := Color(0.86, 0.92, 1.0)
const TEXT_DIM := Color(0.58, 0.66, 0.78)
const ACCENT := Color(0.45, 0.82, 1.0)
const WARN := Color(1.0, 0.72, 0.25)
const ALARM := Color(1.0, 0.36, 0.3)
const GOOD := Color(0.45, 0.95, 0.6)

const MARGIN := 18.0
const PAD := 12.0
const LINE := 20.0

var _canvas: Control
var _font: Font
var _bold: Font


func _ready() -> void:
	_font = ThemeDB.fallback_font
	_bold = ThemeDB.fallback_font
	for label in [_label_mode, _label_lin_x, _label_lin_y, _label_lin_z,
			_label_hint, _label_nav]:
		if label != null:
			label.visible = false

	_canvas = Control.new()
	_canvas.name = "Instruments"
	_canvas.set_anchors_preset(Control.PRESET_FULL_RECT)
	# Or the panel swallows every click meant for the world behind it.
	_canvas.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_canvas.draw.connect(_draw_instruments)
	add_child(_canvas)
	# Behind the quit button, which is a real Control and has to stay clickable.
	move_child(_canvas, 0)


func _process(_delta: float) -> void:
	_canvas.queue_redraw()


# --- Panels ------------------------------------------------------------------

func _panel(rect: Rect2) -> void:
	_canvas.draw_rect(rect, PANEL_FILL, true)
	_canvas.draw_rect(rect, PANEL_EDGE, false, 1.0)
	# Corner ticks. A plain rectangle reads as a debug overlay; the ticks are
	# what make it read as an instrument.
	var tick: float = 10.0
	for corner in [
		[rect.position, Vector2(1, 0), Vector2(0, 1)],
		[rect.position + Vector2(rect.size.x, 0), Vector2(-1, 0), Vector2(0, 1)],
		[rect.position + Vector2(0, rect.size.y), Vector2(1, 0), Vector2(0, -1)],
		[rect.position + rect.size, Vector2(-1, 0), Vector2(0, -1)],
	]:
		var at: Vector2 = corner[0]
		_canvas.draw_line(at, at + (corner[1] as Vector2) * tick, ACCENT, 2.0)
		_canvas.draw_line(at, at + (corner[2] as Vector2) * tick, ACCENT, 2.0)


func _text(at: Vector2, text: String, color: Color = TEXT, size: int = 15) -> float:
	_canvas.draw_string(_font, at, text, HORIZONTAL_ALIGNMENT_LEFT, -1, size, color)
	return _font.get_string_size(text, HORIZONTAL_ALIGNMENT_LEFT, -1, size).x


func _text_right(at: Vector2, text: String, color: Color = TEXT, size: int = 15) -> void:
	var width: float = _font.get_string_size(
		text, HORIZONTAL_ALIGNMENT_LEFT, -1, size).x
	_canvas.draw_string(_font, at - Vector2(width, 0.0), text,
		HORIZONTAL_ALIGNMENT_LEFT, -1, size, color)


## A labelled horizontal bar. `fraction` is clamped, so a gauge cannot draw
## outside its own frame when something reads out of range.
func _bar(rect: Rect2, fraction: float, color: Color, label: String) -> void:
	_canvas.draw_rect(rect, Color(0.1, 0.13, 0.18, 0.9), true)
	var filled := Rect2(rect.position,
		Vector2(rect.size.x * clampf(fraction, 0.0, 1.0), rect.size.y))
	_canvas.draw_rect(filled, color, true)
	_canvas.draw_rect(rect, PANEL_EDGE, false, 1.0)
	if label != "":
		_text(rect.position + Vector2(4.0, rect.size.y - 3.0), label,
			Color(0.05, 0.07, 0.1, 0.9), 11)


# --- The whole panel ---------------------------------------------------------

func _draw_instruments() -> void:
	if _ship == null or _astronaut == null:
		return
	var screen: Vector2 = _canvas.size
	var on_foot: bool = _astronaut.is_active

	_draw_reticle(screen, on_foot)
	_draw_flight(screen, on_foot)
	_draw_navigation(screen, on_foot)
	_draw_hints(screen, on_foot)
	if not on_foot:
		_draw_docking(screen)


## Centre reticle, plus where the ship is actually going.
##
## The velocity marker is the useful half. A spacecraft points wherever it is
## pointed and travels wherever it was already travelling, and the gap between
## the two is the whole of what flying one is about -- it cannot be read off a
## number, but it can be read at a glance off two marks on the glass.
func _draw_reticle(screen: Vector2, on_foot: bool) -> void:
	if on_foot:
		var dot: Vector2 = screen * 0.5
		_canvas.draw_arc(dot, 2.5, 0.0, TAU, 12, Color(1, 1, 1, 0.5), 1.5)
		return

	var centre: Vector2 = screen * 0.5
	for angle in [0.0, PI * 0.5, PI, PI * 1.5]:
		var d := Vector2(cos(angle), sin(angle))
		_canvas.draw_line(centre + d * 9.0, centre + d * 18.0,
			Color(ACCENT, 0.85), 1.5)
	_canvas.draw_arc(centre, 3.0, 0.0, TAU, 10, Color(ACCENT, 0.9), 1.5)

	var camera: Camera3D = get_viewport().get_camera_3d()
	if camera == null:
		return
	var relative: Vector3 = _ship.linear_velocity
	var planet: Planet = SolarSystem.Instance.get_dominant_planet(
		_ship.global_position)
	if planet != null:
		# Against the ground, not against the star. Sitting on a planet doing
		# eight units a second around the sun is not "moving" to a pilot.
		relative -= planet.get_orbital_velocity(_ship.global_position)
	if relative.length() < 0.2:
		return

	_draw_marker(camera, _ship.global_position + relative.normalized() * 100.0,
		GOOD, true)
	_draw_marker(camera, _ship.global_position - relative.normalized() * 100.0,
		Color(GOOD, 0.5), false)


## A marker at a world point, or nothing if it is behind the camera. The
## is_position_behind test is not optional: a projected point behind the viewer
## lands mirrored on the far side of the screen and tracks the wrong way.
func _draw_marker(camera: Camera3D, world: Vector3, color: Color, filled: bool) -> void:
	if camera.is_position_behind(world):
		return
	var at: Vector2 = camera.unproject_position(world)
	if not _canvas.get_rect().has_point(at):
		return
	_canvas.draw_arc(at, 7.0, 0.0, TAU, 16, color, 1.5)
	if filled:
		for angle in [0.0, PI * 0.5, PI]:
			var d := Vector2(cos(angle), sin(angle))
			_canvas.draw_line(at + d * 7.0, at + d * 12.0, color, 1.5)


func _draw_flight(screen: Vector2, on_foot: bool) -> void:
	var width: float = 268.0
	# The EVA panel grows a line when the oxygen warning appears, rather than
	# leaving a gap for it that is empty almost all of the time.
	var height: float = 176.0
	if on_foot:
		height = 176.0 if _astronaut.oxygen_fraction() < 0.18 else 152.0
	var rect := Rect2(Vector2(MARGIN, screen.y - MARGIN - height),
		Vector2(width, height))
	_panel(rect)

	var at: Vector2 = rect.position + Vector2(PAD, PAD + 12.0)
	if on_foot:
		_text(at, "EVA", ACCENT, 16)
		at.y += LINE + 4.0
		var speed: float = _astronaut.velocity.length()
		_text(at, "speed", TEXT_DIM, 13)
		_text_right(Vector2(rect.end.x - PAD, at.y), "%.2f u/s" % speed)
		at.y += LINE
		_text(at, "frame", TEXT_DIM, 13)
		_text_right(Vector2(rect.end.x - PAD, at.y), _frame_name())
		at.y += LINE
		if _astronaut.is_submerged:
			_text(at, "SUBMERGED", ACCENT, 13)
			at.y += LINE
		at.y += 6.0
		_bar(Rect2(Vector2(rect.position.x + PAD, at.y),
			Vector2(width - PAD * 2.0, 14.0)),
			_astronaut.oxygen_fraction(), _oxygen_color(), "OXYGEN")
		at.y += 24.0
		_bar(Rect2(Vector2(rect.position.x + PAD, at.y),
			Vector2(width - PAD * 2.0, 14.0)),
			_astronaut.jetpack_fraction(),
			WARN if _astronaut.is_jetting else ACCENT, "JETPACK")
		if _astronaut.oxygen_fraction() < 0.18:
			at.y += 26.0
			_text(Vector2(rect.position.x + PAD, at.y), "LOW OXYGEN", ALARM, 14)
		return

	_text(at, "FLIGHT", ACCENT, 16)
	if _ship.docked_to != null:
		_text_right(Vector2(rect.end.x - PAD, at.y), "DOCKED", GOOD, 14)
	elif _ship.is_parked():
		_text_right(Vector2(rect.end.x - PAD, at.y), "LANDED", GOOD, 14)
	at.y += LINE + 4.0

	var planet: Planet = SolarSystem.Instance.get_dominant_planet(
		_ship.global_position)
	var ground: Vector3 = _ship.linear_velocity
	if planet != null:
		ground -= planet.get_orbital_velocity(_ship.global_position)
	_text(at, "speed", TEXT_DIM, 13)
	_text_right(Vector2(rect.end.x - PAD, at.y), "%.1f u/s" % ground.length())
	at.y += LINE

	_text(at, "vertical", TEXT_DIM, 13)
	var vertical: float = 0.0
	if planet != null:
		var up: Vector3 = (_ship.global_position - planet.global_position).normalized()
		vertical = ground.dot(up)
	var vertical_color: Color = TEXT
	if vertical < -6.0:
		vertical_color = ALARM
	elif vertical < -2.0:
		vertical_color = WARN
	_text_right(Vector2(rect.end.x - PAD, at.y), "%+.1f u/s" % vertical,
		vertical_color)
	at.y += LINE

	_text(at, "assist", TEXT_DIM, 13)
	_text_right(Vector2(rect.end.x - PAD, at.y),
		"on" if _ship.flight_assist else "OFF",
		TEXT if _ship.flight_assist else WARN)
	at.y += LINE + 6.0

	_bar(Rect2(Vector2(rect.position.x + PAD, at.y),
		Vector2(width - PAD * 2.0, 14.0)), _ship.throttle, ACCENT, "THROTTLE")
	at.y += 22.0
	var fuel: float = _ship.fuel_fraction()
	_bar(Rect2(Vector2(rect.position.x + PAD, at.y),
		Vector2(width - PAD * 2.0, 14.0)), fuel,
		ALARM if fuel < 0.15 else (WARN if fuel < 0.35 else GOOD), "FUEL")


func _oxygen_color() -> Color:
	var fraction: float = _astronaut.oxygen_fraction()
	if fraction < 0.15:
		return ALARM
	if fraction < 0.35:
		return WARN
	return ACCENT


func _frame_name() -> String:
	var frame: Node3D = _astronaut.current_frame()
	if frame == null:
		return "free"
	if frame is Planet:
		return (frame as Planet).label()
	if frame is Station:
		return (frame as Station).label()
	return frame.name


func _draw_navigation(screen: Vector2, on_foot: bool) -> void:
	var system := SolarSystem.Instance
	var body: Node3D = _astronaut if on_foot else _ship
	var world_velocity: Vector3 = _astronaut.get_world_velocity() if on_foot \
		else _ship.linear_velocity

	var width: float = 280.0
	var height: float = 122.0
	var rect := Rect2(Vector2(screen.x - MARGIN - width, MARGIN),
		Vector2(width, height))
	_panel(rect)

	var at: Vector2 = rect.position + Vector2(PAD, PAD + 12.0)
	_text(at, "NAVIGATION", ACCENT, 14)
	at.y += LINE + 2.0

	# Aboard a station, say so. The dominant planet is still four hundred units
	# below and "DEEP SPACE" is not what someone standing on a deck wants read
	# back to them.
	var aboard: Station = null
	for station in system.Stations:
		if station.contains_point(body.global_position):
			aboard = station

	var here: Planet = system.get_dominant_planet(body.global_position)
	if aboard != null:
		_text(at, "aboard %s" % aboard.label())
	elif here == null:
		_text(at, "DEEP SPACE", TEXT_DIM)
	else:
		var local: Vector3 = here.to_local(body.global_position)
		var altitude: float = local.length() \
			- here.get_surface_radius_at(local.normalized())
		_text(at, here.label())
		_text_right(Vector2(rect.end.x - PAD, at.y), "alt " + _distance(altitude))
	at.y += LINE

	if system.target == null:
		_text(at, "no target", TEXT_DIM)
		_text_right(Vector2(rect.end.x - PAD, at.y), "[Tab]", TEXT_DIM, 13)
		return

	var vector: Array = system.get_target_vector(
		body.global_position, world_velocity)
	var range_to: float = vector[0]
	var closing: float = vector[1]
	_text(at, "-> " + _target_name(system.target), ACCENT)
	_text_right(Vector2(rect.end.x - PAD, at.y), _distance(range_to))
	at.y += LINE
	if closing > 0.5:
		_text(at, "closing " + _distance(closing) + "/s", TEXT_DIM, 13)
		_text_right(Vector2(rect.end.x - PAD, at.y),
			"eta " + _duration(range_to / closing), TEXT_DIM, 13)
	elif closing < -0.5:
		_text(at, "opening " + _distance(-closing) + "/s", TEXT_DIM, 13)
	else:
		_text(at, "holding range", TEXT_DIM, 13)


func _target_name(target: Node3D) -> String:
	if target is Planet:
		return (target as Planet).label()
	if target is Station:
		return (target as Station).label()
	return target.name


## Approach guidance, shown whenever a station's port is close enough to fly at.
##
## This exists because the ship docks stern-first and the pilot is facing the
## other way. Everything needed to fly the approach -- how far in front of the
## port face, how far off the centreline, and whether the ship is the right way
## round -- is behind the pilot's head, so it has to be on the glass.
func _draw_docking(screen: Vector2) -> void:
	var station: Station = null
	var best: float = INF
	for candidate in SolarSystem.Instance.Stations:
		var gap: float = _ship.port_position().distance_to(candidate.port_position())
		if gap < best:
			best = gap
			station = candidate
	if station == null or best > station.approach_range * 2.5:
		return

	var size: float = 150.0
	var rect := Rect2(
		Vector2(screen.x - MARGIN - size, screen.y - MARGIN - size - 24.0),
		Vector2(size, size + 24.0))
	_panel(rect)
	_text(rect.position + Vector2(PAD, PAD + 6.0), "APPROACH", ACCENT, 13)

	var normal: Vector3 = station.port_normal()
	var offset: Vector3 = station.port_position() - _ship.port_position()
	var axial: float = -offset.dot(normal)
	var across: Vector3 = offset + normal * axial
	var alignment: float = _ship.port_normal().dot(normal)

	# The cross-hair is the corridor seen down its own axis: centre is on the
	# centreline. Right and up are taken from the station, not the ship, or the
	# display would spin with the pilot's own roll.
	var centre := Vector2(rect.position.x + size * 0.5,
		rect.position.y + 30.0 + (size - 30.0) * 0.5)
	var radius: float = (size - 46.0) * 0.5
	_canvas.draw_arc(centre, radius, 0.0, TAU, 32, Color(ACCENT, 0.35), 1.0)
	_canvas.draw_arc(centre, radius * 0.33, 0.0, TAU, 20, Color(ACCENT, 0.25), 1.0)
	_canvas.draw_line(centre - Vector2(radius, 0), centre + Vector2(radius, 0),
		Color(ACCENT, 0.2), 1.0)
	_canvas.draw_line(centre - Vector2(0, radius), centre + Vector2(0, radius),
		Color(ACCENT, 0.2), 1.0)

	var right: Vector3 = station.global_basis.x
	var up: Vector3 = station.global_basis.y
	var scale: float = maxf(station.approach_range * 0.35, 0.001)
	var mark := centre + Vector2(
		clampf(across.dot(right) / scale, -1.2, 1.2) * radius,
		-clampf(across.dot(up) / scale, -1.2, 1.2) * radius)
	var lined_up: bool = alignment < _ship.dock_align
	var mark_color: Color = GOOD if lined_up else WARN
	_canvas.draw_arc(mark, 6.0, 0.0, TAU, 14, mark_color, 2.0)

	var readout: Vector2 = Vector2(rect.position.x + PAD, rect.end.y - 10.0)
	_text(readout, "%.1f out" % maxf(axial, 0.0), TEXT_DIM, 12)
	_text_right(Vector2(rect.end.x - PAD, readout.y),
		"aligned" if lined_up else "turn", mark_color, 12)


func _draw_hints(screen: Vector2, on_foot: bool) -> void:
	var parts: Array[String] = []
	if on_foot:
		var prompt: String = _astronaut.get_interaction_prompt()
		if prompt != "":
			parts.append(prompt)
		parts.append("[L] lamp %s" % ("off" if _astronaut.is_lamp_on else "on"))
		parts.append("[Space] jets" if not _astronaut.is_on_floor() else "[Space] jump")
	else:
		parts.append("[E] leave seat")
		parts.append("[F] %s ramp" % ("close" if _ship.is_ramp_open else "open"))
		parts.append("[L] headlight %s" % ("off" if _ship.is_headlight_on else "on"))
		parts.append("[Z] assist %s" % ("on" if _ship.flight_assist else "OFF"))
		parts.append(_autopilot_text())
		parts.append("thr %3d%%" % roundi(_ship.throttle * 100.0))

	var text: String = "    ".join(parts)
	var width: float = _font.get_string_size(
		text, HORIZONTAL_ALIGNMENT_LEFT, -1, 14).x
	# Centred on the space to the right of the flight panel, not on the window.
	# Centred on the window it runs underneath the panel, and the two overlap in
	# exactly the corner where the fuel gauge is.
	var left: float = MARGIN + 268.0 + PAD
	var at := Vector2(left + (screen.x - left - width) * 0.5,
		screen.y - MARGIN - 6.0)
	_canvas.draw_rect(Rect2(at - Vector2(PAD, 16.0),
		Vector2(width + PAD * 2.0, 26.0)), PANEL_FILL, true)
	_text(at, text, TEXT_DIM, 14)


## State of the autopilot.
##
## Shown even when there is no policy file, because the alternative is a key
## that silently does nothing and no way to tell that from a broken one. While
## engaged it names the phase and the destination: a ship crossing the system
## under transit guidance and one flying a descent look nothing alike from the
## cockpit, and "AUTO" for both tells the player neither.
func _autopilot_text() -> String:
	var auto: ShipAutopilot = _ship.autopilot
	if auto == null:
		return "[P] autopilot n/a"
	if _ship.docked_to != null:
		return "[P] DOCKED at %s -- thrust to release" % _ship.docked_to.label()
	if auto.is_engaged:
		return "[P] AUTO %s -> %s" % [
			auto.phase_name(),
			_target_name(auto.destination) if auto.destination != null else "?",
		]
	if not auto.is_available:
		return "[P] autopilot untrained"
	return "[P] autopilot"


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
