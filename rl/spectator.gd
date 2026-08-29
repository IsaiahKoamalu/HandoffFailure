extends Node3D
class_name TrainSpectator

## A window onto a training run: chase camera plus a live readout.
##
## Training does not need this and never reads it. It exists because a learning
## curve is a lossy summary of a policy — "mean return 61.6" does not tell you
## that the ship is arriving level and braking late, or that it has found a
## hover, or that half the batch is tumbling. Ten seconds of watching does.
##
## Everything here is observation only. The camera does not touch the ships, the
## overlay does not touch the environment, and the whole node is absent unless
## the trainer was launched with --render. A rendered run and a headless run
## produce the same trajectories from the same seed.

## Agents to draw the strip of altitude bars for, at most.
const STRIP_LIMIT: int = 16
## Seconds of camera smoothing. Low enough to keep up with a descent.
const FOLLOW_LAG: float = 0.12

var _env: LandingEnv
var _camera: Camera3D
var _readout: RichTextLabel
var _strip: RichTextLabel
var _banner: Label

var _focus: int = 0
var _auto_follow: bool = true
## Set when the focus changes, so the camera cuts instead of sweeping across
## the solar system.
var _snap: bool = true
var _cam_position: Vector3 = Vector3.ZERO
var _cam_target: Vector3 = Vector3.ZERO

## Whatever the trainer last told us about itself. Free-form: the trainer owns
## the schema and this only ever formats it.
var _stats: Dictionary = {}
## Rolling outcome tally, kept here rather than asked of the env so that it
## survives resets and matches what the viewer has actually seen happen.
var _tally: Dictionary = {}
var _episodes: int = 0
var _recent: Array[int] = []


func _init(env: LandingEnv) -> void:
	_env = env
	name = "Spectator"


func _ready() -> void:
	# After the env, so the camera frames where the ships ended up this tick
	# rather than where they were at the start of it.
	process_physics_priority = 150

	_camera = Camera3D.new()
	# Planets are hundreds of units across and the ship is a few units long, so
	# the depth range has to be wide. Near is as far out as it can go without
	# clipping the hull the camera is sitting behind.
	_camera.near = 0.5
	_camera.far = 5_000_000.0
	_camera.fov = 60.0
	_camera.current = true
	add_child(_camera)

	var layer := CanvasLayer.new()
	add_child(layer)

	_readout = _make_label(Vector2(16, 12), Vector2(430, 300))
	layer.add_child(_readout)

	_strip = _make_label(Vector2(16, 320), Vector2(430, 360))
	layer.add_child(_strip)

	_banner = Label.new()
	_banner.set_anchors_preset(Control.PRESET_CENTER_BOTTOM)
	_banner.position = Vector2(-320, -46)
	_banner.size = Vector2(640, 26)
	_banner.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_banner.add_theme_color_override("font_color", Color(0.75, 0.78, 0.85))
	_banner.add_theme_color_override("font_shadow_color", Color(0, 0, 0, 0.9))
	_banner.add_theme_constant_override("shadow_offset_y", 1)
	_banner.text = "[ ] focus   A auto-follow   , . speed   H hide"
	layer.add_child(_banner)


func _make_label(at: Vector2, box: Vector2) -> RichTextLabel:
	var label := RichTextLabel.new()
	label.bbcode_enabled = true
	label.scroll_active = false
	label.position = at
	label.size = box
	label.add_theme_font_override("normal_font", ThemeDB.fallback_font)
	label.add_theme_font_size_override("normal_font_size", 13)
	label.add_theme_color_override("default_color", Color(0.86, 0.89, 0.95))
	# The camera looks at whatever is behind the text, which is sometimes a
	# sunlit planet and sometimes empty space. A shadow is what makes the
	# readout legible against both.
	label.add_theme_color_override("font_shadow_color", Color(0, 0, 0, 0.95))
	label.add_theme_constant_override("shadow_offset_x", 1)
	label.add_theme_constant_override("shadow_offset_y", 1)
	return label


# --- Fed by the host ---------------------------------------------------------

## Called with whatever the trainer put in the "hud" field of an action message.
func set_stats(stats: Dictionary) -> void:
	_stats = stats


## Called once per finished episode, before the agent is reset.
func note_outcome(outcome: int) -> void:
	_tally[outcome] = int(_tally.get(outcome, 0)) + 1
	_episodes += 1
	_recent.append(outcome)
	if _recent.size() > 60:
		_recent.remove_at(0)


# --- Camera ------------------------------------------------------------------

func _physics_process(delta: float) -> void:
	if _env == null or _env.ships.is_empty():
		return
	if _auto_follow:
		_focus = _pick_focus()
	_focus = clampi(_focus, 0, _env.ships.size() - 1)
	_track(delta)
	_draw_readout()


## The agent closest to resolving one way or the other.
##
## Following agent 0 forever is a bad watch: most of any tick is sixteen ships
## in mid-descent, and the moments worth seeing — the flare, the touchdown, the
## tumble — are brief and happen to whoever happens to be lowest. This just
## keeps the camera on whoever is nearest the ground, which is a decent proxy
## for "about to be interesting".
func _pick_focus() -> int:
	var best: int = _focus
	var best_score: float = INF
	for i in _env.ships.size():
		var frame: Array = LandingEnv.surface_frame(
			_env.ships[i].global_position, _env.targets[i]
		)
		var altitude: float = frame[0]
		if altitude < 0.0:
			continue
		var normalised: float = altitude / _env.targets[i].surface_radius
		if normalised < best_score:
			best_score = normalised
			best = i
	return best


func _track(delta: float) -> void:
	var ship: Ship = _env.ships[_focus]
	var planet: Planet = _env.targets[_focus]
	var frame: Array = LandingEnv.surface_frame(ship.global_position, planet)
	var altitude: float = maxf(frame[0], 0.0)
	var up: Vector3 = frame[1]

	# Pull back with altitude so the shot holds both the ship and the ground it
	# is aiming at. Close in, the useful thing to see is the flare; high up, it
	# is the angle of the whole approach.
	var reach: float = clampf(altitude * 0.9 + 18.0, 18.0, 420.0)

	# A tangent to orbit around, taken from the ship's own heading so the camera
	# sits behind it. Near the poles of that choice the projection collapses, so
	# fall back to anything perpendicular.
	var heading: Vector3 = -ship.global_basis.z
	var behind: Vector3 = (heading - up * heading.dot(up))
	if behind.length_squared() < 1e-6:
		behind = up.cross(Vector3.RIGHT)
		if behind.length_squared() < 1e-6:
			behind = up.cross(Vector3.FORWARD)
	behind = behind.normalized()

	var wanted_position: Vector3 = ship.global_position + up * (reach * 0.42) - behind * reach
	# Aim below the ship, so the ground it is descending toward is in frame
	# instead of sitting just under the bottom edge.
	var wanted_target: Vector3 = ship.global_position - up * minf(altitude * 0.3, reach * 0.35)

	if _snap:
		_cam_position = wanted_position
		_cam_target = wanted_target
		_snap = false
	else:
		# Frame-rate independent exponential smoothing.
		var blend: float = 1.0 - exp(-delta / FOLLOW_LAG)
		_cam_position = _cam_position.lerp(wanted_position, blend)
		_cam_target = _cam_target.lerp(wanted_target, blend)

	_camera.global_position = _cam_position
	if _cam_position.distance_squared_to(_cam_target) > 1e-6:
		_camera.look_at(_cam_target, up)


# --- Readout -----------------------------------------------------------------

func _draw_readout() -> void:
	var ship: Ship = _env.ships[_focus]
	var planet: Planet = _env.targets[_focus]
	var frame: Array = LandingEnv.surface_frame(ship.global_position, planet)
	var altitude: float = frame[0]
	var up: Vector3 = frame[1]
	var relative: Vector3 = planet.get_relative_velocity_to_surface(
		ship.global_position, ship.linear_velocity
	)
	var descent: float = -relative.dot(up)
	var align: float = ship.global_basis.y.dot(up)

	var lines: Array[String] = []
	lines.append("[b]TRAINING[/b]   %s" % _speed_text())
	lines.append(_stat_line())
	lines.append("")
	lines.append("[b]AGENT %d[/b]%s   -> %s" % [
		_focus, "  (auto)" if _auto_follow else "", planet.name,
	])
	lines.append("  altitude   %s" % _bar_text(altitude, planet.surface_radius * 0.4, "%8.1f" % altitude))
	lines.append("  descent    %s" % _bar_text(
		absf(descent), 8.0, "%8.2f %s" % [absf(descent), "down" if descent > 0.0 else "up  "]
	))
	lines.append("  align      %s" % _bar_text(
		maxf(align, 0.0), 1.0, "%8.3f %s" % [align, "ok" if align >= _env.touchdown_align else "  "]
	))
	lines.append("  throttle   %s" % _bar_text(ship.throttle, 1.0, "%8.2f" % ship.throttle))
	var land := LandingEnv.land_frame(ship.global_position, planet)
	var landness: float = float(land[0])
	lines.append("  ground     %s" % (
		"[color=#7fe08a]LAND[/color]" if landness > 0.0 else "[color=#5aa8d8]WATER[/color]"
	))
	lines.append("")
	lines.append("[b]OUTCOMES[/b]  %d episodes" % _episodes)
	lines.append(_tally_line())
	_readout.text = "\n".join(lines)

	_strip.text = _strip_text()


func _speed_text() -> String:
	var ticks: int = Engine.physics_ticks_per_second
	return "%d Hz  (x%.1f real time)" % [ticks, float(ticks) / 60.0]


func _stat_line() -> String:
	if _stats.is_empty():
		return "  waiting for the trainer's first update"
	var parts: Array[String] = []
	if _stats.has("phase"):
		parts.append("[b]%s[/b]" % str(_stats["phase"]))
	if _stats.has("iter"):
		parts.append("iter %d" % int(_stats["iter"]))
	if _stats.has("steps"):
		parts.append("%s steps" % _compact(int(_stats["steps"])))
	if _stats.has("ret"):
		parts.append("return %.1f" % float(_stats["ret"]))
	if _stats.has("land"):
		parts.append("landing %.1f%%" % (float(_stats["land"]) * 100.0))
	return "  " + "   ".join(parts)


static func _compact(value: int) -> String:
	if value >= 1_000_000:
		return "%.2fM" % (value / 1_000_000.0)
	if value >= 1_000:
		return "%.0fk" % (value / 1_000.0)
	return str(value)


## A bar in text. Godot has real progress bars, but sixteen of them in a layout
## is more machinery than a monospaced row of blocks that says the same thing.
static func _bar_text(value: float, full: float, label: String) -> String:
	var filled: int = clampi(int(round(clampf(value / full, 0.0, 1.0) * 20.0)), 0, 20)
	return "[color=#7fd0ff]%s[/color][color=#33404d]%s[/color]  %s" % [
		"|".repeat(filled), "-".repeat(20 - filled), label,
	]


func _tally_line() -> String:
	if _episodes == 0:
		return "  none finished yet"
	var order: Array[int] = [
		LandingEnv.LANDED, LandingEnv.CRASHED, LandingEnv.DITCHED,
		LandingEnv.TIMEOUT, LandingEnv.ESCAPED,
	]
	var names := {
		LandingEnv.LANDED: "landed",
		LandingEnv.CRASHED: "crashed",
		LandingEnv.DITCHED: "ditched",
		LandingEnv.TIMEOUT: "timeout",
		LandingEnv.ESCAPED: "escaped",
	}
	var parts: Array[String] = []
	for outcome in order:
		var count: int = int(_tally.get(outcome, 0))
		parts.append("%s %s%d  %4.1f%%[/color]" % [
			names[outcome], _outcome_colour(outcome), count,
			100.0 * float(count) / float(_episodes),
		])
	return "  " + "   ".join(parts)


static func _outcome_colour(outcome: int) -> String:
	match outcome:
		LandingEnv.LANDED: return "[color=#7fe08a]"
		LandingEnv.CRASHED: return "[color=#e08080]"
		LandingEnv.DITCHED: return "[color=#5aa8d8]"
		LandingEnv.TIMEOUT: return "[color=#d8c070]"
		_: return "[color=#9090c0]"


## One row per agent: where it is and how fast it is coming down.
##
## This is the view the single-agent chase camera cannot give you. A policy
## that has learned to hover and a policy that has learned to land look
## identical for the first ten seconds, and completely different here.
func _strip_text() -> String:
	var lines: Array[String] = ["[b]BATCH[/b]", ""]
	var shown: int = mini(_env.ships.size(), STRIP_LIMIT)
	for i in shown:
		var ship: Ship = _env.ships[i]
		var planet: Planet = _env.targets[i]
		var frame: Array = LandingEnv.surface_frame(ship.global_position, planet)
		var altitude: float = frame[0]
		var descent: float = -planet.get_relative_velocity_to_surface(
			ship.global_position, ship.linear_velocity
		).dot(frame[1])
		var marker: String = ">" if i == _focus else " "
		lines.append("%s%2d %s %6.1f  %+6.2f" % [
			marker, i,
			_bar_text(altitude, planet.surface_radius * 0.4, ""),
			altitude, -descent,
		])
	if _env.ships.size() > shown:
		lines.append("   ... and %d more" % (_env.ships.size() - shown))
	lines.append("")
	lines.append("[b]LAST 60[/b]")
	var run: String = ""
	for outcome in _recent:
		run += "%s#[/color]" % _outcome_colour(outcome)
	lines.append("  " + run if not run.is_empty() else "  -")
	return "\n".join(lines)


# --- Input -------------------------------------------------------------------

func _unhandled_input(event: InputEvent) -> void:
	if not (event is InputEventKey) or not event.pressed or event.echo:
		return
	match (event as InputEventKey).physical_keycode:
		KEY_BRACKETLEFT:
			_auto_follow = false
			_focus = posmod(_focus - 1, _env.ships.size())
			_snap = true
		KEY_BRACKETRIGHT:
			_auto_follow = false
			_focus = posmod(_focus + 1, _env.ships.size())
			_snap = true
		KEY_A:
			_auto_follow = not _auto_follow
			_snap = true
		KEY_COMMA:
			_set_rate(Engine.physics_ticks_per_second - 60)
		KEY_PERIOD:
			_set_rate(Engine.physics_ticks_per_second + 60)
		KEY_H:
			_readout.visible = not _readout.visible
			_strip.visible = _readout.visible
			_banner.visible = _readout.visible


## Change how fast the simulation runs without changing what it simulates.
##
## The physics step stays 1/60 of a simulated second whatever this is set to,
## so the trajectories are identical to a headless run — only the number of
## them per real second moves. That is the whole trick behind watching training
## at all: the interesting part is thousands of episodes long.
func _set_rate(ticks: int) -> void:
	Engine.physics_ticks_per_second = clampi(ticks, 60, 960)
