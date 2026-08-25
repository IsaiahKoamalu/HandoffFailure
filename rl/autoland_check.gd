extends Node3D

## End-to-end check of the in-game autopilot.
##
## Everything else in rl/ tests a piece: the bridge, the forward pass, the
## policy's score in the training environment. This runs the thing the player
## actually presses — main.tscn, the real Ship node, ShipAutopilot reading
## NeuralPilot — and reports whether the ship gets down.
##
##     Godot.exe --headless --fixed-fps 60 --path . res://rl/autoland_check.tscn
##
## It deliberately does NOT go through LandingEnv. If the observation the
## autopilot builds in the cockpit disagreed with the one the trainer builds,
## every test in rl/ would still pass and only this one would fail.

const MAIN_SCENE := "res://main.tscn"
const ATTEMPTS: int = 8
const MAX_TICKS: int = 2000

var _ship: Ship
var _planet: Planet
var _ticks: int = 0
var _attempt: int = 0
var _landed: int = 0
var _rng := RandomNumberGenerator.new()
var _peak_speed: float = 0.0


func _ready() -> void:
	process_physics_priority = 200
	_rng.seed = 99
	var world: Node = load(MAIN_SCENE).instantiate()
	add_child(world)

	_ship = world.get_node("Ship")
	for planet in SolarSystem.Instance.Planets:
		if planet.terrain_enabled:
			_planet = planet
			break

	if _ship.autopilot == null or not _ship.autopilot.is_available:
		print("RESULT no_policy")
		get_tree().quit(1)
		return
	_place()


func _place() -> void:
	var direction := Vector3(_rng.randfn(), _rng.randfn(), _rng.randfn()).normalized()
	var ground: float = _planet.get_surface_radius_at(direction)
	var altitude: float = _planet.surface_radius * _rng.randf_range(0.12, 0.32)
	var position: Vector3 = _planet.global_position + direction * (ground + altitude)

	var side: Vector3 = direction.cross(Vector3(_rng.randfn(), _rng.randfn(), _rng.randfn())).normalized()
	var basis := Basis(side, direction, side.cross(direction)).orthonormalized()
	basis = basis.rotated(side, _rng.randf_range(-0.3, 0.3))

	_ship.global_transform = Transform3D(basis, position)
	_ship.linear_velocity = _planet.get_surface_velocity(position) \
		+ direction * _rng.randf_range(-5.0, 0.0)
	_ship.angular_velocity = Vector3.ZERO
	_ship.throttle = 0.0
	_ticks = 0
	_peak_speed = 0.0
	_ship.autopilot.toggle()


func _physics_process(_delta: float) -> void:
	if _attempt >= ATTEMPTS:
		return

	_ticks += 1
	var frame := LandingEnv.surface_frame(_ship.global_position, _planet)
	var altitude: float = frame[0]
	var speed: float = _planet.get_relative_velocity_to_surface(
		_ship.global_position, _ship.linear_velocity
	).length()
	_peak_speed = maxf(_peak_speed, speed)

	var finished := false
	var verdict := ""
	if altitude <= 0.0:
		finished = true
		# The same gate LandingEnv scores with, so this number is comparable to
		# everything reported by python/evaluate.py.
		var align: float = _ship.global_basis.y.dot(frame[1])
		if speed <= 2.0 and align >= 0.9:
			_landed += 1
			verdict = "landed  speed %5.2f align %.3f" % [speed, align]
		else:
			verdict = "crashed speed %5.2f align %.3f" % [speed, align]
	elif not _ship.autopilot.is_engaged:
		finished = true
		var align_now: float = _ship.global_basis.y.dot(frame[1])
		if _ship.autopilot.status == "landed" and speed <= 2.0 and align_now >= 0.9:
			_landed += 1
			verdict = "landed  speed %5.2f align %.3f alt %.2f" % [speed, align_now, altitude]
		else:
			verdict = "%s speed %5.2f alt %.1f" % [_ship.autopilot.status, speed, altitude]
	elif _ticks >= MAX_TICKS:
		finished = true
		verdict = "timeout at altitude %.1f" % altitude

	if not finished:
		return

	print("attempt %d  %-46s peak %5.2f  %4d ticks" % [_attempt + 1, verdict, _peak_speed, _ticks])
	_ship.autopilot.disengage("test over")
	_attempt += 1
	if _attempt >= ATTEMPTS:
		print("RESULT %d/%d landed" % [_landed, ATTEMPTS])
		get_tree().quit(0)
		return
	_place()
