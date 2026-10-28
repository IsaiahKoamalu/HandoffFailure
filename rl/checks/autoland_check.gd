extends Node3D

## End-to-end check of the in-game autopilot.
##
## Everything else in rl/ tests a piece: the bridge, the forward pass, the
## policy's score in the training environment. This runs the thing the player
## actually presses — main.tscn, the real Ship node, ShipAutopilot reading
## NeuralPilot — and reports whether the ship gets down.
##
##     Godot.exe --headless --fixed-fps 60 --path . res://rl/checks/autoland_check.tscn
##
## It deliberately does NOT go through LandingEnv. If the observation the
## autopilot builds in the cockpit disagreed with the one the trainer builds,
## every test in rl/ would still pass and only this one would fail.

const MAIN_SCENE := "res://sim/main.tscn"
const ATTEMPTS: int = 12
const MAX_TICKS: int = 10000

var _ship: Ship
var _planet: Planet
var _ticks: int = 0
var _attempt: int = 0
var _landed: int = 0
var _rng := RandomNumberGenerator.new()
var _peak_speed: float = 0.0
## How far the ship started from the target continent, in radians, so the
## report says whether an attempt was a short hop or half a planet.
var _start_angle: float = 0.0


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

	# Which policy is on trial. Without this the check could only ever score the
	# installed one, so a candidate had to be swapped in before it could be
	# tested -- the wrong way round for the question being asked.
	var policy := ""
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--policy="):
			policy = argument.substr("--policy=".length())
	if not policy.is_empty():
		if _ship.autopilot == null or not _ship.autopilot.use_policy(policy):
			print("RESULT no_policy (%s)" % policy)
			get_tree().quit(1)
			return
		print("policy: %s" % policy)

	if _ship.autopilot == null or not _ship.autopilot.is_available:
		print("RESULT no_policy")
		get_tree().quit(1)
		return
	print("target: %s, largest landmass %.1f%% of the sphere, extent %.2f rad" % [
		_planet.name, _planet.largest_landmass_share * 100.0,
		_planet.largest_landmass_extent,
	])
	_place()


func _place() -> void:
	# Anywhere at all. That is the claim being tested: the player can be over
	# the wrong hemisphere, high up, tumbling, carrying orbital velocity, and
	# press the key. Deliberately not biased toward the target continent -- a
	# check that started next to the answer would pass without meaning anything.
	var direction := Vector3(_rng.randfn(), _rng.randfn(), _rng.randfn()).normalized()
	var ground: float = _planet.get_surface_radius_at(direction)
	var altitude: float = _planet.surface_radius * _rng.randf_range(0.05, 1.10)
	var position: Vector3 = _planet.global_position + direction * (ground + altitude)

	var side: Vector3 = direction.cross(Vector3(_rng.randfn(), _rng.randfn(), _rng.randfn())).normalized()
	var basis := Basis(side, direction, side.cross(direction)).orthonormalized()
	# Half the attempts start tumbling and completely off level.
	if _attempt % 2 == 0:
		basis = Basis(
			Vector3(_rng.randfn(), _rng.randfn(), _rng.randfn()).normalized(),
			_rng.randf_range(-PI, PI)
		) * basis
	else:
		basis = basis.rotated(side, _rng.randf_range(-0.3, 0.3))

	var forward: Vector3 = side.cross(direction)
	var circular: float = sqrt(
		_planet.standard_gravitational_parameter / maxf(ground + altitude, 1.0)
	)
	var heading: float = _rng.randf_range(-PI, PI)
	var tangential: float = circular * _rng.randf_range(0.0, 0.8)

	_ship.global_transform = Transform3D(basis, position)
	_ship.linear_velocity = _planet.get_surface_velocity(position) \
		+ (side * cos(heading) + forward * sin(heading)) * tangential \
		+ direction * _rng.randf_range(-6.0, 0.0)
	_ship.angular_velocity = Vector3(
		_rng.randfn(), _rng.randfn(), _rng.randfn()
	) * (0.5 if _attempt % 2 == 0 else 0.0)
	_start_angle = _planet.angle_to_largest_landmass(
		(_planet.global_basis.inverse() * direction).normalized()
	)
	_ship.throttle = 0.0
	_ticks = 0
	_peak_speed = 0.0
	_ship.autopilot.toggle()


func _physics_process(_delta: float) -> void:
	if _attempt >= ATTEMPTS:
		return

	_ticks += 1
	var frame := Surface.surface_frame(_ship.global_position, _planet)
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
		if not Surface.is_over_land(_ship.global_position, _planet):
			verdict = "DITCHED speed %5.2f align %.3f" % [speed, align]
		elif not Surface.is_on_target_site(_ship.global_position, _planet):
			verdict = "MISSED  wrong landmass, speed %5.2f" % speed
		elif speed <= 2.0 and align >= 0.9:
			_landed += 1
			verdict = "landed  speed %5.2f align %.3f" % [speed, align]
		else:
			verdict = "crashed speed %5.2f align %.3f" % [speed, align]
	elif not _ship.autopilot.is_engaged:
		finished = true
		var align_now: float = _ship.global_basis.y.dot(frame[1])
		var on_site: bool = Surface.is_on_target_site(_ship.global_position, _planet)
		var dry: bool = Surface.is_over_land(_ship.global_position, _planet)
		if _ship.autopilot.status == "landed" and speed <= 2.0 and align_now >= 0.9 and on_site:
			_landed += 1
			verdict = "landed  speed %5.2f align %.3f alt %.2f" % [speed, align_now, altitude]
		elif not dry:
			verdict = "DITCHED speed %5.2f alt %.1f" % [speed, altitude]
		elif not on_site:
			verdict = "MISSED  wrong landmass, speed %5.2f" % speed
		else:
			verdict = "%s speed %5.2f alt %.1f" % [_ship.autopilot.status, speed, altitude]
	elif _ticks >= MAX_TICKS:
		finished = true
		verdict = "timeout at altitude %.1f" % altitude

	if not finished:
		return

	print("attempt %2d  start %4.2f rad  %-46s %5d ticks" % [
		_attempt + 1, _start_angle, verdict, _ticks,
	])
	_ship.autopilot.disengage("test over")
	_attempt += 1
	if _attempt >= ATTEMPTS:
		print("RESULT %d/%d landed" % [_landed, ATTEMPTS])
		get_tree().quit(0)
		return
	_place()
