extends Node3D

## What has to be true of a ship that has landed and a pilot who gets out.
##
##     Godot.exe --fixed-fps 60 --path . res://rl/checks/surface_check.tscn
##
## Deliberately NOT headless: the ship and the astronaut are fine headless, but
## running with a display is what the player has, and none of this costs enough
## to be worth two code paths.
##
## Three claims, each of which was false at some point and each of which reads
## as "the ship is broken" rather than as the specific thing it is:
##
##   parks    A ship at rest on the ground latches to the surface, so it rides
##            the planet's spin instead of creeping away while you are outside.
##   holds    Panning the camera while parked does not move the hull. The mouse
##            flies the ship; parked, it must only pan, or looking around the
##            landing site drives the ship across it.
##   boards   The astronaut can walk up the lowered ramp and onto the deck. The
##            geometry either lines up or the ramp is scenery.

const MAIN_SCENE := "res://sim/main.tscn"
## Sites spread across the target continent, so this is not one lucky patch of
## flat ground. Offsets in radians from the landmass centre.
const SITES: Array[Vector2] = [
	Vector2(0.0, 0.0),
	Vector2(0.06, 0.0),
	Vector2(-0.05, 0.04),
	Vector2(0.02, -0.07),
	Vector2(-0.03, -0.03),
]
## Ticks allowed for each phase.
const SETTLE: int = 300
const PAN: int = 240
const WALK: int = 420
## How far a parked hull may move while the pilot pans the camera, in metres.
const DRIFT_ALLOWED: float = 0.05

var _ship: Ship
var _astronaut: Astronaut
var _planet: Planet
var _site: int = 0
var _phase: String = "settle"
var _ticks: int = 0
var _anchor: Vector3 = Vector3.ZERO
var _failures: int = 0


func _ready() -> void:
	# Ahead of the ship, so the faked mouse delta is in place before the ship
	# reads it in the same tick.
	process_physics_priority = -100
	var world: Node = load(MAIN_SCENE).instantiate()
	add_child(world)
	_ship = world.get_node("Ship")
	_astronaut = world.get_node("Astronaut")
	for planet in SolarSystem.Instance.Planets:
		if planet.terrain_enabled:
			_planet = planet
			break
	print("surface: %s" % _planet.name)
	_place()


func _place() -> void:
	var home: Vector3 = _planet.largest_landmass_center
	var east: Vector3 = home.cross(Vector3.UP).normalized()
	var north: Vector3 = home.cross(east)
	var offset: Vector2 = SITES[_site]
	var dir: Vector3 = home.rotated(east, offset.x).rotated(north, offset.y).normalized()

	var ground: float = _planet.get_surface_radius_at(dir)
	var at: Vector3 = _planet.global_position \
		+ _planet.global_basis * (dir * (ground + 4.0))
	var up: Vector3 = (_planet.global_basis * dir).normalized()
	var side: Vector3 = up.cross(Vector3.FORWARD).normalized()
	_ship.global_transform = Transform3D(
		Basis(side, up, side.cross(up)).orthonormalized(), at)
	_ship.linear_velocity = _planet.get_surface_velocity(at)
	_ship.angular_velocity = Vector3.ZERO
	_ship.is_ramp_open = false
	_ship.set_piloted(true)
	# Teleported, so the surface anchor from the last site is now a hundred
	# kilometres away and would drag the ship back to it on the first tick.
	_ship._is_parked = false
	_ship._settled_ticks = 0
	_planet.build_collision_now(_ship, at)
	_phase = "settle"
	_ticks = 0


func _physics_process(_delta: float) -> void:
	if _site >= SITES.size():
		return
	_ticks += 1
	match _phase:
		"settle":
			_settle()
		"pan":
			_pan()
		"walk":
			_walk()


func _settle() -> void:
	if _ship.is_parked():
		_report("parks", true, "after %d ticks" % _ticks)
		_anchor = _ship.global_position
		_phase = "pan"
		_ticks = 0
		return
	if _ticks >= SETTLE:
		var rest: float = _planet.get_relative_velocity_to_surface(
			_ship.global_position, _ship.linear_velocity).length()
		_report("parks", false, "still loose after %d ticks, rest speed %.3f" % [
			_ticks, rest])
		_next()


func _pan() -> void:
	# Anchored on the first full tick of this phase rather than at the end of
	# the last one: this node runs ahead of the ship and the planet, so a
	# position read at a phase boundary is a tick stale in both frames at once.
	if _ticks == 1:
		_anchor = _planet.to_local(_ship.global_position)
		return

	# A pilot swinging the view around the landing site: a full sweep left, then
	# back. Fed straight to the field the mouse writes to.
	_ship._mouse_delta = Vector2(12.0 if _ticks < PAN / 2 else -12.0, 3.0)
	if _ticks % 40 == 0:
		print("    pan t=%d parked=%s local=%v vel=%.2f" % [
			_ticks, _ship.is_parked(), _planet.to_local(_ship.global_position) - _anchor,
			_ship.linear_velocity.length()])
	if _ticks < PAN:
		return
	_ship._mouse_delta = Vector2.ZERO
	var moved: float = _planet.to_local(_ship.global_position).distance_to(_anchor)
	_report("holds", moved <= DRIFT_ALLOWED and _ship.is_parked(),
		"moved %.4f m, parked=%s" % [moved, _ship.is_parked()])
	_start_walk()


func _start_walk() -> void:
	_ship.is_ramp_open = true
	_ship.set_piloted(false)
	_phase = "walk"
	_ticks = 0
	_astronaut.set_active(false)


func _walk() -> void:
	# Give the ramp time to find the ground before stepping onto it.
	if _ticks == 120:
		var ramp: Node3D = _ship.get_node("Ramp")
		var lip: Vector3 = ramp.global_transform * Vector3(0.0, 0.0, _ship._ramp_length + 0.2)
		var up: Vector3 = (lip - _planet.global_position).normalized()
		var toward: Vector3 = (_ship.global_position - lip)
		toward = (toward - up * toward.dot(up)).normalized()
		# Standing on the ground behind the ramp, not on its lip: the player
		# walks up to a ship, and a spawn balanced on the lip itself can start
		# them underneath it wherever the ramp rests high on rising ground.
		var behind: Vector3 = lip - toward * 2.0
		var stand: Vector3 = _planet.global_position + _planet.global_basis * (
			_planet.to_local(behind).normalized() * (_planet.get_surface_radius_at(
				_planet.to_local(behind).normalized()) + 1.1))
		_astronaut.place_at(Transform3D(
			Basis(toward.cross(up), up, -toward).orthonormalized(), stand))
		_astronaut.set_active(true)
		Input.action_press("move_forward")
		return
	if _ticks < 120:
		return

	# Aboard is a position, not a signal: standing on the deck slab, inside the
	# cabin's own footprint.
	var here: Vector3 = _ship.to_local(_astronaut.global_position)
	if absf(here.z) < 2.0 and here.y > 0.5 and here.y < 1.2 and absf(here.x) < 1.4:
		Input.action_release("move_forward")
		_report("boards", true, "reached ship-local %.2f, %.2f, %.2f" % [
			here.x, here.y, here.z])
		_next()
		return
	if _ticks >= WALK:
		Input.action_release("move_forward")
		_report("boards", false, "stuck at ship-local %.2f, %.2f, %.2f" % [
			here.x, here.y, here.z])
		_next()


func _report(claim: String, passed: bool, detail: String) -> void:
	if not passed:
		_failures += 1
	print("site %d  %-7s %s  %s" % [
		_site, claim, "ok  " if passed else "FAIL", detail])


func _next() -> void:
	_site += 1
	if _site < SITES.size():
		_place()
		return
	print("RESULT %s (%d failures)" % [
		"pass" if _failures == 0 else "FAIL", _failures])
	get_tree().quit(0 if _failures == 0 else 1)
