extends Node3D

## End-to-end check of docking in the game, not in the trainer.
##
## journey_check flies to planets; this flies to a station, latches, and then
## puts the astronaut on the deck and walks them about. It goes through
## ShipAutopilot, Ship.try_dock and the real Astronaut, so what it exercises is
## the thing the player presses.
##
##     Godot --headless --fixed-fps 60 --path . res://rl/dock_check.tscn
##
## The astronaut half matters as much as the approach. A station is a frame like
## the ship's cabin: the room orbits a planet at eight units a second while
## someone stands still inside it, and "stands still" has to mean relative to
## the room. A frame bug does not look like a crash, it looks like the astronaut
## sliding into a wall.

const MAIN_SCENE := "res://main.tscn"
const MAX_TICKS: int = 60000

var _ship: Ship
var _astronaut: Astronaut
var _station: Station
var _system: SolarSystem
var _world: Node
var _failures: int = 0


func _ready() -> void:
	process_physics_priority = 200
	_world = load(MAIN_SCENE).instantiate()
	add_child(_world)
	_ship = _world.get_node("Ship")
	_astronaut = _world.get_node("Astronaut")
	_system = SolarSystem.Instance
	for station in _system.Stations:
		_station = station
	var hud: Node = _world.get_node_or_null("HUD")
	if hud != null:
		_world.remove_child(hud)
		hud.queue_free()


func _physics_process(_delta: float) -> void:
	set_physics_process(false)
	await _run()
	print("RESULT %s" % ("PASS" if _failures == 0 else "FAIL (%d)" % _failures))
	get_tree().quit(_failures)


func _check(label: String, ok: bool, detail: String) -> void:
	if not ok:
		_failures += 1
	print("%-40s %-7s %s" % [label, "ok" if ok else "FAILED", detail])


## A world point in the station's own frame, which is the only frame in which
## "did it move" means anything: the room is orbiting at eight units a second.
func _local(world_pos: Vector3) -> Vector3:
	return _station.global_basis.inverse() * (world_pos - _station.global_position)


func _settle(ticks: int) -> void:
	for _i in ticks:
		await get_tree().physics_frame


func _run() -> void:
	if _station == null:
		_check("a station exists", false, "no Station in main.tscn")
		return

	# --- The approach --------------------------------------------------------
	var autopilot: ShipAutopilot = _ship.autopilot
	_check("station is targetable", _system.targets().has(_station),
		"%d targets" % _system.targets().size())

	# Start well out, at rest, tumbling: the autopilot is meant to be engageable
	# from wherever the player happens to be.
	var start: Vector3 = _station.global_position + Vector3(300.0, 120.0, -260.0)
	_ship.global_transform = Transform3D(Basis.IDENTITY, start)
	_ship.linear_velocity = Vector3.ZERO
	_ship.angular_velocity = Vector3(0.15, -0.2, 0.1)
	_ship.refuel()
	_system.target = _station
	autopilot.toggle()
	_check("autopilot engages on a station", autopilot.is_engaged, autopilot.status)
	if not autopilot.is_engaged:
		return

	var ticks: int = 0
	var saw_dock_phase: bool = false
	while autopilot.is_engaged and ticks < MAX_TICKS:
		await get_tree().physics_frame
		ticks += 1
		if autopilot.phase == ShipAutopilot.Phase.DOCK:
			saw_dock_phase = true

	_check("reached the dock phase", saw_dock_phase, "")
	_check("latched", _ship.docked_to == _station,
		"%s after %.1fs, %.2f units from the port"
			% [autopilot.status, ticks / 60.0,
				_ship.port_position().distance_to(_station.port_position())])
	if _ship.docked_to != _station:
		return

	# --- Latched -------------------------------------------------------------
	# The station orbits; a hull pinned in world space is left behind at once.
	var held: Vector3 = _local(_ship.global_position)
	await _settle(1800)
	var drift: float = (_local(_ship.global_position) - held).length()
	_check("stays latched for 30s", _ship.docked_to == _station and drift < 0.05,
		"drift %.4f units in the station's frame" % drift)

	# --- The astronaut -------------------------------------------------------
	_ship.set_piloted(true)
	_ship.disembark()
	await _settle(120)
	var arrival: Vector3 = _local(_astronaut.global_position)
	_check("astronaut arrives inside the station",
		_station.contains_point(_astronaut.global_position),
		"at %.1f, %.1f, %.1f in the station's frame"
			% [arrival.x, arrival.y, arrival.z])
	_check("astronaut rides the station's frame",
		_astronaut.get_parent() == _station,
		"parent is %s" % _astronaut.get_parent().name)

	# Standing still means standing still relative to the room, not the star.
	var floor_local: Vector3 = _local(_astronaut.global_position)
	await _settle(600)
	var slide: float = (_local(_astronaut.global_position) - floor_local).length()
	_check("stands still on the deck", slide < 0.5,
		"moved %.3f units in ten seconds" % slide)
	_check("is on the ground", _astronaut.is_on_floor(), "")

	# ...and can walk, driven through the real input path rather than by writing
	# `velocity` directly -- _physics_process recomputes that from the input
	# every tick, so an injected value is overwritten before it moves anything.
	var before: Vector3 = _local(_astronaut.global_position)
	Input.action_press("move_forward")
	await _settle(180)
	Input.action_release("move_forward")
	await _settle(30)
	var walked: float = (_local(_astronaut.global_position) - before).length()
	_check("walks across the deck", walked > 2.0,
		"%.1f units in three seconds" % walked)
	_check("still aboard after walking",
		_station.contains_point(_astronaut.global_position), "")
	_check("still on the deck after walking", _astronaut.is_on_floor(), "")

	# The deck has to hold up everywhere, not only where the airlock puts you.
	var solid: int = 0
	for spot in [Vector3(5.0, -2.5, 6.0), Vector3(-5.0, -2.5, -6.0),
			Vector3(0.0, -2.5, 8.5), Vector3(5.5, -2.5, -8.0)]:
		_astronaut.place_at(Transform3D(
			_station.global_basis,
			_station.global_position + _station.global_basis * spot))
		await _settle(90)
		if _astronaut.is_on_floor() and _station.contains_point(_astronaut.global_position):
			solid += 1
	_check("deck is solid across the room", solid == 4, "%d of 4 corners" % solid)

	# --- Undocking -----------------------------------------------------------
	_ship.undock()
	_check("undocking inherits the station's motion",
		(_ship.linear_velocity - _station.constant_linear_velocity).length() < 0.01,
		"%.3f units/s relative" % (_ship.linear_velocity
			- _station.constant_linear_velocity).length())
