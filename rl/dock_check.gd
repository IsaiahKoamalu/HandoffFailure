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

	# --- The join closes and its floor is solid ------------------------------
	# The latch tolerates two units of port-to-port error, so what a ship
	# arrives with is not what it keeps: retraction has to pull the ports
	# together, or the doorway is left with a hole in its floor exactly as wide
	# as the docking error. Measured before the retraction existed: a 0.65-unit
	# slot across the full width of the threshold, which is a thing to fall
	# through.
	var arrived: float = _ship.port_position().distance_to(_station.port_position())
	await _settle(int(_ship.dock_capture_seconds * 60.0) + 60)
	var gap: float = _ship.port_position().distance_to(_station.port_position())
	_check("retraction closes the join", gap < 0.05,
		"%.3f units apart on arrival, %.4f after" % [arrived, gap])

	var holes: int = 0
	var probes: int = 0
	var space := get_viewport().world_3d.direct_space_state
	var x: float = -1.0
	while x <= 1.0001:
		var z: float = 2.2
		while z <= 3.2001:
			var from: Vector3 = _ship.global_transform * Vector3(x, 0.6, z)
			var to: Vector3 = _ship.global_transform * Vector3(x, -2.4, z)
			var query := PhysicsRayQueryParameters3D.create(from, to)
			# The astronaut's own mask, not a convenient one: the hull it may
			# not collide with is exactly the hull that cannot hold it up.
			query.collision_mask = _astronaut.collision_mask
			probes += 1
			if space.intersect_ray(query).is_empty():
				holes += 1
			z += 0.05
		x += 0.1
	_check("floor is continuous across the doorway", holes == 0,
		"%d of %d points with nothing under them" % [holes, probes])

	# --- Latched -------------------------------------------------------------
	# The station orbits; a hull pinned in world space is left behind at once.
	# Measured from *after* the retraction, which is motion this asked for --
	# baselined before it, the pull-in reads as two thirds of a unit of drift.
	var held: Vector3 = _local(_ship.global_position)
	await _settle(1800)
	var drift: float = (_local(_ship.global_position) - held).length()
	_check("stays latched for 30s", _ship.docked_to == _station and drift < 0.05,
		"drift %.4f units in the station's frame" % drift)

	# --- The astronaut walks across ------------------------------------------
	# The transfer is simulated: no teleport, no fade. Out of the seat, forward
	# through the aft doorway, along the tunnel and into the room, on foot,
	# through the same input path a player uses.
	_ship.set_piloted(true)
	_ship.disembark()
	await _settle(90)
	_check("astronaut starts in the ship", _astronaut.get_parent() == _ship,
		"parent is %s" % _astronaut.get_parent().name)
	_check("hatches open when docked",
		_ship.is_hatch_open() and not _station.hatch.visible, "")

	# Walking there. The frames are watched the whole way: a gap between the
	# ship's volume and the station's would hand the astronaut to the planet
	# four hundred units below, mid-stride, and that does not look like a bug so
	# much as a stumble.
	var frames := {}
	var handed_to_planet: bool = false
	Input.action_press("move_forward")
	for i in 360:
		await get_tree().physics_frame
		var parent: Node = _astronaut.get_parent()
		frames[parent.name] = true
		if parent is Planet or parent is SolarSystem:
			handed_to_planet = true
		if i % 90 == 0:
			var at: Vector3 = _local(_astronaut.global_position)
			print("    t=%3d  station-local z %6.2f   frame %s" % [i, at.z, parent.name])
	Input.action_release("move_forward")
	await _settle(60)

	var here: Vector3 = _local(_astronaut.global_position)
	_check("walked out of the ship and into the station",
		_astronaut.get_parent() == _station,
		"parent is %s, at %.1f, %.1f, %.1f in the station's frame"
			% [_astronaut.get_parent().name, here.x, here.y, here.z])
	_check("never handed to the planet on the way", not handed_to_planet,
		"frames seen: %s" % ", ".join(PackedStringArray(frames.keys())))
	_check("reached the room, not just the tunnel",
		absf(here.z) <= _station.interior_extents.z, "z %.1f" % here.z)
	_check("is on the ground", _astronaut.is_on_floor(), "")

	# Standing still means standing still relative to the room, not the star.
	var floor_local: Vector3 = _local(_astronaut.global_position)
	await _settle(600)
	var slide: float = (_local(_astronaut.global_position) - floor_local).length()
	_check("stands still on the deck", slide < 0.5,
		"moved %.3f units in ten seconds" % slide)

	# The deck has to hold up everywhere, not only on the walk in.
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
	# Measured before any time passes: the station is on a circular orbit, so
	# its velocity vector turns every tick and comparing a frame later measures
	# that rotation rather than the handover.
	var inherited: float = (_ship.linear_velocity
		- _station.constant_linear_velocity).length()
	await _settle(2)
	_check("hatches close on release",
		not _ship.is_hatch_open() and _station.hatch.visible, "")
	_check("undocking inherits the station's motion", inherited < 0.01,
		"%.4f units/s relative" % inherited)
