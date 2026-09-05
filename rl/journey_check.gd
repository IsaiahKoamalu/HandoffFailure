extends Node3D

## End-to-end check of the full autopilot stack: transit, then descent.
##
## autoland_check.tscn starts the ship already inside the destination's sphere
## of influence, because until now that was the only place the autopilot could
## exist. This one starts it where a player actually would — parked at another
## planet, or out between bodies — points it at a destination with [Tab]'s
## target, and asks whether the ship gets there and lands.
##
##     Godot --headless --fixed-fps 60 --path . res://rl/journey_check.tscn
##
## It goes through ShipAutopilot rather than around it, and through the real
## main.tscn ship. What it is really testing is the handover: transit has to
## deliver the ship into the shell the policy was trained on, at a speed and
## attitude the policy has seen. A journey that arrives outside that
## distribution fails here and nowhere else in rl/.

const MAIN_SCENE := "res://main.tscn"
const MAX_TICKS: int = 70000

var _ship: Ship
var _world: Node
var _system: SolarSystem
var _failures: int = 0
var _attempts: int = 0


func _ready() -> void:
	process_physics_priority = 200
	_world = load(MAIN_SCENE).instantiate()
	add_child(_world)
	_ship = _world.get_node("Ship")
	_system = SolarSystem.Instance
	# The astronaut rides the ship and the HUD reads nodes this check does not
	# keep alive.
	for node_name in ["HUD", "Astronaut"]:
		var node: Node = _world.get_node_or_null(node_name)
		if node != null:
			_world.remove_child(node)
			node.queue_free()


func _physics_process(_delta: float) -> void:
	set_physics_process(false)
	await _run()
	print("RESULT %d/%d journeys completed" % [_attempts - _failures, _attempts])
	get_tree().quit(_failures)


func _planet(planet_name: String) -> Planet:
	for candidate in _system.Planets:
		if candidate.name == planet_name:
			return candidate
	return null


## Fly one journey and report it. `from` is a world position to start at.
func _journey(label: String, from: Vector3, to: Planet) -> void:
	_attempts += 1
	var autopilot: ShipAutopilot = _ship.autopilot
	if autopilot == null or not autopilot.is_available:
		print("%-38s SKIPPED (no policy)" % label)
		_failures += 1
		return

	_ship.global_transform = Transform3D(Basis.IDENTITY, from)
	# Start at rest in the star's frame, tumbling slightly: the point is that
	# transit does not need to be handed a good initial state.
	_ship.linear_velocity = Vector3.ZERO
	_ship.angular_velocity = Vector3(0.1, -0.2, 0.05)
	_ship.refuel()
	_system.target = to
	autopilot.toggle()
	if not autopilot.is_engaged:
		print("%-38s FAILED to engage: %s" % [label, autopilot.status])
		_failures += 1
		return

	var start: Vector3 = from
	var ticks: int = 0
	var transit_ticks: int = 0
	var top_speed: float = 0.0
	while autopilot.is_engaged and ticks < MAX_TICKS:
		await get_tree().physics_frame
		ticks += 1
		if autopilot.phase == ShipAutopilot.Phase.TRANSIT:
			transit_ticks += 1
		top_speed = maxf(top_speed, _ship.linear_velocity.length())

	var landed: bool = autopilot.status == "landed"
	var frame := LandingEnv.surface_frame(_ship.global_position, to)
	var on_site: bool = LandingEnv.is_on_target_site(_ship.global_position, to)
	var dry: bool = LandingEnv.is_over_land(_ship.global_position, to)
	var where: String = "on target" if on_site else ("wrong continent" if dry else "in the sea")
	if not (landed and on_site):
		_failures += 1
	print("%-38s %-9s %-16s %6.0f units  transit %5.1fs  descent %5.1fs  peak %5.1f u/s  alt %4.1f"
		% [
			label,
			"landed" if landed else autopilot.status,
			where,
			(to.global_position - start).length(),
			transit_ticks / 60.0,
			(ticks - transit_ticks) / 60.0,
			top_speed,
			float(frame[0]),
		])
	if autopilot.is_engaged:
		autopilot.disengage("check ended")


func _run() -> void:
	var p1: Planet = _planet("Planet1")
	var p2: Planet = _planet("Planet2")
	var p3: Planet = _planet("Planet3")

	# Deep space, no dominant body at all. This is the case the old autopilot
	# refused to engage in.
	await _journey("deep space -> Planet1", Vector3(0.0, 0.0, 6000.0), p1)

	# Parked at one planet, sent to another: the whole journey, and the longest
	# leg in the system.
	await _journey(
		"Planet3 orbit -> Planet1",
		p3.global_position + Vector3(p3.surface_radius * 1.4, 0.0, 0.0), p1)

	# ...and the reverse, so the result is not a property of one destination.
	await _journey(
		"Planet1 orbit -> Planet2",
		p1.global_position + Vector3(0.0, p1.surface_radius * 1.4, 0.0), p2)

	# Already inside the destination's influence: transit should hand over
	# immediately and this should behave exactly like autoland_check.
	await _journey(
		"already there -> Planet2",
		p2.global_position + Vector3(0.0, 0.0, p2.surface_radius * 1.3), p2)

	# And the two ways of asking for something impossible. Both have to refuse
	# and say so rather than engage and fly somewhere unlandable: a self-driving
	# ship that silently declines is worse than one that says it cannot.
	_refusal("no destination", Vector3(0.0, 0.0, 6000.0), null,
		"no destination — [Tab] to pick one")
	_refusal("destination has no ground", Vector3(0.0, 0.0, 6000.0),
		_planet("Sun"), "cannot land on Sun")


## Engage with a destination that cannot be flown to, and check the refusal.
func _refusal(label: String, from: Vector3, to: Planet, wanted: String) -> void:
	_attempts += 1
	var autopilot: ShipAutopilot = _ship.autopilot
	_ship.global_transform = Transform3D(Basis.IDENTITY, from)
	_ship.linear_velocity = Vector3.ZERO
	_system.target = to
	autopilot.toggle()
	var ok: bool = not autopilot.is_engaged and autopilot.status == wanted
	if not ok:
		_failures += 1
		if autopilot.is_engaged:
			autopilot.disengage("check ended")
	print("%-38s %-9s %s"
		% [label, "refused" if ok else "FAILED", autopilot.status])
