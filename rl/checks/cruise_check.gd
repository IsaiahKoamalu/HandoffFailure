extends Node3D

## Check that flight assist can hold a commanded velocity between bodies.
##
## The landing checks all run inside a sphere of influence, where the assist has
## a surface to hold station against. This one runs where it does not, because
## that is where a self-driving ship spends most of a journey and it is the case
## the old assist did nothing at all in.
##
##     Godot --headless --fixed-fps 60 --path . res://rl/checks/cruise_check.tscn
##
## What it asserts is that the velocity error sits where the physics says it
## should, not that it goes to zero. It cannot go to zero: the assist is
## proportional, so against a gravitational pull it settles where
## `drift * _assist_strength` cancels `g` — an offset of exactly `g / 1.6`.
## That is real and it is fine; the point of a cruise is to stop velocity being
## bled away, not to make the ship inertial.
##
## So a held case is checked against that predicted offset at the local gravity
## where it ends up, which is the only correct target. A first version of this
## check used a flat tolerance and reported a failure for a ship that had simply
## flown somewhere gravity is stronger.

const MAIN_SCENE := "res://sim/main.tscn"
const FIRST: int = 600
const SECOND: int = 1200
## Somewhere outside every capture radius. Planet3 orbits at 3600 and the
## largest capture radius in the system is 2 * 260 = 520.
const START := Vector3(0.0, 0.0, 6000.0)
const CRUISE := Vector3(40.0, 0.0, -25.0)
## Error at 20 seconds over error at 10. A held velocity has converged and
## barely moves; an unheld one is still integrating gravity.
## How far the settled error may sit from `g / _assist_strength`.
const HELD_SLACK: float = 0.25
const HELD_RELATIVE: float = 0.25
## An unheld ship has to be *much* worse than an assisted one, not marginally.
const FREE_MULTIPLE: float = 3.0

var _ship: Ship
var _world: Node
var _failures: int = 0


func _ready() -> void:
	process_physics_priority = 200
	_world = load(MAIN_SCENE).instantiate()
	add_child(_world)
	_ship = _world.get_node("Ship")
	# The HUD reads nodes this check does not keep alive, and the astronaut is
	# irrelevant to it.
	for node_name in ["HUD", "Astronaut"]:
		var node: Node = _world.get_node_or_null(node_name)
		if node != null:
			_world.remove_child(node)
			node.queue_free()
	_ship.external_control = true


func _physics_process(_delta: float) -> void:
	set_physics_process(false)
	await _run()
	print("RESULT %s" % ("PASS" if _failures == 0 else "FAIL (%d)" % _failures))
	get_tree().quit(_failures)


func _place(at: Vector3, velocity: Vector3) -> void:
	_ship.global_transform = Transform3D(Basis.IDENTITY, at)
	_ship.linear_velocity = velocity
	_ship.angular_velocity = Vector3.ZERO
	_ship.command_thrust = Vector3.ZERO
	_ship.command_turn = Vector3.ZERO


func _settle(ticks: int) -> void:
	for _i in ticks:
		await get_tree().physics_frame


## Run one case and report whether the velocity error settled or kept growing.
func _case(label: String, at: Vector3, cruising: bool, held: bool) -> void:
	if cruising:
		_ship.set_cruise(CRUISE)
	else:
		_ship.clear_cruise()
	_place(at, Vector3.ZERO if cruising else CRUISE)

	await _settle(FIRST)
	var early: float = (_ship.linear_velocity - CRUISE).length()
	await _settle(SECOND - FIRST)
	var late: float = (_ship.linear_velocity - CRUISE).length()

	# The equilibrium the assist can actually reach, at the gravity the ship
	# ended up in rather than the one it started in.
	var g: float = SolarSystem.Instance.get_acceleration_at_position(
		_ship.global_position).length()
	var predicted: float = g / _assist_strength()

	var ok: bool
	if held:
		ok = absf(late - predicted) <= maxf(HELD_SLACK, HELD_RELATIVE * predicted)
	else:
		ok = late >= FREE_MULTIPLE * maxf(predicted, 0.05)
	if not ok:
		_failures += 1
	print("%-36s error %6.2f -> %6.2f   g %5.2f, predicted %5.2f   %s"
		% [label, early, late, g, predicted, "ok" if ok else "FAILED"])


## The assist gain is exported and private; read it off the ship rather than
## repeating the number, so retuning the ship cannot quietly invalidate this.
func _assist_strength() -> float:
	return maxf(float(_ship.get("_assist_strength")), 0.001)


func _run() -> void:
	var solar_system := SolarSystem.Instance
	print("dominant at start: %s"
		% ["none" if solar_system.get_dominant_planet(START) == null
			else solar_system.get_dominant_planet(START).name])

	# The control: no cruise and no body to hold station against, so the assist
	# must stay out of the way and the ship falls freely. This is the old
	# behaviour, and it is why a crossing was impossible.
	await _case("no cruise, deep space: free fall", START, false, false)

	# The thing being added.
	await _case("cruise, deep space: held", START, true, true)

	# The reference is world-space, so crossing into a sphere of influence must
	# not step the target velocity. Planet1 both orbits and spins under the
	# ship, and neither may leak into what is being held.
	var planet: Planet = null
	for candidate in solar_system.Planets:
		if candidate.name == "Planet1":
			planet = candidate
	var near: Vector3 = planet.global_position + Vector3(planet.surface_radius * 1.5, 0.0, 0.0)
	print("dominant near Planet1: %s"
		% ["none" if solar_system.get_dominant_planet(near) == null
			else solar_system.get_dominant_planet(near).name])
	await _case("cruise, inside an SOI: held", near, true, true)

	# And with the assist trimming to the surface instead, which is what the
	# ship does today: the same start, no cruise, inside a sphere of influence.
	# The velocity is driven to the ground's, not to the commanded one, so the
	# error settles far from zero -- held, but held at the wrong thing.
	_ship.clear_cruise()
	_place(near, CRUISE)
	await _settle(SECOND)
	print("no cruise, inside an SOI:            v=(%6.2f,%6.2f,%6.2f) -- trimmed to the surface"
		% [_ship.linear_velocity.x, _ship.linear_velocity.y, _ship.linear_velocity.z])
