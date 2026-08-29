extends Node
class_name ShipAutopilot

## Flies the player's ship with the policy trained by python/ppo.py.
##
## This is the whole point of the rl/ directory: the network is not a demo
## sitting beside the game, it is a landing system inside it. It drives the
## exact same `command_thrust` / `command_turn` seam the trainer drives, and it
## reads its observation from `LandingEnv.build_observation` — the one shared
## definition — so what it sees in the cockpit is what it saw in training.
##
## It engages only with a planet under you. In deep space there is no surface to
## land on, no altitude, and no local up, so the observation would be undefined
## and the policy would be extrapolating off the end of everything it has seen.
##
## Within a planet's influence it is meant to work from anywhere: any altitude
## out to three radii, any attitude including tumbling, and carrying orbital
## velocity. It flies to that planet's largest continent and lands there. That
## breadth is a property of the training distribution rather than of this file —
## see LandingEnv.spawn_alt_max, spawn_orbit_fraction and spawn_tumble_fraction,
## which exist to make sure the policy has actually seen what it is now asked to
## survive.

## Ticks after which the autopilot gives up and hands back control. Matches
## LandingEnv.max_steps: the policy has an episode-progress input, so running it
## far past the horizon it trained on is feeding it a number it never saw.
##
## 150 seconds, because the job is no longer "land from just above the pad". It
## is "get to the largest continent from wherever you are and land on it", and
## crossing half a planet at the ship's cruise speed is a minute on its own.
const MAX_TICKS: int = 9000

## How far the ship's origin sits above the ground when it is resting on its
## struts, plus margin.
##
## This is not a fudge factor, it is a mismatch the training environment cannot
## see. LandingEnv ends an episode at altitude zero, measured to the ship's
## origin, because there are no colliders in training. In the game the hull has
## landing legs, so a ship sitting on the ground reads about 1.4 — and an
## autopilot waiting for zero waits forever. It did exactly that: eight for
## eight, descending correctly and then hovering a metre and a half up until it
## timed out.
const HULL_CLEARANCE: float = 2.2
## Speed relative to the surface below which a ship at hull clearance is down.
const SETTLED_SPEED: float = 1.5

signal engaged_changed(engaged: bool)

var is_engaged: bool = false
var is_available: bool = false
## Why the autopilot is not flying, for the HUD to show.
var status: String = "no policy"

var _ship: Ship
var _pilot: NeuralPilot
var _observation := PackedFloat32Array()
var _ticks: int = 0
## Whatever the pilot had set before we took over, so disengaging puts it back.
var _restore_external: bool = false


func _init(ship: Ship) -> void:
	_ship = ship
	name = "Autopilot"


func _ready() -> void:
	# Ahead of Ship, so the commands are already in place when it reads them.
	process_physics_priority = -10
	_observation.resize(LandingEnv.OBS_SIZE)
	_pilot = NeuralPilot.load_from()
	is_available = _pilot != null
	# A fresh checkout has no trained policy and that is fine — the game plays
	# exactly as it did before, and the key does nothing but say so.
	status = "ready" if is_available else "no policy (see rl/README.md)"


func toggle() -> void:
	if is_engaged:
		disengage("pilot took over")
		return
	if not is_available:
		return
	if SolarSystem.Instance.get_dominant_planet(_ship.global_position) == null:
		status = "no planet in range"
		return
	is_engaged = true
	_ticks = 0
	_restore_external = _ship.external_control
	_ship.external_control = true
	status = "engaged"
	engaged_changed.emit(true)


func disengage(reason: String) -> void:
	if not is_engaged:
		return
	is_engaged = false
	_ship.external_control = _restore_external
	_ship.command_thrust = Vector3.ZERO
	_ship.command_turn = Vector3.ZERO
	status = reason
	engaged_changed.emit(false)


func _physics_process(_delta: float) -> void:
	if not is_engaged:
		return

	var planet: Planet = SolarSystem.Instance.get_dominant_planet(_ship.global_position)
	if planet == null:
		disengage("left the planet")
		return

	# Hand back once the ship is actually down. The policy's job ends at
	# touchdown; Ship._process_parking takes it from there.
	var frame := LandingEnv.surface_frame(_ship.global_position, planet)
	var altitude: float = frame[0]
	# is_grounded() reads a RayCast3D, which lags a frame behind a teleport and
	# will happily report the ground the ship was at before it moved. Bounding
	# it by altitude keeps a stale hit from ending the descent on engage.
	var settled: bool = altitude <= HULL_CLEARANCE 		or (_ship.is_grounded() and altitude <= HULL_CLEARANCE * 2.5)
	if settled and planet.get_relative_velocity_to_surface(
		_ship.global_position, _ship.linear_velocity
	).length() < SETTLED_SPEED:
		disengage("landed")
		return

	_ticks += 1
	if _ticks > MAX_TICKS:
		disengage("timed out")
		return

	LandingEnv.build_observation(
		_ship, planet, float(_ticks) / float(MAX_TICKS), _observation, 0
	)
	var action := _pilot.act(_observation)
	_ship.command_thrust = Vector3(action[0], action[1], action[2])
	_ship.command_turn = Vector3(action[3], action[4], action[5])
