extends Node
class_name ShipAutopilot

## Flies the player's ship to a chosen planet and lands it there.
##
## This is the whole point of the rl/ directory: the network is not a demo
## sitting beside the game, it is a landing system inside it. It drives the
## exact same `command_thrust` / `command_turn` seam the trainer drives, and it
## reads its observation from `LandingEnv.build_observation` — the one shared
## definition — so what it sees in the cockpit is what it saw in training.
##
## It is a stack, not a policy. A journey has phases that want different things,
## and the trained policy answers only the last of them:
##
##   TRANSIT   analytic guidance, anywhere in the system, including deep space
##   DESCENT   the trained policy, inside a planet's sphere of influence
##   DOCK      analytic guidance again, onto a station's port
##
## Transit is deliberately *not* learned. It is a proportional guidance law over
## a problem that has a closed form, and rl/README.md records the hand-written
## controller matching or beating PPO in every configuration measured so far.
## RL earns its place where the problem is terrain-shaped and contact-rich —
## the descent — and not before.
##
## The descent phase is meant to work from anywhere inside the destination's
## influence: any altitude out to three radii, any attitude including tumbling,
## and carrying orbital velocity. That breadth is a property of the training
## distribution rather than of this file — see LandingEnv.spawn_alt_max,
## spawn_orbit_fraction and spawn_tumble_fraction, which exist to make sure the
## policy has actually seen what it is now asked to survive.

enum Phase { IDLE, TRANSIT, DESCENT, DOCK }

## Ticks the descent phase may run for. Matches LandingEnv.max_steps: the policy
## has an episode-progress input, so running it far past the horizon it trained
## on is feeding it a number it never saw.
##
## 450 seconds, because the job is no longer "land from just above the pad". It
## is "get to the largest continent from wherever you are and land on it", and
## crossing half a planet at the ship's cruise speed is a minute on its own.
## Measured: at 150 seconds one approach in fifteen was still airborne and on
## course when the clock cut it off. Kept in step with LandingEnv.max_steps.
const DESCENT_MAX_TICKS: int = 27000
## Ticks the transit phase may run for, per attempt. The system is 3,600 units
## across and transit closes that in well under a minute; ten minutes means
## something is wrong rather than slow.
const TRANSIT_MAX_TICKS: int = 36000

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

## Altitude, in destination radii, at which transit hands over to the policy.
##
## The capture radius is 2 R, so the ship becomes the destination's dominant
## body at almost exactly this altitude — the two conditions agree by
## construction rather than by coincidence, and the handover happens once
## instead of chattering between phases at the boundary.
const HANDOFF_ALT_FACTOR: float = 1.0
## Speed relative to the destination that transit aims to arrive at.
##
## Chosen to sit inside the policy's spawn distribution rather than to be fast:
## LandingEnv spawns with a radial component in [-8, +2] units/s, so handing the
## policy a ship falling at 20 would be starting it outside everything it has
## seen. Arriving slow costs a few seconds and keeps the descent in
## distribution.
const ARRIVAL_SPEED: float = 5.0
## Fraction of full thrust transit plans its braking around. The rest is margin
## for gravity, for steering, and for the fact that the destination is moving.
const BRAKE_FRACTION: float = 0.5
## Speed cap, for the leg in the middle where braking distance is not binding.
const CRUISE_SPEED_MAX: float = 300.0
## Velocity error below which transit stops thrusting and lets flight assist
## hold the cruise.
##
## The assist is uncapped — it applies `drift * _assist_strength * mass` with no
## ceiling and no fuel — so commanding a cruise far from the current velocity
## would accelerate the ship for free. Engines do the accelerating and burn for
## it; the assist only trims what is left. See Ship.set_cruise.
const CRUISE_TRIM_BAND: float = 8.0
## Velocity error at which transit asks for full thrust. Below it the command
## tapers, so the ship eases onto its cruise instead of chattering across it.
const THRUST_FULL_ERROR: float = 12.0
## Proportional gain turning a pointing error into a turn rate.
const TURN_GAIN: float = 2.0

# --- Docking gains -----------------------------------------------------------
# A direct port of the controller measured in python/evaluate.py, which scores
# 76.7% against 0.0% for no input at all. The numbers are in world units here
# rather than the observation's scaled ones, but they are the same law and every
# comment on them there applies.

## Ticks the dock phase may run for before giving up.
const DOCK_MAX_TICKS: int = 9000
## Gain on the across-corridor error. High, because flight assist trims drift
## against the planet the station orbits rather than against the station, and a
## proportional law against that standing disturbance settles where its command
## cancels it.
const DOCK_ACROSS_GAIN: float = 3.0
## Ceiling on any commanded speed, units per second.
const DOCK_SPEED_MAX: float = 5.0
## Closing rate per unit of range, so contact is gentle without a braking phase.
const DOCK_ARRIVE_GAIN: float = 0.25
## Closing rate held once lined up. Everything else here tapers to nothing on
## arrival, and a proportional law against a standing disturbance stops where
## its command equals it -- measured, four tenths of a unit short of the gate.
const DOCK_CREEP: float = 0.6
## Lateral offset at which closing is throttled to nothing, in world units.
const DOCK_TIGHT: float = 1.5
## Alignment required before closing at all.
const DOCK_COMMIT_ALIGN: float = -0.95
## Gain on the velocity error, and on pointing and spin.
const DOCK_VELOCITY_GAIN: float = 0.3
const DOCK_TURN_GAIN: float = 2.5
const DOCK_SPIN_GAIN: float = 1.2

signal engaged_changed(engaged: bool)

var is_engaged: bool = false
var is_available: bool = false
var phase: Phase = Phase.IDLE

## Where the autopilot is flying. A Planet to land on or a Station to dock
## with; chosen on engage and held for the journey.
var destination: Node3D = null
## Why the autopilot is not flying, or what it is doing, for the HUD to show.
var status: String = "no policy"

## The action being held, and how many ticks are left of it.
var _command_thrust: Vector3 = Vector3.ZERO
var _command_turn: Vector3 = Vector3.ZERO
var _held: int = 0

var _ship: Ship
var _pilot: NeuralPilot
var _observation := PackedFloat32Array()
var _descent_ticks: int = 0
var _transit_ticks: int = 0
var _dock_ticks: int = 0
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


## Where to fly: the navigation target if one is set, otherwise the planet
## underneath. Tab cycles the target, so "go there" is already a thing the
## player can say without a new control.
func _pick_destination() -> Node3D:
	var solar_system := SolarSystem.Instance
	var chosen: Node3D = solar_system.target
	if chosen == null:
		chosen = solar_system.get_dominant_planet(_ship.global_position)
	return chosen


func toggle() -> void:
	if is_engaged:
		disengage("pilot took over")
		return
	if not is_available:
		return

	var chosen: Node3D = _pick_destination()
	if chosen == null:
		status = "no destination — [Tab] to pick one"
		return
	# The policy lands on terrain and navigates by the largest landmass. A body
	# with neither is not somewhere it can be asked to go. A station is always
	# fair game: it has a port, which is the whole of what docking needs.
	if chosen is Planet and not (chosen as Planet).terrain_enabled:
		status = "cannot land on %s" % chosen.name
		return

	destination = chosen
	is_engaged = true
	phase = Phase.TRANSIT
	_descent_ticks = 0
	_transit_ticks = 0
	_held = 0
	_restore_external = _ship.external_control
	_ship.external_control = true
	status = "engaged"
	engaged_changed.emit(true)


func disengage(reason: String) -> void:
	if not is_engaged:
		return
	is_engaged = false
	phase = Phase.IDLE
	_ship.external_control = _restore_external
	_ship.command_thrust = Vector3.ZERO
	_ship.command_turn = Vector3.ZERO
	# Leaving a cruise set would have flight assist holding a velocity the
	# player never asked for, for the rest of the session.
	_ship.clear_cruise()
	status = reason
	engaged_changed.emit(false)


## Short label for the HUD.
func phase_name() -> String:
	match phase:
		Phase.TRANSIT: return "TRANSIT"
		Phase.DESCENT: return "DESCENT"
		Phase.DOCK: return "DOCK"
	return "IDLE"


func _physics_process(_delta: float) -> void:
	if not is_engaged:
		return
	if not is_instance_valid(destination):
		disengage("destination is gone")
		return

	if destination is Station:
		_fly_to_station(destination as Station)
		return

	var planet: Planet = destination as Planet
	var dominant: Planet = SolarSystem.Instance.get_dominant_planet(_ship.global_position)
	var frame := LandingEnv.surface_frame(_ship.global_position, planet)
	var altitude: float = frame[0]

	if _has_landed(planet, altitude):
		disengage("landed")
		return

	# Arrived: the destination owns the ship and it is inside the shell the
	# policy was trained on. Handing over any earlier would ask the policy to
	# fly a regime it has never seen; any later would waste the approach.
	if dominant == planet and altitude <= HANDOFF_ALT_FACTOR * planet.surface_radius:
		if phase != Phase.DESCENT:
			phase = Phase.DESCENT
			_descent_ticks = 0
			_held = 0
			# The policy expects the assist it trained with, which holds station
			# over the ground rather than holding a cruise.
			_ship.clear_cruise()
		_run_descent(planet)
		return

	# Anywhere else, including deep space and including having been thrown back
	# out of a descent. A self-driving ship that gives up because it drifted out
	# of the shell is not self-driving; it flies the approach again.
	if phase != Phase.TRANSIT:
		phase = Phase.TRANSIT
		_transit_ticks = 0
	_run_transit()


## Station journeys: the same transit law out to the corridor, then the dock
## controller. There is no policy in this branch at all -- see the note at the
## top about where RL earns its place.
func _fly_to_station(station: Station) -> void:
	if _ship.docked_to == station:
		disengage("docked")
		return

	var to_port: float = _ship.port_position().distance_to(station.port_position())
	# Hand over at twice the corridor length. Transit aims at the station's
	# centre and the dock controller aims at its port, so the switch has to
	# happen while there is still room to fix the difference.
	if to_port <= station.approach_range * 2.0:
		if phase != Phase.DOCK:
			phase = Phase.DOCK
			_dock_ticks = 0
			# The dock controller commands velocities directly and the assist
			# would be holding a cruise from the transit leg.
			_ship.clear_cruise()
		_run_dock(station)
		return

	if phase != Phase.TRANSIT:
		phase = Phase.TRANSIT
		_transit_ticks = 0
	_run_transit()


func _has_landed(planet: Planet, altitude: float) -> bool:
	# is_grounded() reads a RayCast3D, which lags a frame behind a teleport and
	# will happily report the ground the ship was at before it moved. Bounding
	# it by altitude keeps a stale hit from ending the descent on engage.
	var settled: bool = altitude <= HULL_CLEARANCE \
		or (_ship.is_grounded() and altitude <= HULL_CLEARANCE * 2.5)
	if not settled:
		return false
	return planet.get_relative_velocity_to_surface(
		_ship.global_position, _ship.linear_velocity
	).length() < SETTLED_SPEED


# --- Transit -----------------------------------------------------------------

## Analytic guidance to the handover point.
##
## One law does the whole leg. The commanded speed is whatever can still be shed
## before arrival — `sqrt(v_arrival^2 + 2 a s)` — capped by CRUISE_SPEED_MAX, so
## acceleration, cruise and braking are the same equation evaluated at different
## ranges rather than three phases with boundaries to tune between them.
func _run_transit() -> void:
	_transit_ticks += 1
	if _transit_ticks > TRANSIT_MAX_TICKS:
		disengage("transit timed out")
		return

	var to_target: Vector3 = destination.global_position - _ship.global_position
	var distance: float = to_target.length()
	if distance < 0.001:
		return
	var direction: Vector3 = to_target / distance
	# Range left to the handover shell, not to the planet's centre or its
	# surface. Braking to a stop at the surface would mean arriving through the
	# ground.
	var handoff_radius: float = SolarSystem.target_radius(destination) * (
		1.0 + HANDOFF_ALT_FACTOR if destination is Planet else 2.0
	)
	var range_left: float = maxf(distance - handoff_radius, 0.0)

	var brake: float = _ship.thrust_acceleration() * BRAKE_FRACTION
	var allowed: float = sqrt(ARRIVAL_SPEED * ARRIVAL_SPEED + 2.0 * brake * range_left)
	var speed: float = minf(CRUISE_SPEED_MAX, allowed)

	# The destination is orbiting. Closing on where it is now, at a speed
	# expressed in the star's frame, aims at where it was: the commanded
	# velocity carries the destination's own motion and adds the closing speed
	# on top of it.
	var wanted: Vector3 = SolarSystem.target_velocity(destination) + direction * speed
	var error: Vector3 = wanted - _ship.linear_velocity

	if error.length() <= CRUISE_TRIM_BAND:
		# On the cruise. Hand it to flight assist and stop burning: the assist
		# only has a small residual to hold, which is the job it does elsewhere.
		_ship.set_cruise(wanted)
		_command_thrust = Vector3.ZERO
	else:
		# Off the cruise by more than the assist should be asked to make up.
		# Engines do this, and pay for it.
		_ship.clear_cruise()
		var authority: float = minf(error.length() / THRUST_FULL_ERROR, 1.0)
		_command_thrust = _to_local(error.normalized()) * authority

	_command_turn = _turn_toward(direction)
	_ship.command_thrust = _command_thrust
	_ship.command_turn = _command_turn


## A world direction in the ship's own axes, which is the frame command_thrust
## is expressed in.
func _to_local(world_direction: Vector3) -> Vector3:
	return _ship.global_basis.inverse() * world_direction


## Turn rates that swing the nose toward a world direction.
##
## Cosmetic, strictly: thrust is six-axis and applied along `global_basis *
## command_thrust`, so the ship can fly any direction whatever it is facing. It
## is here because a ship crossing a solar system sideways does not read as one
## that is driving itself.
##
## The nose is local -Z, so the axis that rotates it onto a local direction `d`
## is `(0, 0, -1) x d = (d.y, -d.x, 0)` — pitch and yaw, no roll.
func _turn_toward(world_direction: Vector3) -> Vector3:
	var d: Vector3 = _to_local(world_direction).normalized()
	return Vector3(
		clampf(d.y * TURN_GAIN, -1.0, 1.0),
		clampf(-d.x * TURN_GAIN, -1.0, 1.0),
		0.0,
	)


# --- Descent -----------------------------------------------------------------

func _run_descent(planet: Planet) -> void:
	_descent_ticks += 1
	if _descent_ticks > DESCENT_MAX_TICKS:
		disengage("descent timed out")
		return

	# Decide at the rate the policy was trained at, and hold in between. The
	# commands are not latched on the ship, so they are re-applied every tick.
	if _held <= 0:
		LandingEnv.build_observation(
			_ship, planet,
			float(_descent_ticks) / float(DESCENT_MAX_TICKS), _observation, 0
		)
		var action := _pilot.act(_observation)
		_command_thrust = Vector3(action[0], action[1], action[2])
		_command_turn = Vector3(action[3], action[4], action[5])
		_held = _pilot.repeat
	_held -= 1
	_ship.command_thrust = _command_thrust
	_ship.command_turn = _command_turn


# --- Dock --------------------------------------------------------------------

## Fly the corridor and latch.
##
## Two commands, and only one of them is ever throttled. Across the corridor is
## corrected at full authority always; closing is gated on being centred, facing
## the right way and in front of the port face. Blending the two instead — a
## single aim point somewhere between the port and a standoff — makes them
## compete, and the lateral error then settles at exactly whatever threshold the
## blend uses, with the ship parked in a perfect corridor until the clock runs
## out. Measured twice, at two different thresholds, before it was believed.
func _run_dock(station: Station) -> void:
	_dock_ticks += 1
	if _dock_ticks > DOCK_MAX_TICKS:
		disengage("docking timed out")
		return

	if _ship.try_dock(station):
		disengage("docked")
		return

	var normal: Vector3 = station.port_normal()
	var offset: Vector3 = station.port_position() - _ship.port_position()
	var distance: float = offset.length()
	var axial: float = -offset.dot(normal)
	var across: Vector3 = offset + normal * axial
	var lateral: float = across.length()
	var alignment: float = _ship.port_normal().dot(normal)

	var wanted: Vector3 = Vector3.ZERO
	if lateral > 0.001:
		wanted = across / lateral * minf(lateral * DOCK_ACROSS_GAIN, DOCK_SPEED_MAX)

	if axial > 0.0 and lateral < DOCK_TIGHT and alignment < DOCK_COMMIT_ALIGN:
		var gate: float = clampf(1.0 - lateral / DOCK_TIGHT, 0.0, 1.0)
		var approach: float = maxf(
			minf(distance * DOCK_ARRIVE_GAIN, DOCK_SPEED_MAX), DOCK_CREEP
		)
		wanted -= normal * (approach * gate)
	elif axial <= 0.0:
		# Behind the port face there is nothing to line up on: get out in front
		# of it first, which is also what keeps this out of the hull.
		wanted += normal * DOCK_SPEED_MAX

	var error: Vector3 = (wanted + station.port_velocity()) - _ship.port_velocity()
	_command_thrust = _to_local(error) * DOCK_VELOCITY_GAIN
	_command_thrust = Vector3(
		clampf(_command_thrust.x, -1.0, 1.0),
		clampf(_command_thrust.y, -1.0, 1.0),
		clampf(_command_thrust.z, -1.0, 1.0),
	)

	# The ship's port is on its nose, so the nose has to end up pointing back
	# down the corridor -- along the inward normal.
	var d: Vector3 = _to_local(-normal).normalized()
	var spin: Vector3 = _to_local(_ship.angular_velocity)
	_command_turn = Vector3(
		clampf(d.y * DOCK_TURN_GAIN - spin.x * DOCK_SPIN_GAIN, -1.0, 1.0),
		clampf(-d.x * DOCK_TURN_GAIN - spin.y * DOCK_SPIN_GAIN, -1.0, 1.0),
		clampf(-spin.z * DOCK_SPIN_GAIN, -1.0, 1.0),
	)
	_ship.command_thrust = _command_thrust
	_ship.command_turn = _command_turn
