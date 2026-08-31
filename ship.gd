extends RigidBody3D
class_name Ship

## A small landable ship with a walk-in cabin and a hinged rear ramp.
##
## The cabin is a frame of reference for the Astronaut in the same way a Planet
## is: while inside, the astronaut is reparented here and rides the ship's
## motion for free. See `contains_point` and `get_surface_velocity`.
##
## The astronaut never touches this body -- see `_build_deck`.

## Collision layers, mirroring the names in project.godot.
const LAYER_WORLD: int = 1 << 0
const LAYER_SHIP: int = 1 << 1
const LAYER_DECK: int = 1 << 3

@export_category("Nodes")
@export var _camera: Camera3D
@export var _camera_pivot: Node3D
@export var _ground_cast: RayCast3D
## Where the astronaut stands when they get out of the pilot seat.
@export var _exit_point: Node3D
@export var _astronaut: Astronaut
## Hinge node for the rear ramp; rotated about its local X axis.
@export var _ramp: AnimatableBody3D
@export var _cockpit: Interactable
@export var _cockpit_light_switch: Interactable
## Forward headlight. Lives under the camera pivot, so it points where you look.
@export var _headlight: SpotLight3D

@export_category("Cabin Light Switch")
@export var _cabin_light: OmniLight3D

@export_category("Cabin")
## Walkable volume in ship-local space, used to decide when the astronaut is
## aboard. Kept as a box rather than an Area3D so the test is deterministic and
## does not depend on physics-signal ordering.
@export var _interior_center: Vector3 = Vector3(0.0, 1.2, 0.0)
@export var _interior_extents: Vector3 = Vector3(1.5, 1.2, 2.5)

@export_category("Ramp")
## Steepest the ramp will go if it never finds ground. Keep at or below the
## astronaut's floor_max_angle or the lowered ramp becomes an unclimbable wall.
@export var _ramp_open_degrees: float = 45.0
@export var _ramp_closed_degrees: float = -90.0
@export var _ramp_speed: float = 55.0
## Distance from the hinge to the lip, used to probe for ground.
@export var _ramp_length: float = 3.2

@export_category("Fuel")
## Seconds of continuous full thrust the tanks hold. Zero or less is unlimited,
## which is what the player flies with unless something says otherwise -- so
## this changes nothing about the game until it is deliberately turned on.
##
## Landing is a task a proportional law already solves well, which is why a
## learned policy has only ever managed to match the hand-written one. A budget
## is what makes the problem worth learning: it turns a control task into a
## constrained optimisation over the whole flight, where crossing early and
## coasting beats holding a fixed cruise speed, and there is no obvious law to
## write down.
@export var fuel_seconds: float = 0.0

## Fuel left, in seconds of full thrust. Meaningless when fuel_seconds <= 0.
var fuel: float = 0.0
## Whether the tanks are finite at all. Cached because it is read every tick.
var has_fuel_limit: bool = false
## Dry, with a limit in force. The engines are dead; gravity is not.
var is_out_of_fuel: bool = false

@export_category("Water")
## Upward force underwater as a multiple of local gravity. Above 1.0 the ship
## surfaces; below 1.0 it settles onto the seabed.
@export var _buoyancy: float = 1.15
## Linear damping applied while submerged, so water actually feels thick.
@export var _water_drag: float = 3.0
## Depth at which the hull counts as fully under and buoyancy reaches full
## strength. Buoyancy ramps up over this distance rather than switching on at
## the waterline, which is what gives the ship a stable float instead of
## launching it out of the sea and bouncing.
@export var _hull_depth: float = 2.0

@export_category("Handling")
## How quickly the ship catches up to where the mouse has asked it to point.
## The old code applied mouse movement straight to the transform, so the ship
## snapped instantly and stopped instantly — 200 kg of mass when you thrust and
## none at all when you turn. Lower values feel heavier; 30 is close to the old
## instant behaviour.
@export_range(0.5, 30.0, 0.1) var _turn_response: float = 18.0
## Seconds after you stop steering before auto-levelling resumes. Without this
## the assist fights every input you make.
@export_range(0.0, 3.0, 0.05) var _level_delay: float = 0.6
## How far the pilot can pitch the camera while parked, in degrees either way.
## Short of 90 so the view never rolls over the top.
@export_range(10.0, 89.0, 1.0) var _free_look_pitch: float = 80.0
## Degrees per second the camera swings back to face forward once the ship is
## flying again. Fast enough not to be a wait, slow enough not to be a jump.
@export_range(30.0, 720.0, 10.0) var _free_look_recentre: float = 240.0
## Seconds for the engines to spool from idle to full.
@export_range(0.0, 3.0, 0.05) var _throttle_response: float = 0.3
## How hard flight assist cancels drift you did not ask for, as a multiple of
## the ship's mass per second.
@export_range(0.0, 6.0, 0.1) var _assist_strength: float = 1.6

@export_category("Settings")
@export var _mouse_sensitivity: float = 0.3
## Degrees per second at full stick deflection, for external controllers. The
## mouse path has no equivalent because a mouse delta is already an angle.
@export var _stick_rate: float = 90.0
## Thrust as an acceleration, so handling does not change with the ship's mass.
##
## Surface gravity on these planets is 9.81, so this is a thrust-to-weight ratio
## of about 1.8. It used to be 10.0 -- a ratio of 1.02 -- and that number could
## not survive contact with a landing. Flight assist deliberately stops damping
## whichever axis you are thrusting along, so a ship braking its descent gets
## thrust minus gravity and nothing else: 0.19 units/s^2, which needs half a
## minute and ninety units of altitude to wash off a six-unit-a-second fall. The
## ship had no way to stop, and the only reason landings ever appeared to work
## was that it was coming down in the sea, where water drag did the braking.
@export var _thrust: float = 18.0
@export var _auto_orient_speed: float = 2.0
## Speed relative to the ground below which a landed ship parks itself.
@export var _landing_speed: float = 0.2

## Gravity of the dominant planet, as a force. Zero when in open space.
var closest_force: Vector3 = Vector3.ZERO
var is_piloted: bool = false
var is_ramp_open: bool = false
var is_cabin_switch: bool = true
var is_headlight_on: bool = true
## Below sea level on the dominant planet.
var is_submerged: bool = false
## Engine spool, 0 to 1. Shown on the HUD.
var throttle: float = 0.0
## When on, the ship cancels sideways drift relative to the body it is near.
var flight_assist: bool = true

# --- External control --------------------------------------------------------
# A seam for any pilot that is not a person at a mouse: the RL trainer, and the
# trained policy flying the ship in game. The dynamics below this line do not
# know or care which one is at the controls, which is the point — a policy
# trained against `command_*` is flying the same ship the player does.

## When true, `command_*` replace mouse and keyboard. Expected to be refreshed
## every physics tick; they are not latched.
var external_control: bool = false
## Local-space thrust request, each component in [-1, 1]: +x right, +y up,
## +z forward. Matches the six translation keys the player has.
var command_thrust: Vector3 = Vector3.ZERO
## Rotation request, each component in [-1, 1] of full stick: x pitch (nose up
## positive), y yaw (left positive), z roll.
var command_turn: Vector3 = Vector3.ZERO
## Skip the camera, exhaust, audio and terrain streaming. Set before the ship
## enters the tree. Training runs many ships at once and none of them are looked
## at, so all of that is pure cost.
var headless_agent: bool = false
## Build the exhaust plumes on a headless agent anyway. Only set when someone
## is actually watching training: the plume is the clearest read on what a
## policy is doing with the throttle, and it is the one piece of the player's
## presentation worth paying for sixteen times over. Audio, the camera and
## terrain streaming stay off.
var agent_visuals: bool = false

## Fill the tanks, and settle whether there are any. Called on spawn and by the
## trainer on every episode reset.
func refuel() -> void:
	has_fuel_limit = fuel_seconds > 0.0
	fuel = fuel_seconds if has_fuel_limit else 0.0
	is_out_of_fuel = false


## Fuel left as a fraction of capacity, and 1.0 when the tanks are unlimited --
## so the observation reads "plenty" rather than "empty" for a ship that has no
## budget to worry about.
func fuel_fraction() -> float:
	if not has_fuel_limit:
		return 1.0
	return clampf(fuel / fuel_seconds, 0.0, 1.0)


## Trained landing autopilot. Null in a checkout with no policy file.
var autopilot: ShipAutopilot = null

var _thrusters: Array[GPUParticles3D] = []
var _exhaust_glow: OmniLight3D = null
var _engine_audio: EngineAudio = null

## Rotation the mouse has asked for but the ship has not yet made, in degrees.
var _pending_turn: Vector2 = Vector2.ZERO
## Where the pilot has aimed the camera while parked, in degrees: x pitch, y yaw.
var _free_look: Vector2 = Vector2.ZERO
## Seconds since the pilot last steered. Auto-levelling waits this out.
var _since_steer: float = 999.0

var _surface_position: Vector3 = Vector3.ZERO
var _is_parked: bool = false
var _ramp_degrees: float = 0.0
var _mouse_delta: Vector2 = Vector2.ZERO


func _ready() -> void:
	# The ramp is a body in its own right so it can swing independently; without
	# this the hull and the ramp shove each other apart where they meet.
	add_collision_exception_with(_ramp)
	_ramp_degrees = _ramp_closed_degrees
	_ramp.rotation.x = deg_to_rad(_ramp_degrees)
	refuel()
	if headless_agent:
		# Terrain collision is replaced by analytic ground in training; see
		# rl/landing_env.gd. Nothing here is heard, and normally nothing is
		# rendered either.
		if agent_visuals:
			_build_thrusters()
		return
	_build_deck()
	_build_thrusters()
	_build_engine_audio()
	autopilot = ShipAutopilot.new(self)
	add_child(autopilot)
	set_piloted(true)
	_register_terrain.call_deferred()


# SolarSystem.Instance is assigned in Main's _ready, which Godot runs after its
# children's, so this cannot be called directly from _ready.
func _register_terrain() -> void:
	SolarSystem.Instance.register_terrain_client(self)


func _physics_process(delta: float) -> void:
	var solar_system := SolarSystem.Instance
	apply_central_force(solar_system.get_acceleration_at_position(global_position) * mass)

	var dominant: Planet = solar_system.get_dominant_planet(global_position)
	closest_force = Vector3.ZERO
	if dominant != null:
		closest_force = dominant.get_acceleration_at_position(global_position) * mass

	_process_buoyancy(dominant)

	# Touching the controls takes the ship back. An autopilot you have to
	# switch off before you can save yourself is not a feature.
	if autopilot != null and autopilot.is_engaged and is_piloted:
		if _get_movement_input() != Vector3.ZERO or not _mouse_delta.is_zero_approx():
			autopilot.disengage("pilot took over")

	var movement := _get_thrust_request()
	if is_piloted or external_control:
		_process_look_inputs(delta)

	_process_parking(movement)

	# Engines spool rather than snapping between off and full.
	var demand := 1.0 if movement != Vector3.ZERO else 0.0
	if _throttle_response <= 0.0:
		throttle = demand
	else:
		throttle = move_toward(throttle, demand, delta / _throttle_response)

	if movement != Vector3.ZERO:
		# Thrust authority. For the player this is always 1.0: a key is down or
		# it is not, and normalising the six-axis input is what keeps a
		# diagonal from being faster than a straight line.
		#
		# For an external controller it is the length of the command, and that
		# is worth the branch. Without it the action-to-force map is a step
		# function: any command at all, however small, fires the engines at
		# full power along its direction. A policy trained through that step
		# learns to depend on its own exploration noise to cross the threshold,
		# and then flies completely differently the moment you stop sampling
		# and take the distribution mean — which is exactly what you do at play
		# time. Measured: 39.7% landings before fine-tuning, 3.0% after, while
		# the training reward went *up* the whole time.
		var authority: float = minf(movement.length(), 1.0) if external_control else 1.0
		# Burn first, then thrust with what was actually available. Charging for
		# the whole tick and firing anyway would let a policy spend its last
		# gram twice.
		if has_fuel_limit:
			var wanted: float = throttle * authority * delta
			var burned: float = minf(fuel, wanted)
			fuel -= burned
			is_out_of_fuel = fuel <= 0.0
			authority *= (burned / wanted) if wanted > 0.0 else 0.0
		if authority > 0.0:
			apply_central_force(
				_thrust * mass * movement.normalized() * throttle * authority
			)

	_process_flight_assist(dominant, movement)
	if not headless_agent or agent_visuals:
		_process_thrusters(movement)

	_process_auto_orientation(delta)
	_process_camera_recentre(delta)
	_process_ramp(delta)

	_mouse_delta = Vector2.ZERO


func _unhandled_input(event: InputEvent) -> void:
	if not is_piloted:
		return
	if event.is_action_pressed("interact"):
		get_viewport().set_input_as_handled()
		disembark()
	elif event.is_action_pressed("toggle_ramp"):
		get_viewport().set_input_as_handled()
		toggle_ramp()
	elif event.is_action_pressed("toggle_light"):
		get_viewport().set_input_as_handled()
		toggle_headlight()
	elif event.is_action_pressed("toggle_assist"):
		get_viewport().set_input_as_handled()
		toggle_flight_assist()
	elif event.is_action_pressed("toggle_autopilot"):
		get_viewport().set_input_as_handled()
		if autopilot != null:
			autopilot.toggle()


# --- Exhaust and engine note -------------------------------------------------

## Main exhaust, built in code so the scene stays free of tuning state.
##
## Only the main drive is modelled. RCS jets for each of the six thrust
## directions would be more honest, but the rear plume is the one you actually
## look at, and it carries almost all of the feedback.
## Give the astronaut a deck to stand on that is not this rigid body.
##
## A CharacterBody3D is kinematic, which the solver reads as infinite mass:
## whatever rigid thing it leans on gets moved, and it feels no reaction. While
## the hull was part of this body, an astronaut braced against a planet could
## walk into a parked ship and drive it across the landscape -- measured at 3.5
## units/s and 15.9 units of ground covered in fifteen seconds of walking, from
## a standing start, against 200 kg of spacecraft.
##
## Inside the cabin this never showed, which is what made it easy to miss: in
## there the astronaut is riding the ship, so it is pushing the very floor it
## is braced against and the two cancel. It needs the planet under its feet.
##
## The astronaut's collision mask now excludes the hull entirely, and this hangs
## a kinematic twin of the hull's shapes off it to stand on instead. A kinematic
## body cannot be pushed, so the ship stays where it was parked and the
## astronaut still cannot walk through it. The twin reuses the same Shape3D
## resources rather than copying them, so it cannot drift out of step with the
## hull it mirrors.
##
## What is being thrown away is the astronaut's ability to move the ship at all,
## which is the right trade: the engine offers infinite mass or nothing, and of
## those two, a ship whose landing gear is planted is much the closer to right.
func _build_deck() -> void:
	var deck := AnimatableBody3D.new()
	deck.name = "Deck"
	# Carried by this node's transform as its parent, not animated in its own
	# right, and the astronaut rides the cabin by reparenting rather than through
	# Godot's moving-platform handling. Neither path needs the sync.
	deck.sync_to_physics = false
	deck.collision_layer = LAYER_DECK
	deck.collision_mask = 0
	for child in get_children():
		var shape := child as CollisionShape3D
		if shape == null or shape.disabled:
			continue
		var twin := CollisionShape3D.new()
		twin.shape = shape.shape
		twin.transform = shape.transform
		deck.add_child(twin)
	add_child(deck)


func _build_thrusters() -> void:
	var process := ParticleProcessMaterial.new()
	process.direction = Vector3(0.0, 0.0, 1.0)
	process.spread = 7.0
	process.initial_velocity_min = 14.0
	process.initial_velocity_max = 24.0
	process.gravity = Vector3.ZERO
	process.scale_min = 0.3
	process.scale_max = 0.7
	process.damping_min = 8.0
	process.damping_max = 14.0

	var ramp := Gradient.new()
	ramp.set_color(0, Color(0.85, 0.93, 1.0, 1.0))
	ramp.set_color(1, Color(0.18, 0.34, 0.95, 0.0))
	var ramp_texture := GradientTexture1D.new()
	ramp_texture.gradient = ramp
	process.color_ramp = ramp_texture

	var flame := SphereMesh.new()
	flame.radius = 0.16
	flame.height = 0.32
	flame.radial_segments = 6
	flame.rings = 3

	var flame_material := StandardMaterial3D.new()
	flame_material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	flame_material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	flame_material.blend_mode = BaseMaterial3D.BLEND_MODE_ADD
	flame_material.vertex_color_use_as_albedo = true
	flame_material.billboard_mode = BaseMaterial3D.BILLBOARD_PARTICLES
	flame.material = flame_material

	for offset in [Vector3(-0.9, 0.15, 2.55), Vector3(0.9, 0.15, 2.55)]:
		var jet := GPUParticles3D.new()
		jet.name = "Thruster"
		jet.position = offset
		jet.amount = 64
		jet.lifetime = 0.45
		jet.process_material = process
		jet.draw_pass_1 = flame
		# World coords, so the plume is left behind rather than dragged along.
		jet.local_coords = false
		jet.emitting = false
		add_child(jet)
		_thrusters.append(jet)

	_exhaust_glow = OmniLight3D.new()
	_exhaust_glow.name = "ExhaustGlow"
	_exhaust_glow.position = Vector3(0.0, 0.15, 2.7)
	_exhaust_glow.light_color = Color(0.5, 0.7, 1.0)
	_exhaust_glow.omni_range = 8.0
	_exhaust_glow.light_energy = 0.0
	add_child(_exhaust_glow)


func _build_engine_audio() -> void:
	_engine_audio = EngineAudio.new()
	_engine_audio.name = "EngineAudio"
	_engine_audio.position = Vector3(0.0, 0.2, 2.4)
	_engine_audio.unit_size = 12.0
	_engine_audio.max_distance = 220.0
	_engine_audio._ship = self
	add_child(_engine_audio)


## Drive the plume from the forward component of thrust only, so it does not
## fire out of the back while the pilot is translating sideways.
func _process_thrusters(movement: Vector3) -> void:
	var forward_share := 0.0
	if movement != Vector3.ZERO:
		forward_share = clampf(movement.normalized().dot(-global_basis.z), 0.0, 1.0)

	var burn: float = throttle * forward_share
	for jet in _thrusters:
		jet.emitting = burn > 0.02
		jet.amount_ratio = clampf(burn, 0.05, 1.0)
	if _exhaust_glow != null:
		_exhaust_glow.light_energy = burn * 2.4


# --- Flight assist -----------------------------------------------------------

## Cancel drift the pilot did not ask for, relative to the body they are near.
##
## Deliberately does nothing in deep space: there is no frame to hold station
## against out there, and killing velocity relative to nothing would just make
## the ship feel stuck. Near a planet it holds you where you point.
##
## This does fight orbital mechanics — with assist on you cannot coast in an
## orbit, because the assist reads orbital velocity as drift. That is what the
## toggle is for.
func _process_flight_assist(dominant: Planet, movement: Vector3) -> void:
	if not flight_assist or dominant == null or is_submerged or _is_parked:
		return

	var drift: Vector3 = linear_velocity - dominant.get_surface_velocity(global_position)
	if movement != Vector3.ZERO:
		# Leave the axis being thrust along alone; only trim the rest.
		var wanted := movement.normalized()
		drift -= wanted * drift.dot(wanted)

	apply_central_force(-drift * _assist_strength * mass)


func toggle_flight_assist() -> void:
	flight_assist = not flight_assist


# --- Water ------------------------------------------------------------------

# The ocean has no collision shape, so there is nothing to splash against: the
# ship flies straight in and buoyancy plus damping take over from there.
func _process_buoyancy(dominant: Planet) -> void:
	if dominant == null:
		is_submerged = false
		return

	# Scale buoyancy by how much of the hull is under, rather than switching it
	# on at the waterline. A binary test has no equilibrium: full buoyancy
	# throws the ship clear of the sea, gravity drops it back, and it never
	# settles. Ramped, it comes to rest where buoyancy cancels gravity, which
	# is a float.
	var fraction: float = clampf(dominant.get_depth_at(global_position) / _hull_depth, 0.0, 1.0)
	is_submerged = fraction > 0.0
	if not is_submerged:
		return

	# closest_force already points down toward the planet, so negating it gives
	# the upward push. No mass factor — it is already a force.
	apply_central_force(-closest_force * _buoyancy * fraction)

	# Drag acts on velocity relative to the WATER, which rides the planet's
	# orbit and spin. Godot's linear_damp damps toward world-stationary, so on
	# an orbiting planet it drags the ship sideways at orbital speed — the same
	# double-count the astronaut avoids by being reparented. The ship lives in
	# world space, so it has to subtract the surface velocity by hand.
	#
	# Not ramped by `fraction`: buoyancy needs the ramp to have an equilibrium,
	# but damping is what settles the ship into it.
	var drift: Vector3 = linear_velocity - dominant.get_surface_velocity(global_position)
	apply_central_force(-drift * _water_drag * mass)


# --- Cabin ------------------------------------------------------------------

## True when a world-space point is inside the walkable cabin volume.
func contains_point(world_pos: Vector3) -> bool:
	var local := to_local(world_pos) - _interior_center
	return absf(local.x) <= _interior_extents.x \
		and absf(local.y) <= _interior_extents.y \
		and absf(local.z) <= _interior_extents.z


## World-space velocity of the cabin at a point, matching Planet's signature so
## the astronaut can convert frames without caring which kind of body it rides.
func get_surface_velocity(world_pos: Vector3) -> Vector3:
	return linear_velocity + angular_velocity.cross(world_pos - global_position)


#--- Turbo ---------------------

func toggle_cabin_light_switch() -> void:
	is_cabin_switch = not is_cabin_switch
	_cabin_light.visible = is_cabin_switch


## Forward headlight, bound to the toggle_light action. The cabin light is an
## Interactable because you walk to the switch; this one is a cockpit control.
func toggle_headlight() -> void:
	is_headlight_on = not is_headlight_on
	_headlight.visible = is_headlight_on


## Push the light flags out to the nodes. Used after a load, where the flags are
## restored directly rather than through the toggles.
func apply_light_state() -> void:
	_headlight.visible = is_headlight_on
	_cabin_light.visible = is_cabin_switch

# --- Ramp -------------------------------------------------------------------

func toggle_ramp() -> void:
	is_ramp_open = not is_ramp_open

func _process_ramp(delta: float) -> void:
	# A ramp lowered to a fixed angle only meets the ground on a flat world. These
	# planets are small enough that the surface curves away noticeably over the
	# ramp's own length, so it keeps lowering until it finds ground and rests
	# there, exactly like a real cargo ramp.
	if is_ramp_open and _ramp_degrees > _ramp_closed_degrees and _ramp_lip_on_ground():
		return

	var target := _ramp_open_degrees if is_ramp_open else _ramp_closed_degrees
	if is_equal_approx(_ramp_degrees, target):
		return
	_ramp_degrees = move_toward(_ramp_degrees, target, _ramp_speed * delta)
	_ramp.rotation.x = deg_to_rad(_ramp_degrees)


func _ramp_lip_on_ground() -> bool:
	var lip: Vector3 = _ramp.global_transform * Vector3(0.0, 0.0, _ramp_length)
	var up: Vector3 = global_basis.y
	var query := PhysicsRayQueryParameters3D.create(lip + up * 0.1, lip - up * 0.15)
	# Ground, and only ground. The deck twin sits exactly where the hull is, so
	# without this the ramp would find itself and never lower.
	query.collision_mask = LAYER_WORLD
	query.exclude = [get_rid(), _ramp.get_rid()]
	return not get_world_3d().direct_space_state.intersect_ray(query).is_empty()


# --- Boarding ---------------------------------------------------------------

func board() -> void:
	_astronaut.set_active(false)
	set_piloted(true)


## Leave the pilot seat and stand up in the cabin. Allowed in flight as well as
## on the ground: the cabin has its own gravity, so it is safe to walk around.
func disembark() -> void:
	set_piloted(false)
	_astronaut.place_at(_exit_point.global_transform)
	_astronaut.set_active(true)


func set_piloted(piloted: bool) -> void:
	is_piloted = piloted
	_camera.current = piloted
	_cockpit.enabled = not piloted
	if piloted:
		Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)


func _on_cockpit_activated(_who: Node3D) -> void:
	board()


func _on_ramp_control_activated(_who: Node3D) -> void:
	toggle_ramp()



# --- Movement ---------------------------------------------------------------

## Thrust the pilot is asking for, in world space, from whichever pilot is
## flying. Not normalised: callers normalise, and the magnitude is meaningless
## because thrust is all-or-nothing per axis.
func _get_thrust_request() -> Vector3:
	if external_control:
		# Magnitude is preserved here and honoured in _physics_process. The
		# keyboard path cannot express a partial command — a key is down or it
		# is not — but an external controller can, and the difference matters
		# more than it looks. See the note on thrust authority below.
		var request := Vector3(
			clampf(command_thrust.x, -1.0, 1.0),
			clampf(command_thrust.y, -1.0, 1.0),
			clampf(command_thrust.z, -1.0, 1.0),
		)
		if request.length() < 0.02:
			return Vector3.ZERO
		return global_basis * request
	if not is_piloted:
		return Vector3.ZERO
	return _get_movement_input()


func _get_movement_input() -> Vector3:
	var movement := Vector3.ZERO

	var forward: Vector3 = -global_transform.basis.z
	var left: Vector3 = -global_transform.basis.x
	var up: Vector3 = global_transform.basis.y

	if Input.is_action_pressed("move_forward"):
		movement += forward
	if Input.is_action_pressed("move_backward"):
		movement -= forward
	if Input.is_action_pressed("move_left"):
		movement += left
	if Input.is_action_pressed("move_right"):
		movement -= left
	if Input.is_action_pressed("move_up"):
		movement += up
	if Input.is_action_pressed("move_down"):
		movement -= up

	return movement


# A parked ship is pinned to surface-local coordinates so it rides the planet's
# orbit and spin instead of drifting away while the astronaut is outside.
func _process_parking(movement: Vector3) -> void:
	# Never park while afloat. Parking pins the ship to a fixed surface-local
	# point, so buoyancy would keep accumulating velocity against a frozen
	# position and then release it all at once when the ship un-parks.
	var ground: Planet = get_ground()
	var should_park: bool = (movement == Vector3.ZERO) \
		and (ground != null) \
		and (not is_submerged) \
		and (ground.get_relative_velocity_to_surface(global_position, linear_velocity).length() < _landing_speed)

	if should_park:
		if _is_parked:
			global_position = ground.to_global(_surface_position)
		else:
			_surface_position = ground.to_local(global_position)
			_is_parked = true
	else:
		_is_parked = false


# Mouse movement is banked as a pending turn rather than applied straight to the
# transform, and the ship works through that backlog at _turn_response. The
# result lags the mouse going in and coasts on coming out, which is what gives
# the rotation any sense of mass. Roll is unchanged: hold `rotate` and steer.
func _process_look_inputs(delta: float) -> void:
	if external_control:
		# A policy outputs a rate, not a mouse delta, so it goes straight to the
		# transform. The pending-turn backlog exists to give a mouse flick some
		# mass; a controller that already emits smooth rates does not need it.
		var stick := Vector3(
			clampf(command_turn.x, -1.0, 1.0),
			clampf(command_turn.y, -1.0, 1.0),
			clampf(command_turn.z, -1.0, 1.0),
		) * _stick_rate * delta
		if not stick.is_zero_approx():
			_since_steer = 0.0
		rotate_object_local(Vector3.RIGHT, deg_to_rad(stick.x))
		rotate_object_local(Vector3.UP, deg_to_rad(stick.y))
		rotate_object_local(Vector3.BACK, deg_to_rad(stick.z))
		return

	var dx: float = _mouse_delta.y * _mouse_sensitivity
	var dy: float = -_mouse_delta.x * _mouse_sensitivity

	# Parked, the mouse looks around instead of flying. A ship sitting on its
	# struts that yaws to follow the mouse is not heavy, it is a turntable --
	# and it was a full 360 degrees of it, because nothing in the turn path
	# cared whether the ship was on the ground. Worse, pivoting the hull grinds
	# it into the terrain, and the ejection that follows is enough to push the
	# ship past the parking speed and set it drifting.
	#
	# The pilot still needs to look around a landing site, so the pan moves to
	# the camera. Un-parking hands the mouse straight back to the hull; see the
	# recentre in _process_camera_recentre.
	if _is_parked:
		_free_look.x = clampf(_free_look.x - dx, -_free_look_pitch, _free_look_pitch)
		_free_look.y = wrapf(_free_look.y + dy, -180.0, 180.0)
		return

	if Input.is_action_pressed("rotate"):
		rotate(_camera_pivot.global_transform.basis.z, deg_to_rad(dy))
		return

	_pending_turn += Vector2(dx, dy)
	if not _mouse_delta.is_zero_approx():
		_since_steer = 0.0

	var caught: Vector2 = _pending_turn * clampf(_turn_response * delta, 0.0, 1.0)
	_pending_turn -= caught

	# Both axes turn the ship. Pitch used to move only the camera, with the hull
	# left to catch up via auto-orientation — which is why aiming felt soft and
	# why it stopped working at all once levelling no longer dragged the nose
	# around. The camera is rigid now and the mouse flies the ship.
	rotate_object_local(Vector3.UP, deg_to_rad(caught.y))
	rotate_object_local(Vector3.RIGHT, deg_to_rad(-caught.x))


# Push the free-look aim onto the camera, and wind it back to straight ahead
# once the ship is flying, so the mouse and the view agree again the moment the
# hull starts answering to it.
func _process_camera_recentre(delta: float) -> void:
	if headless_agent:
		return
	if not _is_parked and _free_look != Vector2.ZERO:
		var step: float = _free_look_recentre * delta
		_free_look.x = move_toward(_free_look.x, 0.0, step)
		_free_look.y = move_toward(_free_look.y, 0.0, step)
	_camera_pivot.rotation = Vector3(
		deg_to_rad(_free_look.x), deg_to_rad(_free_look.y), 0.0
	)


# Keep the wings level without taking the nose off where the pilot pointed it.
#
# This used to rotate the ship's whole orientation toward "belly down" every
# frame, which is a combined pitch and roll correction — so diving at a planet
# meant fighting the assist the entire way in. On the ground it was worse: a
# hard snap to level every frame, leaving almost no authority at all.
func _process_auto_orientation(delta: float) -> void:
	angular_velocity = Vector3.ZERO
	angular_damp = 10.0

	_since_steer += delta

	if closest_force == Vector3.ZERO:
		return

	var up_direction: Vector3 = -closest_force.normalized()
	var ground: Planet = get_ground()

	if ground != null:
		angular_velocity = ground.constant_angular_velocity.project(up_direction)
		angular_damp = 0.0

	# Never level while the pilot is steering, or just after.
	if is_piloted and _since_steer < _level_delay:
		return

	var current_q: Quaternion = global_basis.get_rotation_quaternion()
	var target_q: Quaternion

	if ground != null and linear_velocity.length() < _landing_speed * 4.0:
		# Sitting on the surface: settle flat on the struts.
		target_q = Quaternion(global_basis.y, up_direction) * current_q
	else:
		# Flying: correct roll only. Both vectors below are perpendicular to
		# `forward`, so their rotation axis lies along it — a pure roll that
		# leaves the heading untouched.
		var forward: Vector3 = -global_basis.z
		var desired_up: Vector3 = up_direction - forward * up_direction.dot(forward)
		if desired_up.length_squared() < 0.0001:
			return
		target_q = Quaternion(global_basis.y, desired_up.normalized()) * current_q

	var blend: float = clampf(_auto_orient_speed * delta, 0.0, 1.0)
	# Basis, not global_rotation: assigning Euler angles round-trips the
	# quaternion through a lossy conversion that jitters near the poles.
	global_basis = Basis(current_q.slerp(target_q.normalized(), blend))


func _input(e: InputEvent) -> void:
	if e is InputEventMouseMotion and Input.get_mouse_mode() == Input.MOUSE_MODE_CAPTURED:
		var mm: InputEventMouseMotion = e
		_mouse_delta += mm.relative


# --- Helpers ---------------------------------------------------------------

## Landed on an actual Planet. Deliberately not just `_ground_cast.is_colliding()`:
## the ray can hit the astronaut or the ramp, and the orientation code needs a
## Planet or nothing at all.
func is_grounded() -> bool:
	return get_ground() != null


func get_ground() -> Planet:
	return _ground_cast.get_collider() as Planet


func _on_light_switch_cabin_activated(_who: Node3D) -> void:
	toggle_cabin_light_switch()
