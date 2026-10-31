extends RefCounted
class_name LandingEnv

## Descent and landing. Everything task-specific: where an episode starts, what
## the policy sees, what it is rewarded for, and what ends it. The host owns the
## socket and the clock and nothing else.

## PLACEHOLDER width, until the real observation is designed.
const OBS_SIZE: int = 4

const RUNNING: int = 0
const LANDED: int = 1
const CRASHED: int = 2
const ESCAPED: int = 3
const TIMEOUT: int = 4

# More placeholders. These are task-definition decisions and freeze with it.
const SPAWN_ALT_MIN: float = 0.2 # planet radii
const SPAWN_ALT_MAX: float = 1.0
const TOUCHDOWN_SPEED: float = 3.0 # units/s relative to the ground
const ESCAPE_ALT: float = 4.0 # planet radii

const TASK_NAME := "landing"
const OBS_NAMES: Array[String] = ["altitude", "vertical_speed", "horizontal_speed", "uprightness"]
const OUTCOMES := {
	RUNNING: "running",
	LANDED: "landed",
	CRASHED: "crashed",
	ESCAPED: "escaped",
	TIMEOUT: "timeout",
}
const SUCCESS: int = LANDED

var ships: Array[Ship] = []
var targets: Array[Planet] = []
var _landable: Array[Planet] = []
var _rng: RandomNumberGenerator


func _init(world: SolarSystem, agent_ships: Array[Ship], rng: RandomNumberGenerator) -> void:
	assert(OBS_NAMES.size() == OBS_SIZE, "OBS_NAMES and OBS_SIZE disagree")
	ships = agent_ships
	_rng = rng
	for planet in world.Planets:
		if planet.largest_landmass_center != Vector3.ZERO:
			_landable.append(planet)
	targets.resize(ships.size())


## Start ship `i` on a new episode: a random landable planet, a random point
## above it, at rest relative to the ground below.
func reset(i: int) -> void:
	var ship := ships[i]
	var planet := _landable[_rng.randi_range(0, _landable.size() - 1)]
	targets[i] = planet

	var up := Vector3(_rng.randfn(), _rng.randfn(), _rng.randfn()).normalized()
	var altitude := _rng.randf_range(SPAWN_ALT_MIN, SPAWN_ALT_MAX) * planet.surface_radius
	var position := planet.global_position + up * (planet.surface_radius + altitude)
	var forward := up.cross(Vector3.UP)
	if forward.length_squared() < 1e-6:
		forward = up.cross(Vector3.RIGHT)

	ship.global_transform = Transform3D(Basis.looking_at(forward.normalized(), up), position)
	ship.linear_velocity = planet.get_surface_velocity(position)
	ship.angular_velocity = Vector3.ZERO
	ship.throttle = 0.0
	ship.command_thrust = Vector3.ZERO
	ship.command_turn = Vector3.ZERO
	ship.refuel()


## Score ship `i` for one physics tick. Returns [reward, outcome]; RUNNING means
## the episode goes on.
func evaluate(i: int) -> Array:
	var ship := ships[i]
	var planet := targets[i]
	var frame := Surface.surface_frame(ship.global_position, planet)
	var altitude: float = frame[0]
	if altitude > ESCAPE_ALT * planet.surface_radius:
		return [0.0, ESCAPED]
	if altitude > 0.0:
		return [0.0, RUNNING]
	var relative := ship.linear_velocity - planet.get_surface_velocity(ship.global_position)
	return [0.0, LANDED if relative.length() < TOUCHDOWN_SPEED else CRASHED]


func observe(i: int, into: PackedFloat32Array, at: int) -> void:
	build_observation(ships[i], targets[i], 0.0, into, at)


## The one definition of what the policy sees. Static because the in-game
## autopilot builds it too, and the two must never disagree.
static func build_observation(
	ship: Ship, planet: Planet, _progress: float, into: PackedFloat32Array, at: int
) -> void:
	var frame := Surface.surface_frame(ship.global_position, planet)
	var up: Vector3 = frame[1]
	var relative := ship.linear_velocity - planet.get_surface_velocity(ship.global_position)
	var v_up := relative.dot(up)
	var v_scale := Surface.speed_scale(planet)
	into[at + 0] = float(frame[0]) / planet.surface_radius
	into[at + 1] = v_up / v_scale
	into[at + 2] = (relative - up * v_up).length() / v_scale
	into[at + 3] = ship.global_basis.y.dot(up)
