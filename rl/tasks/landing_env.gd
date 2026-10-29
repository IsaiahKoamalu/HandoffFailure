extends RefCounted
class_name LandingEnv

## Descent and landing. (everything task specific)
## where an episode starts, what the policy sees,
## what it is rewarded for, and what ends it.
## Host owns the socket and the clock and nothing else.


## PLACEHOLDER width
const OBS_SIZE: int = 4

const RUNNING: int = 0
const LANDED: int = 1
const CRASHED: int = 2
const ESCAPED: int = 3
const TIMEOUT: int = 4

# More placeholders (task definition decisions and freeze with it)
const SPAWN_ALT_MIN: float = 0.2 # planet radii
const SPAWN_ALT_MAX: float = 1.0
const TOUCHDOWN_SPEED: float = 3.0 # unit/s relative to the ground
const ESCAPE_ALT: float = 4.0 # planet radii

var ships: Array[Ship] = []
var targets: Array[Planet] = []
var _landable: Array[Planet] = []
var _rng: RandomNumberGenerator

func _init(world: SolarSystem, agent_ships: Array[ship], rng: RandomNumberGenerator) -> void:
    ships = agentic_ships
    _rng = rng
    for planet in world.Planets:
        if planet.largest_landmass_center != Vector3.ZERO:
            _landable.append(planet)
    targets.resize(ships.size())

func reset(i: int) -> void:
    var ship := ships[i]
    var planet := _landable[_rng.randi_range(0, _landable.size() - 1)]
    targets[i] = planet

    var up := Vector3(_rng.randfn(), _rng.randfn(), _rng.randfn()).normalized()
    