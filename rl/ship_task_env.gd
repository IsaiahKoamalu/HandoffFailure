extends Node3D
class_name ShipTaskEnv

## Base for vectorised control tasks flown by the real ship.
##
## Everything here is what the tasks share: a pool of agent ships wired for
## external control, the episode clock, the shaping baseline, and the action
## seam. What a task adds is a *reference target* and a *terminal condition* —
## a planet surface for landing, another vessel for docking, an orbit for a
## transfer. Subclasses supply those and nothing else.
##
## The agent flies `Ship` through `Ship.command_thrust` / `command_turn` — the
## same seam a human's mouse and keyboard feed. Nothing about the dynamics is
## reimplemented: gravity, drag, buoyancy, engine spool and auto-levelling are
## all the shipping code. That is deliberate. A policy trained against a
## simplified copy of the ship would be a policy that cannot fly the real one.
##
## Many ships share one solar system and one clock. They are independent bodies
## under analytic gravity, so this costs nothing in fidelity and buys a whole
## vectorised batch per engine tick.
##
## Subclasses must override: obs_size, task_name, outcome_names, reset_agent,
## observe, _compute_potential and step_result. _bind_world is optional.

## Three translation axes and three rotation axes — exactly the control set the
## player has. Shared by every task, because every task flies the same ship.
const ACT_SIZE: int = 6

## The one outcome every task has: the episode is still going. Task-specific
## outcomes number upward from 1 and are named by outcome_names().
const RUNNING: int = 0

## Ships stepped in parallel. Each has its own episode and its own target.
@export var num_agents: int = 16
## Leave the ship's own drift assist enabled. With it off the ship's
## thrust-to-weight ratio makes a powered descent impossible, so this is really
## a switch for experiments, not a tuning knob.
@export var flight_assist: bool = true
## Give each agent its exhaust plume. Costs nothing when nobody is watching
## and is off by default; see rl/spectator.gd.
@export var visible_agents: bool = false
## Physics ticks before an episode is cut short. At 60 Hz this is 450 seconds.
##
## This is part of the metric, not part of the scenery. Measured on the
## hand-written landing controller, tripling it from 9,000 took landings from
## 92.5% to 99.0% and timeouts from 6.5% to zero -- every one of those episodes
## converted to a landing, and not one to a crash. Three separate PPO runs had
## plateaued at "92%", which looked like a control ceiling and was in fact the
## clock. Any new task should size this against its own worst case and check
## that its failures are not dominated by truncation.
@export var max_steps: int = 27000
## Seconds of full thrust each agent starts an episode with. Zero is unlimited.
##
## No reward term goes with this, deliberately. Fuel is a constraint, not a
## preference: running dry is punished already, by whatever failure follows.
## Paying a bonus for leftover fuel would invent a second objective and make it
## impossible to say afterwards whether the policy learned to fly economically
## or merely learned to hoard.
@export var fuel_seconds: float = 0.0

var ships: Array[Ship] = []

## Last episode outcome per agent, for the trainer's logs.
var last_outcome: PackedInt32Array = PackedInt32Array()
## Task-defined scalar recorded at the end of an episode — impact speed for a
## landing, closing speed for a dock. Reported alongside the outcome so the
## logs can say how badly, not just whether.
var last_impact: PackedFloat32Array = PackedFloat32Array()

# Both are read and written by subclasses: the episode clock and the shaping
# baseline are bookkeeping the base owns but only the task can interpret.
var _steps: PackedInt32Array = PackedInt32Array()
var _potential: PackedFloat32Array = PackedFloat32Array()
var _rng := RandomNumberGenerator.new()


# --- Description, for the handshake ------------------------------------------

## Width of this task's observation vector. Sent to the trainer on connect, so
## a policy is never allocated at the wrong shape for the task it is flying.
func obs_size() -> int:
	push_error("ShipTaskEnv.obs_size() not overridden")
	return 0


## The clock this task wants when the trainer does not name one. A landing
## crossing half a planet and a docking approach from a hundred metres are not
## the same length of episode, and a default sized for one is part of the
## metric for the other -- see the note on max_steps above.
func default_max_steps() -> int:
	return max_steps


## Width of the action vector. The same for every control task.
func act_size() -> int:
	return ACT_SIZE


## Short identifier recorded in policy.json, so a policy trained for one task
## cannot be loaded silently against another.
func task_name() -> String:
	push_error("ShipTaskEnv.task_name() not overridden")
	return "unknown"


## Outcome id -> name, for the trainer's logs and the spectator overlay. The
## trainer takes its labels from this rather than from a mirrored copy, which
## is how the two used to drift.
func outcome_names() -> Dictionary:
	push_error("ShipTaskEnv.outcome_names() not overridden")
	return {}


## Which outcome counts as the task being done well -- the number a run is
## judged on and the one "best policy so far" is selected by.
##
## The task declares it because only the task knows. The trainer used to assume
## "landed", which is fine while there is one task and a crash the moment there
## are two: the first docking run died in the logging with KeyError: 'landed'
## after a clean warm start, having thrown the rollout away.
##
## Defaulted rather than abstract: every outcome enum here numbers success 1
## upward from RUNNING, so a task that does not care gets the right answer.
func success_outcome() -> int:
	return 1


# --- Setup -------------------------------------------------------------------

## Build the agent pool and start every episode. False means the task could not
## bind to the world it needs and the run should not start.
func setup(ship_scene: PackedScene, seed_value: int) -> bool:
	_rng.seed = seed_value
	_steps.resize(num_agents)
	_potential.resize(num_agents)
	last_outcome.resize(num_agents)
	last_impact.resize(num_agents)

	if not _bind_world():
		return false

	for i in num_agents:
		var ship: Ship = ship_scene.instantiate()
		# All three are read by Ship._ready, so they have to be set before the
		# ship enters the tree.
		ship.fuel_seconds = fuel_seconds
		ship.headless_agent = true
		ship.agent_visuals = visible_agents
		ship.external_control = true
		ship.name = "Agent%d" % i
		add_child(ship)
		# Agents fly through each other. They are separate rollouts that happen
		# to share a world; a collision between two of them is not a real event.
		#
		# A task where contact with the target *is* the event — docking — cannot
		# simply reuse this: it needs its target on a layer the agent still sees.
		ship.collision_layer = 1 << 9
		ship.collision_mask = ship.collision_mask & ~(1 << 9)
		var ramp := ship.get_node_or_null("Ramp") as PhysicsBody3D
		if ramp != null:
			ramp.collision_layer = 1 << 9
			ramp.collision_mask = ramp.collision_mask & ~(1 << 9)
		ship.flight_assist = flight_assist
		ships.append(ship)
		reset_agent(i)
	return true


## Find whatever the task needs from the loaded scene, and size any per-agent
## arrays the task owns. Called before the ships exist, because reset_agent runs
## as each one is created. False aborts the run.
func _bind_world() -> bool:
	return true


# --- Episode control ---------------------------------------------------------

## Place agent `i` at a fresh start state and take its shaping baseline.
func reset_agent(_i: int) -> void:
	push_error("ShipTaskEnv.reset_agent() not overridden")


func reset_all() -> void:
	for i in num_agents:
		reset_agent(i)


## Re-take the shaping baseline without paying out for the change.
##
## An agent that ends its episode part way through a held action coasts to the
## next decision boundary before its new episode starts being scored. That coast
## still moves the ship, and without this the movement would be charged to the
## first action of the new episode, which did not cause it.
func rebaseline(i: int) -> void:
	_potential[i] = _compute_potential(i)


## How far through its clock agent `i` is, in [0, 1]. This is an observation
## channel in every task: an agent that cannot see the clock cannot tell a
## truncation coming.
func episode_progress(i: int) -> float:
	return float(_steps[i]) / float(max_steps)


# --- Observation, reward, action ---------------------------------------------

## Write agent `i`'s observation into `into` starting at `at`.
func observe(_i: int, _into: PackedFloat32Array, _at: int) -> void:
	push_error("ShipTaskEnv.observe() not overridden")


## Potential-based shaping. The per-step reward is the *change* in this, which
## (Ng, Harada & Russell, 1999) leaves the optimal policy unchanged — shaping of
## this form can only speed learning up, it cannot bend the agent toward a
## different task. A raw "closer is better" bonus has no such guarantee.
func _compute_potential(_i: int) -> float:
	push_error("ShipTaskEnv._compute_potential() not overridden")
	return 0.0


## Read the result of a tick that has already been simulated, and decide whether
## the episode is over. Returns [reward, terminated, truncated].
func step_result(_i: int, _action: PackedFloat32Array, _at: int) -> Array:
	push_error("ShipTaskEnv.step_result() not overridden")
	return [0.0, false, false]


## The control seam, and the reason none of the dynamics live here.
func apply_action(i: int, action: PackedFloat32Array, at: int) -> void:
	var ship: Ship = ships[i]
	ship.command_thrust = Vector3(action[at + 0], action[at + 1], action[at + 2])
	ship.command_turn = Vector3(action[at + 3], action[at + 4], action[at + 5])


# --- Shared geometry ---------------------------------------------------------

## Altitude above the analytic terrain surface, the local up direction, and the
## distance from the planet's centre. Returned together because all three come
## from the same offset and every caller wants at least two of them.
##
## Static, and taking a world position rather than an agent index, because the
## in-game autopilot needs exactly this and must not compute it a second,
## subtly different way. See LandingEnv.build_observation.
static func surface_frame(world_pos: Vector3, planet: Planet) -> Array:
	var offset: Vector3 = world_pos - planet.global_position
	var distance: float = offset.length()
	if distance < 0.001:
		return [0.0, Vector3.UP, 0.001]
	var up: Vector3 = offset / distance
	# get_surface_radius_at wants a direction in the planet's own frame: the
	# noise is sampled in body space, and the planet spins.
	var local_dir: Vector3 = (planet.global_basis.inverse() * up).normalized()
	return [distance - planet.get_surface_radius_at(local_dir), up, distance]


## Speed scale for a planet: the circular orbital speed at its surface. A
## physical scale rather than a hand-picked constant is what keeps observations
## in the same range across planets of very different sizes.
static func speed_scale(planet: Planet) -> float:
	return maxf(sqrt(planet.standard_gravitational_parameter / planet.surface_radius), 1.0)
