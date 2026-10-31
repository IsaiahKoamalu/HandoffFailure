extends Node

## Headless training host. Listens for the trainer and answers one JSON message
## per line, in strict request-reply order. See docs/protocol.md.
##
## While it waits for the next request the engine is blocked, so physics only
## advances when the trainer asks and a run is reproducible from its seed.

const PORT: int = 11008
const MAIN_SCENE := "res://sim/main.tscn"
const SHIP_SCENE := "res://sim/ship.tscn"
## Nodes in main.tscn that exist only for a person playing the game.
const PLAYER_ONLY: Array[String] = ["Ship", "Astronaut", "HUD"]

var ships: Array[Ship] = []

var _world: SolarSystem = null
var _rng := RandomNumberGenerator.new()
var _repeat: int = 1
var _max_steps: int = 27000
var _task: LandingEnv = null

var _server := TCPServer.new()
var _peer: StreamPeerTCP = null
## Bytes received but not yet ending in a newline.
var _buffer := PackedByteArray()

## Physics ticks left in the action being held. Zero means the host is at a
## decision boundary and waiting for the trainer.
var _hold: int = 0
## Per ship: ticks into the current episode.
var _steps := PackedInt32Array()
## Per ship, for the decision in progress. Flags are Array[bool] because
## JSON.stringify writes a PackedByteArray as a string.
var _done: Array[bool] = []
var _reward := PackedFloat32Array()
var _terminated: Array[bool] = []
var _truncated: Array[bool] = []
var _outcome := PackedInt32Array()
## Ship index (as a string) -> the observation its episode ended on.
var _final_obs := {}


func _ready() -> void:
	var err := _server.listen(PORT, "127.0.0.1")
	if err != OK:
		push_error("host: cannot listen on port %d (error %d)" % [PORT, err])
		get_tree().quit(1)
		return
	print("host: listening on %d" % PORT)


func _physics_process(_delta: float) -> void:
	if _peer == null:
		if _server.is_connection_available():
			_peer = _server.take_connection()
			_peer.set_no_delay(true)
			print("host: trainer connected")
		return

	if _hold > 0:
		_tick()
		_hold -= 1
		if _hold > 0:
			return
		_finish_step()

	while _peer != null and _hold == 0:
		var line: Variant = _read_line()
		if line == null:
			print("host: trainer disconnected")
			get_tree().quit()
			return
		_handle(line)


## Blocks until one whole line has arrived. Null when the trainer has gone.
func _read_line() -> Variant:
	while true:
		var newline := _buffer.find(10)
		if newline >= 0:
			var line := _buffer.slice(0, newline).get_string_from_utf8()
			_buffer = _buffer.slice(newline + 1)
			return line
		var available := maxi(_peer.get_available_bytes(), 1)
		var result: Array = _peer.get_data(available)
		if result[0] != OK:
			return null
		_buffer.append_array(result[1])
	return null


func _handle(line: String) -> void:
	var message = JSON.parse_string(line)
	if typeof(message) != TYPE_DICTIONARY:
		_send({"type": "error", "reason": "not a JSON object: %s" % line})
		return

	match message.get("type", ""):
		"hello":
			_hello(message)
		"reset":
			_reset()
		"step":
			_step(message)
		"close":
			_send({"type": "close"})
			_peer = null
			get_tree().quit()
		_:
			_send({"type": "error", "reason": "unknown message type '%s'" % message.get("type", "")})


func _hello(message: Dictionary) -> void:
	if _world != null:
		_send({"type": "error", "reason": "hello sent twice"})
		return
	var count := int(message.get("ships", 0))
	if count < 1:
		_send({"type": "error", "reason": "hello needs ships >= 1, got %d" % count})
		return
	_rng.seed = int(message.get("seed", 0))
	_repeat = maxi(1, int(message.get("repeat", 1)))
	_max_steps = maxi(1, int(message.get("max_steps", _max_steps)))

	_world = load(MAIN_SCENE).instantiate()
	for node_name in PLAYER_ONLY:
		var node := _world.get_node_or_null(node_name)
		if node != null:
			_world.remove_child(node)
			node.free()
	add_child(_world)

	var ship_scene: PackedScene = load(SHIP_SCENE)
	for i in count:
		var ship: Ship = ship_scene.instantiate()
		ship.headless_agent = true
		ship.external_control = true
		ship.name = "Agent%d" % i
		ship.collision_mask &= ~Ship.LAYER_SHIP
		_world.add_child(ship)
		ships.append(ship)

	_task = LandingEnv.new(_world, ships, _rng)
	_steps.resize(count)
	_done.resize(count)
	_reward.resize(count)
	_terminated.resize(count)
	_truncated.resize(count)
	_outcome.resize(count)

	var planet_names: Array[String] = []
	for planet in _world.Planets:
		planet_names.append(planet.name)
	_send({
		"type": "hello",
		"task": LandingEnv.TASK_NAME,
		"ships": ships.size(),
		"repeat": _repeat,
		"max_steps": _max_steps,
		"obs_size": LandingEnv.OBS_SIZE,
		"act_size": NeuralPilot.ACT_SIZE,
		"obs_names": LandingEnv.OBS_NAMES,
		"outcomes": LandingEnv.OUTCOMES,
		"success": LandingEnv.SUCCESS,
		"planets": planet_names,
	})


func _reset() -> void:
	if _task == null:
		_send({"type": "error", "reason": "reset before hello"})
		return
	for i in ships.size():
		_task.reset(i)
		_steps[i] = 0
	var obs := _observe_all()
	var bad := _first_non_finite(obs)
	if bad >= 0:
		_send({"type": "error", "reason": "ship %d has a non-finite observation after reset" % bad})
		return
	_send({"type": "reset", "obs": obs})


func _step(message: Dictionary) -> void:
	if _task == null:
		_send({"type": "error", "reason": "step before hello"})
		return
	var actions = message.get("actions")
	if typeof(actions) != TYPE_ARRAY or actions.size() != ships.size():
		_send({"type": "error", "reason": "step needs %d actions" % ships.size()})
		return
	# Check every action before applying any, so a bad request changes nothing.
	for i in ships.size():
		var a = actions[i]
		if typeof(a) != TYPE_ARRAY or a.size() != NeuralPilot.ACT_SIZE:
			_send({"type": "error", "reason": "action %d is not a list of %d numbers"
				% [i, NeuralPilot.ACT_SIZE]})
			return
		if not _all_finite(a):
			_send({"type": "error", "reason": "action %d is not finite" % i})
			return

	for i in ships.size():
		var a: Array = actions[i]
		ships[i].command_thrust = Vector3(a[0], a[1], a[2]).clampf(-1.0, 1.0)
		ships[i].command_turn = Vector3(a[3], a[4], a[5]).clampf(-1.0, 1.0)
		_done[i] = false
		_reward[i] = 0.0
		_terminated[i] = false
		_truncated[i] = false
		_outcome[i] = LandingEnv.RUNNING
	_final_obs.clear()
	_hold = _repeat


## One physics tick of scoring, for every ship still in its episode.
func _tick() -> void:
	for i in ships.size():
		if _done[i]:
			continue
		_steps[i] += 1
		var result := _task.evaluate(i)
		_reward[i] += result[0]
		var outcome: int = result[1]
		if outcome == LandingEnv.RUNNING and _steps[i] >= _max_steps:
			_truncated[i] = true
			outcome = LandingEnv.TIMEOUT
		elif outcome != LandingEnv.RUNNING:
			_terminated[i] = true
		if outcome == LandingEnv.RUNNING:
			continue

		# The episode ended on this tick: keep its last observation, then let
		# the ship coast with engines off until the decision boundary.
		_outcome[i] = outcome
		_done[i] = true
		var final := PackedFloat32Array()
		final.resize(LandingEnv.OBS_SIZE)
		_task.observe(i, final, 0)
		_final_obs[str(i)] = final
		ships[i].command_thrust = Vector3.ZERO
		ships[i].command_turn = Vector3.ZERO


## The decision boundary: restart finished ships and reply to the step.
func _finish_step() -> void:
	for i in ships.size():
		if _done[i]:
			_task.reset(i)
			_steps[i] = 0

	var obs := _observe_all()
	var bad := _first_non_finite(obs)
	if bad >= 0:
		_send({"type": "error", "reason": "ship %d has a non-finite observation" % bad})
		return
	for i in ships.size():
		if not is_finite(_reward[i]):
			_send({"type": "error", "reason": "ship %d has a non-finite reward" % i})
			return
	for key in _final_obs:
		if not _all_finite(_final_obs[key]):
			_send({"type": "error", "reason": "ship %s has a non-finite final observation" % key})
			return

	_send({
		"type": "step",
		"obs": obs,
		"reward": _reward,
		"terminated": _terminated,
		"truncated": _truncated,
		"outcome": _outcome,
		"final_obs": _final_obs,
	})


func _observe_all() -> Array:
	var all := []
	var row := PackedFloat32Array()
	row.resize(LandingEnv.OBS_SIZE)
	for i in ships.size():
		_task.observe(i, row, 0)
		all.append(row.duplicate())
	return all


func _send(message: Dictionary) -> void:
	_peer.put_data((JSON.stringify(message) + "\n").to_utf8_buffer())


static func _all_finite(values) -> bool:
	for v in values:
		if typeof(v) != TYPE_FLOAT and typeof(v) != TYPE_INT:
			return false
		if not is_finite(v):
			return false
	return true


## Index of the first row holding a NaN or infinity, or -1 if every row is finite.
static func _first_non_finite(rows: Array) -> int:
	for i in rows.size():
		if not _all_finite(rows[i]):
			return i
	return -1