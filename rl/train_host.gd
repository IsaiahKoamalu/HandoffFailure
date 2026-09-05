extends Node3D

## Headless training host: the Godot half of the RL loop.
##
## Loads the *real* solar system from main.tscn, strips out the parts that only
## exist for a human (the player's ship, the astronaut, the HUD), and replaces
## them with a batch of agent-flown ships. Then it speaks a small line-delimited
## JSON protocol over TCP, one exchange per physics tick:
##
##     Godot -> Python  {"obs": [[...] x N], "rew": [...], "done": [...],
##                       "trunc": [...], "outcome": [...],
##                       "final": {"<i>": [...]}}
##     Python -> Godot  {"act": [[...] x N], "hud": {...}}
##                      or {"cmd": "close"}
##
## "hud" is optional and ignored unless --render is on: it is whatever the
## trainer wants shown on screen, and nothing in the loop reads it.
##
## One exchange per physics tick is the default, but `--repeat=N` holds each
## action for N ticks and exchanges once per hold, summing the rewards in
## between. See `repeat` below for why that matters.
##
## The read blocks inside _physics_process, which stalls the whole engine until
## Python answers. That is the point: the two processes are in lock-step, so a
## rollout is exactly reproducible and no transition is ever dropped.
##
## Run it with:
##     Godot_v4.7.2-stable_win64_console.exe --headless --fixed-fps 60 \
##         --path <project> res://rl/train.tscn -- --agents=16 --port=11008
##
## `--fixed-fps` is what stops the engine sleeping to hit real time; without it
## training runs at 60 steps/second no matter how fast the machine is.
##
## Drop --headless and pass `--render=1` to watch it instead. That opens a
## chase camera and a live readout (rl/spectator.gd) and costs some throughput,
## but changes nothing about the simulation: same seed, same trajectories, same
## gradients. Only the rate they arrive at moves.

const MAIN_SCENE := "res://main.tscn"
const SHIP_SCENE := "res://ship.tscn"

## Nodes in main.tscn that exist only for a human player.
const PLAYER_ONLY := ["HUD", "Ship", "Astronaut"]

@export var port: int = 11008
@export var num_agents: int = 16
@export var seed_value: int = 1
## Passed through to ShipTaskEnv.flight_assist.
@export var assist: bool = true
## Open a window with a chase camera and a readout instead of running blind.
@export var render: bool = false
## Physics ticks per simulated second when rendering. The step stays 1/60 of a
## simulated second, so this is a playback speed and not a physics change.
@export var render_rate: int = 180
## Physics ticks each action is held for: one decision every `repeat` ticks.
##
## The ship is simulated at 60 Hz but does not need deciding at 60 Hz -- the
## expert this task was cloned from is a proportional law that would be
## unchanged at 10. What the decision rate does change is the discount horizon.
## At gamma 0.997 the agent sees 1/(1-gamma) = 333 decisions ahead; at one
## decision per tick that is 5.6 seconds against episodes running 18 to 54, so
## the +120 landing bonus arrives discounted to 120 * 0.997^2000 = 0.3 reward.
## Shaping is worth ~39 over the same episode and is paid densely and early, so
## it survives discounting nearly intact. The agent was not weighing 39 against
## 120; it was weighing 39 against 0.3, and it could not see the landing at all.
##
## Holding an action for 6 ticks puts a whole episode inside the horizon, and
## cuts the round trips per episode by the same factor.
##
## Left at 1 by default: the shipped policy was cloned at one decision per tick
## and evaluating it at another rate would not be measuring the same thing.
@export var repeat: int = 1
## Physics ticks before an episode is truncated. 0 leaves the task's own
## default alone. Exposed because "how often does this controller land?" turns
## out to depend on it: every controller measured so far, the hand-written one
## included, fails 6.5% of episodes by timeout and almost nothing else.
@export var max_steps: int = 0
## Seconds of full thrust per episode, passed to the task. Zero is unlimited.
@export var fuel_seconds: float = 0.0

## Which task to serve. The host owns the socket and the clock and nothing
## else; the task owns the observation, the reward and what ends an episode.
@export var task: String = "landing"

var _env: ShipTaskEnv
# Cached off the task once, because they are read several times per agent per
# tick and a task's widths cannot change while it is running.
var _obs_size: int = 0
var _act_size: int = 0
var _spectator: TrainSpectator = null
var _server := TCPServer.new()
var _peer: StreamPeerTCP = null
var _rx := PackedByteArray()

var _obs := PackedFloat32Array()
var _act := PackedFloat32Array()
var _first_tick := true

# --- Held-action state -------------------------------------------------------
# Physics ticks left before the next exchange, and the batch being summed up
# over them. An agent that ends its episode inside a hold stops accumulating
# and coasts on a zero action until the next decision boundary, so its reward
# never mixes two episodes together.
var _hold: int = 0
var _stepping: Array[bool] = []
var _acc_reward := PackedFloat32Array()
var _acc_done := PackedInt32Array()
var _acc_trunc := PackedInt32Array()
var _acc_outcome := PackedInt32Array()
var _acc_impact := PackedFloat32Array()
var _acc_final := {}


func _ready() -> void:
	_parse_arguments()

	# Run after the solar system has advanced its orbits and after every ship
	# has applied the action it was given, so what this node observes is the
	# state the engine actually just simulated.
	process_physics_priority = 100

	var world: Node = load(MAIN_SCENE).instantiate()
	add_child(world)
	# Remove rather than queue_free: queued frees happen at the end of the
	# frame, and these nodes would get one _process call in between. The HUD
	# reads the ship it is about to lose.
	for node_name in PLAYER_ONLY:
		var node: Node = world.get_node_or_null(node_name)
		if node != null:
			world.remove_child(node)
			node.queue_free()

	_env = _make_env(task)
	if _env == null:
		push_error("Unknown task '%s'." % task)
		get_tree().quit(1)
		return
	_env.name = "Env"
	_env.num_agents = num_agents
	_env.flight_assist = assist
	_env.visible_agents = render
	_env.fuel_seconds = fuel_seconds
	# An explicit --max-steps wins; otherwise the task says how long its own
	# episodes should be.
	_env.max_steps = max_steps if max_steps > 0 else _env.default_max_steps()
	add_child(_env)
	if not _env.setup(load(SHIP_SCENE), seed_value):
		# The task has already said why. It could not bind to the world it
		# needs, so there is nothing to serve.
		get_tree().quit(1)
		return
	_obs_size = _env.obs_size()
	_act_size = _env.act_size()

	if render and _env is LandingEnv:
		# Faster than real time, but still a fixed step. The trainer is on the
		# other end of a blocking socket, so this is a ceiling rather than a
		# target: the loop still runs at whatever the slower half manages.
		Engine.physics_ticks_per_second = render_rate
		_spectator = TrainSpectator.new(_env)
		add_child(_spectator)
	elif render:
		# The overlay reads landing geometry — altitude, coastline, continent
		# bearing — so it is a landing visualiser, not a generic one. Another
		# task renders as a plain window until it has its own.
		Engine.physics_ticks_per_second = render_rate

	_obs.resize(num_agents * _obs_size)
	_act.resize(num_agents * _act_size)
	_acc_reward.resize(num_agents)
	_acc_done.resize(num_agents)
	_acc_trunc.resize(num_agents)
	_acc_outcome.resize(num_agents)
	_acc_impact.resize(num_agents)
	_stepping.resize(num_agents)
	_begin_hold()

	var error := _server.listen(port, "127.0.0.1")
	if error != OK:
		push_error("Could not listen on port %d: %s" % [port, error_string(error)])
		get_tree().quit(1)
		return
	print("[train_host] task '%s', %d agents, obs %d, listening on 127.0.0.1:%d"
		% [_env.task_name(), num_agents, _obs_size, port])
	_wait_for_client()


## The task registry. One line per task; everything else in this file is
## task-agnostic.
func _make_env(name: String) -> ShipTaskEnv:
	match name:
		"landing": return LandingEnv.new()
		"docking": return DockingEnv.new()
	return null


func _parse_arguments() -> void:
	# Everything after a bare `--` belongs to the project, not to the engine.
	for argument in OS.get_cmdline_user_args():
		var pair := argument.split("=", true, 1)
		if pair.size() != 2:
			continue
		match pair[0]:
			"--agents": num_agents = int(pair[1])
			"--task": task = pair[1]
			"--port": port = int(pair[1])
			"--seed": seed_value = int(pair[1])
			"--assist": assist = int(pair[1]) != 0
			"--render": render = int(pair[1]) != 0
			"--rate": render_rate = int(pair[1])
			"--repeat": repeat = maxi(1, int(pair[1]))
			"--max-steps": max_steps = int(pair[1])
			"--fuel": fuel_seconds = float(pair[1])


func _wait_for_client() -> void:
	while not _server.is_connection_available():
		OS.delay_msec(10)
	_peer = _server.take_connection()
	# Nagle would coalesce our one-line-per-tick writes into larger, later
	# packets, which in a lock-step protocol is pure added latency.
	_peer.set_no_delay(true)
	# Describe the task before any data flows. The trainer used to carry its own
	# copy of the observation width and the outcome names, which is fine while
	# there is one task and silently wrong the moment there are two: a policy
	# would be allocated at the wrong shape and the logs would label outcomes
	# from a task nobody was running.
	var outcomes := {}
	for id in _env.outcome_names():
		outcomes[str(id)] = _env.outcome_names()[id]
	_send({
		"type": "hello",
		"task": _env.task_name(),
		"obs_size": _obs_size,
		"act_size": _act_size,
		"num_agents": num_agents,
		"repeat": repeat,
		"max_steps": _env.max_steps,
		"outcomes": outcomes,
	})
	print("[train_host] trainer connected")


# --- The loop ----------------------------------------------------------------

func _physics_process(_delta: float) -> void:
	if _peer == null:
		return

	if not _first_tick:
		_accumulate_tick()
		_hold -= 1
		if _hold > 0:
			# Mid-hold: keep driving with the action already given. Frozen
			# agents have had their row zeroed, so they coast.
			for i in num_agents:
				_env.apply_action(i, _act, i * _act_size)
			return

	if not _exchange():
		return
	_begin_hold()
	_first_tick = false


# Simulate one tick of the current hold into the accumulators.
func _accumulate_tick() -> void:
	for i in num_agents:
		if not _stepping[i]:
			continue

		var result: Array = _env.step_result(i, _act, i * _act_size)
		_acc_reward[i] += result[0]
		_acc_outcome[i] = _env.last_outcome[i]
		_acc_impact[i] = _env.last_impact[i]

		if not (result[1] or result[2]):
			continue

		_acc_done[i] = 1 if result[1] else 0
		_acc_trunc[i] = 1 if result[2] else 0
		if _spectator != null:
			_spectator.note_outcome(_env.last_outcome[i])
		var terminal := PackedFloat32Array()
		terminal.resize(_obs_size)
		_env.observe(i, terminal, 0)
		_acc_final[str(i)] = terminal
		_env.reset_agent(i)

		# Done for this hold. The new episode coasts to the next decision
		# boundary rather than being flown by the action that ended the last one.
		_stepping[i] = false
		var stride: int = i * _act_size
		for k in _act_size:
			_act[stride + k] = 0.0


# Send the batch, read the reply, and hand the new actions to the ships.
# Returns false if the exchange ended the run.
func _exchange() -> bool:
	for i in num_agents:
		_env.observe(i, _obs, i * _obs_size)

	var rows := []
	for i in num_agents:
		rows.append(_obs.slice(i * _obs_size, (i + 1) * _obs_size))

	if not _send({
		"obs": rows,
		"rew": _acc_reward,
		"done": _acc_done,
		"trunc": _acc_trunc,
		"outcome": _acc_outcome,
		"impact": _acc_impact,
		"final": _acc_final,
	}):
		return false

	var line := _read_line()
	if line.is_empty():
		_shutdown("trainer disconnected")
		return false

	var message = JSON.parse_string(line)
	if typeof(message) != TYPE_DICTIONARY:
		_shutdown("unparseable message from trainer")
		return false
	if message.get("cmd", "") == "close":
		_shutdown("trainer asked to close")
		return false
	if message.get("cmd", "") == "reset_all":
		_env.reset_all()
		_first_tick = true
		return false

	if _spectator != null:
		var hud = message.get("hud", null)
		if typeof(hud) == TYPE_DICTIONARY:
			_spectator.set_stats(hud)

	var actions: Array = message.get("act", [])
	if actions.size() != num_agents:
		_shutdown("expected %d action rows, got %d" % [num_agents, actions.size()])
		return false

	for i in num_agents:
		var row: Array = actions[i]
		var stride: int = i * _act_size
		for k in _act_size:
			_act[stride + k] = clampf(float(row[k]), -1.0, 1.0)
		_env.apply_action(i, _act, stride)
	return true


# Open a fresh hold: clear the accumulators and put every agent back in play.
func _begin_hold() -> void:
	for i in num_agents:
		# An agent that sat out the tail of the last hold has drifted since its
		# reward was last read. Re-take its shaping baseline so that drift is
		# not charged to the action about to be taken.
		if not _stepping[i] and not _first_tick:
			_env.rebaseline(i)
		_stepping[i] = true
		_acc_reward[i] = 0.0
		_acc_done[i] = 0
		_acc_trunc[i] = 0
		_acc_outcome[i] = ShipTaskEnv.RUNNING
		_acc_impact[i] = 0.0
	_acc_final = {}
	_hold = repeat


# --- Transport ---------------------------------------------------------------

func _send(payload: Dictionary) -> bool:
	var bytes := (JSON.stringify(payload) + "\n").to_utf8_buffer()
	if _peer.put_data(bytes) != OK:
		_shutdown("write failed")
		return false
	return true


## Block until a whole line has arrived. Returns "" if the peer went away.
##
## The leftover bytes are kept in _rx: a single read can straddle a newline, and
## dropping the tail would silently desynchronise the protocol.
func _read_line() -> String:
	while true:
		var newline := _rx.find(10)
		if newline >= 0:
			var line := _rx.slice(0, newline).get_string_from_utf8()
			_rx = _rx.slice(newline + 1)
			return line

		_peer.poll()
		if _peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
			return ""
		var available := _peer.get_available_bytes()
		if available <= 0:
			# Sleep rather than spin: the trainer's forward pass is milliseconds
			# and a busy loop would fight it for a core.
			OS.delay_usec(50)
			continue
		var chunk: Array = _peer.get_data(available)
		if chunk[0] != OK:
			return ""
		_rx.append_array(chunk[1])
	return ""


func _shutdown(reason: String) -> void:
	print("[train_host] closing: %s" % reason)
	_peer = null
	get_tree().quit()
