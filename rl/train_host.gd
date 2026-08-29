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
## Passed through to LandingEnv.flight_assist.
@export var assist: bool = true
## Open a window with a chase camera and a readout instead of running blind.
@export var render: bool = false
## Physics ticks per simulated second when rendering. The step stays 1/60 of a
## simulated second, so this is a playback speed and not a physics change.
@export var render_rate: int = 180

var _env: LandingEnv
var _spectator: TrainSpectator = null
var _server := TCPServer.new()
var _peer: StreamPeerTCP = null
var _rx := PackedByteArray()

var _obs := PackedFloat32Array()
var _act := PackedFloat32Array()
var _first_tick := true


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

	var landable: Array[Planet] = []
	for planet in SolarSystem.Instance.Planets:
		if planet.terrain_enabled:
			landable.append(planet)
	if landable.is_empty():
		push_error("No planet in main.tscn has terrain_enabled; nothing to land on.")
		get_tree().quit(1)
		return

	_env = LandingEnv.new()
	_env.name = "Env"
	_env.num_agents = num_agents
	_env.flight_assist = assist
	_env.visible_agents = render
	add_child(_env)
	_env.setup(load(SHIP_SCENE), landable, seed_value)

	if render:
		# Faster than real time, but still a fixed step. The trainer is on the
		# other end of a blocking socket, so this is a ceiling rather than a
		# target: the loop still runs at whatever the slower half manages.
		Engine.physics_ticks_per_second = render_rate
		_spectator = TrainSpectator.new(_env)
		add_child(_spectator)

	_obs.resize(num_agents * LandingEnv.OBS_SIZE)
	_act.resize(num_agents * LandingEnv.ACT_SIZE)

	var error := _server.listen(port, "127.0.0.1")
	if error != OK:
		push_error("Could not listen on port %d: %s" % [port, error_string(error)])
		get_tree().quit(1)
		return
	print("[train_host] %d agents on %d planets, listening on 127.0.0.1:%d"
		% [num_agents, landable.size(), port])
	_wait_for_client()


func _parse_arguments() -> void:
	# Everything after a bare `--` belongs to the project, not to the engine.
	for argument in OS.get_cmdline_user_args():
		var pair := argument.split("=", true, 1)
		if pair.size() != 2:
			continue
		match pair[0]:
			"--agents": num_agents = int(pair[1])
			"--port": port = int(pair[1])
			"--seed": seed_value = int(pair[1])
			"--assist": assist = int(pair[1]) != 0
			"--render": render = int(pair[1]) != 0
			"--rate": render_rate = int(pair[1])


func _wait_for_client() -> void:
	while not _server.is_connection_available():
		OS.delay_msec(10)
	_peer = _server.take_connection()
	# Nagle would coalesce our one-line-per-tick writes into larger, later
	# packets, which in a lock-step protocol is pure added latency.
	_peer.set_no_delay(true)
	print("[train_host] trainer connected")


# --- The loop ----------------------------------------------------------------

func _physics_process(_delta: float) -> void:
	if _peer == null:
		return

	var rewards := PackedFloat32Array()
	var dones := PackedInt32Array()
	var truncs := PackedInt32Array()
	var outcomes := PackedInt32Array()
	var impacts := PackedFloat32Array()
	rewards.resize(num_agents)
	dones.resize(num_agents)
	truncs.resize(num_agents)
	outcomes.resize(num_agents)
	impacts.resize(num_agents)
	# Observations from just before a reset, which the trainer needs to
	# bootstrap the value function across a time-limit cut. Only ended episodes
	# appear here, so it is empty on almost every tick.
	var final_obs := {}

	for i in num_agents:
		var stride: int = i * LandingEnv.OBS_SIZE

		if _first_tick:
			rewards[i] = 0.0
			dones[i] = 0
			truncs[i] = 0
			outcomes[i] = LandingEnv.RUNNING
			impacts[i] = 0.0
			_env.observe(i, _obs, stride)
			continue

		var result: Array = _env.step_result(i, _act, i * LandingEnv.ACT_SIZE)
		rewards[i] = result[0]
		dones[i] = 1 if result[1] else 0
		truncs[i] = 1 if result[2] else 0
		outcomes[i] = _env.last_outcome[i]
		impacts[i] = _env.last_impact[i]

		if result[1] or result[2]:
			if _spectator != null:
				_spectator.note_outcome(_env.last_outcome[i])
			var terminal := PackedFloat32Array()
			terminal.resize(LandingEnv.OBS_SIZE)
			_env.observe(i, terminal, 0)
			final_obs[str(i)] = terminal
			_env.reset_agent(i)

		_env.observe(i, _obs, stride)

	_first_tick = false

	var rows := []
	for i in num_agents:
		rows.append(_obs.slice(i * LandingEnv.OBS_SIZE, (i + 1) * LandingEnv.OBS_SIZE))

	if not _send({
		"obs": rows,
		"rew": rewards,
		"done": dones,
		"trunc": truncs,
		"outcome": outcomes,
		"impact": impacts,
		"final": final_obs,
	}):
		return

	var line := _read_line()
	if line.is_empty():
		_shutdown("trainer disconnected")
		return

	var message = JSON.parse_string(line)
	if typeof(message) != TYPE_DICTIONARY:
		_shutdown("unparseable message from trainer")
		return
	if message.get("cmd", "") == "close":
		_shutdown("trainer asked to close")
		return
	if message.get("cmd", "") == "reset_all":
		_env.reset_all()
		_first_tick = true
		return

	if _spectator != null:
		var hud = message.get("hud", null)
		if typeof(hud) == TYPE_DICTIONARY:
			_spectator.set_stats(hud)

	var actions: Array = message.get("act", [])
	if actions.size() != num_agents:
		_shutdown("expected %d action rows, got %d" % [num_agents, actions.size()])
		return

	for i in num_agents:
		var row: Array = actions[i]
		var stride: int = i * LandingEnv.ACT_SIZE
		for k in LandingEnv.ACT_SIZE:
			_act[stride + k] = clampf(float(row[k]), -1.0, 1.0)
		_env.apply_action(i, _act, stride)


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
