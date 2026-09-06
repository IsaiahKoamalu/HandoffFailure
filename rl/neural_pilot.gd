extends RefCounted
class_name NeuralPilot

## Runs a policy trained by python/ppo.py, in-engine, with no dependencies.
##
## There is no ONNX runtime here and no GDExtension. The policy is a two-hidden-
## layer MLP of 64 units — four small matrix-vector products — so the forward
## pass is a handful of nested loops and a `.json` file loaded as a resource.
## Adding a native inference runtime to save a few microseconds on a network
## this size would be all cost and no benefit.
##
## The weights file is written by ppo.save(). Layout:
##     { "format": 1, "task": "landing", "obs_size": 20, "act_size": 6,
##       "activation": "tanh",
##       "layers": [ {"w": [[...] x out][in], "b": [...]}, ... ],
##       "obs_mean": [...], "obs_std": [...] }
##
## `obs_mean` / `obs_std` are as much a part of the policy as the weights. PPO
## trains on normalised observations, so feeding raw ones here would put the
## network far outside the distribution it ever saw, and it would fly like it.

const DEFAULT_PATH := "res://rl/runs/landing/policy.json"

## Which task these weights were trained for. Checked on load, because a
## policy trained for one task will happily run against another: it produces
## six numbers in range either way, and flies something it has never seen.
var task: String = "landing"
var obs_size: int = 0
## Physics ticks each action was held for in training; the autopilot matches it.
var repeat: int = 1

## Below this, an observation channel counts as constant. Must match
## python/ppo.py.
const CONSTANT_STD: float = 1e-3
var act_size: int = 0
var trained_steps: int = 0

var _weights: Array[Array] = []
var _biases: Array[PackedFloat32Array] = []
var _mean: PackedFloat32Array = PackedFloat32Array()
var _inverse_std: PackedFloat32Array = PackedFloat32Array()
var _scratch_a: PackedFloat32Array = PackedFloat32Array()
var _scratch_b: PackedFloat32Array = PackedFloat32Array()


## Returns null when there is no policy to load, which is the normal state of a
## fresh checkout — nobody should have to train before they can play.
static func load_from(path: String = DEFAULT_PATH) -> NeuralPilot:
	if not FileAccess.file_exists(path):
		return null
	var file := FileAccess.open(path, FileAccess.READ)
	if file == null:
		push_warning("NeuralPilot: cannot open %s" % path)
		return null
	var parsed = JSON.parse_string(file.get_as_text())
	if typeof(parsed) != TYPE_DICTIONARY:
		push_warning("NeuralPilot: %s is not valid JSON" % path)
		return null

	var pilot := NeuralPilot.new()
	if not pilot._adopt(parsed):
		return null
	return pilot


func _adopt(data: Dictionary) -> bool:
	if int(data.get("format", 0)) != 1:
		push_warning("NeuralPilot: unsupported weights format")
		return false

	# Policies predating the multi-task split have no task field, and every one
	# of them was a lander.
	task = str(data.get("task", "landing"))
	obs_size = int(data.get("obs_size", 0))
	act_size = int(data.get("act_size", 0))
	trained_steps = int(data.get("trained_steps", 0))
	# Older policies predate the field and were trained one decision per tick.
	repeat = maxi(1, int(data.get("repeat", 1)))
	if task != "landing":
		push_warning("NeuralPilot: policy was trained for '%s'; this loader flies landings."
			% task)
		return false
	if obs_size != LandingEnv.OBS_SIZE or act_size != LandingEnv.ACT_SIZE:
		# Almost always means the observation was changed without retraining.
		# Better to refuse than to silently feed the network the wrong columns.
		push_warning("NeuralPilot: policy is %d->%d but this build is %d->%d; retrain."
			% [obs_size, act_size, LandingEnv.OBS_SIZE, LandingEnv.ACT_SIZE])
		return false

	for layer in data.get("layers", []):
		var rows: Array[PackedFloat32Array] = []
		for row in layer["w"]:
			rows.append(PackedFloat32Array(row))
		_weights.append(rows)
		_biases.append(PackedFloat32Array(layer["b"]))
	if _weights.is_empty():
		push_warning("NeuralPilot: policy has no layers")
		return false

	_mean = PackedFloat32Array(data.get("obs_mean", []))
	var deviations := PackedFloat32Array(data.get("obs_std", []))
	_inverse_std.resize(deviations.size())
	for i in deviations.size():
		# A channel that never moved is treated as unit scale rather than as
		# something to divide by. CONSTANT_STD is not an epsilon guarding a
		# divide by zero: it is a decision about what counts as constant, and it
		# has to match python/ppo.py's, which verify_pilot checks.
		#
		# A standard deviation a thousandth of a channel's own range means it
		# never varied. Measured: this policy's fuel channel has obs_std 1.0e-4,
		# because it trained with unlimited tanks and always saw exactly 1.0.
		# Giving the player ship a fuel budget fed it 0.985, which divided out
		# to -147 against inputs that otherwise live in [-3, 3]. The clamp below
		# caught it at -10, which is still a value that channel never once saw,
		# and the ship stopped being able to land.
		_inverse_std[i] = 1.0 / deviations[i] if deviations[i] > CONSTANT_STD else 1.0
	if _mean.size() != obs_size or _inverse_std.size() != obs_size:
		push_warning("NeuralPilot: normalisation statistics do not match obs_size")
		return false

	var widest := obs_size
	for bias in _biases:
		widest = maxi(widest, bias.size())
	_scratch_a.resize(widest)
	_scratch_b.resize(widest)
	return true


## Mean action for an observation. The trained policy is a Gaussian; at play
## time we take the mean rather than sampling, because the exploration noise
## has no job left to do and shows up as visible twitching.
func act(observation: PackedFloat32Array) -> PackedFloat32Array:
	for i in obs_size:
		_scratch_a[i] = clampf((observation[i] - _mean[i]) * _inverse_std[i], -10.0, 10.0)

	var width := obs_size
	for layer in _weights.size():
		var rows: Array = _weights[layer]
		var bias: PackedFloat32Array = _biases[layer]
		var out_width := bias.size()
		for o in out_width:
			var row: PackedFloat32Array = rows[o]
			var sum: float = bias[o]
			for k in width:
				sum += row[k] * _scratch_a[k]
			# tanh everywhere except the output layer, which is the raw action.
			_scratch_b[o] = tanh(sum) if layer < _weights.size() - 1 else sum
		var swap := _scratch_a
		_scratch_a = _scratch_b
		_scratch_b = swap
		width = out_width

	var action := PackedFloat32Array()
	action.resize(act_size)
	for i in act_size:
		action[i] = clampf(_scratch_a[i], -1.0, 1.0)
	return action
