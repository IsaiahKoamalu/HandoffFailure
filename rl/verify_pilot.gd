extends SceneTree

## Prints NeuralPilot's output for a fixed observation, so it can be diffed
## against the numpy forward pass in python/evaluate.py.
##
## There are two implementations of the same network — one in GDScript for the
## game, one in numpy for evaluation — and nothing about a wrong one looks
## wrong. A transposed weight matrix or a missing normalisation still produces
## six plausible numbers in [-1, 1], and the ship still flies, just badly. The
## only way to know they agree is to make them answer the same question.
##
##     Godot.exe --headless --path . --script res://rl/verify_pilot.gd

const PROBES: Array = [
	[0.20, 0.90, 0.10, -0.30, 0.05, 0.10, 0.95, -0.20, -0.15, 0.08, 1.00, 0.50, 0.30, 0.00, 0.62, 0.10, 0.05, 0.99, 0.35],
	[0.02, 0.35, -0.05, 0.02, 0.01, -0.02, 0.99, 0.05, -0.03, 0.01, 1.00, 0.90, 0.80, 0.00, -0.80, -0.60, 0.20, 0.77, 0.82],
	[1.20, 1.00, 0.40, -0.10, -0.25, 0.30, 0.70, 0.60, -0.40, 0.35, 0.55, 0.10, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10],
]


func _initialize() -> void:
	var pilot := NeuralPilot.load_from()
	if pilot == null:
		print("NO_POLICY")
		quit(1)
		return

	print("trained_steps=%d obs=%d act=%d" % [pilot.trained_steps, pilot.obs_size, pilot.act_size])
	for probe in PROBES:
		var observation := PackedFloat32Array(probe)
		var action := pilot.act(observation)
		var parts := PackedStringArray()
		for value in action:
			parts.append("%.6f" % value)
		print("ACT " + ",".join(parts))
	quit(0)
