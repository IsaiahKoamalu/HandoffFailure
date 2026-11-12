## Runs observations through NeuralPilot.act and writes the actions back,
## so python/verify_policy.py can compare the engines forward pass with PyTorch's.
##
##     "$GODOT" --headless --path . --script res://rl/checks/policy_check.gd -- \
##          <policy.json> <inputs.json> <outputs.json>
##
## The policy path comes from the caller and never from NeuralPilot's default, so the
## check cannot quietly compare against whatever policy happens to be installed.

func _initialize() -> void:
    var args := OS.get_cmdline_args()
    if args.size() != 3:
        printerr("Policy check: expected <policy.json> <inputs.json> <outputs.json>")
        quit(2)
        return
    
    var inputs = JSON.parse_string(FileAccess.get_file_as_string(args[1]))
    if typeof(inputs) != TYPE_ARRAY or inputs.is_empty():
        printerr("policy_check: %s is a non-empty list of observations" % args[1])
        quit(2)
        return
    
    var pilot := NeuralPilot.load_from(inputs[0].size(), args[0])
    if pilot == null:
        printerr("policy_check: NeuralPilot refused %s" % args[0])
        quit(1)
        return
    
    var outputs := []
    for row in inputs:
        outputs.append(pilot.act(PackedFloat32Array(row)))
    var file := FileAccess.open(args[2], FileAccess.WRITE)
    file.store_string(JSON.stringify(outputs, "", false, true))
    file.close()
    quit(0)