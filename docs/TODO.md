## Step 11: export `policy.json` and check it against the engine

Roadmap item: "check that numpy inference and the in-engine GDScript forward pass agree
to about 1e-6 before believing any in-engine result."

This step adds three new files and makes three edits to `ppo.py`. When it's done, every
training run ends with a `policy.json` that `rl/neural_pilot.gd` can load, together with
proof that the engine computes the same actions as PyTorch.

### What was run before writing this
In a scratch copy of the repo, `ppo.py --total-timesteps 20000 --max-steps 600` with
everything below in place:

```
iter 2/2  episodes 97  landed 0.0%  explained_var 0.858  SPS 3020
policy saved to rl/runs/ppo__7__…/policy.json; engine agrees with PyTorch to 1.5e-07
```

- **The exported file** has layers `(64, 4)`, `(64, 64)`, `(6, 64)`, the normalizer
  statistics, the observation names and `trained_steps: 16384`.
- **Negative control:** with `obs_mean` shifted by 0.01 in a copy of the file, the
  engine's actions changed by 0.0043, more than 4,000 times the tolerance. A wrong export
  is caught.
- **One real bug came up and is fixed in the code below.** CleanRL builds its layers
  with sizes from `np.prod(...)`, so `in_features` is a numpy integer, which `json`
  refuses to write.

---

### 11a. `python/export.py`

```python
"""Write a trained policy as the policy.json that rl/neural_pilot.gd loads.

NeuralPilot runs the actor's mean in GDScript: inputs normalized with obs_mean and
obs_std, tanh after every layer but the last, output clamped to [-1, 1]. The critic
and the exploration noise are only needed for training and are not exported.
"""

import json

import torch.nn as nn

FORMAT = 1


def export_policy(path, agent, normalizer, *, task, obs_names, repeat, trained_steps):
    modules = list(agent.actor_mean)
    linears = modules[0::2]
    expected = [nn.Linear, nn.Tanh] * (len(linears) - 1) + [nn.Linear]
    if [type(m) for m in modules] != expected:
        raise ValueError("NeuralPilot runs Linear, Tanh, ..., Linear; the actor has a different shape")

    state = normalizer.state()
    policy = {
        "format": FORMAT,
        "task": task,
        "obs_size": int(linears[0].in_features),
        "act_size": int(linears[-1].out_features),
        "activation": "tanh",
        "repeat": repeat,
        "trained_steps": trained_steps,
        "layers": [
            {"w": layer.weight.detach().cpu().tolist(), "b": layer.bias.detach().cpu().tolist()}
            for layer in linears
        ],
        "obs_mean": state["obs_mean"],
        "obs_std": state["obs_std"],
        "obs_names": obs_names,
        "normalizer_count": state["count"],
    }
    with open(path, "w") as f:
        json.dump(policy, f)
```

- **Only `actor_mean` is exported.** In play the ship flies the mean action, so the
  critic and `actor_logstd` (the exploration noise) have no use in the engine.
- **It checks the network's shape before exporting.** `neural_pilot.gd` hard-codes "tanh
  after every layer except the last". Changing the network (ReLU, an extra layer norm)
  would otherwise load fine and compute something different. `modules[0::2]` takes every
  other module, which are the `Linear` layers.
- **The keys are exactly the ones `neural_pilot.gd` reads** (its header comment, lines
  9–16).
- **`int(...)` around the sizes** is the fix for the numpy-integer bug. It's simpler to
  convert here than to change CleanRL's `Agent`.
- **`layer.weight` is already `[out][in]`,** the layout `neural_pilot.gd` loops over. No
  transpose is needed, and the engine check is what proves it.
- **`.tolist()` writes float32 values exactly,** and the engine loads them back into a
  `PackedFloat32Array` unchanged.
- **`obs_names` and `normalizer_count` are extra.** `neural_pilot.gd` ignores keys it
  doesn't read, and `docs/protocol.md` says the names must be saved alongside the weights
  for E3.

---

### 11b. `rl/checks/policy_check.gd`

The engine side of the comparison: a script Godot runs without loading any scene.

```gdscript
extends SceneTree

## Runs observations through NeuralPilot.act and writes the actions back, so
## python/verify_policy.py can compare the engine's forward pass with PyTorch's.
##
##     "$GODOT" --headless --path . --script res://rl/checks/policy_check.gd -- \
##         <policy.json> <inputs.json> <outputs.json>
##
## The policy path comes from the caller, never from NeuralPilot's default, so the
## check cannot quietly compare against whatever policy happens to be installed.


func _initialize() -> void:
	var args := OS.get_cmdline_user_args()
	if args.size() != 3:
		printerr("policy_check: expected <policy.json> <inputs.json> <outputs.json>")
		quit(2)
		return

	var inputs = JSON.parse_string(FileAccess.get_file_as_string(args[1]))
	if typeof(inputs) != TYPE_ARRAY or inputs.is_empty():
		printerr("policy_check: %s is not a non-empty list of observations" % args[1])
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
```

- **`extends SceneTree` with `_initialize`** is how Godot runs a plain script with
  `--script`. No world gets built, so it starts in about a second.
- **`OS.get_cmdline_user_args()`** returns everything after the `--` in the command.
- **The policy path is an argument and goes straight into `load_from`,** with no fallback
  to `NeuralPilot.DEFAULT_PATH`. This is the trap from the roadmap: a path that reaches
  only one side of a check once produced a 0.91 disagreement that was pure artefact.
- **It calls `NeuralPilot.act`, the exact function the autopilot calls in play,** not a
  copy of the forward pass.
- **`JSON.stringify(outputs, "", false, true)`:** the last argument is `full_precision`.
  Without it Godot rounds floats when writing JSON, and at 1e-6 that could cause a
  failure by itself.
- **Exit codes:** 2 means called wrong, 1 means the policy was refused, 0 means it worked.
- **No reimport is needed,** since there's no `class_name`. Commit the `.uid` file Godot
  creates for it.

---

### 11c. `python/verify_policy.py`

```python
"""Check that rl/neural_pilot.gd computes the same actions as PyTorch.

Runs the same observations through the trained agent here and through NeuralPilot
in a headless Godot, both reading the one policy.json path given, and fails if any
action differs by more than TOLERANCE. Until this passes, nothing measured in the
engine can be trusted to be the policy that was trained.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from launch import ROOT

TOLERANCE = 1e-6


def engine_actions(policy_path, observations):
    """Run observations through NeuralPilot.act in a headless Godot."""
    godot = os.environ.get("GODOT")
    if not godot:
        raise RuntimeError("GODOT is not set; see 'Running the simulator' in README.md")
    with tempfile.TemporaryDirectory() as tmp:
        inputs = Path(tmp) / "inputs.json"
        outputs = Path(tmp) / "outputs.json"
        inputs.write_text(json.dumps(observations.tolist()))
        result = subprocess.run(
            [godot, "--headless", "--path", str(ROOT), "--script", "res://rl/checks/policy_check.gd",
             "--", str(Path(policy_path).resolve()), str(inputs), str(outputs)],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not outputs.exists():
            raise RuntimeError(f"policy_check failed ({result.returncode}):\n{result.stdout}{result.stderr}")
        return np.array(json.loads(outputs.read_text()), dtype=np.float32)


def verify_policy(policy_path, agent, normalizer, samples=1000, seed=0):
    """Return the largest difference between engine and PyTorch actions; raise if over TOLERANCE."""
    rng = np.random.default_rng(seed)
    # Spread observations to three standard deviations, so the clamp is exercised too.
    raw = normalizer.mean + normalizer.std * rng.normal(scale=3.0, size=(samples, len(normalizer.mean)))
    raw = raw.astype(np.float32)
    with torch.no_grad():
        mean_action = agent.actor_mean(torch.tensor(normalizer(raw)))
    expected = mean_action.clamp(-1.0, 1.0).cpu().numpy()
    actual = engine_actions(policy_path, raw)
    worst = float(np.max(np.abs(actual - expected)))
    if worst > TOLERANCE:
        raise RuntimeError(f"engine and PyTorch disagree by {worst:.3g} (tolerance {TOLERANCE:g})")
    return worst
```

- **Files rather than the socket.** This is a one-off batch computation, so it writes
  inputs to a file, runs Godot once and reads the outputs back. It doesn't touch the
  bridge.
- **`TemporaryDirectory`** cleans up both files when the block ends, even if Godot fails.
- **`.resolve()` makes the path absolute,** because Godot runs with `--path` set to the
  repo root and would read a relative path from somewhere else.
- **On failure the error includes Godot's own output,** so a "NeuralPilot refused…"
  message reaches you.
- **The PyTorch side is the live agent, not something rebuilt from the JSON.** That
  compares what was trained against what the engine loaded, which covers the export, the
  weight layout, the normalizer statistics and the engine's arithmetic. A numpy
  reimplementation reading the same JSON would share any export mistake and agree with it.
- **Raw observations spread to 3 standard deviations,** so some go far enough out that
  the ±10 input clamp and the ±1 action clamp both apply.
- **The expected action is clamped to [−1, 1]** because `NeuralPilot.act` clamps its
  output.
- **`seed=0`** makes the check reproducible, and **it returns the worst difference** so
  `ppo.py` can print it and the write-up can quote a number.

---

### 11d. Three edits to `python/ppo.py`

**1. Imports**, under `from normalizer import ObsNormalizer`:
```python
from export import export_policy
from verify_policy import verify_policy
```

**2. Remove the `save_model` argument** from `Args`, both lines:
```python
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
```
Every run now exports instead. The old path (`runs/{run_name}/...`) didn't match the
run directory anyway.

**3. Replace the `if args.save_model:` block** (the four lines before `envs.close()`)
with:
```python
        policy_path = run_dir / "policy.json"
        export_policy(policy_path, agent, normalizer, task=envs.task, obs_names=envs.obs_names,
                      repeat=args.repeat, trained_steps=global_step)
        worst = verify_policy(policy_path, agent, normalizer)
        print(f"policy saved to {policy_path}; engine agrees with PyTorch to {worst:.2g}")
```
- **Every run now ends with a verified `policy.json`.** If the engine disagrees, the run
  raises an error instead of handing over a policy that would fly differently in the game.
- **It runs inside the `try`,** so the host is still killed by the `finally` if the check
  raises.
- **`trained_steps=global_step`** records the steps actually taken, which can be less
  than `--total-timesteps` because of rounding to whole iterations (16,384 of 20,000).

---

### 11e. Run it

```bash
"$GODOT" --headless --path . --import       # registers the new script's uid
uv run python python/ppo.py --total-timesteps 20000 --max-steps 600
```

The last line should read `engine agrees with PyTorch to …`, with a value well under
1e-6.

### Once it passes
- Add to the changes list in `ppo.py`'s header:
  ```
      - Every run exports policy.json in rl/neural_pilot.gd's format and checks
        that the engine computes the same actions as PyTorch (verify_policy.py).
  ```
- Tick the normalizer item in `ROADMAP.md`, since it's now saved next to the weights.
- Commit `python/export.py`, `python/verify_policy.py`, `rl/checks/policy_check.gd` and
  its `.uid`, and `python/ppo.py`.

### Not covered by this step
Installing a chosen policy where the game looks for it (`rl/runs/landing/policy.json`,
`NeuralPilot.DEFAULT_PATH`). Do that deliberately once there's a policy worth installing,
rather than having every run overwrite it. The roadmap's own note: "a policy chosen
deliberately beats a policy left wherever the last training run put it."

### After this
The trainer side of week 1 is complete apart from the reward-scale unit test. What remains
is the real landing task: its observation, reward, spawn distribution and termination.
Its reward is zero everywhere right now, which is why `landed 0.0%` can't move yet.
