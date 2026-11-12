"""Check that rl/neural_pilot.gd computes the same actions as PyTorch.

Runs the same observations through the trained agent here and through 
NeuralPilot in a headless Godot, both read one policy.json path (that is given)
and fails if any action differes by more than set TOLERANCE.
Nothing measured in the engine is able to be confidently trusted as the 
policy that was trained until this passes.
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
    """Return the largest difference between engine and PyTorch actions; raise if over tolerance."""
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