"""Diff the GDScript forward pass against the numpy one.

Runs `rl/verify_pilot.gd` headless, runs the same observations through
`PolicyController`, and compares. The two implementations of the network — one
for the game, one for evaluation — are the kind of duplication where a bug is
invisible: a transposed weight or a dropped normalisation still yields six
plausible numbers in [-1, 1] and a ship that flies, just badly.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from evaluate import PolicyController

PROBES = np.array(
    [
        [0.20, 0.90, 0.10, -0.30, 0.05, 0.10, 0.95, -0.20, -0.15, 0.08, 1.00, 0.50, 0.30, 0.00],
        [0.02, 0.35, -0.05, 0.02, 0.01, -0.02, 0.99, 0.05, -0.03, 0.01, 1.00, 0.90, 0.80, 0.00],
        [1.20, 1.00, 0.40, -0.10, -0.25, 0.30, 0.70, 0.60, -0.40, 0.35, 0.55, 0.10, 0.05, 0.00],
    ],
    dtype=np.float32,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--godot", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--policy-path", default="rl/runs/landing/policy.json")
    parser.add_argument("--tolerance", type=float, default=2e-4)
    args = parser.parse_args()

    result = subprocess.run(
        [args.godot, "--headless", "--path", args.project, "--script", "res://rl/verify_pilot.gd"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    rows = [
        line.split(" ", 1)[1] for line in result.stdout.splitlines() if line.startswith("ACT ")
    ]
    if len(rows) != len(PROBES):
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        print(f"FAIL: expected {len(PROBES)} actions from Godot, got {len(rows)}")
        return 1

    godot = np.array([[float(v) for v in row.split(",")] for row in rows])
    numpy_actions = PolicyController(Path(args.policy_path))(PROBES)

    worst = float(np.max(np.abs(godot - numpy_actions)))
    for i in range(len(PROBES)):
        print(f"probe {i}")
        print(f"  gdscript {np.round(godot[i], 5)}")
        print(f"  numpy    {np.round(numpy_actions[i], 5)}")
    print(f"\nlargest disagreement: {worst:.2e} (tolerance {args.tolerance:.0e})")

    if worst > args.tolerance:
        print("FAIL: the two forward passes disagree")
        return 1
    # float32 in GDScript against float32 in numpy will not be bit-identical;
    # the tolerance is for accumulation order, not for a real difference.
    print("PASS: in-game inference matches the evaluated policy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
