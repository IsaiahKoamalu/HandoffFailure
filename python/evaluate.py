"""Score a controller on the landing task.

Runs a fixed number of episodes and reports landing rate, touchdown speed and
return. Two controllers are built in:

``--controller baseline``
    A hand-written proportional descent controller: hold a descent rate that
    tapers with altitude, and torque the belly toward the ground. This is the
    thing you would write *instead* of training something, and it is the number
    the policy has to beat for any of this to have been worth doing.

``--controller policy``
    The trained network, evaluated at its distribution mean rather than sampled.
    Exploration noise has no job left once training is over and it shows up as
    visible twitching.

``--controller zero``
    Hands off the controls. Establishes what the ship's own flight assist does
    unaided, which is the real floor — not a random policy.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from godot_env import ACT_SIZE, OBS_SIZE, OUTCOME_NAMES, RUNNING, LANDED, GodotVecEnv

# Observation layout, mirroring LandingEnv.build_observation.
ALT = 0
V_LOCAL = slice(2, 5)
UP_LOCAL = slice(5, 8)
V_UP = 8


def baseline_controller(obs: np.ndarray) -> np.ndarray:
    """Proportional descent hold plus attitude levelling, in observation units.

    Everything here is expressed in the ship's own frame, which is what makes it
    short: `up_local` is where the ground is from the ship's point of view, so
    thrusting along it is thrusting away from the ground, and rotating it onto
    +Y is levelling out.
    """
    action = np.zeros((obs.shape[0], ACT_SIZE), dtype=np.float32)

    altitude = obs[:, ALT]
    up_local = obs[:, UP_LOCAL]
    v_up = obs[:, V_UP]

    # Descent rate tapers to near zero at the surface. Observations are scaled
    # by orbital speed at the surface (~42 units/s), hence the small numbers.
    desired = -0.036 - 0.09 * np.clip(altitude / 0.2, 0.0, 1.0)
    lift = np.clip(15.0 * (desired - v_up), -1.0, 1.0)

    # Thrust along local up, expressed in the ship frame.
    action[:, 0:3] = up_local * lift[:, None]

    # Rotate the ship's +Y onto up_local. The axis of that rotation is
    # (0,1,0) x up_local = (up_local.z, 0, -up_local.x), so its X component is
    # pitch and its Z component is roll.
    action[:, 3] = np.clip(2.0 * up_local[:, 2], -1.0, 1.0)
    action[:, 5] = np.clip(-2.0 * up_local[:, 0], -1.0, 1.0)
    return action


class PolicyController:
    """Forward pass over the exported JSON, in numpy.

    Reads the same file Godot reads, so a disagreement between this number and
    in-game behaviour is a bug in the GDScript forward pass, not in the policy.
    """

    def __init__(self, path: Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert data["obs_size"] == OBS_SIZE and data["act_size"] == ACT_SIZE
        self.layers = [
            (np.asarray(layer["w"], dtype=np.float32), np.asarray(layer["b"], dtype=np.float32))
            for layer in data["layers"]
        ]
        self.mean = np.asarray(data["obs_mean"], dtype=np.float32)
        self.std = np.asarray(data["obs_std"], dtype=np.float32)
        self.trained_steps = data.get("trained_steps", 0)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = np.clip((obs - self.mean) / np.maximum(self.std, 1e-6), -10.0, 10.0)
        for index, (weight, bias) in enumerate(self.layers):
            x = x @ weight.T + bias
            if index < len(self.layers) - 1:
                x = np.tanh(x)
        return np.clip(x, -1.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--godot", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--controller", choices=["policy", "baseline", "zero"], default="policy")
    parser.add_argument("--policy-path", default="rl/runs/landing/policy.json")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--agents", type=int, default=32)
    parser.add_argument("--port", type=int, default=11040)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-ticks", type=int, default=20000)
    args = parser.parse_args()

    if args.controller == "policy":
        controller = PolicyController(Path(args.policy_path))
        label = f"policy ({controller.trained_steps:,} steps)"
    elif args.controller == "baseline":
        controller = baseline_controller
        label = "hand-written baseline"
    else:
        controller = lambda obs: np.zeros((obs.shape[0], ACT_SIZE), dtype=np.float32)
        label = "no input (assist only)"

    outcomes: collections.Counter[int] = collections.Counter()
    impacts: list[float] = []
    landed_impacts: list[float] = []
    returns: list[float] = []
    lengths: list[int] = []

    with GodotVecEnv(
        args.godot, args.project, num_agents=args.agents, port=args.port, seed=args.seed
    ) as env:
        obs = env.reset()
        running_return = np.zeros(args.agents)
        running_length = np.zeros(args.agents, dtype=int)

        for _ in range(args.max_ticks):
            obs, reward, terminated, truncated, info = env.step(controller(obs))
            running_return += reward
            running_length += 1
            for i in range(args.agents):
                outcome = int(info["outcome"][i])
                if outcome != RUNNING:
                    outcomes[outcome] += 1
                    impacts.append(float(info["impact"][i]))
                    if outcome == LANDED:
                        landed_impacts.append(float(info["impact"][i]))
                if terminated[i] or truncated[i]:
                    returns.append(float(running_return[i]))
                    lengths.append(int(running_length[i]))
                    running_return[i] = 0.0
                    running_length[i] = 0
            if len(returns) >= args.episodes:
                break

    total = max(len(returns), 1)
    print(f"\n=== {label} ===")
    print(f"episodes            {len(returns)}")
    print(f"landing rate        {outcomes[LANDED] / total:.1%}")
    print(f"mean return         {np.mean(returns):.2f}")
    print(f"mean episode length {np.mean(lengths):.0f} ticks ({np.mean(lengths) / 60:.1f}s)")
    if impacts:
        print(f"touchdown speed     mean {np.mean(impacts):.2f}  median {np.median(impacts):.2f}"
              f"  p90 {np.percentile(impacts, 90):.2f} units/s")
    if landed_impacts:
        print(f"  of the landings   mean {np.mean(landed_impacts):.2f} units/s")
    for outcome, count in sorted(outcomes.items()):
        print(f"  {OUTCOME_NAMES.get(outcome, outcome):<8} {count / total:6.1%}  ({count})")


if __name__ == "__main__":
    main()
