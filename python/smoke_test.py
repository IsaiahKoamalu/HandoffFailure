"""Connect to the Godot env, drive it with random actions, and report throughput.

Run this before training. It answers the two questions that decide whether the
bridge is usable at all: does an episode ever end, and how many transitions a
second does the loop actually deliver?
"""

from __future__ import annotations

import argparse
import collections
import time

import numpy as np

from godot_env import OUTCOME_NAMES, RUNNING, GodotVecEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--godot", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--agents", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--fuel", type=float, default=0.0)
    parser.add_argument("--ticks", type=int, default=1500)
    parser.add_argument("--port", type=int, default=11008)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    outcomes: collections.Counter[int] = collections.Counter()
    returns = np.zeros(args.agents, dtype=np.float64)
    finished: list[float] = []

    with GodotVecEnv(
        args.godot, args.project, num_agents=args.agents, port=args.port, verbose=args.verbose,
        repeat=args.repeat,
        fuel_seconds=args.fuel,
    ) as env:
        obs = env.reset()
        print(f"obs batch {obs.shape}, first row:\n  {np.round(obs[0], 3)}")

        start = time.time()
        for _ in range(args.ticks):
            action = rng.uniform(-1.0, 1.0, size=(args.agents, env.act_size)).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            returns += reward
            for i, outcome in enumerate(info["outcome"]):
                if outcome != RUNNING:
                    outcomes[int(outcome)] += 1
                if terminated[i] or truncated[i]:
                    finished.append(returns[i])
                    returns[i] = 0.0
        elapsed = time.time() - start

    steps = args.ticks * args.agents
    print(f"\n{steps} transitions in {elapsed:.1f}s = {steps / elapsed:,.0f} steps/s")
    print(f"episodes finished: {len(finished)}")
    if finished:
        print(f"random-policy return: mean {np.mean(finished):.1f}  min {min(finished):.1f}  max {max(finished):.1f}")
    for outcome, count in outcomes.most_common():
        print(f"  {OUTCOME_NAMES.get(outcome, outcome):<8} {count}")
    if not np.isfinite(obs).all():
        raise SystemExit("FAIL: non-finite values in observations")
    print("observations finite: ok")


if __name__ == "__main__":
    main()
