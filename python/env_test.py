"""Smoke test for the bridge: launches the headless host, flys random actions, reports.

SHOULD**** run in under a minute (lord willing). A random policy should end nearly every
episode badly; if it lands often, the termination logic is wrong.

    run with: uv run python python/env_test.py
"""

import collections
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from godot_env import GodotVectorEnv

ROOT = Path(__file__).resolve().parent.parent
SHIPS = 32
SEED = 7
REPEAT = 6
MAX_STEPS = 600 # short episodes, so plenty end inside to test
STEPS = 400
TIME_LIMIT = 60.0

def launch_host():
    godot = os.environ.get("GODOT")
    if not godot:
        sys.exit("GODOT is not set; see 'Running the simulator' in README.md")
    return subprocess.Popen(
        [godot, "--headless", "--fixed-fps", "60", "--path", str(ROOT),
         "res://rl/train/host.tscn"]
    )

def connect(host, timeout=30.0):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return GodotVectorEnv(num_envs=SHIPS, seed=SEED, repeat=REPEAT, max_steps=MAX_STEPS)
        except ConnectionRefusedError:
            if host.poll() is not None:
                sys.exit(f"host exited with code {host.returncode} before it was listening")
            if time.monotonic() > deadline:
                sys.exit(f"host was not listening after {timeout:.0f} s")
            time.sleep(0.2)

def main():
    started = time.monotonic()
    host = launch_host()
    try:
        env = connect(host)
        env.action_space.seed(SEED)
        obs, _ = env.reset(seed=SEED)
        if obs.shape != env.observation_space.shape:
            sys.exit(f"reset returned {obs.shape}, expected {env.observation_space.shape}")

        endings = collections.Counter()
        stepping = time.monotonic()
        for _ in range(STEPS):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            if np.any(terminated & truncated):
                sys.exit("a ship was both terminated and truncated")
            for i in np.flatnonzero(info["_final_obs"]):
                endings[env.outcomes[int(info["outcome"][i])]] += 1
        stepping = time.monotonic() - stepping
        env.close()
        host.wait(timeout=10)
    finally:
        if host.poll() is None:
            host.kill()
    elapsed = time.monotonic() - started

    episodes = sum(endings.values())
    print(f"\n{episodes} episodes ended in {STEPS} decisions")
    for name, count in endings.most_common():
        print(f"    {name:<10} {count:>5} {count / max(episodes, 1):6.1%}")
    ticks = STEPS * SHIPS * REPEAT
    print(f"{ticks/stepping:,.0f} ship-ticks/s, {STEPS/stepping:,.0f} decisions/s, "
          f"{elapsed:.1f} s in total")

    success = endings[env.outcomes[env.success]]
    failures = []
    if episodes == 0:
        failures.append("no episode ended; the termination logic never fires")
    elif success / episodes > 0.05:
        failures.append(f"a random policy succeeded {success/episodes:.0%} of the time")
    if elapsed > TIME_LIMIT:
        failures.append(f"took {elapsed:.0f} s, over the {TIME_LIMIT:.0f} s budget")
    if host.returncode != 0:
        failures.append(f"host exited with the code {host.returncode} after the close")

    for failure in failures:
        print("FAIL:", failure)
    print("PASS" if not failures else "")
    sys.exit(1 if failures else 0)

if __name__ == "__main__":
    main()
