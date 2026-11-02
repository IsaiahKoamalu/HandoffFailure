import os
import subprocess
import sys
import time
from pathlib import Path
from godot_env import GodotVectorEnv

ROOT = Path(__file__).resolve().parent.parent

def launch_host():
    godot = os.environ.get("GODOT")
    if not godot:
        sys.exit("GODOT is not set; see 'Running the simulator' in README.md")
    return subprocess.Popen(
        [godot, "--headless", "--fixed-fps", "60", "--path", str(ROOT),
         "res://rl/train/host.tscn"]
    )

def connect(host, num_envs, seed, repeat, max_steps, timeout=30.0):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return GodotVectorEnv(num_envs, seed, repeat, max_steps)
        except ConnectionRefusedError:
            if host.poll() is not None:
                sys.exit(f"host exited with code {host.returncode} before it was listening")
            if time.monotonic() > deadline:
                sys.exit(f"host was not listening after {timeout:.0f} s")
            time.sleep(0.2)