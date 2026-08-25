"""Python half of the Godot lock-step RL bridge.

Launches a headless Godot process running ``rl/train.tscn`` and talks to it over
a loopback socket, one line-delimited JSON exchange per physics tick. Every tick
carries the whole batch of agents, so a single round trip yields ``num_agents``
transitions.

The interface is deliberately close to a Gymnasium ``VectorEnv``: ``reset()``
returns the first observation batch and ``step(actions)`` returns
``(obs, reward, terminated, truncated, info)``. It is not an actual
``VectorEnv`` subclass because there is nothing to gain from the dependency.

Autoreset happens on the Godot side. When an episode ends, the observation
returned for that agent is already the first frame of its next episode, and the
final frame of the one that ended is handed back in ``info["final_obs"]`` — the
value function needs it to bootstrap correctly across a time-limit cut.
"""

from __future__ import annotations

import json
import socket
import subprocess
import time
from pathlib import Path

import numpy as np

OBS_SIZE = 14
ACT_SIZE = 6

# Mirrors the outcome constants in rl/landing_env.gd.
RUNNING, LANDED, CRASHED, ESCAPED, TIMEOUT = range(5)
OUTCOME_NAMES = {
    LANDED: "landed",
    CRASHED: "crashed",
    ESCAPED: "escaped",
    TIMEOUT: "timeout",
}


class GodotVecEnv:
    def __init__(
        self,
        godot_bin: str | Path,
        project: str | Path,
        num_agents: int = 16,
        port: int = 11008,
        seed: int = 1,
        fixed_fps: int = 60,
        launch: bool = True,
        verbose: bool = False,
        step_timeout: float = 30.0,
    ) -> None:
        self.num_agents = num_agents
        self.obs_size = OBS_SIZE
        self.act_size = ACT_SIZE
        self._process: subprocess.Popen | None = None

        if launch:
            command = [
                str(godot_bin),
                "--headless",
                # Without --fixed-fps the engine sleeps to hit real time, and
                # training crawls at 60 steps a second regardless of the CPU.
                "--fixed-fps",
                str(fixed_fps),
                "--path",
                str(project),
                "res://rl/train.tscn",
                "--",
                f"--agents={num_agents}",
                f"--port={port}",
                f"--seed={seed}",
            ]
            self._process = subprocess.Popen(
                command,
                stdout=None if verbose else subprocess.DEVNULL,
                stderr=None if verbose else subprocess.DEVNULL,
            )

        self._sock = self._connect(port)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # A read timeout, because the failure it guards against is a silent
        # one. If a GDScript error aborts _physics_process, the engine never
        # sends its line and never closes the socket, and this side blocks
        # forever having printed nothing at all. Failing loudly after a few
        # seconds is worth far more than the timeout costs.
        self._sock.settimeout(step_timeout)
        self._rx = b""

    def _connect(self, port: int, timeout: float = 45.0) -> socket.socket:
        """Retry until the engine has finished building terrain and is listening.

        First contact is slow: Godot has to build every planet mesh before it
        opens the port, which is seconds of CPU work at mesh_resolution 160.
        """
        deadline = time.time() + timeout
        last_error: OSError | None = None
        while time.time() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    f"Godot exited with code {self._process.returncode} before listening. "
                    "Re-run with verbose=True to see its output."
                )
            try:
                sock = socket.create_connection(("127.0.0.1", port), timeout=5.0)
                sock.settimeout(None)
                return sock
            except OSError as error:  # not up yet
                last_error = error
                time.sleep(0.25)
        raise TimeoutError(f"Godot never started listening on {port}") from last_error

    # --- Protocol ------------------------------------------------------------

    def _read_message(self) -> dict:
        while b"\n" not in self._rx:
            try:
                chunk = self._sock.recv(1 << 16)
            except socket.timeout as error:
                raise TimeoutError(
                    "Godot stopped responding mid-episode. The usual cause is a "
                    "GDScript error aborting _physics_process before it replies; "
                    "run rl/train.tscn directly to see the engine output."
                ) from error
            if not chunk:
                raise ConnectionError("Godot closed the connection")
            self._rx += chunk
        line, self._rx = self._rx.split(b"\n", 1)
        return json.loads(line)

    def _send(self, payload: dict) -> None:
        self._sock.sendall((json.dumps(payload, separators=(",", ":")) + "\n").encode())

    @staticmethod
    def _unpack(message: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        obs = np.asarray(message["obs"], dtype=np.float32)
        reward = np.asarray(message["rew"], dtype=np.float32)
        terminated = np.asarray(message["done"], dtype=bool)
        truncated = np.asarray(message["trunc"], dtype=bool)
        info = {
            "outcome": np.asarray(message["outcome"], dtype=np.int64),
            "impact": np.asarray(message["impact"], dtype=np.float32),
            "final_obs": {
                int(k): np.asarray(v, dtype=np.float32)
                for k, v in message.get("final", {}).items()
            },
        }
        return obs, reward, terminated, truncated, info

    # --- Gym-ish surface -----------------------------------------------------

    def reset(self) -> np.ndarray:
        """First observation batch.

        Godot speaks first on every tick including the very first, so a reset is
        just a read — there is no request to make.
        """
        obs, _, _, _, _ = self._unpack(self._read_message())
        return obs

    def step(self, actions: np.ndarray):
        actions = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        assert actions.shape == (self.num_agents, ACT_SIZE), actions.shape
        self._send({"act": actions.tolist()})
        return self._unpack(self._read_message())

    def close(self) -> None:
        try:
            self._send({"cmd": "close"})
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass
        if self._process is not None:
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()

    def __enter__(self) -> "GodotVecEnv":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
