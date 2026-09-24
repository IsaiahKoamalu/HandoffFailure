"""Python half of the Godot lock-step RL bridge.

Launches a headless Godot process running ``rl/train.tscn`` and talks to it over
a loopback socket, one line-delimited JSON exchange per physics tick. Every tick
carries the whole batch of agents, so a single round trip yields ``num_agents``
transitions.

The interface is deliberately close to a Gymnasium ``VectorEnv``: ``reset()``
returns the first observation batch and ``step(actions)`` returns
``(obs, reward, terminated, truncated, info)``. It is not an actual
``VectorEnv`` subclass because there is nothing to gain from the dependency.

One ``step()`` is one *decision*. With ``repeat > 1`` the engine holds that
action for several physics ticks and sums the rewards, so a step covers more
simulated time without the trainer seeing any more transitions.

Autoreset happens on the Godot side. When an episode ends, the observation
returned for that agent is already the first frame of its next episode, and the
final frame of the one that ended is handed back in ``info["final_obs"]`` — the
value function needs it to bootstrap correctly across a time-limit cut.
"""

from __future__ import annotations

import json
import math
import socket
import subprocess
import time
from pathlib import Path

import numpy as np

# Every control task flies the same ship, so the action width is fixed. The
# observation width is not: it comes from the engine's hello message, because
# it is a property of the task and this module does not know which one is
# running. See GodotVecEnv.obs_size.
ACT_SIZE = 6

# The one outcome every task has. The rest are named by the hello message and
# land in GodotVecEnv.outcome_names -- there is deliberately no mirrored copy
# of a task's outcome list here, because a mirror is a thing that can drift.
RUNNING = 0

# Landing outcome ids, for callers that test for a specific one. These are
# LandingEnv's constants and mean nothing under another task; anything that
# only needs to *label* an outcome should use env.outcome_names instead.
LANDED, CRASHED, ESCAPED, TIMEOUT, DITCHED, MISSED = range(1, 7)


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
        render: bool = False,
        render_rate: int = 180,
        repeat: int = 1,
        fuel_seconds: float = 0.0,
        max_steps: int = 0,
        planets: str = "",
        holdout: str = "",
        task: str = "landing",
    ) -> None:
        self.num_agents = num_agents
        # Physics ticks each action is held for. One `step()` here is one
        # decision, which is `repeat` ticks of simulation with the rewards
        # summed. See the note on `repeat` in rl/train_host.gd.
        self.repeat = repeat
        # Filled in from the engine's hello message, below.
        self.obs_size = 0
        self.act_size = ACT_SIZE
        self.task = task
        self.outcome_names: dict[int, str] = {}
        self.max_steps = 0
        self._process: subprocess.Popen | None = None
        self._hud: dict | None = None

        if launch:
            command = [str(godot_bin)]
            if render:
                # A window, and no --fixed-fps: that flag makes the engine run
                # the main loop flat out, which off screen is the whole point
                # and on screen is an unreadable blur. Pace is set instead by
                # Engine.physics_ticks_per_second on the Godot side, which
                # speeds up the run without altering the step.
                command += ["--path", str(project), "res://rl/train.tscn"]
            else:
                command += [
                    "--headless",
                    # Without --fixed-fps the engine sleeps to hit real time,
                    # and training crawls at 60 steps a second regardless of
                    # how fast the CPU is.
                    "--fixed-fps",
                    str(fixed_fps),
                    "--path",
                    str(project),
                    "res://rl/train.tscn",
                ]
            command += [
                "--",
                f"--agents={num_agents}",
                f"--task={task}",
                f"--port={port}",
                f"--seed={seed}",
                f"--render={1 if render else 0}",
                f"--rate={render_rate}",
                f"--repeat={repeat}",
                f"--fuel={fuel_seconds}",
                f"--max-steps={max_steps}",
                f"--planets={planets}",
                f"--holdout={holdout}",
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
        self._hello = self._read_message()
        if self._hello.get("type") != "hello":
            raise RuntimeError(
                f"expected a hello message from Godot, got keys {sorted(self._hello)}"
            )
        self.task = self._hello["task"]
        self.obs_size = int(self._hello["obs_size"])
        self.act_size = int(self._hello["act_size"])
        self.max_steps = int(self._hello["max_steps"])
        self.outcome_names = {
            int(key): value for key, value in self._hello["outcomes"].items()
        }
        # Which outcome the run is judged on. The engine says, because only the
        # task knows: "landed" for landing, "docked" for docking. Older engines
        # predate the field, and for those every task numbers success 1.
        self.success_code = int(self._hello.get("success", 1))
        self.success_name = self.outcome_names.get(self.success_code, "success")
        # Both sides were told the batch size separately, on the command line
        # and here. If they disagree the protocol desynchronises a few frames
        # later with a confusing message, so check it while it is still cheap.
        if int(self._hello["num_agents"]) != num_agents:
            raise RuntimeError(
                f"engine is running {self._hello['num_agents']} agents, "
                f"expected {num_agents}"
            )
        if int(self._hello["repeat"]) != repeat:
            raise RuntimeError(
                f"engine is holding actions for {self._hello['repeat']} ticks, "
                f"expected {repeat}"
            )

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
        # allow_nan=False so a non-finite value raises here instead of going out
        # as a bare NaN. That is valid Python and invalid JSON, and Godot answers
        # an unparseable message by hanging up — which surfaces over here as
        # "Godot closed the connection" some frames later, pointing at nothing.
        line = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        self._sock.sendall((line + "\n").encode())

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

    def set_hud(self, **stats) -> None:
        """Stage a line of text for the on-screen readout.

        Sent with the next action message and dropped by a headless engine, so
        the trainer can call this unconditionally and never branch on whether
        anyone is watching.

        Non-finite values are dropped rather than sent. ``json.dumps`` writes a
        bare ``NaN``, which is valid Python and invalid JSON, and Godot responds
        to an unparseable message by closing the connection — so a mean return
        over an empty window, which is exactly what iteration one has, would
        take down the run. A cosmetic field must not be able to do that.
        """
        self._hud = {
            key: value
            for key, value in stats.items()
            if not (isinstance(value, float) and not math.isfinite(value))
        }

    def step(self, actions: np.ndarray):
        actions = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        assert actions.shape == (self.num_agents, ACT_SIZE), actions.shape
        message: dict = {"act": actions.tolist()}
        if self._hud is not None:
            message["hud"] = self._hud
            self._hud = None
        self._send(message)
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
