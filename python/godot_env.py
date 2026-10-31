"""Gymnasium vecotr environment over the bridge to the headless Godot host.

One Godot process simulates every ship, so this is a native VectorEnv rather
than SyncVectorEnv over single environments. The host resets finished ships
itself (SAME_STEP autoreset): `step` returns the next episode's first
observation, and the observation each finished episode ended on is in
infos["final_obs"], masked by infos["_final_obs"].
"""

import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode

from bridge import Bridge


class GodotVectorEnv(gym.vector.VectorEnv):
    metadata = {"autoreset_mode": AutoresetMode.SAME_STEP}

    def __init__(self, num_envs=32, seed=7, repeat=6, max_steps=27000, port=11008):
        self._seed = seed
        self._bridge = Bridge(port=port)
        hello = self._bridge.request({
            "type": "hello",
            "seed": seed,
            "ships": num_envs,
            "repeat": repeat,
            "max_steps": max_steps
        })

        #checking everything the host echos back
        for key, asked in (("ships", num_envs), ("repeat", repeat), ("max_steps", max_steps)):
            if hello[key] != asked:
                raise RuntimeError(f"aksed for {key}={asked}, host built {hello[key]}")

        # the task describes itself; Python keepd no copy of these
        self.task = hello["task"]
        self.obs_names = hello["obs_names"]
        self.outcomes = {int(k): v for k, v in hello["outcomes"].items()}
        self.success = hello["success"]

        self.num_envs = num_envs
        self.single_observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(hello["obs_size"],), dtype=np.float32
        )
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(hello["act_size"],), dtype=np.float32)
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, num_envs)

    def reset(self, *, seed=None, options=None):
        if seed is not None and seed != self._seed:
            raise ValueError(f"the host was seeded with {self._seed} at hello; got reset(seed={seed})")
        reply = self._bridge.request({"type": "reset"})
        obs = np.asarray(reply["obs"], dtype=np.float32)
        # rule 2 of the protocol.md, checked on way in
        if not np.all(np.isfinite(obs)):
            raise RuntimeError("host sent a non-finite observation on reset")
        return obs, {}

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != self.action_space.shape:
            raise ValueError(f"actions have shape {actions.shape}, expected {self.action_space.shape}")

        # never senf NaN to the engine either
        if not np.all(np.isfinite(actions)):
            raise ValueError("actions contain NaN or infinity")
        reply = self._bridge.request({"type": "step", "actions": actions.tolist()})

        obs = np.asarray(reply["obs"], dtype=np.float32)
        rewards = np.asarray(reply["reward"], dtype=np.float32)
        # rule 2 on the way in
        if not (np.all(np.isfinite(obs)) and np.all(np.isfinite(rewards))):
            raise RuntimeError("host sent a non-finite observation or reward")
        terminations = np.asarray(reply["terminated"], dtype=bool)
        truncations = np.asarray(reply["truncated"], dtype=bool)
        ended = terminations | truncations

        final_obs = np.zeros_like(obs)
        for key, row in reply["final_obs"].items():
            final_obs[int(key)] = row
        received = np.zeros(self.num_envs, dtype=bool)
        received[[int(k) for k in reply["final_obs"]]] = True
        if not np.array_equal(received, ended):
            raise RuntimeError("host sent final_obs for a different set of ships than ended")

        infos = {
            "final_obs": final_obs,
            "_final_obs": ended,
            "outcome": np.asarray(reply["outcome"], dtype=np.int32),
        }
        return obs, rewards, terminations, truncations, infos

    def close_extras(self, **kwargs):
        try:
            self._bridge.request({"type": "close"})
        finally:
            self._bridge.close()
