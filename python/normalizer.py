"""Observation normalizer that applies exactly what rl/neural_pilot.gd applies

The policy trains on normalized observations, so these statistics are as much a part of it as the weights.
Two rules have to match NeuralPilot or the in-engine policy flies a different
input than it trained on:
  - a channel whose std is at most CONSTANT_STD counts as constant and is left
    at unit scale instead of divided by its tiny std;
  - the result is clamped to +-CLIP.
Arithmetic is float32, as in the engine, so the two agree to ~1e-6.
"""
import numpy as np

CONSTANT_STD = 1e-3 # NeuralPilot.CONSTANT_STD
CLIP = 10.0 # the clamp in NeuralPilot.act

class ObsNormalizer:
    def __init__(self, size):
        self.mean = np.zeros(size, dtype=np.float64)
        self.var = np.ones(size, dtype=np.float64)
        self.count = 0
        self.frozen = False

    def update(self, batch):
        """Fold a (num_envs, size) batch into the running mean and variance."""
        if self.frozen:
            return
        batch = np.asarray(batch, dtype=np.float64)
        n = batch.shape[0]
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        if self.count == 0:
            self.mean, self.var, self.count = batch_mean, batch_var, n
            return
        total = self.count + n
        delta = batch_mean - self.mean
        m2 = self.var * self.count + batch_var * n + delta**2 * self.count * n / total
        self.mean = self.mean + delta * n / total
        self.var = m2 / total
        self.count = total

    @property
    def std(self):
        return np.sqrt(self.var)

    def __call__(self, obs):
        """Normalize observations the way NeuralPilot.act does."""
        mean = self.mean.astype(np.float32)
        std = self.std.astype(np.float32)
        inverse_std = np.ones_like(std)
        varying = std > CONSTANT_STD
        inverse_std[varying] = np.float32(1.0) / std[varying]
        normalized = (np.asarray(obs, dtype=np.float32) - mean) * inverse_std
        return np.clip(normalized, -CLIP, CLIP)

    def state(self):
        """What gets saved beside the weights: mean and std, as NeuralPilot reads them."""
        return {"obs_mean": self.mean.tolist(), "obs_std": self.std.tolist(), "count": self.count}

    
