"""Deterministic SB3 PPO MlpPolicy actor using NumPy, no Torch/SB3 at runtime.

SB3 PPO action distribution is unbounded diagonal Gaussian. For deterministic
predict(), the mean is chosen, then clipped to Box(-1,1). No tanh transform.
"""
from pathlib import Path
import numpy as np


class PPONumpyPolicy:
    def __init__(self, path):
        self.path = str(Path(path).expanduser())
        with np.load(self.path, allow_pickle=False) as w:
            self.w1 = w['mlp_extractor.policy_net.0.weight']
            self.b1 = w['mlp_extractor.policy_net.0.bias']
            self.w2 = w['mlp_extractor.policy_net.2.weight']
            self.b2 = w['mlp_extractor.policy_net.2.bias']
            self.w3 = w['action_net.weight']
            self.b3 = w['action_net.bias']
        assert self.w1.shape == (64, 49)
        assert self.w2.shape == (64, 64)
        assert self.w3.shape == (6, 64)

    def predict(self, observation):
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (49,) or not np.isfinite(obs).all():
            raise ValueError('PPO requires finite 49-dimensional observation')
        h1 = np.tanh(self.w1 @ obs + self.b1)
        h2 = np.tanh(self.w2 @ h1 + self.b2)
        mean = self.w3 @ h2 + self.b3
        # SB3 PPO deterministic action (Gaussian mean), Box action clipping.
        return np.clip(mean, -1.0, 1.0).astype(np.float64)
