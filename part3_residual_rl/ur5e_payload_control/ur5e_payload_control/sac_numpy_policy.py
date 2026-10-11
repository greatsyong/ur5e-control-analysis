"""Deterministic SAC 49->256->256->6 actor (ReLU/ReLU/Tanh) in NumPy.

Stable-Baselines3 SAC deterministic prediction uses tanh of the mean output.
The stochastic log_std branch is NOT used in deterministic evaluation.
"""
from pathlib import Path
import numpy as np

class SACNumpyPolicy:
    def __init__(self, path):
        self.path = str(Path(path).expanduser())
        with np.load(self.path,allow_pickle=False) as w:
            self.w1=w['actor.latent_pi.0.weight']; self.b1=w['actor.latent_pi.0.bias']
            self.w2=w['actor.latent_pi.2.weight']; self.b2=w['actor.latent_pi.2.bias']
            self.w3=w['actor.mu.weight']; self.b3=w['actor.mu.bias']
        if self.w1.shape!=(256,49) or self.w2.shape!=(256,256) or self.w3.shape!=(6,256):
            raise ValueError('Unexpected SAC actor sizes')

    def predict(self, observation):
        obs=np.asarray(observation,dtype=np.float32)
        if obs.shape!=(49,) or not np.isfinite(obs).all():
            raise ValueError('SAC requires finite 49-dimensional observation')
        x=np.maximum(self.w1@obs+self.b1,0)
        x=np.maximum(self.w2@x+self.b2,0)
        return np.tanh(self.w3@x+self.b3).astype(np.float64)
