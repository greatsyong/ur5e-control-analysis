"""NumPy-only TD3 actor inference for ROS 2; no torch/SB3 dependency."""
from pathlib import Path
import numpy as np


class TD3NumpyPolicy:
    def __init__(self, policy_path):
        path = Path(policy_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'TD3 exported policy not found: {path}')
        with np.load(path, allow_pickle=False) as data:
            if int(data['input_dim']) != 49 or int(data['output_dim']) != 6:
                raise ValueError('TD3 policy dimensions must be 49 -> 6')
            self.residual_limits = data['residual_limits'].astype(np.float64)
            if not np.allclose(self.residual_limits, [5, 5, 5, .5, .5, .01]):
                raise ValueError('TD3 residual scale differs from training')
            self.model_mass = float(data['model_mass_kg'])
            self.actual_mass = float(data['actual_mass_kg'])
            if self.model_mass != 3.0 or self.actual_mass != 5.0:
                raise ValueError('TD3 policy masses do not match experiment')
            architecture = str(data['architecture']).split(',')
            self.layers = []
            idx = 0
            for kind in architecture:
                if kind == 'linear':
                    w = data[f'w{idx}'].astype(np.float32)
                    b = data[f'b{idx}'].astype(np.float32)
                    if w.ndim != 2 or b.shape != (w.shape[0],):
                        raise ValueError('Invalid TD3 network weight dimensions')
                    self.layers.append((kind, w, b))
                    idx += 1
                elif kind in ('relu', 'tanh'):
                    self.layers.append((kind, None, None))
                else:
                    raise ValueError('Unexpected layer ' + kind)
            if self.layers[-1][0] != 'tanh':
                raise ValueError('TD3 action must terminate with tanh')
        self.path = path

    def predict(self, observation):
        x = np.asarray(observation, dtype=np.float32)
        if x.shape != (49,) or not np.isfinite(x).all():
            raise ValueError('TD3 observation must be finite 49-D vector')
        for kind, w, b in self.layers:
            if kind == 'linear':
                x = w @ x + b
            elif kind == 'relu':
                x = np.maximum(x, 0)
            elif kind == 'tanh':
                x = np.tanh(x)
        if x.shape != (6,) or not np.isfinite(x).all():
            raise ValueError('TD3 produced invalid action')
        return np.clip(x.astype(np.float64), -1., 1.)
