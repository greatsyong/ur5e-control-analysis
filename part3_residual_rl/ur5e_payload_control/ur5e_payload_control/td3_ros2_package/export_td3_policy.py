#!/usr/bin/env python3
"""Export SB3 TD3 MLP to dependency-light NumPy for ROS 2 inference.

Run inside the RL .venv, not the ROS 2 environment.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import TD3


def forward_numpy(obs, layers):
    x = np.asarray(obs, dtype=np.float32)
    for kind, w, b in layers:
        if kind == 'linear':
            x = x @ w.T + b
        elif kind == 'relu':
            x = np.maximum(x, 0)
        elif kind == 'tanh':
            x = np.tanh(x)
        else:
            raise ValueError(kind)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, required=True, help='TD3 best_policy.zip')
    ap.add_argument('--output', type=Path, required=True, help='Exported .npz path')
    args = ap.parse_args()
    model = TD3.load(str(args.model), device='cpu')
    modules = list(model.actor.mu.children())
    layers = []
    archive = {}
    kinds = []
    linear_idx = 0
    for layer in modules:
        if isinstance(layer, torch.nn.Linear):
            w = layer.weight.detach().cpu().numpy().astype(np.float32)
            b = layer.bias.detach().cpu().numpy().astype(np.float32)
            layers.append(('linear', w, b))
            archive[f'w{linear_idx}'] = w
            archive[f'b{linear_idx}'] = b
            linear_idx += 1
            kinds.append('linear')
        elif isinstance(layer, torch.nn.ReLU):
            layers.append(('relu', None, None))
            kinds.append('relu')
        elif isinstance(layer, torch.nn.Tanh):
            layers.append(('tanh', None, None))
            kinds.append('tanh')
        else:
            raise ValueError(f'Unsupported TD3 network layer: {layer}')
    if not layers or layers[0][1].shape[1] != 49 or layers[-1][1] is not None:
        raise ValueError('Expected a 49-dimensional actor with final Tanh')
    if kinds[-1] != 'tanh' or archive[f'w{linear_idx-1}'].shape[0] != 6:
        raise ValueError('Expected 6-dimensional action bounded by Tanh')
    rng = np.random.default_rng(42)
    test_obs = np.vstack([np.zeros((1, 49), np.float32), rng.normal(0, 0.5, (16, 49)).astype(np.float32)])
    predicted, _ = model.predict(test_obs, deterministic=True)
    reproduced = forward_numpy(test_obs, layers)
    maxdiff = float(np.max(np.abs(predicted - reproduced)))
    if maxdiff > 2e-5:
        raise RuntimeError(f'NumPy inference mismatch vs SB3: max |delta|={maxdiff}')
    archive['architecture'] = np.array(','.join(kinds))
    archive['input_dim'] = np.int64(49)
    archive['output_dim'] = np.int64(6)
    archive['residual_limits'] = np.array([5.0, 5.0, 5.0, 0.5, 0.5, 0.01], dtype=np.float64)
    archive['model_mass_kg'] = np.float64(3.0)
    archive['actual_mass_kg'] = np.float64(5.0)
    archive['export_check_max_abs_error'] = np.float64(maxdiff)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **archive)
    print('Verified TD3 actor conversion. Max action difference:', maxdiff)
    print('Saved:', args.output)
    print('Architecture:', archive['architecture'])


if __name__ == '__main__':
    main()
