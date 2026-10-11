
#!/usr/bin/env python3
"""UR5e Residual RL: PPO / SAC / TD3 benchmark.

Shared conditions:
- G025 computed-torque PD
- UR5e analytical dynamics
- Identical reference, observation, action, and reward
- 30,000 environment steps by default
- Evaluation and checkpoint every 3,000 steps

Training is offline Python only. Isaac Sim / ROS2 are reserved
for later physical-simulation validation.
"""

import argparse
import csv
import json
import time

from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from stable_baselines3 import PPO, SAC, TD3

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from ur5e_residual_env import UR5eResidualEnv


BASELINE_RMSE_MM = 10.6358
EVAL_INTERVAL = 3000


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algo",
        choices=["ppo", "sac", "td3"],
        required=True,
    )

    parser.add_argument(
        "--model-mass",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--actual-mass",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=30000,
    )

    parser.add_argument(
        "--eval-interval",
        type=int,
        default=EVAL_INTERVAL,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/rl"),
    )

    return parser.parse_args()


def make_env(model_mass, actual_mass):
    return UR5eResidualEnv(
        model_mass=model_mass,
        actual_mass=actual_mass,
    )


def evaluate_policy(model, env, seed, save_path=None):
    """Evaluate a deterministic policy for one complete episode."""

    obs, _ = env.reset(seed=seed)

    records = []
    terminated = False
    truncated = False

    while not (terminated or truncated):

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        row = {
            "t_s": float(info["t_s"]),
            "tool_error_mm": float(info["tool_error_mm"]),
            "tool_z_error_mm": float(
                info["tool_z_error_mm"]
            ),
            "reward": float(reward),
            "saturated": int(info["saturated"]),
        }

        for j in range(6):

            row[f"residual_tau{j+1}_Nm"] = float(
                info["residual_torque_Nm"][j]
            )

            row[f"command_tau{j+1}_Nm"] = float(
                info["command_torque_Nm"][j]
            )

        records.append(row)

    errors = np.array(
        [r["tool_error_mm"] for r in records],
        dtype=np.float64,
    )

    residuals = np.array(
        [
            [
                r[f"residual_tau{j+1}_Nm"]
                for j in range(6)
            ]
            for r in records
        ],
        dtype=np.float64,
    )

    completed = bool(info["completed"])

    # A partial episode cannot be ranked against a full
    # three-second trajectory.
    rmse = (
        float(np.sqrt(np.mean(errors**2)))
        if completed
        else None
    )

    result = {
        "completed": completed,
        "control_steps": len(records),
        "final_time_s": float(info["t_s"]),
        "tool_rmse_mm": rmse,
        "partial_rmse_mm": float(
            np.sqrt(np.mean(errors**2))
        ),
        "max_tool_error_mm": float(np.max(errors)),
        "final_tool_error_mm": float(errors[-1]),
        "final_z_error_mm": float(
            records[-1]["tool_z_error_mm"]
        ),
        "residual_rms_Nm": float(
            np.sqrt(np.mean(residuals**2))
        ),
        "residual_peak_Nm": float(
            np.max(np.abs(residuals))
        ),
        "saturation_steps": int(
            sum(r["saturated"] for r in records)
        ),
    }

    if completed:
        result["improvement_pct"] = float(
            100.0
            * (
                BASELINE_RMSE_MM - rmse
            )
            / BASELINE_RMSE_MM
        )
    else:
        result["improvement_pct"] = None

    if save_path is not None:
        with Path(save_path).open("w", newline="") as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(records[0].keys()),
            )

            writer.writeheader()
            writer.writerows(records)

    return result


class BenchmarkCallback(BaseCallback):

    def __init__(
        self,
        model_mass,
        actual_mass,
        seed,
        run_dir,
        eval_interval,
    ):
        super().__init__(verbose=0)

        self.eval_env = make_env(
            model_mass,
            actual_mass,
        )

        self.seed = int(seed)
        self.run_dir = Path(run_dir)
        self.eval_interval = int(eval_interval)

        self.checkpoint_dir = (
            self.run_dir / "checkpoints"
        )
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.history = []
        self.best_rmse = float("inf")
        self.best_step = None

        self.wall_start = None
        self.last_eval_step = 0

    def _on_training_start(self):
        self.wall_start = time.perf_counter()

    def _write_history(self):

        if not self.history:
            return

        path = self.run_dir / "learning_curve.csv"

        with path.open("w", newline="") as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    self.history[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(self.history)

    def run_evaluation(self, step):

        result = evaluate_policy(
            self.model,
            self.eval_env,
            seed=self.seed + 1000,
            save_path=(
                self.run_dir
                / f"evaluation_step_{step:06d}.csv"
            ),
        )

        elapsed = (
            time.perf_counter() - self.wall_start
        )

        record = {
            "training_steps": int(step),
            "elapsed_s": float(elapsed),
            **result,
        }

        self.history.append(record)
        self._write_history()

        self.model.save(
            str(
                self.checkpoint_dir
                / f"policy_step_{step:06d}"
            )
        )

        if (
            result["completed"]
            and result["tool_rmse_mm"] < self.best_rmse
        ):

            self.best_rmse = result["tool_rmse_mm"]
            self.best_step = int(step)

            self.model.save(
                str(self.run_dir / "best_policy")
            )

        rmse_text = (
            f'{result["tool_rmse_mm"]:.4f} mm'
            if result["completed"]
            else "INCOMPLETE"
        )

        print(
            f"\n[EVALUATION] "
            f"Step={step} "
            f"RMSE={rmse_text} "
            f"Complete={result['completed']} "
            f"ResidualPeak="
            f"{result['residual_peak_Nm']:.3f} Nm "
            f"Elapsed={elapsed/60:.1f} min",
            flush=True,
        )

        self.last_eval_step = int(step)

        return result

    def _on_step(self):

        step = int(self.num_timesteps)

        if (
            step > 0
            and step % self.eval_interval == 0
            and step != self.last_eval_step
        ):
            self.run_evaluation(step)

        return True

    def close(self):
        self.eval_env.close()


def make_model(algo, env, seed):

    common = {
        "policy": "MlpPolicy",
        "env": env,
        "gamma": 0.99,
        "seed": seed,
        "device": "cpu",
        "verbose": 1,
    }

    if algo == "ppo":

        return PPO(
            **common,
            learning_rate=3e-4,
            n_steps=300,
            batch_size=60,
            n_epochs=5,
        )

    if algo == "sac":

        return SAC(
            **common,
            learning_rate=3e-4,
            buffer_size=30000,
            learning_starts=600,
            batch_size=256,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
        )

    if algo == "td3":

        noise = NormalActionNoise(
            mean=np.zeros(6),
            sigma=0.1 * np.ones(6),
        )

        return TD3(
            **common,
            learning_rate=1e-3,
            buffer_size=30000,
            learning_starts=600,
            batch_size=256,
            train_freq=1,
            gradient_steps=1,
            policy_delay=2,
            action_noise=noise,
        )

    raise ValueError(f"Unknown algorithm: {algo}")


def main():

    args = parse_args()

    if args.steps <= 0:
        raise ValueError("Training steps must be positive.")

    if args.eval_interval <= 0:
        raise ValueError("Evaluation interval must be positive.")

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    run_dir = (
        args.output
        / (
            f"{args.algo.upper()}_"
            f"model{args.model_mass:g}_"
            f"actual{args.actual_mass:g}_"
            f"seed{args.seed}_"
            f"{timestamp}"
        )
    )

    run_dir.mkdir(parents=True)

    # Avoid excessive CPU thread contention for small MLPs.
    torch.set_num_threads(1)

    print("=" * 70, flush=True)
    print("UR5e RESIDUAL RL BENCHMARK", flush=True)
    print("=" * 70, flush=True)

    print("Algorithm:", args.algo.upper(), flush=True)
    print("Model mass:", args.model_mass, flush=True)
    print("Actual mass:", args.actual_mass, flush=True)
    print("Training steps:", args.steps, flush=True)
    print("Evaluation interval:", args.eval_interval, flush=True)
    print("Seed:", args.seed, flush=True)
    print("Network device: CPU", flush=True)
    print("CUDA available:", torch.cuda.is_available(), flush=True)
    print("Output:", run_dir, flush=True)

    env = Monitor(
        make_env(
            args.model_mass,
            args.actual_mass,
        ),
        filename=str(run_dir / "monitor.csv"),
    )

    callback = BenchmarkCallback(
        model_mass=args.model_mass,
        actual_mass=args.actual_mass,
        seed=args.seed,
        run_dir=run_dir,
        eval_interval=args.eval_interval,
    )

    model = make_model(
        args.algo,
        env,
        args.seed,
    )

    start = time.perf_counter()

    try:

        model.learn(
            total_timesteps=args.steps,
            callback=callback,
            progress_bar=False,
        )

        total_elapsed = (
            time.perf_counter() - start
        )

        actual_steps = int(model.num_timesteps)

        # Save the policy at the end of training.
        model.save(
            str(run_dir / "final_policy")
        )

        # Evaluate at the final step if it was not
        # already evaluated by the callback.
        if callback.last_eval_step != actual_steps:
            final_result = callback.run_evaluation(
                actual_steps
            )
        else:
            final_result = {
                key: value
                for key, value
                in callback.history[-1].items()
                if key not in (
                    "training_steps",
                    "elapsed_s",
                )
            }

        summary = {
            "algorithm": args.algo,
            "model_mass_kg": args.model_mass,
            "actual_mass_kg": args.actual_mass,
            "seed": args.seed,
            "requested_steps": args.steps,
            "actual_training_steps": actual_steps,
            "elapsed_s": total_elapsed,
            "environment": {
                "reference_duration_s": 3.0,
                "physics_dt_s": 0.002,
                "control_dt_s": 0.020,
                "controller": "G025",
                "kp": 256.0,
                "kd": 41.6,
                "baseline_rmse_mm": BASELINE_RMSE_MM,
            },
            "best_step": callback.best_step,
            "best_rmse_mm": (
                callback.best_rmse
                if callback.best_step is not None
                else None
            ),
            "final_evaluation": final_result,
        }

        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )

        print("\n" + "=" * 70, flush=True)
        print("BENCHMARK COMPLETED", flush=True)
        print("=" * 70, flush=True)

        print(
            "Algorithm:",
            args.algo.upper(),
            flush=True,
        )

        print(
            "Training steps:",
            actual_steps,
            flush=True,
        )

        print(
            "Elapsed:",
            f"{total_elapsed/60:.2f} minutes",
            flush=True,
        )

        print(
            "Best step:",
            callback.best_step,
            flush=True,
        )

        print(
            "Best RMSE:",
            (
                f"{callback.best_rmse:.4f} mm"
                if callback.best_step is not None
                else "No completed evaluation"
            ),
            flush=True,
        )

        print(
            "Final RMSE:",
            final_result["tool_rmse_mm"],
            flush=True,
        )

        print(
            "Results:",
            run_dir,
            flush=True,
        )

    finally:
        callback.close()
        env.close()


if __name__ == "__main__":
    main()

