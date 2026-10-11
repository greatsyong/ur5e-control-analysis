
#!/usr/bin/env python3
"""UR5e residual-torque RL environment.

Shared Gymnasium environment for PPO, SAC, and TD3.
Reuses the analytical UR5e dynamics and the accepted 3-s reference.

Plant mass and controller-model mass are intentionally independent.
No Isaac Sim or ROS 2 dependency is required for offline training.
"""

from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces

PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE = Path.home() / "robotics_ws"

MODEL_DIR = (
    WORKSPACE
    / "src/ur5e_control_analysis/part1_modeling_control_analysis"
)

REFERENCE = (
    Path.home()
    / "Documents/projects/12-rl-isaac-ur5e/references"
    / "inverse_dynamics_1kg_cube_3s_retimed_v2.csv"
)

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from optimization.pd_gain_search import (
    INITIAL_ERROR_RAD,
    COM_TOOL,
    BASE_TO_WORLD,
    TORQUE_LIMITS,
    load_reference,
    interp6,
)

from dynamics.payload_dynamics import (
    mass_matrix_payload,
    coriolis_centrifugal_vector_payload,
    gravity_vector_payload,
)

from kinematics.forward_kinematics import forward_kinematics


class UR5eResidualEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        model_mass=3.0,
        actual_mass=5.0,
        reference=REFERENCE,
        dt=0.002,
        control_dt=0.020,
    ):
        super().__init__()

        self.model_mass = float(model_mass)
        self.actual_mass = float(actual_mass)

        self.dt = float(dt)
        self.control_dt = float(control_dt)

        self.kp = 256.0
        self.kd = 41.6

        if self.model_mass <= 0 or self.actual_mass <= 0:
            raise ValueError("Payload masses must be positive.")

        self.substeps = round(self.control_dt / self.dt)

        if (
            self.dt <= 0
            or self.control_dt <= 0
            or abs(
                self.substeps * self.dt - self.control_dt
            ) > 1e-10
        ):
            raise ValueError(
                "control_dt must be an integer multiple of dt."
            )

        self.ts, self.qref, self.vref, self.aref = (
            load_reference(Path(reference))
        )

        self.duration = float(self.ts[-1])
        self.max_steps = round(
            self.duration / self.control_dt
        )

        if abs(
            self.max_steps * self.control_dt - self.duration
        ) > 1e-10:
            raise ValueError(
                "Duration must be divisible by control_dt."
            )

        self.residual_limits = np.array(
            [5.0, 5.0, 5.0, 0.5, 0.5, 0.01],
            dtype=np.float64,
        )

        # All algorithms receive the same normalized 6-D action.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        # q, qd, qref, qdref, position error,
        # velocity error, previous residual,
        # nominal PD torque, normalized time.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(49,),
            dtype=np.float32,
        )

        # Reference feedforward uses the MODEL mass.
        # Existing 1-kg CSV torque values are NOT reused.
        self.ffref = np.empty(
            (len(self.ts), 6),
            dtype=np.float64,
        )

        for i in range(len(self.ts)):
            qr = self.qref[i]
            vr = self.vref[i]
            ar = self.aref[i]

            self.ffref[i] = (
                self.mass_matrix(qr, self.model_mass) @ ar
                + self.coriolis(qr, vr, self.model_mass)
                + self.gravity(qr, self.model_mass)
            )

        self.q = None
        self.v = None
        self.t = 0.0
        self.step_count = 0
        self.prev_residual = np.zeros(6)
        self.last_pd = np.zeros(6)

        self.tool0 = self.tool_xyz(self.qref[0])

    def mass_matrix(self, q, mass):
        return np.asarray(
            mass_matrix_payload(
                q,
                payload_mass=mass,
                payload_com_tool=COM_TOOL,
            ),
            dtype=np.float64,
        )

    def coriolis(self, q, v, mass):
        return np.asarray(
            coriolis_centrifugal_vector_payload(
                q,
                v,
                payload_mass=mass,
                payload_com_tool=COM_TOOL,
            ),
            dtype=np.float64,
        )

    def gravity(self, q, mass):
        return np.asarray(
            gravity_vector_payload(
                q,
                payload_mass=mass,
                payload_com_tool=COM_TOOL,
            ),
            dtype=np.float64,
        )

    def tool_xyz(self, q):
        return (
            BASE_TO_WORLD
            @ forward_kinematics(q)[:3, 3]
        )

    def reference_at(self, t):
        qr = interp6(t, self.ts, self.qref)
        vr = interp6(t, self.ts, self.vref)
        ff = interp6(t, self.ts, self.ffref)
        return qr, vr, ff

    def nominal_pd(self, t):
        qr, vr, ff = self.reference_at(t)

        tau_fb = self.mass_matrix(
            self.q, self.model_mass
        ) @ (
            self.kp * (qr - self.q)
            + self.kd * (vr - self.v)
        )

        return ff + tau_fb

    def observation(self):
        qr, vr, _ = self.reference_at(self.t)

        obs = np.concatenate([
            self.q,
            self.v,
            qr,
            vr,
            qr - self.q,
            vr - self.v,
            self.prev_residual / self.residual_limits,
            self.last_pd / TORQUE_LIMITS,
            np.array([self.t / self.duration]),
        ])

        return obs.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.q = (
            self.qref[0].copy() + INITIAL_ERROR_RAD
        )

        self.v = self.vref[0].copy()
        self.t = 0.0
        self.step_count = 0

        self.prev_residual = np.zeros(6)
        self.last_pd = self.nominal_pd(0.0)

        return self.observation(), {
            "model_mass_kg": self.model_mass,
            "actual_mass_kg": self.actual_mass,
        }

    def step(self, action):
        action = np.asarray(
            action, dtype=np.float64
        ).reshape(6)

        if not np.isfinite(action).all():
            raise ValueError("Non-finite RL action.")

        action = np.clip(action, -1.0, 1.0)

        residual = action * self.residual_limits

        # Nominal computed-torque PD at control instant.
        tau_pd = self.nominal_pd(self.t)

        raw_torque = tau_pd + residual

        torque = np.clip(
            raw_torque,
            -TORQUE_LIMITS,
            TORQUE_LIMITS,
        )

        saturated = bool(
            np.any(
                np.abs(raw_torque)
                > TORQUE_LIMITS + 1e-10
            )
        )

        # Physical integration: 2 ms.
        # Control command: zero-order held for 20 ms.
        for _ in range(self.substeps):

            M = self.mass_matrix(
                self.q, self.actual_mass
            )

            c = self.coriolis(
                self.q, self.v, self.actual_mass
            )

            g = self.gravity(
                self.q, self.actual_mass
            )

            qdd = np.linalg.solve(
                M, torque - c - g
            )

            self.v = self.v + self.dt * qdd
            self.q = self.q + self.dt * self.v

            if (
                not np.isfinite(self.q).all()
                or not np.isfinite(self.v).all()
            ):
                raise RuntimeError(
                    "Non-finite plant state."
                )

        self.step_count += 1

        self.t = min(
            self.step_count * self.control_dt,
            self.duration,
        )

        qr, vr, _ = self.reference_at(self.t)

        tool_error_m = (
            self.tool_xyz(self.q)
            - self.tool_xyz(qr)
        )

        tool_error_mm = (
            1000.0 * np.linalg.norm(tool_error_m)
        )

        joint_error = qr - self.q

        # Initial reward design.
        # Cartesian accuracy is the primary objective.
        position_cost = (
            np.linalg.norm(tool_error_m) / 0.010
        ) ** 2

        joint_cost = np.mean(joint_error**2)

        residual_cost = np.mean(
            (residual / self.residual_limits) ** 2
        )

        change_cost = np.mean(
            (
                (residual - self.prev_residual)
                / self.residual_limits
            ) ** 2
        )

        reward = -(
            position_cost
            + 0.1 * joint_cost
            + 0.01 * residual_cost
            + 0.01 * change_cost
        )

        self.prev_residual = residual.copy()
        self.last_pd = tau_pd.copy()

        joint_error_deg = np.rad2deg(joint_error)

        failed = bool(
            np.max(np.abs(joint_error_deg)) > 30.0
            or np.max(np.abs(self.v)) > 20.0
        )

        completed = (
            self.step_count >= self.max_steps
        )

        # Gymnasium: terminated for failure,
        # truncated for the fixed 3-second horizon.
        terminated = failed or completed
        truncated = False

        info = {
            "t_s": self.t,
            "tool_error_mm": float(tool_error_mm),
            "tool_z_error_mm": float(
                1000.0 * tool_error_m[2]
            ),
            "max_joint_error_deg": float(
                np.max(np.abs(joint_error_deg))
            ),
            "nominal_pd_torque_Nm": tau_pd.copy(),
            "residual_torque_Nm": residual.copy(),
            "command_torque_Nm": torque.copy(),
            "saturated": saturated,
            "completed": bool(completed and not failed),
        }

        return (
            self.observation(),
            float(reward),
            terminated,
            truncated,
            info,
        )

