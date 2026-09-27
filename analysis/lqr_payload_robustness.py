import time
import numpy as np

from scipy.signal import cont2discrete

from controllers.lqr import DiscreteJointLQRController
from dynamics.state_space import (
    equilibrium_input,
    linearize_dynamics,
)
from dynamics.payload_dynamics import (
    state_derivative_payload as state_derivative,
)
from trajectories.joint_trajectory import quintic_joint_trajectory
from analysis.controller_metrics import summarize_tcp_errors


# ============================================================
# Final nominal LQR configuration
# ============================================================

DT = 0.002
DURATION = 4.0
RHO_U = 100.0

TORQUE_LIMITS = np.array([
    150.0, 150.0, 150.0,
    28.0, 28.0, 28.0,
])

q_start = np.deg2rad([
    0.0, -90.0, 90.0,
    -90.0, -90.0, 0.0,
])

q_goal = np.deg2rad([
    20.0, -60.0, 60.0,
    -70.0, -70.0, 20.0,
])


# ============================================================
# Fixed equilibrium linearization
# ============================================================

x_eq = np.concatenate([
    q_start,
    np.zeros(6),
])

tau_eq = equilibrium_input(q_start)

A, B = linearize_dynamics(
    x_eq,
    tau_eq,
)


# ============================================================
# Exact ZOH discretization
# ============================================================

C_dummy = np.eye(12)
D_dummy = np.zeros((12, 6))

Ad, Bd, _, _, _ = cont2discrete(
    (A, B, C_dummy, D_dummy),
    DT,
    method="zoh",
)


# ============================================================
# Bryson-normalized LQR weights
# ============================================================

q_allow = np.deg2rad(
    np.full(6, 2.0)
)

qdot_allow = np.deg2rad(
    np.full(6, 10.0)
)

Q = np.diag(
    np.concatenate([
        1.0 / q_allow**2,
        1.0 / qdot_allow**2,
    ])
)

R_bryson = np.diag(
    1.0 / TORQUE_LIMITS**2
)

R = RHO_U * R_bryson


# ============================================================
# Controller design
# ============================================================

controller = DiscreteJointLQRController(
    Ad=Ad,
    Bd=Bd,
    Q=Q,
    R=R,
    torque_limits=TORQUE_LIMITS,
)

eig_d = np.linalg.eigvals(
    Ad - Bd @ controller.K
)

spectral_radius = np.max(
    np.abs(eig_d)
)


# ============================================================
# Nonlinear nominal benchmark
# ============================================================

times = np.arange(
    0.0,
    DURATION + DT,
    DT,
)

x = np.concatenate([
    q_start,
    np.zeros(6),
])

q_history = []
q_ref_history = []
tau_history = []
compute_time_history = []

for t in times:

    q = x[:6]
    qdot = x[6:]

    q_ref, qdot_ref, _ = quintic_joint_trajectory(
        t,
        DURATION,
        q_start,
        q_goal,
    )

    # --------------------------------------------------------
    # Measure online controller computation only:
    #
    # 1. reference-dependent gravity feedforward
    # 2. LQR state feedback
    #
    # Offline DARE / gain calculation is excluded.
    # --------------------------------------------------------

    tic = time.perf_counter()

    tau_ff = equilibrium_input(
        q_ref
    )

    tau = controller.compute(
        q=q,
        qdot=qdot,
        q_ref=q_ref,
        qdot_ref=qdot_ref,
        tau_ff=tau_ff,
    )

    toc = time.perf_counter()

    compute_time_history.append(
        toc - tic
    )

    q_history.append(
        q.copy()
    )

    q_ref_history.append(
        q_ref.copy()
    )

    tau_history.append(
        tau.copy()
    )

    if t >= DURATION:
        continue

    # RK4 nonlinear plant with ZOH control input
    k1 = state_derivative(
        x,
        tau,
    )

    k2 = state_derivative(
        x + 0.5 * DT * k1,
        tau,
    )

    k3 = state_derivative(
        x + 0.5 * DT * k2,
        tau,
    )

    k4 = state_derivative(
        x + DT * k3,
        tau,
    )

    x = x + (
        DT / 6.0
    ) * (
        k1
        + 2.0 * k2
        + 2.0 * k3
        + k4
    )


# ============================================================
# Metrics
# ============================================================

q_history = np.asarray(
    q_history
)

q_ref_history = np.asarray(
    q_ref_history
)

tau_history = np.asarray(
    tau_history
)

compute_time_history = np.asarray(
    compute_time_history
)

error = (
    q_ref_history
    - q_history
)

joint_rmse = np.sqrt(
    np.mean(
        error**2,
        axis=0,
    )
)

joint_max = np.max(
    np.abs(error),
    axis=0,
)

joint_final = error[-1]

overall_rmse = np.sqrt(
    np.mean(
        error**2
    )
)

tcp = summarize_tcp_errors(
    q_ref_history,
    q_history,
)

max_torque = np.max(
    np.abs(tau_history),
    axis=0,
)

rms_torque = np.sqrt(
    np.mean(
        tau_history**2,
        axis=0,
    )
)

torque_utilization = (
    100.0
    * max_torque
    / TORQUE_LIMITS
)

saturation_count = np.sum(
    np.abs(tau_history)
    >= 0.999 * TORQUE_LIMITS,
    axis=0,
)

compute_ms = (
    compute_time_history
    * 1000.0
)


# ============================================================
# Results
# ============================================================

print("\n" + "=" * 88)
print("FINAL NOMINAL DISCRETE LQR BENCHMARK")
print("=" * 88)

print(
    f"\nSampling period       : {DT:.6f} s"
)

print(
    f"Sampling rate         : {1.0 / DT:.1f} Hz"
)

print(
    f"rho_u                 : {RHO_U:.1f}"
)

print(
    f"Closed-loop radius    : {spectral_radius:.10f}"
)

print("\nJoint RMSE [deg]:")
print(
    np.rad2deg(
        joint_rmse
    )
)

print("\nJoint maximum error [deg]:")
print(
    np.rad2deg(
        joint_max
    )
)

print("\nJoint final error [deg]:")
print(
    np.rad2deg(
        joint_final
    )
)

print(
    "\nOverall joint RMSE [deg]: "
    f"{np.rad2deg(overall_rmse):.6f}"
)

print(
    "\nTCP position RMSE / max / final [mm]:"
)

print(
    f"{tcp['tcp_position_rmse_mm']:.6f} / "
    f"{tcp['tcp_position_max_mm']:.6f} / "
    f"{tcp['tcp_position_final_mm']:.6f}"
)

print(
    "\nTCP orientation RMSE / max / final [deg]:"
)

print(
    f"{tcp['tcp_orientation_rmse_deg']:.6f} / "
    f"{tcp['tcp_orientation_max_deg']:.6f} / "
    f"{tcp['tcp_orientation_final_deg']:.6f}"
)

print("\nMaximum torque [Nm]:")
print(
    max_torque
)

print("\nRMS torque [Nm]:")
print(
    rms_torque
)

print("\nTorque utilization [%]:")
print(
    torque_utilization
)

print("\nSaturation sample count:")
print(
    saturation_count
)

print("\nController computation time [ms]:")

print(
    f"Mean : {np.mean(compute_ms):.6f}"
)

print(
    f"Max  : {np.max(compute_ms):.6f}"
)

print(
    f"P99  : {np.percentile(compute_ms, 99):.6f}"
)

print(
    "\nFraction of 2 ms control period [%]:"
)

print(
    f"Mean : "
    f"{100.0 * np.mean(compute_time_history) / DT:.3f}"
)

print(
    f"Max  : "
    f"{100.0 * np.max(compute_time_history) / DT:.3f}"
)


# ============================================================
# Save final nominal result
# ============================================================

import os

results_dir = (
    "results/lqr/nominal"
)

os.makedirs(
    results_dir,
    exist_ok=True,
)

np.savez(
    f"{results_dir}/"
    "lqr_nominal_benchmark.npz",

    time=times,
    q=q_history,
    q_ref=q_ref_history,
    tau=tau_history,

    K=controller.K,
    Q=Q,
    R=R,

    joint_rmse=joint_rmse,
    joint_max_error=joint_max,
    joint_final_error=joint_final,
    overall_joint_rmse=overall_rmse,

    max_torque=max_torque,
    rms_torque=rms_torque,
    torque_utilization=torque_utilization,
    saturation_count=saturation_count,

    compute_time=compute_time_history,

    spectral_radius=spectral_radius,
)

print(
    "\nSaved results to:",
    results_dir,
)