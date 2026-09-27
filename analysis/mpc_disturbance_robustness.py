import os
import time
import numpy as np
import scipy.sparse as sp
import osqp

from scipy.signal import cont2discrete

from dynamics.state_space import (
    state_derivative,
    equilibrium_input,
    linearize_dynamics,
)
from dynamics.disturbance_dynamics import state_derivative_disturbed
from trajectories.joint_trajectory import quintic_joint_trajectory
from analysis.controller_metrics import summarize_tcp_errors
from controllers.mpc import build_prediction_matrices


# ============================================================
# Final nominal MPC configuration
# ============================================================

DT = 0.002
DURATION = 4.0

NP = 160
NC = 40
RHO_U = 10.0

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

tau_eq = equilibrium_input(
    q_start
)

A, B = linearize_dynamics(
    x_eq,
    tau_eq,
)


# ============================================================
# Exact ZOH discretization
# ============================================================

Ad, Bd, _, _, _ = cont2discrete(
    (
        A,
        B,
        np.eye(12),
        np.zeros((12, 6)),
    ),
    DT,
    method="zoh",
)

nx = Ad.shape[0]
nu = Bd.shape[1]


# ============================================================
# Prediction matrices
#
# z = x - x_eq
# v = tau - tau_eq
#
# Z = F z_k + G V
# ============================================================

F, G = build_prediction_matrices(
    Ad,
    Bd,
    NP,
)


# ============================================================
# Control-horizon mapping
#
# W_full = S Wc
#
# First NC correction inputs are independent.
# After NC, the final correction is held constant.
# ============================================================

def build_control_horizon_matrix(
    Np,
    Nc,
    nu,
):
    if Nc <= 0:
        raise ValueError(
            "Nc must be positive."
        )

    if Nc > Np:
        raise ValueError(
            "Nc cannot exceed Np."
        )

    S = np.zeros(
        (
            Np * nu,
            Nc * nu,
        )
    )

    Iu = np.eye(nu)

    for k in range(Np):

        if k < Nc:
            j = k
        else:
            j = Nc - 1

        row = slice(
            k * nu,
            (k + 1) * nu,
        )

        col = slice(
            j * nu,
            (j + 1) * nu,
        )

        S[row, col] = Iu

    return S


S = build_control_horizon_matrix(
    NP,
    NC,
    nu,
)

Gc = G @ S


# ============================================================
# Bryson-normalized MPC weights
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

Qbar = np.kron(
    np.eye(NP),
    Q,
)

Rbar_full = np.kron(
    np.eye(NP),
    R,
)


# ============================================================
# Reduced Hessian
#
# Cost:
#
# (Z - Zref)' Qbar (Z - Zref)
# +
# W_full' Rbar W_full
#
# W_full = S Wc
# ============================================================

H = (
    Gc.T
    @ Qbar
    @ Gc
    +
    S.T
    @ Rbar_full
    @ S
)

P = sp.csc_matrix(
    2.0 * H
)

# Fixed QP gradient map: precompute offline
GRADIENT_MAP = (
    2.0
    * Gc.T
    @ Qbar
)

# Fixed online state-to-gradient map.
STATE_GRADIENT_MAP = (
    GRADIENT_MAP
    @ F
)



# ============================================================
# OSQP setup
#
# Bounds are updated online because gravity feedforward
# changes along the reference trajectory.
# ============================================================

n_decision = NC * nu

A_constraint = sp.eye(
    n_decision,
    format="csc",
)

solver = osqp.OSQP()

solver.setup(
    P=P,
    q=np.zeros(n_decision),
    A=A_constraint,
    l=np.full(n_decision, -np.inf),
    u=np.full(n_decision, np.inf),
    verbose=False,
    polishing=False,
    eps_abs=1e-6,
    eps_rel=1e-6,
    warm_starting=True,
)


# ============================================================
# Precompute nominal reference
#
# The reference trajectory itself is supplied to the
# controller, just as q_ref/qdot_ref are supplied to LQR.
#
# Gravity feedforward is precomputed here for the known
# nominal trajectory. Solver timing is measured separately
# below.
# ============================================================

times = np.arange(
    0.0,
    DURATION + DT,
    DT,
)

n_samples = len(times)

q_ref_all = np.zeros(
    (n_samples, 6)
)

qdot_ref_all = np.zeros(
    (n_samples, 6)
)

tau_ff_all = np.zeros(
    (n_samples, 6)
)

for k, t in enumerate(times):

    q_ref, qdot_ref, _ = quintic_joint_trajectory(
        t,
        DURATION,
        q_start,
        q_goal,
    )

    q_ref_all[k] = q_ref
    qdot_ref_all[k] = qdot_ref

    tau_ff_all[k] = equilibrium_input(
        q_ref
    )


# ============================================================
# Vectorized prediction-horizon reference
# ============================================================

x_ref_all = np.hstack([
    q_ref_all,
    qdot_ref_all,
])

x_ref_padded = np.vstack([
    x_ref_all,
    np.repeat(
        x_ref_all[-1][None, :],
        NP,
        axis=0,
    ),
])

tau_ff_padded = np.vstack([
    tau_ff_all,
    np.repeat(
        tau_ff_all[-1][None, :],
        NP,
        axis=0,
    ),
])


def build_reference_horizon(k):

    Z_ref_matrix = (
        x_ref_padded[
            k + 1:k + 1 + NP
        ]
        - x_eq
    )

    tau_ff_horizon = (
        tau_ff_padded[
            k:k + NP
        ]
    )

    return (
        Z_ref_matrix.reshape(-1),
        tau_ff_horizon,
    )


# ============================================================
# Offline reference-dependent QP gradient
#
# q_qp(k) = STATE_GRADIENT_MAP @ z_k
#           + reference_gradient_all[k]
# ============================================================

reference_gradient_all = np.zeros(
    (n_samples, n_decision)
)

for k in range(n_samples):
    Z_ref_k, tau_ff_horizon_k = (
        build_reference_horizon(k)
    )

    V_ff_k = (
        tau_ff_horizon_k
        - tau_eq.reshape(1, nu)
    ).reshape(-1)

    reference_gradient_all[k] = (
        GRADIENT_MAP
        @ (
            G @ V_ff_k
            - Z_ref_k
        )
    )


# ============================================================
# Helper: correction-input bounds
#
# tau = tau_ff + w
#
# Therefore:
#
# -tau_max - tau_ff <= w
# w <= tau_max - tau_ff
#
# For the final control move, w_{NC-1} is held from
# NC-1 through NP-1. Its admissible interval must therefore
# satisfy every torque bound over that tail.
# ============================================================

def build_reduced_bounds(
    tau_ff_horizon
):

    lower_full = (
        -TORQUE_LIMITS.reshape(1, nu)
        - tau_ff_horizon
    )

    upper_full = (
        TORQUE_LIMITS.reshape(1, nu)
        - tau_ff_horizon
    )

    lower_reduced = np.zeros(
        (NC, nu)
    )

    upper_reduced = np.zeros(
        (NC, nu)
    )

    # First NC-1 moves are independent
    lower_reduced[:NC - 1] = (
        lower_full[:NC - 1]
    )

    upper_reduced[:NC - 1] = (
        upper_full[:NC - 1]
    )

    # Final correction is held over the remaining horizon.
    # It must satisfy all future bounds.
    lower_reduced[NC - 1] = np.max(
        lower_full[NC - 1:],
        axis=0,
    )

    upper_reduced[NC - 1] = np.min(
        upper_full[NC - 1:],
        axis=0,
    )

    if np.any(
        lower_reduced
        > upper_reduced
    ):
        raise RuntimeError(
            "Infeasible reduced MPC input bounds."
        )

    return (
        lower_reduced.reshape(-1),
        upper_reduced.reshape(-1),
    )


# ============================================================
# Nonlinear nominal benchmark
# ============================================================

x = np.concatenate([
    q_start,
    np.zeros(6),
])

q_history = []
qdot_history = []
q_ref_history = []
tau_history = []

compute_time_history = []
osqp_iteration_history = []

constraint_active_history = []


for k, t in enumerate(times):

    q = x[:6]
    qdot = x[6:]

    q_ref = q_ref_all[k]
    qdot_ref = qdot_ref_all[k]

    # --------------------------------------------------------
    # Build current MPC problem
    # --------------------------------------------------------

    tic = time.perf_counter()

    # Reference-dependent prediction/gradient work is offline.
    # Online gradient now depends only on the measured state.
    tau_ff_horizon = (
        tau_ff_padded[
            k:k + NP
        ]
    )

    z = (
        x
        - x_eq
    )

    q_qp = (
        STATE_GRADIENT_MAP
        @ z
        + reference_gradient_all[k]
    )

    lower, upper = (
        build_reduced_bounds(
            tau_ff_horizon
        )
    )

    solver.update(
        q=q_qp,
        l=lower,
        u=upper,
    )

    result = solver.solve()

    if result.info.status not in (
        "solved",
        "solved inaccurate",
    ):
        raise RuntimeError(
            "OSQP failed at "
            f"t={t:.6f} s: "
            + result.info.status
        )

    Wc = result.x.copy()

    # First MPC correction
    w0 = Wc[:nu]

    # Actual current gravity feedforward
    tau_ff_current = (
        tau_ff_horizon[0]
    )

    tau = (
        tau_ff_current
        + w0
    )

    toc = time.perf_counter()

    compute_time_history.append(
        toc - tic
    )

    osqp_iteration_history.append(
        result.info.iter
    )

    # --------------------------------------------------------
    # Constraint activity
    # --------------------------------------------------------

    active = (
        np.abs(tau)
        >= 0.999 * TORQUE_LIMITS
    )

    constraint_active_history.append(
        active
    )

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    q_history.append(
        q.copy()
    )

    qdot_history.append(
        qdot.copy()
    )

    q_ref_history.append(
        q_ref.copy()
    )

    tau_history.append(
        tau.copy()
    )

    if t >= DURATION:
        continue

    # --------------------------------------------------------
    # RK4 nonlinear plant with ZOH control input
    # --------------------------------------------------------

    k1 = state_derivative_disturbed(
        x,
        tau,
        t,
    )

    k2 = state_derivative_disturbed(
        x + 0.5 * DT * k1,
        tau,
        t + 0.5 * DT,
    )

    k3 = state_derivative_disturbed(
        x + 0.5 * DT * k2,
        tau,
        t + 0.5 * DT,
    )

    k4 = state_derivative_disturbed(
        x + DT * k3,
        tau,
        t + DT,
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

qdot_history = np.asarray(
    qdot_history
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

osqp_iteration_history = np.asarray(
    osqp_iteration_history
)

constraint_active_history = np.asarray(
    constraint_active_history
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

constraint_active_count = np.sum(
    constraint_active_history,
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
print("FINAL NOMINAL MPC BENCHMARK")
print("=" * 88)

print(
    f"\nSampling period       : {DT:.6f} s"
)

print(
    f"Sampling rate         : {1.0 / DT:.1f} Hz"
)

print(
    f"Prediction horizon Np : {NP}"
)

print(
    f"Prediction window     : "
    f"{NP * DT * 1000.0:.1f} ms"
)

print(
    f"Control horizon Nc    : {NC}"
)

print(
    f"Control window        : "
    f"{NC * DT * 1000.0:.1f} ms"
)

print(
    f"rho_u                 : {RHO_U:.1f}"
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

print("\nConstraint-active sample count:")
print(
    constraint_active_count
)


print("\nOSQP iterations:")

print(
    f"Mean : "
    f"{np.mean(osqp_iteration_history):.3f}"
)

print(
    f"Max  : "
    f"{np.max(osqp_iteration_history)}"
)

print(
    f"P99  : "
    f"{np.percentile(osqp_iteration_history, 99):.1f}"
)


print("\nOnline MPC computation time [ms]:")

print(
    f"Mean : {np.mean(compute_ms):.6f}"
)

print(
    f"Max  : {np.max(compute_ms):.6f}"
)

print(
    f"P99  : "
    f"{np.percentile(compute_ms, 99):.6f}"
)


print(
    "\nFraction of 2 ms control period [%]:"
)

print(
    f"Mean : "
    f"{100.0 * np.mean(compute_time_history) / DT:.3f}"
)

print(
    f"P99  : "
    f"{100.0 * np.percentile(compute_time_history, 99) / DT:.3f}"
)

print(
    f"Max  : "
    f"{100.0 * np.max(compute_time_history) / DT:.3f}"
)


# ============================================================
# Save final nominal result
# ============================================================

results_dir = (
    "results/mpc/nominal"
)

os.makedirs(
    results_dir,
    exist_ok=True,
)

np.savez(
    f"{results_dir}/"
    "mpc_nominal_benchmark.npz",

    time=times,
    q=q_history,
    qdot=qdot_history,
    q_ref=q_ref_history,
    tau=tau_history,

    Q=Q,
    R=R,

    prediction_horizon=NP,
    control_horizon=NC,
    rho_u=RHO_U,
    osqp_eps_abs=1e-6,
    osqp_eps_rel=1e-6,
    offline_reference_gradient=True,

    joint_rmse=joint_rmse,
    joint_max_error=joint_max,
    joint_final_error=joint_final,
    overall_joint_rmse=overall_rmse,

    max_torque=max_torque,
    rms_torque=rms_torque,
    torque_utilization=torque_utilization,
    saturation_count=saturation_count,
    constraint_active_count=constraint_active_count,

    compute_time=compute_time_history,
    osqp_iterations=osqp_iteration_history,
)

print(
    "\nSaved results to:",
    results_dir,
)