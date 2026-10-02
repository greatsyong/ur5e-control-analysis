import os

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import cont2discrete

from controllers.lqr import DiscreteJointLQRController
from dynamics.state_space import (
    state_derivative,
    equilibrium_input,
    linearize_dynamics,
)
from trajectories.joint_trajectory import quintic_joint_trajectory
from analysis.controller_metrics import summarize_tcp_errors


# ============================================================
# Discrete LQR — nonlinear rho_u sweep
# ============================================================

DT = 0.002
DURATION = 4.0

RHO_VALUES = [
    100.0,
    1000.0,
    10000.0,
]

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
# Bryson normalization
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


# ============================================================
# Common trajectory
# ============================================================

times = np.arange(
    0.0,
    DURATION + DT,
    DT,
)

results = {}


# ============================================================
# Nonlinear benchmark sweep
# ============================================================

for rho_u in RHO_VALUES:

    print(
        f"\nRunning rho_u = {rho_u:.0f} ..."
    )

    R = rho_u * R_bryson

    controller = DiscreteJointLQRController(
        Ad=Ad,
        Bd=Bd,
        Q=Q,
        R=R,
        torque_limits=TORQUE_LIMITS,
    )

    # Discrete closed-loop poles
    eig_d = np.linalg.eigvals(
        Ad - Bd @ controller.K
    )

    spectral_radius = np.max(
        np.abs(eig_d)
    )

    # --------------------------------------------------------
    # Initial nonlinear state
    # --------------------------------------------------------

    x = np.concatenate([
        q_start,
        np.zeros(6),
    ])

    q_history = []
    qdot_history = []

    q_ref_history = []
    qdot_ref_history = []

    tau_history = []

    for t in times:

        q = x[:6]
        qdot = x[6:]

        q_ref, qdot_ref, _ = quintic_joint_trajectory(
            t,
            DURATION,
            q_start,
            q_goal,
        )

        # Fixed-equilibrium feedforward remains unchanged.
        #
        # tau = tau_eq - K dx
        #
        # This isolates the effect of rho_u.
        tau = controller.compute(
            q=q,
            qdot=qdot,
            q_ref=q_ref,
            qdot_ref=qdot_ref,
            tau_ff=tau_eq,
        )

        q_history.append(
            q.copy()
        )

        qdot_history.append(
            qdot.copy()
        )

        q_ref_history.append(
            q_ref.copy()
        )

        qdot_ref_history.append(
            qdot_ref.copy()
        )

        tau_history.append(
            tau.copy()
        )

        if t >= DURATION:
            continue

        # RK4 with zero-order-held torque
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

    # --------------------------------------------------------
    # Convert histories
    # --------------------------------------------------------

    q_history = np.asarray(
        q_history
    )

    qdot_history = np.asarray(
        qdot_history
    )

    q_ref_history = np.asarray(
        q_ref_history
    )

    qdot_ref_history = np.asarray(
        qdot_ref_history
    )

    tau_history = np.asarray(
        tau_history
    )

    # --------------------------------------------------------
    # Joint metrics
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # TCP metrics
    # --------------------------------------------------------

    tcp = summarize_tcp_errors(
        q_ref_history,
        q_history,
    )

    # --------------------------------------------------------
    # Torque metrics
    # --------------------------------------------------------

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

    # Number of samples at/near saturation
    saturation_mask = (
        np.abs(tau_history)
        >= 0.999 * TORQUE_LIMITS
    )

    saturation_count = np.sum(
        saturation_mask,
        axis=0,
    )

    results[rho_u] = {
        "K": controller.K,
        "spectral_radius": spectral_radius,
        "q": q_history,
        "qdot": qdot_history,
        "q_ref": q_ref_history,
        "qdot_ref": qdot_ref_history,
        "tau": tau_history,
        "error": error,
        "joint_rmse": joint_rmse,
        "joint_max": joint_max,
        "joint_final": joint_final,
        "overall_rmse": overall_rmse,
        "tcp": tcp,
        "max_torque": max_torque,
        "rms_torque": rms_torque,
        "torque_utilization": torque_utilization,
        "saturation_count": saturation_count,
    }


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 120)
print("DISCRETE LQR — NONLINEAR CONTROL-PENALTY SWEEP")
print("=" * 120)

print(
    f"\n{'rho_u':>10}"
    f"{'Joint RMSE [deg]':>20}"
    f"{'TCP RMSE [mm]':>18}"
    f"{'TCP max [mm]':>16}"
    f"{'Ori RMSE [deg]':>18}"
    f"{'Max torque util [%]':>22}"
)

print("-" * 120)

for rho_u in RHO_VALUES:

    r = results[rho_u]

    print(
        f"{rho_u:10.0f}"
        f"{np.rad2deg(r['overall_rmse']):20.6f}"
        f"{r['tcp']['tcp_position_rmse_mm']:18.6f}"
        f"{r['tcp']['tcp_position_max_mm']:16.6f}"
        f"{r['tcp']['tcp_orientation_rmse_deg']:18.6f}"
        f"{np.max(r['torque_utilization']):22.3f}"
    )


# ============================================================
# Detailed results
# ============================================================

for rho_u in RHO_VALUES:

    r = results[rho_u]

    print("\n" + "=" * 86)
    print(
        f"rho_u = {rho_u:.0f}"
    )
    print("=" * 86)

    print("\nJoint RMSE [deg]:")
    print(
        np.rad2deg(
            r["joint_rmse"]
        )
    )

    print("\nJoint maximum error [deg]:")
    print(
        np.rad2deg(
            r["joint_max"]
        )
    )

    print("\nJoint final error [deg]:")
    print(
        np.rad2deg(
            r["joint_final"]
        )
    )

    print(
        "\nOverall joint RMSE [deg]: "
        f"{np.rad2deg(r['overall_rmse']):.6f}"
    )

    print(
        "\nTCP position "
        "RMSE / max / final [mm]:"
    )

    print(
        f"{r['tcp']['tcp_position_rmse_mm']:.6f} / "
        f"{r['tcp']['tcp_position_max_mm']:.6f} / "
        f"{r['tcp']['tcp_position_final_mm']:.6f}"
    )

    print(
        "\nTCP orientation "
        "RMSE / max / final [deg]:"
    )

    print(
        f"{r['tcp']['tcp_orientation_rmse_deg']:.6f} / "
        f"{r['tcp']['tcp_orientation_max_deg']:.6f} / "
        f"{r['tcp']['tcp_orientation_final_deg']:.6f}"
    )

    print("\nMaximum torque [Nm]:")
    print(
        r["max_torque"]
    )

    print("\nTorque utilization [%]:")
    print(
        r["torque_utilization"]
    )

    print("\nSaturation sample count:")
    print(
        r["saturation_count"]
    )


# ============================================================
# Save
# ============================================================

results_dir = (
    "results/lqr/rho_nonlinear_sweep"
)

os.makedirs(
    results_dir,
    exist_ok=True,
)

for rho_u in RHO_VALUES:

    r = results[rho_u]

    np.savez(
        os.path.join(
            results_dir,
            f"lqr_rho_{int(rho_u)}.npz",
        ),
        time=times,
        q=r["q"],
        qdot=r["qdot"],
        q_ref=r["q_ref"],
        qdot_ref=r["qdot_ref"],
        tau=r["tau"],
        error=r["error"],
        K=r["K"],
        Q=Q,
        R=rho_u * R_bryson,
        tau_eq=tau_eq,
        joint_rmse=r["joint_rmse"],
        joint_max_error=r["joint_max"],
        joint_final_error=r["joint_final"],
        overall_joint_rmse=r["overall_rmse"],
        max_torque=r["max_torque"],
        rms_torque=r["rms_torque"],
        torque_utilization=r["torque_utilization"],
        saturation_count=r["saturation_count"],
    )


# ============================================================
# Figure 1 — overall joint RMSE vs rho
# ============================================================

rho_array = np.asarray(
    RHO_VALUES
)

rmse_array = np.asarray([
    np.rad2deg(
        results[rho]["overall_rmse"]
    )
    for rho in RHO_VALUES
])

plt.figure(
    figsize=(8, 5)
)

plt.semilogx(
    rho_array,
    rmse_array,
    marker="o",
)

plt.xlabel(
    r"Control penalty $\rho_u$"
)

plt.ylabel(
    "Overall joint RMSE [deg]"
)

plt.title(
    "Discrete LQR — Control Penalty Sweep"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        results_dir,
        "lqr_rho_joint_rmse.png",
    ),
    dpi=300,
)

plt.close()


# ============================================================
# Figure 2 — per-joint RMSE
# ============================================================

plt.figure(
    figsize=(9, 6)
)

for joint in range(6):

    values = [
        np.rad2deg(
            results[rho]["joint_rmse"][joint]
        )
        for rho in RHO_VALUES
    ]

    plt.semilogx(
        rho_array,
        values,
        marker="o",
        label=f"J{joint + 1}",
    )

plt.xlabel(
    r"Control penalty $\rho_u$"
)

plt.ylabel(
    "Joint RMSE [deg]"
)

plt.title(
    "Discrete LQR — Per-Joint Tracking Error"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        results_dir,
        "lqr_rho_joint_rmse_per_joint.png",
    ),
    dpi=300,
)

plt.close()


print(
    "\nSaved results to:",
    results_dir,
)