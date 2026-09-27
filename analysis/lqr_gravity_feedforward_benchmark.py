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
# Configuration
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

R = RHO_U * R_bryson


# ============================================================
# Controller
# ============================================================

controller = DiscreteJointLQRController(
    Ad=Ad,
    Bd=Bd,
    Q=Q,
    R=R,
    torque_limits=TORQUE_LIMITS,
)


# ============================================================
# Common trajectory
# ============================================================

times = np.arange(
    0.0,
    DURATION + DT,
    DT,
)


def run_benchmark(feedforward_mode):

    x = np.concatenate([
        q_start,
        np.zeros(6),
    ])

    q_history = []
    qdot_history = []

    q_ref_history = []
    qdot_ref_history = []

    tau_history = []
    tau_ff_history = []

    for t in times:

        q = x[:6]
        qdot = x[6:]

        q_ref, qdot_ref, _ = quintic_joint_trajectory(
            t,
            DURATION,
            q_start,
            q_goal,
        )

        # ----------------------------------------------------
        # Feedforward choice
        # ----------------------------------------------------

        if feedforward_mode == "fixed":

            tau_ff = tau_eq.copy()

        elif feedforward_mode == "reference_gravity":

            # Gravity compensation evaluated at
            # the reference configuration.
            tau_ff = equilibrium_input(
                q_ref
            )

        else:
            raise ValueError(
                f"Unknown feedforward mode: "
                f"{feedforward_mode}"
            )

        # ----------------------------------------------------
        # LQR feedback
        # ----------------------------------------------------

        tau = controller.compute(
            q=q,
            qdot=qdot,
            q_ref=q_ref,
            qdot_ref=qdot_ref,
            tau_ff=tau_ff,
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

        tau_ff_history.append(
            tau_ff.copy()
        )

        if t >= DURATION:
            continue

        # ----------------------------------------------------
        # Nonlinear plant — RK4, ZOH torque
        # ----------------------------------------------------

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

    # ========================================================
    # Histories
    # ========================================================

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

    tau_ff_history = np.asarray(
        tau_ff_history
    )

    # ========================================================
    # Joint metrics
    # ========================================================

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

    # ========================================================
    # TCP metrics
    # ========================================================

    tcp = summarize_tcp_errors(
        q_ref_history,
        q_history,
    )

    # ========================================================
    # Torque metrics
    # ========================================================

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

    return {
        "q": q_history,
        "qdot": qdot_history,
        "q_ref": q_ref_history,
        "qdot_ref": qdot_ref_history,
        "tau": tau_history,
        "tau_ff": tau_ff_history,
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
# Run A/B comparison
# ============================================================

print("\nRunning fixed gravity feedforward ...")

fixed = run_benchmark(
    "fixed"
)

print(
    "Running reference-dependent "
    "gravity feedforward ..."
)

reference_gravity = run_benchmark(
    "reference_gravity"
)


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 110)
print("DISCRETE LQR — GRAVITY FEEDFORWARD COMPARISON")
print("=" * 110)

print(
    f"\nSampling period : {DT:.6f} s"
)

print(
    f"rho_u           : {RHO_U:.1f}"
)

print(
    "\n"
    f"{'Feedforward':>22}"
    f"{'Joint RMSE [deg]':>20}"
    f"{'TCP RMSE [mm]':>18}"
    f"{'TCP max [mm]':>16}"
    f"{'Ori RMSE [deg]':>18}"
    f"{'Max util [%]':>16}"
)

print("-" * 110)

for name, result in [
    ("Fixed g(q_start)", fixed),
    ("Reference g(q_ref)", reference_gravity),
]:

    print(
        f"{name:>22}"
        f"{np.rad2deg(result['overall_rmse']):20.6f}"
        f"{result['tcp']['tcp_position_rmse_mm']:18.6f}"
        f"{result['tcp']['tcp_position_max_mm']:16.6f}"
        f"{result['tcp']['tcp_orientation_rmse_deg']:18.6f}"
        f"{np.max(result['torque_utilization']):16.3f}"
    )


# ============================================================
# Detailed results
# ============================================================

for name, result in [
    ("FIXED g(q_start)", fixed),
    ("REFERENCE g(q_ref)", reference_gravity),
]:

    print("\n" + "=" * 86)
    print(name)
    print("=" * 86)

    print("\nJoint RMSE [deg]:")
    print(
        np.rad2deg(
            result["joint_rmse"]
        )
    )

    print("\nJoint maximum error [deg]:")
    print(
        np.rad2deg(
            result["joint_max"]
        )
    )

    print("\nJoint final error [deg]:")
    print(
        np.rad2deg(
            result["joint_final"]
        )
    )

    print(
        "\nOverall joint RMSE [deg]: "
        f"{np.rad2deg(result['overall_rmse']):.6f}"
    )

    print(
        "\nTCP position "
        "RMSE / max / final [mm]:"
    )

    print(
        f"{result['tcp']['tcp_position_rmse_mm']:.6f} / "
        f"{result['tcp']['tcp_position_max_mm']:.6f} / "
        f"{result['tcp']['tcp_position_final_mm']:.6f}"
    )

    print(
        "\nTCP orientation "
        "RMSE / max / final [deg]:"
    )

    print(
        f"{result['tcp']['tcp_orientation_rmse_deg']:.6f} / "
        f"{result['tcp']['tcp_orientation_max_deg']:.6f} / "
        f"{result['tcp']['tcp_orientation_final_deg']:.6f}"
    )

    print("\nMaximum torque [Nm]:")
    print(
        result["max_torque"]
    )

    print("\nRMS torque [Nm]:")
    print(
        result["rms_torque"]
    )

    print("\nTorque utilization [%]:")
    print(
        result["torque_utilization"]
    )

    print("\nSaturation sample count:")
    print(
        result["saturation_count"]
    )


# ============================================================
# Improvement
# ============================================================

joint_improvement = (
    100.0
    * (
        fixed["overall_rmse"]
        - reference_gravity["overall_rmse"]
    )
    / fixed["overall_rmse"]
)

tcp_improvement = (
    100.0
    * (
        fixed["tcp"]["tcp_position_rmse_mm"]
        - reference_gravity["tcp"]["tcp_position_rmse_mm"]
    )
    / fixed["tcp"]["tcp_position_rmse_mm"]
)

print("\n" + "=" * 110)
print("IMPROVEMENT FROM REFERENCE-DEPENDENT GRAVITY FEEDFORWARD")
print("=" * 110)

print(
    "\nOverall joint RMSE reduction : "
    f"{joint_improvement:.2f} %"
)

print(
    "TCP position RMSE reduction  : "
    f"{tcp_improvement:.2f} %"
)


# ============================================================
# Plots
# ============================================================

results_dir = (
    "results/lqr/gravity_feedforward"
)

import os

os.makedirs(
    results_dir,
    exist_ok=True,
)


# Joint tracking error comparison
plt.figure(
    figsize=(10, 6)
)

fixed_norm = np.linalg.norm(
    np.rad2deg(
        fixed["error"]
    ),
    axis=1,
)

reference_norm = np.linalg.norm(
    np.rad2deg(
        reference_gravity["error"]
    ),
    axis=1,
)

plt.plot(
    times,
    fixed_norm,
    label="Fixed g(q_start)",
)

plt.plot(
    times,
    reference_norm,
    label="Reference g(q_ref)",
)

plt.xlabel(
    "Time [s]"
)

plt.ylabel(
    "Joint error norm [deg]"
)

plt.title(
    "Discrete LQR — Gravity Feedforward Comparison"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    f"{results_dir}/"
    "lqr_gravity_feedforward_error.png",
    dpi=300,
)

plt.close()


# ============================================================
# Save numerical data
# ============================================================

np.savez(
    f"{results_dir}/"
    "lqr_gravity_feedforward_comparison.npz",

    time=times,

    q_fixed=fixed["q"],
    tau_fixed=fixed["tau"],

    q_reference_gravity=reference_gravity["q"],
    tau_reference_gravity=reference_gravity["tau"],

    q_ref=fixed["q_ref"],

    K=controller.K,
    Q=Q,
    R=R,
)

print(
    "\nSaved results to:",
    results_dir,
)