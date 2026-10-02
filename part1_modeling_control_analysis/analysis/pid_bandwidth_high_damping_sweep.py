import numpy as np

from controllers.pid import JointPIDController
from dynamics.rigid_body_dynamics import mass_matrix
from dynamics.state_space import state_derivative
from trajectories.joint_trajectory import quintic_joint_trajectory
from analysis.controller_metrics import summarize_tcp_errors


# ---------------------------------------------------------------------
# Common benchmark
# ---------------------------------------------------------------------

DT = 0.002
DURATION = 4.0

TORQUE_LIMITS = np.array([
    150.0,
    150.0,
    150.0,
    28.0,
    28.0,
    28.0,
])

q_start = np.deg2rad([
    0.0, -90.0, 90.0, -90.0, -90.0, 0.0,
])

q_goal = np.deg2rad([
    20.0, -60.0, 60.0, -70.0, -70.0, 20.0,
])


# ---------------------------------------------------------------------
# High-damping bandwidth sweep
# ---------------------------------------------------------------------
#
# Purpose:
#   Evaluate whether a higher damping setting allows the PID controller
#   to use a higher closed-loop design bandwidth while maintaining
#   stable and well-damped tracking.
#
# Design choices:
#   - zeta is fixed at 3.0 based on the previous damping-ratio sweep.
#   - wn is varied from 14 to 20 rad/s.
#   - 20 rad/s is the predefined practical upper design bound.
#   - Ki is held numerically constant at the value obtained from the
#     previous wn = 16 rad/s, alpha_i = 2.0 design.
#
# Holding Ki fixed isolates the effect of increasing the proportional
# and derivative bandwidth without simultaneously strengthening the
# integral action.
# ---------------------------------------------------------------------

ZETA = 3.0

WN_VALUES = [
    14.0,
    16.0,
    18.0,
    20.0,
]


# ---------------------------------------------------------------------
# Effective joint inertia
# ---------------------------------------------------------------------

M0 = mass_matrix(q_start)
effective_inertia = np.diag(M0)


# ---------------------------------------------------------------------
# Fixed integral gain
# ---------------------------------------------------------------------
#
# Reference integral design:
#
#   wn_ref    = 16 rad/s
#   alpha_i   = 2.0
#
#   Kp_ref = M_d * wn_ref^2
#   Ki     = alpha_i * Kp_ref
#
# Ki remains fixed throughout this sweep.
# ---------------------------------------------------------------------

WN_KI_REFERENCE = 16.0
ALPHA_I_REFERENCE = 2.0

kp_ki_reference = (
    effective_inertia
    * WN_KI_REFERENCE**2
)

KI_FIXED = (
    ALPHA_I_REFERENCE
    * kp_ki_reference
)


# ---------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------

def run_pid_simulation(wn):

    # Gain design for current bandwidth
    kp = (
        effective_inertia
        * wn**2
    )

    kd = (
        2.0
        * ZETA
        * effective_inertia
        * wn
    )

    # Integral gain intentionally held constant
    ki = KI_FIXED.copy()

    controller = JointPIDController(
        kp=kp,
        ki=ki,
        kd=kd,
        dt=DT,
        torque_limits=TORQUE_LIMITS,
    )

    controller.reset()

    x = np.concatenate([
        q_start,
        np.zeros(6),
    ])

    times = np.arange(
        0.0,
        DURATION + DT,
        DT,
    )

    q_history = []
    q_ref_history = []
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

        tau = controller.compute(
            q,
            qdot,
            q_ref,
            qdot_ref,
        )

        q_history.append(q.copy())
        q_ref_history.append(q_ref.copy())
        tau_history.append(tau.copy())

        if t >= DURATION:
            continue

        # RK4 integration with zero-order-held torque
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

    q_history = np.asarray(
        q_history
    )

    q_ref_history = np.asarray(
        q_ref_history
    )

    tau_history = np.asarray(
        tau_history
    )

    error = (
        q_ref_history
        - q_history
    )

    rmse = np.sqrt(
        np.mean(
            error**2,
            axis=0,
        )
    )

    max_error = np.max(
        np.abs(error),
        axis=0,
    )

    final_error = error[-1]

    overall_rmse = np.sqrt(
        np.mean(error**2)
    )

    max_torque = np.max(
        np.abs(tau_history),
        axis=0,
    )

    utilization = (
        100.0
        * max_torque
        / TORQUE_LIMITS
    )

    tcp_metrics = summarize_tcp_errors(
        q_ref_history,
        q_history,
    )

    return {
        "wn": wn,
        "kp": kp,
        "ki": ki,
        "kd": kd,
        "rmse": rmse,
        "overall_rmse": overall_rmse,
        "max_error": max_error,
        "final_error": final_error,
        "max_torque": max_torque,
        "utilization": utilization,
        "tcp_metrics": tcp_metrics,
    }


# ---------------------------------------------------------------------
# Run sweep
# ---------------------------------------------------------------------

results = []

for wn in WN_VALUES:

    print(
        f"\nRunning wn = {wn:.2f} rad/s ..."
    )

    results.append(
        run_pid_simulation(wn)
    )


# ---------------------------------------------------------------------
# Design summary
# ---------------------------------------------------------------------

print("\n" + "=" * 86)
print("PID HIGH-DAMPING BANDWIDTH SWEEP — NONLINEAR UR5e")
print("=" * 86)

print(
    f"\nFixed zeta          : "
    f"{ZETA:.2f}"
)

print(
    f"Ki reference wn     : "
    f"{WN_KI_REFERENCE:.2f} rad/s"
)

print(
    f"Ki reference alpha_i: "
    f"{ALPHA_I_REFERENCE:.2f}"
)

print(
    "\nEffective joint inertia:"
)

print(
    effective_inertia
)

print(
    "\nFixed Ki:"
)

print(
    KI_FIXED
)


# ---------------------------------------------------------------------
# Gain comparison
# ---------------------------------------------------------------------

print("\n" + "=" * 86)
print("PID GAIN COMPARISON")
print("=" * 86)

for result in results:

    print(
        f"\nwn = "
        f"{result['wn']:.2f} rad/s"
    )

    print(
        "Kp:"
    )

    print(
        result["kp"]
    )

    print(
        "Kd:"
    )

    print(
        result["kd"]
    )


# ---------------------------------------------------------------------
# Joint-space results
# ---------------------------------------------------------------------

print("\n" + "=" * 86)
print("JOINT RMSE COMPARISON [deg]")
print("=" * 86)

print(
    "wn         J1       J2       J3       J4       J5       J6"
)

for result in results:

    rmse_deg = np.rad2deg(
        result["rmse"]
    )

    print(
        f"{result['wn']:>4.1f}  "
        + " ".join(
            f"{value:8.3f}"
            for value in rmse_deg
        )
    )


print("\n" + "=" * 86)
print("OVERALL JOINT TRACKING RMSE [deg]")
print("=" * 86)

for result in results:

    overall_deg = np.rad2deg(
        result["overall_rmse"]
    )

    print(
        f"wn = {result['wn']:.1f}: "
        f"{overall_deg:.4f} deg"
    )


print("\n" + "=" * 86)
print("MAXIMUM JOINT ERROR [deg]")
print("=" * 86)

print(
    "wn         J1       J2       J3       J4       J5       J6"
)

for result in results:

    max_deg = np.rad2deg(
        result["max_error"]
    )

    print(
        f"{result['wn']:>4.1f}  "
        + " ".join(
            f"{value:8.3f}"
            for value in max_deg
        )
    )


print("\n" + "=" * 86)
print("FINAL JOINT ERROR [deg]")
print("=" * 86)

print(
    "wn         J1       J2       J3       J4       J5       J6"
)

for result in results:

    final_deg = np.rad2deg(
        result["final_error"]
    )

    print(
        f"{result['wn']:>4.1f}  "
        + " ".join(
            f"{value:8.3f}"
            for value in final_deg
        )
    )


# ---------------------------------------------------------------------
# TCP task-space results
# ---------------------------------------------------------------------

print("\n" + "=" * 86)
print("TCP TASK-SPACE TRACKING PERFORMANCE")
print("=" * 86)

print(
    "wn      Pos RMSE    Pos Max   Pos Final   "
    "Ori RMSE   Ori Max  Ori Final"
)

print(
    "        [mm]        [mm]       [mm]       "
    "[deg]      [deg]      [deg]"
)

for result in results:

    tcp = result[
        "tcp_metrics"
    ]

    print(
        f"{result['wn']:>4.1f}  "
        f"{tcp['tcp_position_rmse_mm']:10.3f} "
        f"{tcp['tcp_position_max_mm']:10.3f} "
        f"{tcp['tcp_position_final_mm']:10.3f} "
        f"{tcp['tcp_orientation_rmse_deg']:10.3f} "
        f"{tcp['tcp_orientation_max_deg']:9.3f} "
        f"{tcp['tcp_orientation_final_deg']:10.3f}"
    )


# ---------------------------------------------------------------------
# Maximum torque
# ---------------------------------------------------------------------

print("\n" + "=" * 86)
print("MAXIMUM TORQUE [Nm]")
print("=" * 86)

print(
    "wn         J1       J2       J3       J4       J5       J6"
)

for result in results:

    print(
        f"{result['wn']:>4.1f}  "
        + " ".join(
            f"{value:8.3f}"
            for value in result[
                "max_torque"
            ]
        )
    )


# ---------------------------------------------------------------------
# Torque utilization
# ---------------------------------------------------------------------

print("\n" + "=" * 86)
print("MAXIMUM TORQUE UTILIZATION [%]")
print("=" * 86)

print(
    "wn         J1       J2       J3       J4       J5       J6"
)

for result in results:

    print(
        f"{result['wn']:>4.1f}  "
        + " ".join(
            f"{value:8.2f}"
            for value in result[
                "utilization"
            ]
        )
    )


# ---------------------------------------------------------------------
# Best result within predefined design range
# ---------------------------------------------------------------------

best_joint = min(
    results,
    key=lambda result:
    result["overall_rmse"],
)

best_tcp_position = min(
    results,
    key=lambda result:
    result["tcp_metrics"][
        "tcp_position_rmse_mm"
    ],
)

best_tcp_orientation = min(
    results,
    key=lambda result:
    result["tcp_metrics"][
        "tcp_orientation_rmse_deg"
    ],
)


print("\n" + "=" * 86)
print("BEST PERFORMANCE WITHIN PREDEFINED DESIGN RANGE")
print("=" * 86)

print(
    "\nLowest overall joint RMSE:"
)

print(
    f"wn = {best_joint['wn']:.1f} rad/s, "
    f"RMSE = "
    f"{np.rad2deg(best_joint['overall_rmse']):.4f} deg"
)

print(
    "\nLowest TCP position RMSE:"
)

print(
    f"wn = {best_tcp_position['wn']:.1f} rad/s, "
    f"RMSE = "
    f"{best_tcp_position['tcp_metrics']['tcp_position_rmse_mm']:.3f} mm"
)

print(
    "\nLowest TCP orientation RMSE:"
)

print(
    f"wn = {best_tcp_orientation['wn']:.1f} rad/s, "
    f"RMSE = "
    f"{best_tcp_orientation['tcp_metrics']['tcp_orientation_rmse_deg']:.3f} deg"
)