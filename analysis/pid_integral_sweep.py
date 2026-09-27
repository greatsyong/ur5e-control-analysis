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
# Fixed P/D design
# ---------------------------------------------------------------------

WN = 16.0
ZETA = 1.0

ALPHA_I_VALUES = [
    0.0,
    0.25,
    0.5,
    1.0,
    2.0,
]

M0 = mass_matrix(q_start)
effective_inertia = np.diag(M0)

kp = effective_inertia * WN**2

kd = (
    2.0
    * ZETA
    * effective_inertia
    * WN
)


# ---------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------

def run_pid_simulation(alpha_i):

    ki = alpha_i * kp

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

        q_ref, qdot_ref, _ = (
            quintic_joint_trajectory(
                t,
                DURATION,
                q_start,
                q_goal,
            )
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

        # RK4 with zero-order-held torque

        k1 = state_derivative(x, tau)

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

    q_history = np.asarray(q_history)
    q_ref_history = np.asarray(q_ref_history)
    tau_history = np.asarray(tau_history)

    error = q_ref_history - q_history

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

    max_torque = np.max(
        np.abs(tau_history),
        axis=0,
    )

    utilization = (
        100.0
        * max_torque
        / TORQUE_LIMITS
    )

    # One scalar for convenient overall comparison.
    overall_rmse = np.sqrt(
        np.mean(error**2)
    )

    # TCP task-space tracking metrics
    tcp_metrics = summarize_tcp_errors(
        q_ref_history,
        q_history,
    )

    return {
        "alpha_i": alpha_i,
        "ki": ki,
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

for alpha_i in ALPHA_I_VALUES:

    print(
        f"\nRunning alpha_i = {alpha_i:.2f} ..."
    )

    results.append(
        run_pid_simulation(alpha_i)
    )


# ---------------------------------------------------------------------
# Detailed results
# ---------------------------------------------------------------------

print("\n" + "=" * 78)
print("PID INTEGRAL-STRENGTH SWEEP — NONLINEAR UR5e")
print("=" * 78)

print("\nFixed Kp:")
print(kp)

print("\nFixed Kd:")
print(kd)


for result in results:

    print("\n" + "-" * 78)

    print(
        f"alpha_i = {result['alpha_i']:.2f}"
    )

    print("\nKi:")
    print(result["ki"])

    print("\nRMSE [deg]:")
    print(
        np.rad2deg(
            result["rmse"]
        )
    )

    print("\nMaximum absolute error [deg]:")
    print(
        np.rad2deg(
            result["max_error"]
        )
    )

    print("\nFinal error [deg]:")
    print(
        np.rad2deg(
            result["final_error"]
        )
    )

    print("\nMaximum torque [Nm]:")
    print(result["max_torque"])

    print("\nTorque utilization [%]:")
    print(result["utilization"])


# ---------------------------------------------------------------------
# Compact RMSE comparison
# ---------------------------------------------------------------------

print("\n" + "=" * 78)
print("RMSE COMPARISON [deg]")
print("=" * 78)

print(
    "alpha      J1       J2       J3       J4       J5       J6"
)

for result in results:

    rmse_deg = np.rad2deg(
        result["rmse"]
    )

    print(
        f"{result['alpha_i']:>5.2f}  "
        + " ".join(
            f"{value:8.3f}"
            for value in rmse_deg
        )
    )


# ---------------------------------------------------------------------
# Final-error comparison
# ---------------------------------------------------------------------

print("\n" + "=" * 78)
print("FINAL ERROR COMPARISON [deg]")
print("=" * 78)

print(
    "alpha      J1       J2       J3       J4       J5       J6"
)

for result in results:

    final_deg = np.rad2deg(
        result["final_error"]
    )

    print(
        f"{result['alpha_i']:>5.2f}  "
        + " ".join(
            f"{value:8.3f}"
            for value in final_deg
        )
    )


# ---------------------------------------------------------------------
# Overall RMSE
# ---------------------------------------------------------------------

print("\n" + "=" * 78)
print("OVERALL TRACKING RMSE [deg]")
print("=" * 78)

for result in results:

    overall_deg = np.rad2deg(
        result["overall_rmse"]
    )

    print(
        f"alpha_i = {result['alpha_i']:>4.2f}: "
        f"{overall_deg:.4f} deg"
    )


# ---------------------------------------------------------------------
# Torque utilization
# ---------------------------------------------------------------------

print("\n" + "=" * 78)
print("MAXIMUM TORQUE UTILIZATION [%]")
print("=" * 78)

print(
    "alpha      J1       J2       J3       J4       J5       J6"
)

for result in results:

    print(
        f"{result['alpha_i']:>5.2f}  "
        + " ".join(
            f"{value:8.2f}"
            for value in result["utilization"]
        )
    )

# ---------------------------------------------------------------------
# TCP task-space comparison
# ---------------------------------------------------------------------

print("\n" + "=" * 78)
print("TCP TASK-SPACE TRACKING PERFORMANCE")
print("=" * 78)

print(
    "alpha    Pos RMSE    Pos Max   Pos Final   "
    "Ori RMSE   Ori Max  Ori Final"
)

print(
    "           [mm]        [mm]       [mm]       "
    "[deg]      [deg]      [deg]"
)

for result in results:

    tcp = result["tcp_metrics"]

    print(
        f"{result['alpha_i']:>5.2f}  "
        f"{tcp['tcp_position_rmse_mm']:10.3f} "
        f"{tcp['tcp_position_max_mm']:10.3f} "
        f"{tcp['tcp_position_final_mm']:10.3f} "
        f"{tcp['tcp_orientation_rmse_deg']:10.3f} "
        f"{tcp['tcp_orientation_max_deg']:9.3f} "
        f"{tcp['tcp_orientation_final_deg']:10.3f}"
    )