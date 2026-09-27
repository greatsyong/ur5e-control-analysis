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
# Fixed PID design parameters
# ---------------------------------------------------------------------

WN = 16.0
ALPHA_I = 2.0

ZETA_VALUES = [
    1.2,
    1.5,
    2.0,
    3.0,
]

M0 = mass_matrix(q_start)
effective_inertia = np.diag(M0)

kp = effective_inertia * WN**2
ki = ALPHA_I * kp


# ---------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------

def run_pid_simulation(zeta):

    kd = (
        2.0
        * zeta
        * effective_inertia
        * WN
    )

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
        np.mean(error**2, axis=0)
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
        "zeta": zeta,
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

for zeta in ZETA_VALUES:

    print(
        f"\nRunning zeta = {zeta:.2f} ..."
    )

    results.append(
        run_pid_simulation(zeta)
    )


# ---------------------------------------------------------------------
# Joint-space results
# ---------------------------------------------------------------------

print("\n" + "=" * 86)
print("PID DAMPING-RATIO SWEEP — NONLINEAR UR5e")
print("=" * 86)

print(f"\nFixed wn      : {WN:.2f} rad/s")
print(f"Fixed alpha_i : {ALPHA_I:.2f}")

print("\nFixed Kp:")
print(kp)

print("\nFixed Ki:")
print(ki)


print("\n" + "=" * 86)
print("JOINT RMSE COMPARISON [deg]")
print("=" * 86)

print(
    "zeta       J1       J2       J3       J4       J5       J6"
)

for result in results:

    rmse_deg = np.rad2deg(
        result["rmse"]
    )

    print(
        f"{result['zeta']:>4.2f}  "
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
        f"zeta = {result['zeta']:.2f}: "
        f"{overall_deg:.4f} deg"
    )


print("\n" + "=" * 86)
print("MAXIMUM JOINT ERROR [deg]")
print("=" * 86)

print(
    "zeta       J1       J2       J3       J4       J5       J6"
)

for result in results:

    max_deg = np.rad2deg(
        result["max_error"]
    )

    print(
        f"{result['zeta']:>4.2f}  "
        + " ".join(
            f"{value:8.3f}"
            for value in max_deg
        )
    )


print("\n" + "=" * 86)
print("FINAL JOINT ERROR [deg]")
print("=" * 86)

print(
    "zeta       J1       J2       J3       J4       J5       J6"
)

for result in results:

    final_deg = np.rad2deg(
        result["final_error"]
    )

    print(
        f"{result['zeta']:>4.2f}  "
        + " ".join(
            f"{value:8.3f}"
            for value in final_deg
        )
    )


# ---------------------------------------------------------------------
# TCP results
# ---------------------------------------------------------------------

print("\n" + "=" * 86)
print("TCP TASK-SPACE TRACKING PERFORMANCE")
print("=" * 86)

print(
    "zeta    Pos RMSE    Pos Max   Pos Final   "
    "Ori RMSE   Ori Max  Ori Final"
)

print(
    "          [mm]        [mm]       [mm]       "
    "[deg]      [deg]      [deg]"
)

for result in results:

    tcp = result["tcp_metrics"]

    print(
        f"{result['zeta']:>4.2f}  "
        f"{tcp['tcp_position_rmse_mm']:10.3f} "
        f"{tcp['tcp_position_max_mm']:10.3f} "
        f"{tcp['tcp_position_final_mm']:10.3f} "
        f"{tcp['tcp_orientation_rmse_deg']:10.3f} "
        f"{tcp['tcp_orientation_max_deg']:9.3f} "
        f"{tcp['tcp_orientation_final_deg']:10.3f}"
    )


# ---------------------------------------------------------------------
# Torque utilization
# ---------------------------------------------------------------------

print("\n" + "=" * 86)
print("MAXIMUM TORQUE UTILIZATION [%]")
print("=" * 86)

print(
    "zeta       J1       J2       J3       J4       J5       J6"
)

for result in results:

    print(
        f"{result['zeta']:>4.2f}  "
        + " ".join(
            f"{value:8.2f}"
            for value in result["utilization"]
        )
    )