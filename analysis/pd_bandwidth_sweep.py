import numpy as np

from controllers.pid import JointPIDController
from dynamics.rigid_body_dynamics import mass_matrix
from dynamics.state_space import state_derivative
from trajectories.joint_trajectory import quintic_joint_trajectory


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
    0.0,
    -90.0,
    90.0,
    -90.0,
    -90.0,
    0.0,
])

q_goal = np.deg2rad([
    20.0,
    -60.0,
    60.0,
    -70.0,
    -70.0,
    20.0,
])

WN_VALUES = [
    4.0,
    8.0,
    12.0,
    16.0,
]

ZETA = 1.0


# ---------------------------------------------------------------------
# Nominal effective inertia
# ---------------------------------------------------------------------

M0 = mass_matrix(q_start)
effective_inertia = np.diag(M0)


# ---------------------------------------------------------------------
# Simulation function
# ---------------------------------------------------------------------

def run_pd_simulation(wn):

    kp = effective_inertia * wn**2

    kd = (
        2.0
        * ZETA
        * effective_inertia
        * wn
    )

    ki = np.zeros(6)

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

    max_torque = np.max(
        np.abs(tau_history),
        axis=0,
    )

    utilization = (
        100.0
        * max_torque
        / TORQUE_LIMITS
    )

    return {
        "wn": wn,
        "kp": kp,
        "kd": kd,
        "rmse": rmse,
        "max_error": max_error,
        "final_error": final_error,
        "max_torque": max_torque,
        "utilization": utilization,
    }


# ---------------------------------------------------------------------
# Run sweep
# ---------------------------------------------------------------------

results = []

for wn in WN_VALUES:

    print(
        f"\nRunning wn = {wn:.1f} rad/s ..."
    )

    result = run_pd_simulation(wn)

    results.append(result)


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

print("\n" + "=" * 78)
print("PD BANDWIDTH SWEEP — NONLINEAR UR5e")
print("=" * 78)

for result in results:

    wn = result["wn"]

    print(
        "\n"
        + "-" * 78
    )

    print(
        f"wn = {wn:.1f} rad/s"
    )

    print("\nKp:")
    print(result["kp"])

    print("\nKd:")
    print(result["kd"])

    print("\nRMSE [deg]:")
    print(
        np.rad2deg(
            result["rmse"]
        )
    )

    print(
        "\nMaximum absolute error [deg]:"
    )

    print(
        np.rad2deg(
            result["max_error"]
        )
    )

    print(
        "\nFinal error [deg]:"
    )

    print(
        np.rad2deg(
            result["final_error"]
        )
    )

    print(
        "\nMaximum torque [Nm]:"
    )

    print(
        result["max_torque"]
    )

    print(
        "\nTorque utilization [%]:"
    )

    print(
        result["utilization"]
    )


# ---------------------------------------------------------------------
# Compact comparison
# ---------------------------------------------------------------------

print("\n" + "=" * 78)
print("COMPACT RMSE COMPARISON [deg]")
print("=" * 78)

print(
    "wn       J1       J2       J3       J4       J5       J6"
)

for result in results:

    rmse_deg = np.rad2deg(
        result["rmse"]
    )

    print(
        f"{result['wn']:>4.0f}  "
        + " ".join(
            f"{value:8.3f}"
            for value in rmse_deg
        )
    )


print("\n" + "=" * 78)
print("MAXIMUM TORQUE UTILIZATION [%]")
print("=" * 78)

print(
    "wn       J1       J2       J3       J4       J5       J6"
)

for result in results:

    utilization = (
        result["utilization"]
    )

    print(
        f"{result['wn']:>4.0f}  "
        + " ".join(
            f"{value:8.2f}"
            for value in utilization
        )
    )