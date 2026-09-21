import numpy as np

from trajectories.joint_trajectory import (
    quintic_joint_trajectory,
)

from dynamics.rigid_body_dynamics import (
    mass_matrix,
    gravity_vector,
    coriolis_centrifugal_vector,
)


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


times = np.arange(
    0.0,
    DURATION + DT,
    DT
)


tau_history = []


for t in times:

    q, qdot, qddot = (
        quintic_joint_trajectory(
            t,
            DURATION,
            q_start,
            q_goal,
        )
    )

    M = mass_matrix(q)

    c = coriolis_centrifugal_vector(
        q,
        qdot
    )

    g = gravity_vector(q)

    tau = (
        M @ qddot
        + c
        + g
    )

    tau_history.append(tau)


tau_history = np.asarray(tau_history)


max_abs_torque = np.max(
    np.abs(tau_history),
    axis=0
)


utilization = (
    100.0
    * max_abs_torque
    / TORQUE_LIMITS
)


peak_indices = np.argmax(
    np.abs(tau_history),
    axis=0
)


peak_times = times[
    peak_indices
]


print("=" * 72)
print("UR5e BENCHMARK INVERSE-DYNAMICS CHECK")
print("=" * 72)

print("\nSimulation torque limits [N m]:")
print(TORQUE_LIMITS)

print("\nMaximum required torque [N m]:")
print(max_abs_torque)

print("\nTorque utilization [%]:")
print(utilization)

print("\nTime of peak torque [s]:")
print(peak_times)


if np.all(
    max_abs_torque <= TORQUE_LIMITS
):

    print(
        "\nPASS: benchmark trajectory is "
        "within all simulation torque limits."
    )

else:

    print(
        "\nFAIL: benchmark trajectory requires "
        "torque above at least one simulation limit."
    )