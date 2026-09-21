import numpy as np

from trajectories.joint_trajectory import (
    quintic_joint_trajectory,
)


DT = 0.002
DURATION = 4.0

VELOCITY_LIMIT = np.pi


q_start_deg = np.array([
    0.0,
    -90.0,
    90.0,
    -90.0,
    -90.0,
    0.0,
])

q_goal_deg = np.array([
    20.0,
    -60.0,
    60.0,
    -70.0,
    -70.0,
    20.0,
])

q_start = np.deg2rad(q_start_deg)
q_goal = np.deg2rad(q_goal_deg)


times = np.arange(
    0.0,
    DURATION + DT,
    DT
)


q_history = []
qdot_history = []
qddot_history = []


for t in times:

    q, qdot, qddot = (
        quintic_joint_trajectory(
            t,
            DURATION,
            q_start,
            q_goal,
        )
    )

    q_history.append(q)
    qdot_history.append(qdot)
    qddot_history.append(qddot)


q_history = np.asarray(q_history)
qdot_history = np.asarray(qdot_history)
qddot_history = np.asarray(qddot_history)


max_velocity = np.max(
    np.abs(qdot_history),
    axis=0
)

max_acceleration = np.max(
    np.abs(qddot_history),
    axis=0
)


print("=" * 70)
print("UR5e COMMON BENCHMARK TRAJECTORY")
print("=" * 70)

print("\nStart configuration [deg]:")
print(q_start_deg)

print("\nGoal configuration [deg]:")
print(q_goal_deg)

print(
    f"\nDuration: {DURATION:.3f} s"
)

print(
    f"Sample time: {DT:.4f} s"
)

print(
    f"Samples: {len(times)}"
)


print("\nMaximum reference velocity [deg/s]:")

print(
    np.rad2deg(max_velocity)
)


print("\nMaximum reference acceleration [deg/s^2]:")

print(
    np.rad2deg(max_acceleration)
)


print("\nVelocity utilization [% of 180 deg/s]:")

print(
    100.0
    * max_velocity
    / VELOCITY_LIMIT
)


if np.all(
    max_velocity <= VELOCITY_LIMIT
):

    print(
        "\nPASS: reference trajectory "
        "satisfies the velocity constraint."
    )

else:

    print(
        "\nFAIL: reference trajectory "
        "exceeds the velocity constraint."
    )