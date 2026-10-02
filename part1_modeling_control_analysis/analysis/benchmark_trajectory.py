from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from trajectories.joint_trajectory import (
    quintic_joint_trajectory,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "results" / "data"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Benchmark definition
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Generate trajectory
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------

print("=" * 70)
print("UR5e COMMON BENCHMARK TRAJECTORY")
print("=" * 70)

print("\nStart configuration [deg]:")
print(q_start_deg)

print("\nGoal configuration [deg]:")
print(q_goal_deg)

print(f"\nDuration: {DURATION:.3f} s")
print(f"Sample time: {DT:.4f} s")
print(f"Samples: {len(times)}")

print("\nMaximum reference velocity [deg/s]:")
print(np.rad2deg(max_velocity))

print("\nMaximum reference acceleration [deg/s^2]:")
print(np.rad2deg(max_acceleration))

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


# ---------------------------------------------------------------------
# Save benchmark reference
#
# Store SI units so the same file can be reused by controller code.
# ---------------------------------------------------------------------

benchmark_data = np.column_stack([
    times,
    q_history,
    qdot_history,
    qddot_history,
])

column_names = (
    ["time_s"]
    + [f"q{i}_rad" for i in range(1, 7)]
    + [f"qdot{i}_rad_s" for i in range(1, 7)]
    + [f"qddot{i}_rad_s2" for i in range(1, 7)]
)

data_path = (
    DATA_DIR
    / "benchmark_trajectory.csv"
)

np.savetxt(
    data_path,
    benchmark_data,
    delimiter=",",
    header=",".join(column_names),
    comments="",
    fmt="%.12e",
)


# ---------------------------------------------------------------------
# Figure: Joint-position reference
# ---------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(7.5, 4.8)
)

q_history_deg = np.rad2deg(
    q_history
)

for joint_index in range(6):

    ax.plot(
        times,
        q_history_deg[:, joint_index],
        linewidth=1.8,
        label=f"J{joint_index + 1}",
    )


ax.set_xlabel("Time [s]")
ax.set_ylabel("Joint position [deg]")

ax.set_title(
    "Common Controller Benchmark Trajectory"
)

ax.grid(
    True,
    alpha=0.3,
)

ax.legend(
    ncol=3,
)

fig.tight_layout()

figure_path = (
    FIGURE_DIR
    / "benchmark_joint_trajectory.png"
)

fig.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


print("\nSaved:")
print(data_path)
print(figure_path)