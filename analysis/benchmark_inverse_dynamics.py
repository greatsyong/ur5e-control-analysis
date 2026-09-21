from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from trajectories.joint_trajectory import (
    quintic_joint_trajectory,
)

from dynamics.rigid_body_dynamics import (
    mass_matrix,
    gravity_vector,
    coriolis_centrifugal_vector,
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

# Simulation constraints adopted consistently for analytical
# controller benchmarking and later Isaac Sim validation.
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


# ---------------------------------------------------------------------
# Inverse dynamics
# ---------------------------------------------------------------------

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


tau_history = np.asarray(
    tau_history
)


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


# ---------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Save numerical results
# ---------------------------------------------------------------------

torque_data = np.column_stack([
    times,
    tau_history,
])

column_names = (
    ["time_s"]
    + [f"tau{i}_Nm" for i in range(1, 7)]
)

data_path = (
    DATA_DIR
    / "benchmark_inverse_dynamics.csv"
)

np.savetxt(
    data_path,
    torque_data,
    delimiter=",",
    header=",".join(column_names),
    comments="",
    fmt="%.12e",
)


summary_path = (
    DATA_DIR
    / "benchmark_torque_summary.csv"
)

summary_data = np.column_stack([
    np.arange(1, 7),
    TORQUE_LIMITS,
    max_abs_torque,
    utilization,
    peak_times,
])

np.savetxt(
    summary_path,
    summary_data,
    delimiter=",",
    header=(
        "joint,"
        "simulation_torque_limit_Nm,"
        "max_required_torque_Nm,"
        "utilization_percent,"
        "peak_time_s"
    ),
    comments="",
    fmt=[
        "%d",
        "%.12e",
        "%.12e",
        "%.12e",
        "%.12e",
    ],
)


# ---------------------------------------------------------------------
# Figure: Inverse-dynamics feedforward torque
# ---------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(7.5, 4.8)
)

for joint_index in range(6):

    ax.plot(
        times,
        tau_history[:, joint_index],
        linewidth=1.8,
        label=f"J{joint_index + 1}",
    )


ax.set_xlabel("Time [s]")
ax.set_ylabel("Inverse-dynamics torque [N m]")

ax.set_title(
    "Nominal Feedforward Torque for Benchmark Trajectory"
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
    / "benchmark_inverse_dynamics.png"
)

fig.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ---------------------------------------------------------------------
# Figure: Peak torque utilization
# ---------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(7.0, 4.5)
)

joint_labels = [
    "J1",
    "J2",
    "J3",
    "J4",
    "J5",
    "J6",
]

ax.bar(
    joint_labels,
    utilization,
)

ax.axhline(
    100.0,
    linestyle="--",
    linewidth=1.5,
    label="Simulation limit",
)

ax.set_ylabel(
    "Peak torque utilization [%]"
)

ax.set_title(
    "Benchmark Torque Constraint Utilization"
)

ax.grid(
    True,
    axis="y",
    alpha=0.3,
)

ax.legend()

fig.tight_layout()

utilization_path = (
    FIGURE_DIR
    / "benchmark_torque_utilization.png"
)

fig.savefig(
    utilization_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


print("\nSaved:")
print(data_path)
print(summary_path)
print(figure_path)
print(utilization_path)