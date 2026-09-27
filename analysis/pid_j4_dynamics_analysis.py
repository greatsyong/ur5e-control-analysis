import os

import numpy as np
import matplotlib.pyplot as plt

from dynamics.rigid_body_dynamics import (
    mass_matrix,
    gravity_vector,
    coriolis_centrifugal_vector,
)


# ============================================================
# Load final PID benchmark
# ============================================================

DATA_PATH = "results/pid/pid_final_benchmark.npz"

data = np.load(DATA_PATH)

time = data["time"]
q = data["q"]
qdot = data["qdot"]
q_ref = data["q_ref"]
qdot_ref = data["qdot_ref"]
tau_pid = data["tau"]

DT = time[1] - time[0]

# J4 -> Python index 3
J = 3


# ============================================================
# Reference acceleration
#
# qddot_ref was not saved in the original PID benchmark,
# so reconstruct it from qdot_ref.
# ============================================================

qddot_ref = np.gradient(
    qdot_ref,
    DT,
    axis=0,
    edge_order=2,
)


# ============================================================
# J4 dynamics decomposition
#
# tau_4 =
#
#   M44 * qddot_ref_4
# + sum(M4j * qddot_ref_j), j != 4
# + c4(q, qdot)
# + g4(q)
#
# This evaluates the inverse-dynamics demand along the
# ACTUAL PID state q(t), qdot(t), using the REFERENCE
# acceleration as the desired acceleration.
# ============================================================

n = len(time)

self_inertia = np.zeros(n)
inertial_coupling = np.zeros(n)

coriolis = np.zeros(n)
gravity = np.zeros(n)

tau_inverse_dynamics = np.zeros(n)

for k in range(n):

    M = mass_matrix(q[k])

    c = coriolis_centrifugal_vector(
        q[k],
        qdot[k],
    )

    g = gravity_vector(q[k])

    # J4 self-inertia contribution
    self_inertia[k] = (
        M[J, J]
        * qddot_ref[k, J]
    )

    # J4 inertial coupling from all other joints
    coupling = 0.0

    for j in range(6):

        if j == J:
            continue

        coupling += (
            M[J, j]
            * qddot_ref[k, j]
        )

    inertial_coupling[k] = coupling

    # Velocity-dependent dynamics
    coriolis[k] = c[J]

    # Gravity
    gravity[k] = g[J]

    # Total inverse-dynamics torque required for
    # reference acceleration at the actual PID state
    tau_inverse_dynamics[k] = (
        self_inertia[k]
        + inertial_coupling[k]
        + coriolis[k]
        + gravity[k]
    )


# ============================================================
# J4 tracking error
# ============================================================

j4_error = (
    q_ref[:, J]
    - q[:, J]
)

j4_error_deg = np.rad2deg(
    j4_error
)


# ============================================================
# PID torque deficit relative to inverse dynamics
# ============================================================

j4_pid_torque = tau_pid[:, J]

torque_difference = (
    tau_inverse_dynamics
    - j4_pid_torque
)


# ============================================================
# Peak-error location
# ============================================================

peak_idx = np.argmax(
    np.abs(j4_error)
)

peak_time = time[peak_idx]


# ============================================================
# Print numerical summary
# ============================================================

print("\n" + "=" * 88)
print("PID J4 DYNAMICS DECOMPOSITION")
print("=" * 88)

print(
    f"\nPeak |J4 tracking error| time : "
    f"{peak_time:.6f} s"
)

print(
    f"J4 tracking error at peak    : "
    f"{j4_error_deg[peak_idx]:.6f} deg"
)

print("\nDynamics at peak-error time:")

print(
    f"  Self inertia       : "
    f"{self_inertia[peak_idx]: .6f} Nm"
)

print(
    f"  Inertial coupling  : "
    f"{inertial_coupling[peak_idx]: .6f} Nm"
)

print(
    f"  Coriolis/centrif.  : "
    f"{coriolis[peak_idx]: .6f} Nm"
)

print(
    f"  Gravity            : "
    f"{gravity[peak_idx]: .6f} Nm"
)

print(
    f"  Total inv. dynamics: "
    f"{tau_inverse_dynamics[peak_idx]: .6f} Nm"
)

print(
    f"  Actual PID torque  : "
    f"{j4_pid_torque[peak_idx]: .6f} Nm"
)

print(
    f"  Torque difference  : "
    f"{torque_difference[peak_idx]: .6f} Nm"
)


print("\nMaximum absolute contribution over trajectory:")

print(
    f"  Self inertia       : "
    f"{np.max(np.abs(self_inertia)):.6f} Nm"
)

print(
    f"  Inertial coupling  : "
    f"{np.max(np.abs(inertial_coupling)):.6f} Nm"
)

print(
    f"  Coriolis/centrif.  : "
    f"{np.max(np.abs(coriolis)):.6f} Nm"
)

print(
    f"  Gravity            : "
    f"{np.max(np.abs(gravity)):.6f} Nm"
)

print(
    f"  Torque difference  : "
    f"{np.max(np.abs(torque_difference)):.6f} Nm"
)


# ============================================================
# Figure 1
# J4 tracking error + dynamics decomposition
# ============================================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(11, 8),
    sharex=True,
)

axes[0].plot(
    time,
    j4_error_deg,
    label="J4 tracking error",
)

axes[0].axvline(
    peak_time,
    linestyle="--",
    label="Peak error",
)

axes[0].set_ylabel(
    "J4 error [deg]"
)

axes[0].grid(True)
axes[0].legend()


axes[1].plot(
    time,
    self_inertia,
    label="Self inertia",
)

axes[1].plot(
    time,
    inertial_coupling,
    label="Inertial coupling",
)

axes[1].plot(
    time,
    coriolis,
    label="Coriolis / centrifugal",
)

axes[1].plot(
    time,
    gravity,
    label="Gravity",
)

axes[1].axvline(
    peak_time,
    linestyle="--",
)

axes[1].set_xlabel(
    "Time [s]"
)

axes[1].set_ylabel(
    "J4 torque contribution [Nm]"
)

axes[1].grid(True)
axes[1].legend()


fig.suptitle(
    "PID J4 Tracking Error and Dynamic Torque Contributions"
)

fig.tight_layout()


# ============================================================
# Figure 2
# PID torque vs inverse-dynamics demand
# ============================================================

fig2, axes2 = plt.subplots(
    2,
    1,
    figsize=(11, 8),
    sharex=True,
)


axes2[0].plot(
    time,
    j4_pid_torque,
    label="PID torque",
)

axes2[0].plot(
    time,
    tau_inverse_dynamics,
    "--",
    label="Inverse-dynamics demand",
)

axes2[0].axvline(
    peak_time,
    linestyle="--",
)

axes2[0].set_ylabel(
    "J4 torque [Nm]"
)

axes2[0].grid(True)
axes2[0].legend()


axes2[1].plot(
    time,
    torque_difference,
    label="Inverse dynamics - PID",
)

axes2[1].axhline(
    0.0,
    linestyle="--",
)

axes2[1].axvline(
    peak_time,
    linestyle="--",
)

axes2[1].set_xlabel(
    "Time [s]"
)

axes2[1].set_ylabel(
    "Torque difference [Nm]"
)

axes2[1].grid(True)
axes2[1].legend()


fig2.suptitle(
    "PID J4 Torque vs Model-Based Dynamic Demand"
)

fig2.tight_layout()


# ============================================================
# Save
# ============================================================

output_dir = (
    "results/pid/j4_dynamics"
)

os.makedirs(
    output_dir,
    exist_ok=True,
)

fig.savefig(
    os.path.join(
        output_dir,
        "pid_j4_dynamics_decomposition.png",
    ),
    dpi=300,
)

fig2.savefig(
    os.path.join(
        output_dir,
        "pid_j4_torque_comparison.png",
    ),
    dpi=300,
)

plt.close(fig)
plt.close(fig2)


np.savez(
    os.path.join(
        output_dir,
        "pid_j4_dynamics_analysis.npz",
    ),
    time=time,
    j4_error=j4_error,
    self_inertia=self_inertia,
    inertial_coupling=inertial_coupling,
    coriolis=coriolis,
    gravity=gravity,
    tau_inverse_dynamics=tau_inverse_dynamics,
    tau_pid=j4_pid_torque,
    torque_difference=torque_difference,
)


print("\nFiles saved:")
print(
    "  results/pid/j4_dynamics/"
    "pid_j4_dynamics_decomposition.png"
)

print(
    "  results/pid/j4_dynamics/"
    "pid_j4_torque_comparison.png"
)

print(
    "  results/pid/j4_dynamics/"
    "pid_j4_dynamics_analysis.npz"
)