import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

RESULT_FILE = (
    "results/lqr/nominal/"
    "lqr_nominal_benchmark.npz"
)

OUTPUT_DIR = "results/lqr/figures"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# Load nominal LQR result
# ============================================================

data = np.load(
    RESULT_FILE
)

time = data["time"]
q = data["q"]
q_ref = data["q_ref"]
tau = data["tau"]


# ============================================================
# Unit conversion
# ============================================================

q_deg = np.rad2deg(q)
q_ref_deg = np.rad2deg(q_ref)

error_deg = np.rad2deg(
    q_ref - q
)


# ============================================================
# Plot configuration
#
# Intentionally larger than Matplotlib defaults because these
# figures will be scaled down when inserted into the report.
# ============================================================

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "lines.linewidth": 1.8,
})


joint_names = [
    "J1",
    "J2",
    "J3",
    "J4",
    "J5",
    "J6",
]


# ============================================================
# 1. Joint trajectory tracking
# ============================================================

fig, axes = plt.subplots(
    3,
    2,
    figsize=(11, 9),
    sharex=True,
)

axes = axes.flatten()

for i, ax in enumerate(axes):

    ax.plot(
        time,
        q_ref_deg[:, i],
        "--",
        label="Reference",
    )

    ax.plot(
        time,
        q_deg[:, i],
        label="LQR",
    )

    ax.set_title(
        joint_names[i]
    )

    ax.set_ylabel(
        "Position [deg]"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    if i >= 4:
        ax.set_xlabel(
            "Time [s]"
        )

    if i == 0:
        ax.legend(
            loc="best"
        )


fig.suptitle(
    "Discrete LQR Joint-Space Trajectory Tracking",
    fontsize=16,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.96]
)

tracking_file = os.path.join(
    OUTPUT_DIR,
    "lqr_joint_tracking.png",
)

fig.savefig(
    tracking_file,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 2. Joint tracking error
# ============================================================

fig, axes = plt.subplots(
    3,
    2,
    figsize=(11, 9),
    sharex=True,
)

axes = axes.flatten()

for i, ax in enumerate(axes):

    ax.plot(
        time,
        error_deg[:, i],
    )

    ax.axhline(
        0.0,
        linewidth=1.0,
    )

    ax.set_title(
        joint_names[i]
    )

    ax.set_ylabel(
        "Error [deg]"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    if i >= 4:
        ax.set_xlabel(
            "Time [s]"
        )


fig.suptitle(
    "Discrete LQR Joint Tracking Error",
    fontsize=16,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.96]
)

error_file = os.path.join(
    OUTPUT_DIR,
    "lqr_joint_error.png",
)

fig.savefig(
    error_file,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 3. Joint torque histories
# ============================================================

torque_limits = np.array([
    150.0,
    150.0,
    150.0,
    28.0,
    28.0,
    28.0,
])


fig, axes = plt.subplots(
    3,
    2,
    figsize=(11, 9),
    sharex=True,
)

axes = axes.flatten()

for i, ax in enumerate(axes):

    ax.plot(
        time,
        tau[:, i],
        label="LQR torque",
    )

    ax.axhline(
        torque_limits[i],
        linestyle="--",
        linewidth=1.2,
        label="Torque limit"
        if i == 0
        else None,
    )

    ax.axhline(
        -torque_limits[i],
        linestyle="--",
        linewidth=1.2,
    )

    ax.set_title(
        joint_names[i]
    )

    ax.set_ylabel(
        "Torque [N m]"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    if i >= 4:
        ax.set_xlabel(
            "Time [s]"
        )

    if i == 0:
        ax.legend(
            loc="best"
        )


fig.suptitle(
    "Discrete LQR Joint Torque Histories",
    fontsize=16,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.96]
)

torque_file = os.path.join(
    OUTPUT_DIR,
    "lqr_joint_torque.png",
)

fig.savefig(
    torque_file,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Summary
# ============================================================

print("\nReport figures generated:")
print(tracking_file)
print(error_file)
print(torque_file)

print("\nData source:")
print(RESULT_FILE)

print("\nSamples:")
print(len(time))

print("\nTime range:")
print(
    f"{time[0]:.3f} s "
    f"to {time[-1]:.3f} s"
)