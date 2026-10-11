import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.spatial.transform import Rotation

from kinematics.forward_kinematics import forward_kinematics
from kinematics.inverse_kinematics import inverse_kinematics_dls
from kinematics.jacobian import geometric_jacobian


# ============================================================
# UR5e Heavy-Payload Transport
# Full 501-Point IK Validation: Tool0 Local X +25 Degrees
# Date: 2026-10-09
#
# Cartesian payload-center path: unchanged
# Tool0 orientation: C2 tilt envelope
# Solver: existing analytical DLS IK
#
# This is kinematic validation, NOT dynamic validation.
# ============================================================

PROJECT = Path(
    "/home/sooyong/Documents/projects/12-rl-isaac-ur5e"
)

INPUT = PROJECT / "cartesian_path_c2_candidate.csv"
OUTPUT = PROJECT / "transport_ik_xplus25_full.csv"
PLOT_JOINTS = PROJECT / "transport_ik_xplus25_joints.png"
PLOT_DIAG = PROJECT / "transport_ik_xplus25_diagnostics.png"

Q_PICK_DEG = np.array([
    -63.065,
     25.309,
    -39.914,
     14.604,
    -63.065,
      0.0
])

R_BW = np.diag([-1.0, -1.0, 1.0])
OFFSET_TOOL = np.array([0.0, 0.0, 0.025])

TILT_DEG = 25.0
RAMP_UP_END = 0.20
RAMP_DOWN_START = 0.80

IK_SETTINGS = dict(
    damping=0.05,
    max_iterations=500,
    tolerance=1e-6,
    step_size=0.5,
    max_joint_step=0.2
)

JOINT_STEP_WARNING_DEG = 5.0
SIGMA_WARNING = 0.05
CONDITION_WARNING = 100.0


def smootherstep(t):
    t = np.clip(t, 0.0, 1.0)
    return 6*t**5 - 15*t**4 + 10*t**3


def tilt_weight(s):
    if s <= RAMP_UP_END:
        return float(
            smootherstep(s / RAMP_UP_END)
        )

    if s < RAMP_DOWN_START:
        return 1.0

    return float(
        1.0 - smootherstep(
            (s - RAMP_DOWN_START)
            / (1.0 - RAMP_DOWN_START)
        )
    )


def target_orientation(R_initial, s):
    tilt_rad = np.deg2rad(TILT_DEG * tilt_weight(s))

    R_delta = Rotation.from_rotvec(
        [tilt_rad, 0.0, 0.0]
    ).as_matrix()

    return R_initial @ R_delta


# ============================================================
# 1. Load original Cartesian trajectory
# ============================================================

if not INPUT.is_file():
    raise FileNotFoundError(INPUT)

data = np.genfromtxt(
    INPUT,
    delimiter=",",
    names=True
)

s = np.atleast_1d(
    np.asarray(data["s"], dtype=float)
)

payload_world = np.column_stack([
    data["x_m"],
    data["y_m"],
    data["z_m"]
])

assert len(s) == 501, (
    f"Expected 501 samples, found {len(s)}"
)
assert np.all(np.isfinite(payload_world))
assert np.all(np.diff(s) > 0)

payload_base = (R_BW @ payload_world.T).T

q_initial = np.deg2rad(Q_PICK_DEG)
R_initial = forward_kinematics(q_initial)[:3, :3]

N = len(s)

# ============================================================
# 2. Allocate results
# ============================================================

q_result = np.full((N, 6), np.nan)
tool_base = np.full((N, 3), np.nan)

position_errors = np.full(N, np.nan)
orientation_errors = np.full(N, np.nan)

sigma_min = np.full(N, np.nan)
condition = np.full(N, np.nan)
iterations = np.zeros(N, dtype=int)
tilt_angles = np.zeros(N)

success = np.zeros(N, dtype=bool)

q_previous = q_initial.copy()

print("\n========== FULL IK VALIDATION ==========")
print("Samples:", N)
print("Tool0 tilt: local X +25 degrees")
print("Tilt envelope:", RAMP_UP_END, RAMP_DOWN_START)
print("Payload path: UNCHANGED")
print()

# ============================================================
# 3. Solve continuous IK
# ============================================================

for i, si in enumerate(s):

    R_target = target_orientation(R_initial, si)

    tilt_angles[i] = TILT_DEG * tilt_weight(si)

    p_tool = (
        payload_base[i] - R_target @ OFFSET_TOOL
    )

    tool_base[i] = p_tool

    T_target = np.eye(4)
    T_target[:3, :3] = R_target
    T_target[:3, 3] = p_tool

    q, converged, history, _ = inverse_kinematics_dls(
        T_target,
        q_previous,
        **IK_SETTINGS
    )

    iterations[i] = len(history)

    if not converged:
        print(
            f"[IK FAILURE] i={i}, s={si:.6f}, "
            f"error={history[-1]:.6e}"
        )
        break

    q_result[i] = q
    success[i] = True

    T_fk = forward_kinematics(q)

    position_errors[i] = np.linalg.norm(
        T_fk[:3, 3] - p_tool
    )

    R_error = R_target.T @ T_fk[:3, :3]

    cos_angle = np.clip(
        (np.trace(R_error) - 1.0) / 2.0,
        -1.0, 1.0
    )

    orientation_errors[i] = np.arccos(cos_angle)

    J = geometric_jacobian(q)
    singular_values = np.linalg.svd(
        J, compute_uv=False
    )

    sigma_min[i] = singular_values[-1]
    condition[i] = (
        singular_values[0]
        / max(singular_values[-1], 1e-12)
    )

    q_previous = q.copy()

    if i % 50 == 0 or i == N - 1:
        print(
            f"{i:3d}/{N-1} | "
            f"tilt={tilt_angles[i]:6.2f} deg | "
            f"sigma_min={sigma_min[i]:.5f} | "
            f"cond={condition[i]:.2f}"
        )


# ============================================================
# 4. Joint continuity
# ============================================================

valid_count = int(np.sum(success))

if valid_count == 0:
    raise RuntimeError("No IK points converged")

q_valid = np.unwrap(
    q_result[:valid_count],
    axis=0
)

if valid_count > 1:
    dq = np.diff(q_valid, axis=0)

    max_step_each_deg = np.rad2deg(
        np.max(np.abs(dq), axis=0)
    )

    max_step_deg = float(
        np.max(max_step_each_deg)
    )

    discontinuities = np.where(
        np.rad2deg(np.max(np.abs(dq), axis=1))
        > JOINT_STEP_WARNING_DEG
    )[0]

else:
    max_step_each_deg = np.full(6, np.nan)
    max_step_deg = np.nan
    discontinuities = np.array([], dtype=int)


# ============================================================
# 5. Summary
# ============================================================

print("\n========== VALIDATION SUMMARY ==========")

print(f"IK convergence: {valid_count}/{N}")
print("Full path passed:", valid_count == N)

print(
    "Maximum position error [mm]:",
    np.nanmax(position_errors) * 1000
)

print(
    "Maximum orientation error [deg]:",
    np.rad2deg(np.nanmax(orientation_errors))
)

print("Minimum singular value:", np.nanmin(sigma_min))
print("Maximum condition number:", np.nanmax(condition))
print("Maximum joint step [deg]:", max_step_deg)
print("Maximum step per joint [deg]:", max_step_each_deg)

print("Joint discontinuity warnings:", len(discontinuities))
print(
    "Jacobian warning points:",
    int(np.sum(
        (sigma_min[:valid_count] < SIGMA_WARNING)
        | (condition[:valid_count] > CONDITION_WARNING)
    ))
)

print("Start joint angles [deg]:")
print(np.rad2deg(q_valid[0]))

print("Last valid joint angles [deg]:")
print(np.rad2deg(q_valid[-1]))

print("\nManufacturer joint limits: NOT VERIFIED")
print("Robot/table collisions: NOT VERIFIED")
print("Time-based velocity/acceleration: NOT VERIFIED")
print("Required joint torques: NOT VERIFIED")
print("Payload tilt collision/clearance: NOT VERIFIED")


# ============================================================
# 6. Save detailed CSV
# ============================================================

output = np.column_stack([
    s,
    payload_world,
    payload_base,
    tool_base,
    tilt_angles,
    np.rad2deg(q_result),
    success.astype(int),
    position_errors,
    orientation_errors,
    sigma_min,
    condition,
    iterations
])

header = ",".join([
    "s",
    "payload_world_x",
    "payload_world_y",
    "payload_world_z",
    "payload_base_x",
    "payload_base_y",
    "payload_base_z",
    "tool_base_x",
    "tool_base_y",
    "tool_base_z",
    "tool_tilt_deg",
    "q1_deg",
    "q2_deg",
    "q3_deg",
    "q4_deg",
    "q5_deg",
    "q6_deg",
    "success",
    "fk_position_error_m",
    "fk_orientation_error_rad",
    "sigma_min",
    "condition_number",
    "iterations"
])

np.savetxt(
    OUTPUT,
    output,
    delimiter=",",
    header=header,
    comments=""
)


# ============================================================
# 7. Plot joint trajectory
# ============================================================

fig, axes = plt.subplots(
    3, 2,
    figsize=(13, 10),
    sharex=True
)

for j, ax in enumerate(axes.flat):

    ax.plot(
        s[:valid_count],
        np.rad2deg(q_valid[:, j]),
        linewidth=1.8
    )

    ax.set_title(f"Joint {j+1}")
    ax.set_ylabel("Angle [deg]")
    ax.set_xlabel("Path parameter s")
    ax.grid(True, alpha=0.3)

fig.suptitle(
    "UR5e Heavy-Payload Transport - X+25 IK"
)

fig.tight_layout()
fig.savefig(PLOT_JOINTS, dpi=180)
plt.close(fig)


# ============================================================
# 8. Plot diagnostics
# ============================================================

fig, axes = plt.subplots(
    4, 1,
    figsize=(11, 12),
    sharex=True
)

axes[0].plot(s, tilt_angles)
axes[0].set_ylabel("Tool tilt [deg]")

axes[1].plot(s[:valid_count], sigma_min[:valid_count])
axes[1].axhline(SIGMA_WARNING, linestyle="--")
axes[1].set_ylabel("Jacobian sigma min")

axes[2].plot(s[:valid_count], condition[:valid_count])
axes[2].axhline(CONDITION_WARNING, linestyle="--")
axes[2].set_ylabel("Condition number")

axes[3].plot(
    s[:valid_count],
    position_errors[:valid_count] * 1000
)
axes[3].set_ylabel("FK error [mm]")
axes[3].set_xlabel("Path parameter s")

for ax in axes:
    ax.grid(True, alpha=0.3)

fig.suptitle(
    "UR5e X+25 Orientation - Kinematic Diagnostics"
)

fig.tight_layout()
fig.savefig(PLOT_DIAG, dpi=180)
plt.close(fig)

print("\n========== SAVED FILES ==========")
print(OUTPUT)
print(PLOT_JOINTS)
print(PLOT_DIAG)
