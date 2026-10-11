import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

from kinematics.forward_kinematics import forward_kinematics
from kinematics.inverse_kinematics import inverse_kinematics_dls
from kinematics.jacobian import geometric_jacobian


# ============================================================
# UR5e Heavy-Payload Transport
# Tool Orientation Candidate Screening
# Date: 2026-10-09
#
# Cartesian payload path: unchanged
# Tool orientation: variable
# IK: existing DLS solver
# ============================================================

PROJECT = Path(
    "/home/sooyong/Documents/projects/12-rl-isaac-ur5e"
)

INPUT_FILE = PROJECT / "cartesian_path_c2_candidate.csv"
OUTPUT_FILE = PROJECT / "orientation_candidate_results.csv"

Q_PICK_DEG = np.array([
    -63.065,
    25.309,
    -39.914,
    14.604,
    -63.065,
    0.0
])

OFFSET_TOOL = np.array([0.0, 0.0, 0.025])

# World -> Analytical Base
R_BW = np.diag([-1.0, -1.0, 1.0])

# Start and finish with the original tool orientation.
# Smoothly tilt during the middle of the transport.
RAMP_UP_END = 0.20
RAMP_DOWN_START = 0.80

# Screening samples, including both endpoints.
SCREEN_SAMPLES = 101

# Candidate rotations about Tool0 local X or Y.
# Degrees and signs are deliberately tested rather than assumed.
CANDIDATES = [
    ("X+15", "x", +15.0),
    ("X-15", "x", -15.0),
    ("Y+15", "y", +15.0),
    ("Y-15", "y", -15.0),
    ("X+25", "x", +25.0),
    ("X-25", "x", -25.0),
    ("Y+25", "y", +25.0),
    ("Y-25", "y", -25.0),
]

# Numerical IK settings
IK_SETTINGS = dict(
    damping=0.05,
    max_iterations=500,
    tolerance=1e-6,
    step_size=0.5,
    max_joint_step=0.2
)


def smootherstep(t):
    """Quintic C2 transition from 0 to 1."""
    t = np.clip(t, 0.0, 1.0)
    return 6*t**5 - 15*t**4 + 10*t**3


def tilt_weight(s):
    """C2 tilt envelope: 0 -> 1 -> 0."""
    if s <= RAMP_UP_END:
        return float(smootherstep(s / RAMP_UP_END))

    if s < RAMP_DOWN_START:
        return 1.0

    return float(
        1.0 - smootherstep(
            (s - RAMP_DOWN_START)
            / (1.0 - RAMP_DOWN_START)
        )
    )


def orientation_at_s(R_initial, s, axis, angle_deg):
    """Rotate about a Tool0-local axis."""
    weight = tilt_weight(s)

    rotvec = np.zeros(3)
    axis_index = {"x": 0, "y": 1}[axis]

    rotvec[axis_index] = np.deg2rad(angle_deg) * weight

    R_delta = Rotation.from_rotvec(
        rotvec
    ).as_matrix()

    return R_initial @ R_delta


def evaluate_candidate(
    name,
    axis,
    angle_deg,
    s_values,
    payload_base,
    R_initial,
    q_initial
):
    q_previous = q_initial.copy()

    q_history = []
    sigma_history = []
    condition_history = []

    failure_s = np.nan

    for i, si in enumerate(s_values):

        R_target = orientation_at_s(
            R_initial,
            si,
            axis,
            angle_deg
        )

        # Convert payload-center target to Tool0 target.
        p_tool = (
            payload_base[i]
            - R_target @ OFFSET_TOOL
        )

        T_target = np.eye(4)
        T_target[:3, :3] = R_target
        T_target[:3, 3] = p_tool

        q, converged, history, _ = (
            inverse_kinematics_dls(
                T_target,
                q_previous,
                **IK_SETTINGS
            )
        )

        if not converged:
            failure_s = float(si)
            break

        J = geometric_jacobian(q)
        sv = np.linalg.svd(
            J, compute_uv=False
        )

        sigma_history.append(float(sv[-1]))

        condition_history.append(
            float(sv[0] / max(sv[-1], 1e-12))
        )

        q_history.append(q.copy())
        q_previous = q.copy()

    completed = len(q_history)
    total = len(s_values)

    if completed > 1:
        q_array = np.unwrap(
            np.asarray(q_history),
            axis=0
        )

        max_step_deg = float(
            np.rad2deg(
                np.max(
                    np.abs(np.diff(q_array, axis=0))
                )
            )
        )
    else:
        max_step_deg = np.nan

    result = {
        "candidate": name,
        "axis": axis,
        "angle_deg": angle_deg,
        "success": completed == total,
        "completed": completed,
        "total": total,
        "failure_s": failure_s,
        "min_sigma": (
            min(sigma_history)
            if sigma_history else np.nan
        ),
        "max_condition": (
            max(condition_history)
            if condition_history else np.nan
        ),
        "max_step_deg": max_step_deg,
    }

    return result


# ============================================================
# Load original Cartesian path
# ============================================================

data = np.genfromtxt(
    INPUT_FILE,
    delimiter=",",
    names=True
)

s_all = np.asarray(data["s"], dtype=float)

payload_world = np.column_stack([
    data["x_m"],
    data["y_m"],
    data["z_m"]
])

payload_base_all = (
    R_BW @ payload_world.T
).T

# Select equally distributed original path samples.
indices = np.linspace(
    0, len(s_all) - 1,
    SCREEN_SAMPLES
).round().astype(int)

s_screen = s_all[indices]
payload_base_screen = payload_base_all[indices]

q_seed = np.deg2rad(Q_PICK_DEG)

R_pick = forward_kinematics(
    q_seed
)[:3, :3]

print("\n========== ORIENTATION SCREENING ==========")
print("Input points:", len(s_all))
print("Screening points:", len(s_screen))
print("Cartesian path: UNCHANGED")
print("Tool offset [m]:", OFFSET_TOOL)
print("Candidates:", len(CANDIDATES))
print()

results = []

for name, axis, angle in CANDIDATES:

    result = evaluate_candidate(
        name,
        axis,
        angle,
        s_screen,
        payload_base_screen,
        R_pick,
        q_seed
    )

    results.append(result)

    print(
        f"{name:5s} | "
        f"IK {result['completed']:3d}/{result['total']} | "
        f"failure s={result['failure_s']:.3f} | "
        f"min sigma={result['min_sigma']:.5f} | "
        f"max cond={result['max_condition']:.2f} | "
        f"max step={result['max_step_deg']:.3f} deg"
    )


# ============================================================
# Candidate ranking
# ============================================================

# Prioritize full IK completion.
# Among full solutions, prioritize larger minimum singular
# value and smaller maximum joint increment.

results_sorted = sorted(
    results,
    key=lambda r: (
        r["completed"],
        (
            r["min_sigma"]
            if np.isfinite(r["min_sigma"])
            else -1.0
        ),
        -(
            r["max_step_deg"]
            if np.isfinite(r["max_step_deg"])
            else 1e9
        )
    ),
    reverse=True
)

print("\n========== SCREENING RANKING ==========")

for rank, r in enumerate(results_sorted, start=1):
    print(
        f"{rank}. {r['candidate']}  "
        f"completed={r['completed']}/{r['total']}, "
        f"min_sigma={r['min_sigma']:.6f}"
    )

best = results_sorted[0]

print("\nBest screening candidate:", best["candidate"])
print(
    "Full screening pass:",
    bool(best["success"])
)

if not best["success"]:
    print(
        "\nNo candidate passed the entire screening path."
    )
    print(
        "Do NOT treat the best-ranked candidate "
        "as a valid full trajectory."
    )

# Save candidate summary.
with OUTPUT_FILE.open("w") as f:
    f.write(
        "candidate,axis,angle_deg,success,"
        "completed,total,failure_s,min_sigma,"
        "max_condition,max_step_deg\n"
    )

    for r in results:
        f.write(
            f"{r['candidate']},{r['axis']},"
            f"{r['angle_deg']},{int(r['success'])},"
            f"{r['completed']},{r['total']},"
            f"{r['failure_s']},{r['min_sigma']},"
            f"{r['max_condition']},"
            f"{r['max_step_deg']}\n"
        )

print("\nSaved:", OUTPUT_FILE)

print("\nValidation limitations:")
print("- Full 501-point validation: NOT YET")
print("- Collision checking: NOT YET")
print("- Joint limits: NOT YET")
print("- Dynamic torque feasibility: NOT YET")
