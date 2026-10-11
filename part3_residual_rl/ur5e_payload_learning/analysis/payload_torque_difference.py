
#!/usr/bin/env python3
"""UR5e ideal payload-mismatch torque analysis.

Compute inverse-dynamics torque differences between
3 kg and 5 kg payload models along the accepted
3-second reference trajectory.

This is a reference-model comparison, not an RL
policy evaluation or proof of optimal residual action.
"""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path.home() / "robotics_ws/src/ur5e_payload_learning"

MODEL_DIR = (
    Path.home()
    / "robotics_ws/src/ur5e_control_analysis"
    / "part1_modeling_control_analysis"
)

REFERENCE = (
    Path.home()
    / "Documents/projects/12-rl-isaac-ur5e/references"
    / "inverse_dynamics_1kg_cube_3s_retimed_v2.csv"
)

OUTPUT = ROOT / "results/rl/payload_torque_difference.csv"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MODEL_DIR))

from optimization.pd_gain_search import load_reference, interp6

from dynamics.payload_dynamics import (
    mass_matrix_payload,
    coriolis_centrifugal_vector_payload,
    gravity_vector_payload,
)

COM = np.array([0.0, 0.0, 0.025])

LIMITS = np.array([
    5.0, 5.0, 5.0, 0.5, 0.5, 0.01
])


def inverse_dynamics(q, qd, qdd, payload_mass):

    M = mass_matrix_payload(
        q,
        payload_mass=payload_mass,
        payload_com_tool=COM,
    )

    c = coriolis_centrifugal_vector_payload(
        q,
        qd,
        payload_mass=payload_mass,
        payload_com_tool=COM,
    )

    g = gravity_vector_payload(
        q,
        payload_mass=payload_mass,
        payload_com_tool=COM,
    )

    return (
        np.asarray(M) @ qdd
        + np.asarray(c)
        + np.asarray(g)
    )


def main():

    ts, qs, qds, qdds = load_reference(REFERENCE)

    # Match the 150 control instants of the RL evaluation.
    times = np.arange(1, 151, dtype=float) * 0.02

    ideal = []
    gravity_only = []
    records = []

    for t in times:

        q = interp6(t, ts, qs)
        qd = interp6(t, ts, qds)
        qdd = interp6(t, ts, qdds)

        tau3 = inverse_dynamics(q, qd, qdd, 3.0)
        tau5 = inverse_dynamics(q, qd, qdd, 5.0)

        g3 = gravity_vector_payload(
            q, payload_mass=3.0, payload_com_tool=COM
        )

        g5 = gravity_vector_payload(
            q, payload_mass=5.0, payload_com_tool=COM
        )

        delta = tau5 - tau3
        delta_g = np.asarray(g5) - np.asarray(g3)

        ideal.append(delta)
        gravity_only.append(delta_g)

        row = {"t_s": round(float(t), 8)}

        for j in range(6):

            k = j + 1

            row[f"ideal_tau{k}_Nm"] = float(delta[j])
            row[f"gravity_diff{k}_Nm"] = float(delta_g[j])

        records.append(row)

    ideal = np.asarray(ideal)
    gravity_only = np.asarray(gravity_only)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0]),
        )

        writer.writeheader()
        writer.writerows(records)

    print("\nUR5e PAYLOAD MISMATCH TORQUE")
    print("Nominal model: 3 kg")
    print("Actual model: 5 kg")
    print("Reference: accepted 3-second trajectory")
    print("Samples:", len(times))

    print("\n" + "-" * 91)
    print(
        f"{'Joint':<9}"
        f"{'Mean [Nm]':>13}"
        f"{'RMS [Nm]':>13}"
        f"{'Min [Nm]':>13}"
        f"{'Max [Nm]':>13}"
        f"{'g-diff RMS':>14}"
        f"{'Action limit':>15}"
    )
    print("-" * 91)

    for j in range(6):

        x = ideal[:, j]
        g = gravity_only[:, j]

        rms = np.sqrt(np.mean(x**2))
        g_rms = np.sqrt(np.mean(g**2))

        print(
            f"J{j+1:<8}"
            f"{np.mean(x):13.4f}"
            f"{rms:13.4f}"
            f"{np.min(x):13.4f}"
            f"{np.max(x):13.4f}"
            f"{g_rms:14.4f}"
            f"{LIMITS[j]:15.4f}"
        )

    print("\nSaved:", OUTPUT)
    print(
        "\nNOTE: Reference inverse dynamics is not "
        "identical to the optimal closed-loop RL correction."
    )


if __name__ == "__main__":
    main()

