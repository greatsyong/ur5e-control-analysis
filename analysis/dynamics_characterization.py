from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
# Configuration analysis
# ---------------------------------------------------------------------

def analyze_configuration(
    name,
    q_deg,
    qdot_deg_s
):
    q_deg = np.asarray(q_deg, dtype=float)
    qdot_deg_s = np.asarray(
        qdot_deg_s,
        dtype=float
    )

    q = np.deg2rad(q_deg)
    qdot = np.deg2rad(qdot_deg_s)

    M = mass_matrix(q)
    g = gravity_vector(q)

    c = coriolis_centrifugal_vector(
        q,
        qdot
    )

    eigenvalues = np.linalg.eigvalsh(M)

    condition_number = (
        eigenvalues[-1]
        / eigenvalues[0]
    )

    gravity_norm = np.linalg.norm(g)
    coriolis_norm = np.linalg.norm(c)

    if gravity_norm > 1e-12:
        c_to_g_ratio = (
            coriolis_norm
            / gravity_norm
        )
    else:
        c_to_g_ratio = np.nan


    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)

    print("q [deg]:")
    print(q_deg)

    print("\nqdot [deg/s]:")
    print(qdot_deg_s)

    print("\nMass matrix M(q) [kg m^2]:")
    print(
        np.array2string(
            M,
            precision=6,
            suppress_small=True
        )
    )

    print("\nMass-matrix eigenvalues:")
    print(eigenvalues)

    print(
        "\nMass-matrix condition number: "
        f"{condition_number:.6f}"
    )

    print("\nGravity torque g(q) [N m]:")
    print(g)

    print(
        "Gravity torque norm [N m]: "
        f"{gravity_norm:.6f}"
    )

    print(
        "\nCoriolis/centrifugal torque "
        "c(q, qdot) [N m]:"
    )
    print(c)

    print(
        "Coriolis/centrifugal torque norm [N m]: "
        f"{coriolis_norm:.6f}"
    )

    print(
        "\n||c|| / ||g||: "
        f"{c_to_g_ratio:.6f}"
    )


    return {
        "name": name,
        "q_deg": q_deg,
        "qdot_deg_s": qdot_deg_s,
        "M": M,
        "eigenvalues": eigenvalues,
        "condition_number": condition_number,
        "g": g,
        "gravity_norm": gravity_norm,
        "c": c,
        "coriolis_norm": coriolis_norm,
        "c_to_g_ratio": c_to_g_ratio,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":

    qdot_test = np.array([
        20.0,
        -30.0,
        40.0,
        -20.0,
        30.0,
        10.0
    ])

    configurations = {

        "validated_test_pose": [
            10.0,
            -40.0,
            60.0,
            -30.0,
            45.0,
            20.0
        ],

        "nominal_working_pose": [
            0.0,
            -90.0,
            90.0,
            -90.0,
            -90.0,
            0.0
        ],

        "zero_configuration": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
    }


    results = []

    for name, q_deg in configurations.items():

        result = analyze_configuration(
            name,
            q_deg,
            qdot_test
        )

        results.append(result)


    # -----------------------------------------------------------------
    # Save summary data
    # -----------------------------------------------------------------

    summary_rows = []

    for result in results:

        summary_rows.append([
            result["condition_number"],
            result["eigenvalues"][0],
            result["eigenvalues"][-1],
            result["gravity_norm"],
            result["coriolis_norm"],
            result["c_to_g_ratio"],
        ])


    summary_data = np.asarray(summary_rows)

    summary_path = (
        DATA_DIR
        / "dynamics_characterization_summary.csv"
    )

    header = (
        "configuration,"
        "mass_matrix_condition_number,"
        "mass_matrix_lambda_min,"
        "mass_matrix_lambda_max,"
        "gravity_norm_Nm,"
        "coriolis_norm_Nm,"
        "coriolis_to_gravity_ratio"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(header + "\n")

        for result, row in zip(
            results,
            summary_data
        ):

            file.write(
                result["name"]
                + ","
                + ",".join(
                    f"{value:.12e}"
                    for value in row
                )
                + "\n"
            )


    # -----------------------------------------------------------------
    # Save full mass matrices
    # -----------------------------------------------------------------

    for result in results:

        matrix_path = (
            DATA_DIR
            / f"mass_matrix_{result['name']}.csv"
        )

        np.savetxt(
            matrix_path,
            result["M"],
            delimiter=",",
            fmt="%.12e",
        )


    # -----------------------------------------------------------------
    # Figure:
    # Configuration dependence of generalized dynamic torques
    # -----------------------------------------------------------------

    labels = [
        "Validated\npose",
        "Nominal\npose",
        "Zero\nconfiguration",
    ]

    gravity_norms = np.array([
        result["gravity_norm"]
        for result in results
    ])

    coriolis_norms = np.array([
        result["coriolis_norm"]
        for result in results
    ])

    x = np.arange(len(labels))

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(7.0, 4.5)
    )

    ax.bar(
        x - width / 2.0,
        gravity_norms,
        width,
        label=r"$\|g(q)\|_2$",
    )

    ax.bar(
        x + width / 2.0,
        coriolis_norms,
        width,
        label=r"$\|c(q,\dot{q})\|_2$",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylabel("Generalized torque norm [N m]")

    ax.set_title(
        "Configuration-Dependent Dynamic Terms"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    figure_path = (
        FIGURE_DIR
        / "dynamics_configuration_dependence.png"
    )

    fig.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    print("\nSaved:")
    print(summary_path)

    for result in results:
        print(
            DATA_DIR
            / f"mass_matrix_{result['name']}.csv"
        )

    print(figure_path)