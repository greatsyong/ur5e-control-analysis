import numpy as np

from dynamics.rigid_body_dynamics import (
    mass_matrix,
    gravity_vector,
    coriolis_centrifugal_vector,
)


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
        f"{np.linalg.norm(g):.6f}"
    )

    print(
        "\nCoriolis/centrifugal torque "
        "c(q, qdot) [N m]:"
    )
    print(c)

    print(
        "Coriolis/centrifugal torque norm [N m]: "
        f"{np.linalg.norm(c):.6f}"
    )

    if np.linalg.norm(g) > 1e-12:

        ratio = (
            np.linalg.norm(c)
            / np.linalg.norm(g)
        )

        print(
            "\n||c|| / ||g||: "
            f"{ratio:.6f}"
        )


if __name__ == "__main__":

    qdot_test = [
        20.0,
        -30.0,
        40.0,
        -20.0,
        30.0,
        10.0
    ]

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

    for name, q_deg in configurations.items():

        analyze_configuration(
            name,
            q_deg,
            qdot_test
        )