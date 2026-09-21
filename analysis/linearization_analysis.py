import numpy as np

from dynamics.state_space import (
    equilibrium_input,
    linearize_dynamics,
)


def controllability_matrix(A, B):
    """
    Construct the controllability matrix:

        C = [B, AB, A^2 B, ..., A^(n-1) B]
    """

    n = A.shape[0]

    blocks = [B]

    A_power = np.eye(n)

    for _ in range(1, n):

        A_power = A_power @ A
        blocks.append(A_power @ B)

    return np.hstack(blocks)


def analyze_operating_point(name, q_deg):

    q_deg = np.asarray(q_deg, dtype=float)
    q = np.deg2rad(q_deg)

    qdot = np.zeros(6)

    x_eq = np.concatenate([
        q,
        qdot
    ])

    tau_eq = equilibrium_input(q)

    A, B = linearize_dynamics(
        x_eq,
        tau_eq
    )

    eigenvalues = np.linalg.eigvals(A)

    Ctrb = controllability_matrix(
        A,
        B
    )

    controllability_rank = np.linalg.matrix_rank(
        Ctrb,
        tol=1e-8
    )

    print("\n" + "=" * 78)
    print(name)
    print("=" * 78)

    print("q [deg]:")
    print(q_deg)

    print("\nEquilibrium torque [N m]:")
    print(tau_eq)

    print("\nA matrix:")
    print(
        np.array2string(
            A,
            precision=5,
            suppress_small=True
        )
    )

    print("\nB matrix:")
    print(
        np.array2string(
            B,
            precision=5,
            suppress_small=True
        )
    )

    print("\nEigenvalues of A:")
    for eig in eigenvalues:
        print(
            f"{eig.real:+.8f} "
            f"{eig.imag:+.8f}j"
        )

    print(
        "\nControllability rank: "
        f"{controllability_rank} / {A.shape[0]}"
    )


if __name__ == "__main__":

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

        analyze_operating_point(
            name,
            q_deg
        )