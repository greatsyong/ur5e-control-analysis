from pathlib import Path

import numpy as np

from dynamics.rigid_body_dynamics import (
    mass_matrix,
    coriolis_matrix,
    coriolis_centrifugal_vector,
)


RESULTS_DIR = Path("results/data")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def mass_matrix_time_derivative(q, qdot, epsilon=1e-6):
    """
    Directional time derivative:

        M_dot = sum_k (dM/dq_k) qdot_k

    evaluated by central difference along qdot.
    """
    M_plus = mass_matrix(q + epsilon * qdot)
    M_minus = mass_matrix(q - epsilon * qdot)

    return (M_plus - M_minus) / (2.0 * epsilon)


def validate_configuration(name, q, qdot, epsilon=1e-6):
    C = coriolis_matrix(q, qdot, epsilon=epsilon)

    c_vector_reference = coriolis_centrifugal_vector(
        q, qdot, epsilon=epsilon
    )

    c_vector_from_C = C @ qdot

    c_consistency_error = np.max(
        np.abs(c_vector_reference - c_vector_from_C)
    )

    M_dot = mass_matrix_time_derivative(
        q, qdot, epsilon=epsilon
    )

    N = M_dot - 2.0 * C

    skew_residual = N + N.T

    skew_max_error = np.max(np.abs(skew_residual))
    skew_fro_error = np.linalg.norm(skew_residual, ord="fro")

    print("=" * 70)
    print(name)
    print("=" * 70)

    print("\nC(q, qdot) @ qdot:")
    print(c_vector_from_C)

    print("\nExisting coriolis_centrifugal_vector:")
    print(c_vector_reference)

    print(
        "\nC @ qdot consistency max error:"
        f" {c_consistency_error:.12e}"
    )

    print(
        "Skew-symmetry max residual:"
        f" {skew_max_error:.12e}"
    )

    print(
        "Skew-symmetry Frobenius residual:"
        f" {skew_fro_error:.12e}"
    )

    return {
        "configuration": name,
        "c_consistency_max_error": c_consistency_error,
        "skew_max_error": skew_max_error,
        "skew_fro_error": skew_fro_error,
    }


def main():
    qdot = np.radians(
        [20.0, -30.0, 40.0, -20.0, 30.0, 10.0]
    )

    configurations = {
        "validated_test_pose":
            np.radians([10, -40, 60, -30, 45, 20]),

        "nominal_working_pose":
            np.radians([0, -90, 90, -90, -90, 0]),

        "zero_configuration":
            np.zeros(6),
    }

    results = []

    for name, q in configurations.items():
        results.append(
            validate_configuration(
                name,
                q,
                qdot,
                epsilon=1e-6,
            )
        )

    output_file = (
        RESULTS_DIR
        / "coriolis_structural_validation.csv"
    )

    with output_file.open("w") as f:
        f.write(
            "configuration,"
            "c_consistency_max_error,"
            "skew_max_error,"
            "skew_fro_error\n"
        )

        for result in results:
            f.write(
                f"{result['configuration']},"
                f"{result['c_consistency_max_error']:.16e},"
                f"{result['skew_max_error']:.16e},"
                f"{result['skew_fro_error']:.16e}\n"
            )

    print("\n" + "=" * 70)

    max_c_error = max(
        r["c_consistency_max_error"]
        for r in results
    )

    max_skew_error = max(
        r["skew_max_error"]
        for r in results
    )

    print(
        "Maximum C @ qdot consistency error:"
        f" {max_c_error:.12e}"
    )

    print(
        "Maximum skew-symmetry residual:"
        f" {max_skew_error:.12e}"
    )

    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()