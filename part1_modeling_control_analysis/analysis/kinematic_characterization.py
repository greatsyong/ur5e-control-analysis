import numpy as np

from kinematics.forward_kinematics import forward_kinematics
from kinematics.jacobian import geometric_jacobian


def analyze_configuration(name, q_deg):
    q = np.deg2rad(np.asarray(q_deg, dtype=float))

    T = forward_kinematics(q)
    J = geometric_jacobian(q)

    Jv = J[:3, :]
    Jw = J[3:, :]

    sv_linear = np.linalg.svd(
        Jv,
        compute_uv=False
    )

    sv_angular = np.linalg.svd(
        Jw,
        compute_uv=False
    )

    rank_linear = np.linalg.matrix_rank(
        Jv,
        tol=1e-10
    )

    rank_angular = np.linalg.matrix_rank(
        Jw,
        tol=1e-10
    )

    U, singular_values, Vt = np.linalg.svd(
        J,
        full_matrices=True
    )

    sigma_max = singular_values[0]
    sigma_min = singular_values[-1]

    if sigma_min < 1e-12:
        condition_number = np.inf
    else:
        condition_number = sigma_max / sigma_min

    manipulability = np.prod(singular_values)

    rank = np.linalg.matrix_rank(
        J,
        tol=1e-10
    )

    weak_task_direction = U[:, -1]
    weak_joint_direction = Vt[-1, :]

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print("q [deg]:")
    print(np.asarray(q_deg, dtype=float))

    print("\nTCP position [m]:")
    print(T[:3, 3])

    print("\nSingular values:")
    print(singular_values)

    print(f"\nMinimum singular value : {sigma_min:.8f}")
    print(f"Condition number       : {condition_number:.6f}")
    print(f"Manipulability         : {manipulability:.8e}")

    print(f"Jacobian rank          : {rank}")

    print("\nWeakest task-space direction:")
    print(weak_task_direction)

    print("\nCorresponding joint-space direction:")
    print(weak_joint_direction)

    print("\n--- Translational Jacobian Jv ---")
    print("Singular values:")
    print(sv_linear)
    print(f"Rank                   : {rank_linear}")
    print(
        f"Minimum singular value : "
        f"{sv_linear[-1]:.8f}"
    )

    print("\n--- Rotational Jacobian Jw ---")
    print("Singular values:")
    print(sv_angular)
    print(f"Rank                   : {rank_angular}")
    print(
        f"Minimum singular value : "
        f"{sv_angular[-1]:.8f}"
    )

if __name__ == "__main__":

    configurations = {
        # Pre-Part 1 cross-validation configuration
        "validated_test_pose": [
            10.0, -40.0, 60.0, -30.0, 45.0, 20.0
        ],

        # Useful nominal working configuration
        "nominal_working_pose": [
            0.0, -90.0, 90.0, -90.0, -90.0, 0.0
        ],

        # Deliberately simple configuration.
        # We will inspect whether it is singular rather than assume it.
        "zero_configuration": [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    }

    for name, q_deg in configurations.items():
        analyze_configuration(
            name,
            q_deg
        )