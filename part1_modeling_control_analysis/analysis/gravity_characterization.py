import numpy as np

from dynamics.rigid_body_dynamics import gravity_vector


def analyze_configuration(name, q_deg):

    q_deg = np.asarray(q_deg, dtype=float)
    q = np.deg2rad(q_deg)

    g = gravity_vector(q)

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print("q [deg]:")
    print(q_deg)

    print("\nGravity compensation torque [N m]:")
    print(g)

    print(
        "\nTorque norm [N m]: "
        f"{np.linalg.norm(g):.6f}"
    )

    print(
        "Maximum absolute joint torque [N m]: "
        f"{np.max(np.abs(g)):.6f}"
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
        analyze_configuration(
            name,
            q_deg
        )