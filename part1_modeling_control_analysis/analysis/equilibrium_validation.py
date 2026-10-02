import numpy as np

from dynamics.state_space import (
    state_derivative,
    equilibrium_input,
)


def test_equilibrium(name, q_deg):

    q_deg = np.asarray(q_deg, dtype=float)
    q = np.deg2rad(q_deg)

    qdot = np.zeros(6)

    x_eq = np.concatenate([
        q,
        qdot
    ])

    tau_eq = equilibrium_input(q)

    xdot = state_derivative(
        x_eq,
        tau_eq
    )

    qdot_result = xdot[:6]
    qddot_result = xdot[6:]

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print("q [deg]:")
    print(q_deg)

    print("\nEquilibrium torque [N m]:")
    print(tau_eq)

    print("\nResulting qdot [rad/s]:")
    print(qdot_result)

    print("\nResulting qddot [rad/s^2]:")
    print(qddot_result)

    print(
        "\n||xdot||: "
        f"{np.linalg.norm(xdot):.12e}"
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

        test_equilibrium(
            name,
            q_deg
        )