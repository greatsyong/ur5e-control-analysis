import numpy as np

from dynamics.rigid_body_dynamics import (
    coriolis_centrifugal_vector,
)


def analyze_case(name, q_deg, qdot_deg_s):

    q_deg = np.asarray(q_deg, dtype=float)
    qdot_deg_s = np.asarray(
        qdot_deg_s,
        dtype=float
    )

    q = np.deg2rad(q_deg)
    qdot = np.deg2rad(qdot_deg_s)

    c = coriolis_centrifugal_vector(
        q,
        qdot
    )

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print("q [deg]:")
    print(q_deg)

    print("\nqdot [deg/s]:")
    print(qdot_deg_s)

    print(
        "\nCoriolis/centrifugal torque [N m]:"
    )
    print(c)

    print(
        "\nTorque norm [N m]: "
        f"{np.linalg.norm(c):.8f}"
    )

    print(
        "Maximum absolute torque [N m]: "
        f"{np.max(np.abs(c)):.8f}"
    )


if __name__ == "__main__":

    q_test = [
        10.0,
        -40.0,
        60.0,
        -30.0,
        45.0,
        20.0
    ]

    cases = {

        "static_case": (
            q_test,
            [0, 0, 0, 0, 0, 0]
        ),

        "moderate_joint_motion": (
            q_test,
            [20, -30, 40, -20, 30, 10]
        ),

        "faster_joint_motion": (
            q_test,
            [40, -60, 80, -40, 60, 20]
        ),
    }

    for name, (q_deg, qdot_deg_s) in cases.items():

        analyze_case(
            name,
            q_deg,
            qdot_deg_s
        )