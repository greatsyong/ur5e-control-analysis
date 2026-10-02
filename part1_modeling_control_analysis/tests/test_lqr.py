import numpy as np

from controllers.lqr import JointLQRController


def test_simple_lqr():
    """
    First verify the LQR controller implementation using
    a known 6-DOF double-integrator system.

        qddot = u

    State:
        x = [q, qdot]
    """

    n = 6

    A = np.block([
        [
            np.zeros((n, n)),
            np.eye(n),
        ],
        [
            np.zeros((n, n)),
            np.zeros((n, n)),
        ],
    ])

    B = np.vstack([
        np.zeros((n, n)),
        np.eye(n),
    ])

    Q = np.diag(
        [100.0] * 6
        + [10.0] * 6
    )

    R = np.eye(6)

    torque_limits = np.array([
        150.0,
        150.0,
        150.0,
        28.0,
        28.0,
        28.0,
    ])

    controller = JointLQRController(
        A=A,
        B=B,
        Q=Q,
        R=R,
        torque_limits=torque_limits,
    )

    print("\nLQR gain shape:")
    print(controller.K.shape)

    print("\nLQR gain K:")
    print(controller.K)

    # ---------------------------------------------------------
    # Test 1: zero tracking error
    # ---------------------------------------------------------

    q = np.zeros(6)
    qdot = np.zeros(6)

    q_ref = np.zeros(6)
    qdot_ref = np.zeros(6)

    tau = controller.compute(
        q,
        qdot,
        q_ref,
        qdot_ref,
    )

    print("\nTest 1 - zero error:")
    print(tau)

    assert np.allclose(
        tau,
        np.zeros(6),
        atol=1e-12,
    )

    # ---------------------------------------------------------
    # Test 2: +1 degree J2 reference error
    # ---------------------------------------------------------

    q_ref = np.zeros(6)
    q_ref[1] = np.deg2rad(1.0)

    tau = controller.compute(
        q,
        qdot,
        q_ref,
        qdot_ref,
    )

    print("\nTest 2 - +1 deg J2 reference error:")
    print(tau)

    assert tau[1] > 0.0

    # ---------------------------------------------------------
    # Closed-loop eigenvalues
    # ---------------------------------------------------------

    A_cl = A - B @ controller.K

    eigvals = np.linalg.eigvals(A_cl)

    print("\nClosed-loop eigenvalues:")
    print(eigvals)

    assert np.all(
        np.real(eigvals) < 0.0
    )

    print("\nSimple LQR controller test: PASS")


if __name__ == "__main__":
    test_simple_lqr()