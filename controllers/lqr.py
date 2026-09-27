import numpy as np
from scipy.linalg import solve_continuous_are


class JointLQRController:
    """
    Continuous-time full-state LQR controller.

    Linearized system:
        dx_dot = A dx + B du

    Cost:
        J = integral(
            dx.T @ Q @ dx
            + du.T @ R @ du
        ) dt

    Control law:
        du = -K dx

    For trajectory tracking:
        dx = [q - q_ref,
              qdot - qdot_ref]

        tau = tau_ff - K dx

    The feedforward torque tau_ff is supplied externally so that
    different feedforward strategies can be studied independently
    from the LQR feedback design.
    """

    def __init__(
        self,
        A,
        B,
        Q,
        R,
        torque_limits=None,
    ):
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)

        # Continuous-time Algebraic Riccati Equation
        self.P = solve_continuous_are(
            self.A,
            self.B,
            self.Q,
            self.R,
        )

        # Optimal state-feedback gain
        self.K = np.linalg.solve(
            self.R,
            self.B.T @ self.P,
        )

        if torque_limits is None:
            self.torque_limits = None
        else:
            self.torque_limits = np.asarray(
                torque_limits,
                dtype=float,
            )

    def compute(
        self,
        q,
        qdot,
        q_ref,
        qdot_ref,
        tau_ff=None,
    ):
        q = np.asarray(q, dtype=float)
        qdot = np.asarray(qdot, dtype=float)
        q_ref = np.asarray(q_ref, dtype=float)
        qdot_ref = np.asarray(qdot_ref, dtype=float)

        if tau_ff is None:
            tau_ff = np.zeros(6)
        else:
            tau_ff = np.asarray(
                tau_ff,
                dtype=float,
            )

        position_error = q - q_ref
        velocity_error = qdot - qdot_ref

        dx = np.concatenate([
            position_error,
            velocity_error,
        ])

        tau = tau_ff - self.K @ dx

        if self.torque_limits is not None:
            tau = np.clip(
                tau,
                -self.torque_limits,
                self.torque_limits,
            )

        return tau

class DiscreteJointLQRController:
    """
    Discrete-time full-state LQR controller.

        x[k+1] = Ad x[k] + Bd u[k]

    Cost:
        J = sum(
            dx[k].T Q dx[k]
            + du[k].T R du[k]
        )

    Control:
        tau = tau_ff - K dx
    """

    def __init__(
        self,
        Ad,
        Bd,
        Q,
        R,
        torque_limits=None,
    ):
        from scipy.linalg import solve_discrete_are

        self.Ad = np.asarray(Ad, dtype=float)
        self.Bd = np.asarray(Bd, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)

        self.P = solve_discrete_are(
            self.Ad,
            self.Bd,
            self.Q,
            self.R,
        )

        self.K = np.linalg.solve(
            self.R
            + self.Bd.T @ self.P @ self.Bd,
            self.Bd.T @ self.P @ self.Ad,
        )

        if torque_limits is None:
            self.torque_limits = None
        else:
            self.torque_limits = np.asarray(
                torque_limits,
                dtype=float,
            )

    def compute(
        self,
        q,
        qdot,
        q_ref,
        qdot_ref,
        tau_ff=None,
    ):
        q = np.asarray(q, dtype=float)
        qdot = np.asarray(qdot, dtype=float)
        q_ref = np.asarray(q_ref, dtype=float)
        qdot_ref = np.asarray(qdot_ref, dtype=float)

        if tau_ff is None:
            tau_ff = np.zeros(6)
        else:
            tau_ff = np.asarray(
                tau_ff,
                dtype=float,
            )

        dx = np.concatenate([
            q - q_ref,
            qdot - qdot_ref,
        ])

        tau = tau_ff - self.K @ dx

        if self.torque_limits is not None:
            tau = np.clip(
                tau,
                -self.torque_limits,
                self.torque_limits,
            )

        return tau