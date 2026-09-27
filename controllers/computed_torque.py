import numpy as np

from dynamics.rigid_body_dynamics import (
    mass_matrix,
    coriolis_centrifugal_vector,
    gravity_vector,
)


class ComputedTorqueController:
    """
    Model-based computed-torque controller for the UR5e.

    Control law:

        e     = q_ref - q
        e_dot = qdot_ref - qdot

        v = qddot_ref + Kd * e_dot + Kp * e

        tau = M(q) @ v + c(q, qdot) + g(q)

    where:
        M(q)       : joint-space inertia matrix
        c(q,qdot)  : Coriolis/centrifugal vector
        g(q)       : gravity vector

    Optional actuator torque saturation is applied after
    the model-based torque command is calculated.
    """

    def __init__(
        self,
        kp,
        kd,
        torque_limits=None,
    ):
        self.kp = np.asarray(
            kp,
            dtype=float,
        )

        self.kd = np.asarray(
            kd,
            dtype=float,
        )

        if self.kp.shape != (6,):
            raise ValueError(
                "kp must be a 6-element vector."
            )

        if self.kd.shape != (6,):
            raise ValueError(
                "kd must be a 6-element vector."
            )

        if torque_limits is None:
            self.torque_limits = None
        else:
            self.torque_limits = np.asarray(
                torque_limits,
                dtype=float,
            )

            if self.torque_limits.shape != (6,):
                raise ValueError(
                    "torque_limits must be a 6-element vector."
                )

    def compute(
        self,
        q,
        qdot,
        q_ref,
        qdot_ref,
        qddot_ref,
    ):
        q = np.asarray(
            q,
            dtype=float,
        )

        qdot = np.asarray(
            qdot,
            dtype=float,
        )

        q_ref = np.asarray(
            q_ref,
            dtype=float,
        )

        qdot_ref = np.asarray(
            qdot_ref,
            dtype=float,
        )

        qddot_ref = np.asarray(
            qddot_ref,
            dtype=float,
        )

        # ----------------------------------------------------
        # Tracking error
        # ----------------------------------------------------

        error = q_ref - q
        error_dot = qdot_ref - qdot

        # ----------------------------------------------------
        # Desired auxiliary acceleration
        #
        # If the rigid-body model is exact:
        #
        #     qddot = v
        #
        # and therefore:
        #
        #     e_ddot + Kd e_dot + Kp e = 0
        #
        # ----------------------------------------------------

        v = (
            qddot_ref
            + self.kd * error_dot
            + self.kp * error
        )

        # ----------------------------------------------------
        # Nonlinear rigid-body dynamics
        # ----------------------------------------------------

        M = mass_matrix(q)
        c = coriolis_centrifugal_vector(
            q,
            qdot,
        )
        g = gravity_vector(q)

        # ----------------------------------------------------
        # Computed-torque control law
        # ----------------------------------------------------

        tau = (
            M @ v
            + c
            + g
        )

        # ----------------------------------------------------
        # Actuator saturation
        # ----------------------------------------------------

        if self.torque_limits is not None:
            tau = np.clip(
                tau,
                -self.torque_limits,
                self.torque_limits,
            )

        return tau