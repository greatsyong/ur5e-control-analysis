import numpy as np


class JointPIDController:
    """
    Independent joint-space PID controller with
    torque saturation and conditional-integration anti-windup.

    Control law:
        e     = q_ref - q
        edot  = qdot_ref - qdot

        tau = Kp*e + Kd*edot + Ki*eint

    The integral state is updated only when:
      1. the unsaturated torque is inside the actuator limit, or
      2. the current error would drive a saturated actuator
         back toward the admissible torque range.
    """

    def __init__(
        self,
        kp,
        ki,
        kd,
        dt,
        torque_limits,
        integral_limits=None,
    ):
        self.kp = np.asarray(kp, dtype=float)
        self.ki = np.asarray(ki, dtype=float)
        self.kd = np.asarray(kd, dtype=float)

        self.dt = float(dt)

        self.torque_limits = np.asarray(
            torque_limits,
            dtype=float,
        )

        if integral_limits is None:
            self.integral_limits = np.full(
                6,
                np.inf,
            )
        else:
            self.integral_limits = np.asarray(
                integral_limits,
                dtype=float,
            )

        self.integral_error = np.zeros(6)

    def reset(self):
        self.integral_error[:] = 0.0

    def compute(
        self,
        q,
        qdot,
        q_ref,
        qdot_ref,
    ):
        q = np.asarray(q, dtype=float)
        qdot = np.asarray(qdot, dtype=float)

        q_ref = np.asarray(q_ref, dtype=float)
        qdot_ref = np.asarray(
            qdot_ref,
            dtype=float,
        )

        error = q_ref - q
        error_dot = qdot_ref - qdot

        # ---------------------------------------------------------
        # Torque using the current integral state
        # ---------------------------------------------------------

        tau_current = (
            self.kp * error
            + self.kd * error_dot
            + self.ki * self.integral_error
        )

        # ---------------------------------------------------------
        # Conditional-integration anti-windup
        #
        # Integrate if:
        #   - actuator is not saturated, or
        #   - error drives the saturated torque back toward
        #     the admissible range.
        # ---------------------------------------------------------

        below_upper = (
            tau_current
            < self.torque_limits
        )

        above_lower = (
            tau_current
            > -self.torque_limits
        )

        integrate_positive = (
            below_upper
            | (error < 0.0)
        )

        integrate_negative = (
            above_lower
            | (error > 0.0)
        )

        integrate = (
            integrate_positive
            & integrate_negative
        )

        self.integral_error[integrate] += (
            error[integrate]
            * self.dt
        )

        self.integral_error = np.clip(
            self.integral_error,
            -self.integral_limits,
            self.integral_limits,
        )

        # ---------------------------------------------------------
        # Final torque with updated integral state
        # ---------------------------------------------------------

        tau_unsaturated = (
            self.kp * error
            + self.kd * error_dot
            + self.ki * self.integral_error
        )

        tau = np.clip(
            tau_unsaturated,
            -self.torque_limits,
            self.torque_limits,
        )

        return tau