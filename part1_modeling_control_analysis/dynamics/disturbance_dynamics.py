import numpy as np

from dynamics.state_space import state_derivative


DISTURBANCE_START = 2.0
DISTURBANCE_END = 2.2

DISTURBANCE_TORQUE = np.array([
    0.0,
    20.0,
    0.0,
    0.0,
    0.0,
    0.0,
])


def external_disturbance(t):
    """
    R2 external joint disturbance:
        J2: +20 Nm
        2.0 <= t < 2.2 s
    """
    if DISTURBANCE_START <= t < DISTURBANCE_END:
        return DISTURBANCE_TORQUE.copy()

    return np.zeros(6)


def state_derivative_disturbed(x, tau, t):
    """
    Actual plant input:
        tau_actual = tau_controller + tau_disturbance
    """
    tau_actual = (
        np.asarray(tau, dtype=float)
        + external_disturbance(t)
    )

    return state_derivative(
        x,
        tau_actual,
    )
