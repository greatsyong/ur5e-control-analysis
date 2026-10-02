import numpy as np


def quintic_joint_trajectory(t, T, q_start, q_goal):
    """
    Quintic point-to-point joint trajectory.

    Boundary conditions:
        q(0)     = q_start
        q(T)     = q_goal
        qdot(0)  = qdot(T)  = 0
        qddot(0) = qddot(T) = 0
    """

    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)

    if t <= 0.0:
        return (
            q_start.copy(),
            np.zeros_like(q_start),
            np.zeros_like(q_start),
        )

    if t >= T:
        return (
            q_goal.copy(),
            np.zeros_like(q_goal),
            np.zeros_like(q_goal),
        )

    s = t / T

    h = (
        10.0 * s**3
        - 15.0 * s**4
        + 6.0 * s**5
    )

    h_dot = (
        30.0 * s**2
        - 60.0 * s**3
        + 30.0 * s**4
    ) / T

    h_ddot = (
        60.0 * s
        - 180.0 * s**2
        + 120.0 * s**3
    ) / T**2

    delta_q = q_goal - q_start

    q = q_start + delta_q * h
    qdot = delta_q * h_dot
    qddot = delta_q * h_ddot

    return q, qdot, qddot