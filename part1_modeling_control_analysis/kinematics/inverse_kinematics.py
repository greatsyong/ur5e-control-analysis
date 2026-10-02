"""
Numerical inverse kinematics for the nominal UR5e model.
"""

import numpy as np

from kinematics.forward_kinematics import forward_kinematics
from kinematics.jacobian import geometric_jacobian


def rotation_error(R_current, R_target):
    """
    Orientation error expressed in the base frame.

    Parameters
    ----------
    R_current : np.ndarray, shape (3, 3)
    R_target : np.ndarray, shape (3, 3)

    Returns
    -------
    error : np.ndarray, shape (3,)
        Orientation error vector.
    """

    return 0.5 * (
        np.cross(R_current[:, 0], R_target[:, 0])
        + np.cross(R_current[:, 1], R_target[:, 1])
        + np.cross(R_current[:, 2], R_target[:, 2])
    )


def pose_error(T_current, T_target):
    """
    Compute 6D Cartesian pose error.

    Returns
    -------
    error : np.ndarray, shape (6,)
        [position_error, orientation_error]
    """

    p_current = T_current[:3, 3]
    p_target = T_target[:3, 3]

    R_current = T_current[:3, :3]
    R_target = T_target[:3, :3]

    e_position = p_target - p_current

    e_orientation = rotation_error(
        R_current,
        R_target
    )

    return np.concatenate([
        e_position,
        e_orientation
    ])


def inverse_kinematics(
    T_target,
    q_initial,
    max_iterations=500,
    tolerance=1e-6,
    step_size=0.5
):
    """
    Numerical IK using the Jacobian pseudoinverse.

    Parameters
    ----------
    T_target : np.ndarray, shape (4, 4)
        Desired TCP pose.

    q_initial : array_like, shape (6,)
        Initial joint configuration [rad].

    max_iterations : int
        Maximum number of iterations.

    tolerance : float
        Pose error convergence threshold.

    step_size : float
        Iterative update gain.

    Returns
    -------
    q : np.ndarray
        IK solution [rad].

    converged : bool

    history : list
        Pose error norm at each iteration.
    """

    q = np.asarray(
        q_initial,
        dtype=float
    ).copy()

    history = []

    for _ in range(max_iterations):

        T_current = forward_kinematics(q)

        error = pose_error(
            T_current,
            T_target
        )

        error_norm = np.linalg.norm(error)
        history.append(error_norm)

        if error_norm < tolerance:
            return q, True, history

        J = geometric_jacobian(q)

        dq = (
            step_size
            * np.linalg.pinv(J)
            @ error
        )

        q += dq

    return q, False, history

def inverse_kinematics_dls(
    T_target,
    q_initial,
    damping=0.05,
    max_iterations=500,
    tolerance=1e-6,
    step_size=0.5,
    max_joint_step=0.2
):
    """
    Numerical IK using Damped Least Squares.

    Parameters
    ----------
    T_target : np.ndarray, shape (4, 4)
        Desired TCP pose.

    q_initial : array_like, shape (6,)
        Initial joint configuration [rad].

    damping : float
        DLS damping coefficient.

    max_iterations : int
        Maximum number of iterations.

    tolerance : float
        Pose-error convergence threshold.

    step_size : float
        Iterative update gain.

    max_joint_step : float
        Maximum Euclidean joint update per iteration [rad].

    Returns
    -------
    q : np.ndarray
        IK solution [rad].

    converged : bool

    history : list
        Pose-error norm.

    step_history : list
        Joint-update norm.
    """

    q = np.asarray(q_initial, dtype=float).copy()

    history = []
    step_history = []

    I = np.eye(6)

    for _ in range(max_iterations):

        T_current = forward_kinematics(q)

        error = pose_error(
            T_current,
            T_target
        )

        error_norm = np.linalg.norm(error)
        history.append(error_norm)

        if error_norm < tolerance:
            return q, True, history, step_history

        J = geometric_jacobian(q)

        J_dls = (
            J.T
            @ np.linalg.solve(
                J @ J.T + damping**2 * I,
                I
            )
        )

        dq = step_size * (J_dls @ error)

        dq_norm = np.linalg.norm(dq)

        # Limit excessive joint update
        if dq_norm > max_joint_step:
            dq *= max_joint_step / dq_norm
            dq_norm = max_joint_step

        step_history.append(dq_norm)

        q += dq

    return q, False, history, step_history