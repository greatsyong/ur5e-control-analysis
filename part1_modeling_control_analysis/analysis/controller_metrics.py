# analysis/controller_metrics.py

import numpy as np

from kinematics.forward_kinematics import forward_kinematics


def rotation_error_angle(R_ref, R_actual):
    """
    Geodesic orientation error between two rotation matrices.

    Parameters
    ----------
    R_ref : ndarray, shape (3, 3)
        Reference orientation.
    R_actual : ndarray, shape (3, 3)
        Actual orientation.

    Returns
    -------
    float
        Orientation error angle [rad].
    """
    R_err = R_ref.T @ R_actual

    cos_theta = (np.trace(R_err) - 1.0) / 2.0

    # Numerical protection against values such as 1.0000000002
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return np.arccos(cos_theta)


def compute_tcp_errors(q_ref_history, q_history):
    """
    Compute TCP position and orientation tracking errors.

    Parameters
    ----------
    q_ref_history : ndarray, shape (N, 6)
        Reference joint trajectory [rad].

    q_history : ndarray, shape (N, 6)
        Actual joint trajectory [rad].

    Returns
    -------
    dict
        Time histories of TCP tracking errors.
    """

    q_ref_history = np.asarray(q_ref_history, dtype=float)
    q_history = np.asarray(q_history, dtype=float)

    if q_ref_history.shape != q_history.shape:
        raise ValueError(
            "Reference and actual joint trajectories must have identical shapes."
        )

    if q_ref_history.ndim != 2 or q_ref_history.shape[1] != 6:
        raise ValueError(
            "Joint trajectories must have shape (N, 6)."
        )

    n_samples = q_history.shape[0]

    position_error = np.zeros(n_samples)
    orientation_error = np.zeros(n_samples)

    reference_position = np.zeros((n_samples, 3))
    actual_position = np.zeros((n_samples, 3))

    for k in range(n_samples):

        T_ref = forward_kinematics(q_ref_history[k])
        T_actual = forward_kinematics(q_history[k])

        p_ref = T_ref[:3, 3]
        p_actual = T_actual[:3, 3]

        R_ref = T_ref[:3, :3]
        R_actual = T_actual[:3, :3]

        reference_position[k] = p_ref
        actual_position[k] = p_actual

        # Euclidean TCP position error [m]
        position_error[k] = np.linalg.norm(
            p_ref - p_actual
        )

        # Geodesic orientation error [rad]
        orientation_error[k] = rotation_error_angle(
            R_ref,
            R_actual,
        )

    return {
        "position_error_m": position_error,
        "orientation_error_rad": orientation_error,
        "reference_position_m": reference_position,
        "actual_position_m": actual_position,
    }


def summarize_tcp_errors(q_ref_history, q_history):
    """
    Compute summary TCP tracking metrics.

    Returns
    -------
    dict
        Position metrics in millimetres.
        Orientation metrics in degrees.
    """

    errors = compute_tcp_errors(
        q_ref_history,
        q_history,
    )

    position_error_m = errors["position_error_m"]
    orientation_error_rad = errors["orientation_error_rad"]

    position_error_mm = position_error_m * 1000.0
    orientation_error_deg = np.rad2deg(
        orientation_error_rad
    )

    metrics = {
        # Position
        "tcp_position_rmse_mm":
            np.sqrt(np.mean(position_error_mm ** 2)),

        "tcp_position_max_mm":
            np.max(position_error_mm),

        "tcp_position_final_mm":
            position_error_mm[-1],

        # Orientation
        "tcp_orientation_rmse_deg":
            np.sqrt(np.mean(orientation_error_deg ** 2)),

        "tcp_orientation_max_deg":
            np.max(orientation_error_deg),

        "tcp_orientation_final_deg":
            orientation_error_deg[-1],
    }

    return metrics


def print_tcp_summary(metrics):
    """
    Print TCP tracking metrics in a consistent format.
    """

    print("\n" + "=" * 78)
    print("TCP TRACKING PERFORMANCE")
    print("=" * 78)

    print("\nPosition error:")
    print(
        f"  RMSE  : "
        f"{metrics['tcp_position_rmse_mm']:.6f} mm"
    )
    print(
        f"  Max   : "
        f"{metrics['tcp_position_max_mm']:.6f} mm"
    )
    print(
        f"  Final : "
        f"{metrics['tcp_position_final_mm']:.6f} mm"
    )

    print("\nOrientation error:")
    print(
        f"  RMSE  : "
        f"{metrics['tcp_orientation_rmse_deg']:.6f} deg"
    )
    print(
        f"  Max   : "
        f"{metrics['tcp_orientation_max_deg']:.6f} deg"
    )
    print(
        f"  Final : "
        f"{metrics['tcp_orientation_final_deg']:.6f} deg"
    )

    print("=" * 78)