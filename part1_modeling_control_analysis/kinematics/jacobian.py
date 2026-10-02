"""
Geometric Jacobian for the nominal UR5e model.
"""

import numpy as np

from kinematics.forward_kinematics import forward_kinematics


def geometric_jacobian(q):
    """
    Compute the 6x6 geometric Jacobian of the UR5e.

    Parameters
    ----------
    q : array_like, shape (6,)
        Joint angles [rad].

    Returns
    -------
    J : np.ndarray, shape (6, 6)
        Geometric Jacobian expressed in the base frame.

        J[:3, :] -> linear velocity Jacobian
        J[3:, :] -> angular velocity Jacobian
    """

    T06, transforms = forward_kinematics(
        q,
        return_all=True
    )

    p_tcp = T06[:3, 3]

    J = np.zeros((6, 6))

    for i in range(6):

        # Joint i+1 rotates around z_i
        z = transforms[i][:3, 2]

        # Origin of frame i
        p = transforms[i][:3, 3]

        # Linear velocity contribution
        J[:3, i] = np.cross(
            z,
            p_tcp - p
        )

        # Angular velocity contribution
        J[3:, i] = z

    return J