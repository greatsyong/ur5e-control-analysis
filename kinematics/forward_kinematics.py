"""
Forward kinematics for the nominal UR5e model.
"""

import numpy as np

from models.ur5e_parameters import A, D, ALPHA, N_JOINTS


def dh_transform(theta, a, d, alpha):
    """
    Standard Denavit-Hartenberg homogeneous transformation.

    Parameters
    ----------
    theta : float
        Joint angle [rad].
    a : float
        Link length [m].
    d : float
        Link offset [m].
    alpha : float
        Link twist [rad].

    Returns
    -------
    T : np.ndarray, shape (4, 4)
        Homogeneous transformation matrix.
    """

    ct = np.cos(theta)
    st = np.sin(theta)

    ca = np.cos(alpha)
    sa = np.sin(alpha)

    T = np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0]
    ])

    return T


def forward_kinematics(q, return_all=False):
    """
    Compute UR5e forward kinematics.

    Parameters
    ----------
    q : array_like, shape (6,)
        Joint angles [rad].

    return_all : bool
        If True, return transformations from the base
        to every joint frame.

    Returns
    -------
    T06 : np.ndarray, shape (4, 4)
        Base-to-tool transformation.

    transforms : list of np.ndarray
        Returned only when return_all=True.
        Contains T00, T01, ..., T06.
    """

    q = np.asarray(q, dtype=float)

    if q.shape != (N_JOINTS,):
        raise ValueError(
            f"Expected {N_JOINTS} joint angles, got shape {q.shape}"
        )

    T = np.eye(4)

    transforms = [T.copy()]

    for i in range(N_JOINTS):

        Ti = dh_transform(
            theta=q[i],
            a=A[i],
            d=D[i],
            alpha=ALPHA[i]
        )

        T = T @ Ti

        transforms.append(T.copy())

    if return_all:
        return T, transforms

    return T