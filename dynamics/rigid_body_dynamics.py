import numpy as np

from models.ur5e_parameters import (
    COM_POSITIONS,
    N_JOINTS
)


def rot_x(theta):
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s,  c]
    ])


def rot_z(theta):
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])


def make_transform(R=None, p=None):

    T = np.eye(4)

    if R is not None:
        T[:3, :3] = R

    if p is not None:
        T[:3, 3] = p

    return T


def physical_link_transforms(q):
    """
    UR5e physical-link transforms using the joint-frame
    convention imported into Isaac Sim from the official URDF.

    Returns
    -------
    transforms : list of 4x4 arrays
        Base-frame transforms of:
        shoulder_link,
        upper_arm_link,
        forearm_link,
        wrist_1_link,
        wrist_2_link,
        wrist_3_link
    """

    q = np.asarray(q, dtype=float)

    if q.shape != (N_JOINTS,):
        raise ValueError(
            f"Expected q shape ({N_JOINTS},), got {q.shape}"
        )

    transforms = []

    T = make_transform(R=rot_z(np.pi))

    # Joint 1: shoulder_pan
    T = (
        T
        @ make_transform(p=np.array([0.0, 0.0, 0.1625]))
        @ make_transform(R=rot_z(q[0]))
    )
    transforms.append(T.copy())

    # Joint 2: shoulder_lift
    T = (
        T
        @ make_transform(R=rot_x(np.pi / 2))
        @ make_transform(R=rot_z(q[1]))
    )
    transforms.append(T.copy())

    # Joint 3: elbow
    T = (
        T
        @ make_transform(p=np.array([-0.425, 0.0, 0.0]))
        @ make_transform(R=rot_z(q[2]))
    )
    transforms.append(T.copy())

    # Joint 4: wrist_1
    T = (
        T
        @ make_transform(p=np.array([-0.3922, 0.0, 0.1333]))
        @ make_transform(R=rot_z(q[3]))
    )
    transforms.append(T.copy())

    # Joint 5: wrist_2
    T = (
        T
        @ make_transform(p=np.array([0.0, -0.0997, 0.0]))
        @ make_transform(R=rot_x(np.pi / 2))
        @ make_transform(R=rot_z(q[4]))
    )
    transforms.append(T.copy())

    # Joint 6: wrist_3
    T = (
        T
        @ make_transform(p=np.array([0.0, 0.0996, 0.0]))
        @ make_transform(R=rot_x(-np.pi / 2))
        @ make_transform(R=rot_z(q[5]))
    )
    transforms.append(T.copy())

    return transforms


def link_com_positions(q):

    transforms = physical_link_transforms(q)

    positions = np.zeros((N_JOINTS, 3))

    for i, T in enumerate(transforms):

        R = T[:3, :3]
        p = T[:3, 3]

        positions[i] = (
            p
            + R @ COM_POSITIONS[i]
        )

    return positions

from models.ur5e_parameters import (
    LINK_MASSES,
    COM_POSITIONS,
    INERTIA_TENSORS,
    N_JOINTS,
)


def joint_origins_axes(q):
    """
    Joint origins and rotation axes expressed in the base/world frame.

    Returns
    -------
    origins : (6, 3)
    axes    : (6, 3)
    """

    q = np.asarray(q, dtype=float)

    if q.shape != (N_JOINTS,):
        raise ValueError(
            f"Expected q shape ({N_JOINTS},), got {q.shape}"
        )

    # Same base-frame convention validated against Isaac Sim.
    T = make_transform(R=rot_z(np.pi))

    origins = np.zeros((N_JOINTS, 3))
    axes = np.zeros((N_JOINTS, 3))

    # --------------------------------------------------
    # Joint 1
    # --------------------------------------------------
    T = T @ make_transform(
        p=np.array([0.0, 0.0, 0.1625])
    )

    origins[0] = T[:3, 3]
    axes[0] = T[:3, :3] @ np.array([0.0, 0.0, 1.0])

    T = T @ make_transform(R=rot_z(q[0]))

    # --------------------------------------------------
    # Joint 2
    # --------------------------------------------------
    T = T @ make_transform(R=rot_x(np.pi / 2))

    origins[1] = T[:3, 3]
    axes[1] = T[:3, :3] @ np.array([0.0, 0.0, 1.0])

    T = T @ make_transform(R=rot_z(q[1]))

    # --------------------------------------------------
    # Joint 3
    # --------------------------------------------------
    T = T @ make_transform(
        p=np.array([-0.425, 0.0, 0.0])
    )

    origins[2] = T[:3, 3]
    axes[2] = T[:3, :3] @ np.array([0.0, 0.0, 1.0])

    T = T @ make_transform(R=rot_z(q[2]))

    # --------------------------------------------------
    # Joint 4
    # --------------------------------------------------
    T = T @ make_transform(
        p=np.array([-0.3922, 0.0, 0.1333])
    )

    origins[3] = T[:3, 3]
    axes[3] = T[:3, :3] @ np.array([0.0, 0.0, 1.0])

    T = T @ make_transform(R=rot_z(q[3]))

    # --------------------------------------------------
    # Joint 5
    # --------------------------------------------------
    T = (
        T
        @ make_transform(
            p=np.array([0.0, -0.0997, 0.0])
        )
        @ make_transform(R=rot_x(np.pi / 2))
    )

    origins[4] = T[:3, 3]
    axes[4] = T[:3, :3] @ np.array([0.0, 0.0, 1.0])

    T = T @ make_transform(R=rot_z(q[4]))

    # --------------------------------------------------
    # Joint 6
    # --------------------------------------------------
    T = (
        T
        @ make_transform(
            p=np.array([0.0, 0.0996, 0.0])
        )
        @ make_transform(R=rot_x(-np.pi / 2))
    )

    origins[5] = T[:3, 3]
    axes[5] = T[:3, :3] @ np.array([0.0, 0.0, 1.0])

    return origins, axes


def com_jacobians(q):
    """
    Linear and angular Jacobians of every link CoM.

    Returns
    -------
    Jv : (6, 3, 6)
    Jw : (6, 3, 6)
    """

    q = np.asarray(q, dtype=float)

    transforms = physical_link_transforms(q)
    com_positions = link_com_positions(q)
    joint_origins, joint_axes = joint_origins_axes(q)

    Jv = np.zeros((N_JOINTS, 3, N_JOINTS))
    Jw = np.zeros((N_JOINTS, 3, N_JOINTS))

    for i in range(N_JOINTS):

        p_com = com_positions[i]

        # Link i is affected only by joints 0 ... i.
        for j in range(i + 1):

            z = joint_axes[j]
            p_joint = joint_origins[j]

            Jv[i, :, j] = np.cross(
                z,
                p_com - p_joint
            )

            Jw[i, :, j] = z

    return Jv, Jw


def mass_matrix(q):
    """
    Joint-space mass matrix M(q).

    M(q) = sum_i [
        m_i Jv_i^T Jv_i
        +
        Jw_i^T I_i_world Jw_i
    ]
    """

    q = np.asarray(q, dtype=float)

    if q.shape != (N_JOINTS,):
        raise ValueError(
            f"Expected q shape ({N_JOINTS},), got {q.shape}"
        )

    transforms = physical_link_transforms(q)
    Jv, Jw = com_jacobians(q)

    M = np.zeros((N_JOINTS, N_JOINTS))

    for i in range(N_JOINTS):

        R = transforms[i][:3, :3]

        # Inertia tensor expressed in world/base frame.
        I_world = (
            R
            @ INERTIA_TENSORS[i]
            @ R.T
        )

        M += (
            LINK_MASSES[i]
            * Jv[i].T
            @ Jv[i]
            +
            Jw[i].T
            @ I_world
            @ Jw[i]
        )

    # Remove only floating-point asymmetry.
    M = 0.5 * (M + M.T)

    return M

def gravity_vector(q, gravity=9.81):
    """
    Compute the generalized gravity compensation torque vector.

    Dynamics convention:
        M(q) qddot + C(q, qdot) qdot + g(q) = tau

    Therefore, g(q) represents the joint torques required to
    statically compensate for gravity.

    Parameters
    ----------
    q : array_like, shape (6,)
        Joint angles [rad].

    gravity : float
        Gravitational acceleration magnitude [m/s^2].

    Returns
    -------
    g : ndarray, shape (6,)
        Gravity compensation joint torques [N m].
    """

    q = np.asarray(q, dtype=float)

    # Translational and rotational Jacobians
    # evaluated at each link center of mass.
    Jv_list, _ = com_jacobians(q)

    # Positive vector here represents the compensation force
    # required against physical gravity acting in -Z.
    gravity_compensation = np.array([
        0.0,
        0.0,
        gravity
    ])

    g = np.zeros(N_JOINTS)

    for i in range(N_JOINTS):

        g += (
            Jv_list[i].T
            @ (
                LINK_MASSES[i]
                * gravity_compensation
            )
        )

    return g

def coriolis_centrifugal_vector(q, qdot, epsilon=1e-6):
    """
    Compute the Coriolis and centrifugal generalized torque vector.

    Dynamics convention:
        M(q) qddot + c(q, qdot) + g(q) = tau

    where:
        c(q, qdot) = C(q, qdot) qdot

    The vector is computed from the Christoffel symbols using
    numerical derivatives of the mass matrix.

    Parameters
    ----------
    q : array_like, shape (6,)
        Joint angles [rad].

    qdot : array_like, shape (6,)
        Joint velocities [rad/s].

    epsilon : float
        Central finite-difference perturbation [rad].

    Returns
    -------
    c : ndarray, shape (6,)
        Coriolis and centrifugal generalized torques [N m].
    """

    q = np.asarray(q, dtype=float)
    qdot = np.asarray(qdot, dtype=float)

    # dM_dq[k, i, j] = d M_ij / d q_k
    dM_dq = np.zeros(
        (N_JOINTS, N_JOINTS, N_JOINTS)
    )

    for k in range(N_JOINTS):

        dq = np.zeros(N_JOINTS)
        dq[k] = epsilon

        M_plus = mass_matrix(q + dq)
        M_minus = mass_matrix(q - dq)

        dM_dq[k] = (
            M_plus - M_minus
        ) / (2.0 * epsilon)

    c = np.zeros(N_JOINTS)

    for i in range(N_JOINTS):
        for j in range(N_JOINTS):
            for k in range(N_JOINTS):

                gamma_ijk = 0.5 * (
                    dM_dq[k, i, j]
                    + dM_dq[j, i, k]
                    - dM_dq[i, j, k]
                )

                c[i] += (
                    gamma_ijk
                    * qdot[j]
                    * qdot[k]
                )

    return c