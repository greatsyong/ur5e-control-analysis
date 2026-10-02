"""
Payload-perturbed dynamics for UR5e robustness analysis.

The nominal UR5e model remains unchanged. A rigid payload is attached
to the nominal tool frame T06. The payload is intentionally unknown
to the controller and is applied only to the simulated plant.

Initial robustness model:
    - point-mass payload
    - configurable mass
    - configurable CoM offset in the tool frame
"""

import numpy as np

from models.ur5e_parameters import N_JOINTS

from kinematics.forward_kinematics import forward_kinematics

from dynamics.rigid_body_dynamics import (
    mass_matrix as nominal_mass_matrix,
    gravity_vector as nominal_gravity_vector,
)


DEFAULT_PAYLOAD_MASS = 2.0

DEFAULT_PAYLOAD_COM_TOOL = np.array([
    0.0,
    0.0,
    0.10,
])


def payload_com_position(
    q,
    payload_com_tool=DEFAULT_PAYLOAD_COM_TOOL,
):
    """
    Payload center-of-mass position expressed in the base frame.
    """

    q = np.asarray(q, dtype=float)

    if q.shape != (N_JOINTS,):
        raise ValueError(
            f"Expected q shape ({N_JOINTS},), got {q.shape}"
        )

    r_tool = np.asarray(
        payload_com_tool,
        dtype=float,
    )

    if r_tool.shape != (3,):
        raise ValueError(
            "payload_com_tool must have shape (3,)"
        )

    T06 = forward_kinematics(q)

    R06 = T06[:3, :3]
    p06 = T06[:3, 3]

    return p06 + R06 @ r_tool


def payload_com_jacobian(
    q,
    payload_com_tool=DEFAULT_PAYLOAD_COM_TOOL,
    epsilon=1e-7,
):
    """
    Translational Jacobian of the payload CoM.

    Computed numerically from the same validated forward-kinematics
    chain used elsewhere in the project.
    """

    q = np.asarray(q, dtype=float)

    Jv = np.zeros((3, N_JOINTS))

    for j in range(N_JOINTS):

        dq = np.zeros(N_JOINTS)
        dq[j] = epsilon

        p_plus = payload_com_position(
            q + dq,
            payload_com_tool,
        )

        p_minus = payload_com_position(
            q - dq,
            payload_com_tool,
        )

        Jv[:, j] = (
            p_plus - p_minus
        ) / (2.0 * epsilon)

    return Jv


def mass_matrix_payload(
    q,
    payload_mass=DEFAULT_PAYLOAD_MASS,
    payload_com_tool=DEFAULT_PAYLOAD_COM_TOOL,
):
    """
    Actual plant mass matrix with point-mass payload.

        M_actual = M_nominal + m_p J_p^T J_p
    """

    q = np.asarray(q, dtype=float)

    M = nominal_mass_matrix(q)

    Jp = payload_com_jacobian(
        q,
        payload_com_tool,
    )

    M_payload = (
        payload_mass
        * Jp.T
        @ Jp
    )

    M_actual = M + M_payload

    return 0.5 * (
        M_actual + M_actual.T
    )


def gravity_vector_payload(
    q,
    payload_mass=DEFAULT_PAYLOAD_MASS,
    payload_com_tool=DEFAULT_PAYLOAD_COM_TOOL,
    gravity=9.81,
):
    """
    Gravity compensation vector with payload.
    """

    q = np.asarray(q, dtype=float)

    g = nominal_gravity_vector(
        q,
        gravity=gravity,
    )

    Jp = payload_com_jacobian(
        q,
        payload_com_tool,
    )

    gravity_compensation = np.array([
        0.0,
        0.0,
        gravity,
    ])

    g_payload = (
        Jp.T
        @ (
            payload_mass
            * gravity_compensation
        )
    )

    return g + g_payload


def coriolis_centrifugal_vector_payload(
    q,
    qdot,
    payload_mass=DEFAULT_PAYLOAD_MASS,
    payload_com_tool=DEFAULT_PAYLOAD_COM_TOOL,
    epsilon=1e-6,
):
    """
    Coriolis/centrifugal vector of the payload-perturbed plant.

    Computed from numerical derivatives of the perturbed mass matrix
    using Christoffel symbols.
    """

    q = np.asarray(q, dtype=float)
    qdot = np.asarray(qdot, dtype=float)

    dM_dq = np.zeros(
        (N_JOINTS, N_JOINTS, N_JOINTS)
    )

    for k in range(N_JOINTS):

        dq = np.zeros(N_JOINTS)
        dq[k] = epsilon

        M_plus = mass_matrix_payload(
            q + dq,
            payload_mass,
            payload_com_tool,
        )

        M_minus = mass_matrix_payload(
            q - dq,
            payload_mass,
            payload_com_tool,
        )

        dM_dq[k] = (
            M_plus - M_minus
        ) / (2.0 * epsilon)

    c = np.zeros(N_JOINTS)

    for i in range(N_JOINTS):

        for j in range(N_JOINTS):

            for k in range(N_JOINTS):

                christoffel = 0.5 * (
                    dM_dq[k, i, j]
                    + dM_dq[j, i, k]
                    - dM_dq[i, j, k]
                )

                c[i] += (
                    christoffel
                    * qdot[j]
                    * qdot[k]
                )

    return c


def state_derivative_payload(
    x,
    tau,
    payload_mass=DEFAULT_PAYLOAD_MASS,
    payload_com_tool=DEFAULT_PAYLOAD_COM_TOOL,
):
    """
    Nonlinear perturbed plant:

        M_p(q) qddot
        + c_p(q,qdot)
        + g_p(q)
        = tau
    """

    x = np.asarray(x, dtype=float)
    tau = np.asarray(tau, dtype=float)

    q = x[:N_JOINTS]
    qdot = x[N_JOINTS:]

    M = mass_matrix_payload(
        q,
        payload_mass,
        payload_com_tool,
    )

    c = coriolis_centrifugal_vector_payload(
        q,
        qdot,
        payload_mass,
        payload_com_tool,
    )

    g = gravity_vector_payload(
        q,
        payload_mass,
        payload_com_tool,
    )

    qddot = np.linalg.solve(
        M,
        tau - c - g,
    )

    return np.concatenate([
        qdot,
        qddot,
    ])