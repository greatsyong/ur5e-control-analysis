import numpy as np

from dynamics.rigid_body_dynamics import (
    mass_matrix,
    gravity_vector,
    coriolis_centrifugal_vector,
)


N_JOINTS = 6
N_STATES = 12


def state_derivative(x, tau):
    """
    Nonlinear UR5e state-space dynamics.

    State:
        x = [q, qdot]

    Input:
        tau = joint torque vector

    Dynamics:
        M(q) qddot + c(q, qdot) + g(q) = tau

    Therefore:
        qddot = M(q)^(-1) [tau - c(q, qdot) - g(q)]

    Parameters
    ----------
    x : array_like, shape (12,)
        State vector:
        [q1 ... q6, qdot1 ... qdot6]

    tau : array_like, shape (6,)
        Applied joint torques [N m].

    Returns
    -------
    xdot : ndarray, shape (12,)
        State derivative:
        [qdot, qddot]
    """

    x = np.asarray(x, dtype=float)
    tau = np.asarray(tau, dtype=float)

    q = x[:N_JOINTS]
    qdot = x[N_JOINTS:]

    M = mass_matrix(q)

    c = coriolis_centrifugal_vector(
        q,
        qdot
    )

    g = gravity_vector(q)

    qddot = np.linalg.solve(
        M,
        tau - c - g
    )

    xdot = np.concatenate([
        qdot,
        qddot
    ])

    return xdot


def equilibrium_input(q):
    """
    Return the joint torque required for static equilibrium
    at configuration q.

    At static equilibrium:
        qdot = 0
        qddot = 0

    Therefore:
        tau_eq = g(q)
    """

    q = np.asarray(q, dtype=float)

    return gravity_vector(q)

def linearize_dynamics(
    x_eq,
    tau_eq,
    epsilon_x=1e-6,
    epsilon_u=1e-6,
):
    """
    Numerically linearize the nonlinear dynamics:

        xdot = f(x, tau)

    around an operating point:

        (x_eq, tau_eq)

    The resulting perturbation model is:

        delta_xdot = A delta_x + B delta_tau

    using central finite differences.

    Parameters
    ----------
    x_eq : array_like, shape (12,)
        Equilibrium state.

    tau_eq : array_like, shape (6,)
        Equilibrium joint torque.

    epsilon_x : float
        State perturbation used for numerical differentiation.

    epsilon_u : float
        Input perturbation used for numerical differentiation.

    Returns
    -------
    A : ndarray, shape (12, 12)
        State matrix.

    B : ndarray, shape (12, 6)
        Input matrix.
    """

    x_eq = np.asarray(x_eq, dtype=float)
    tau_eq = np.asarray(tau_eq, dtype=float)

    A = np.zeros((N_STATES, N_STATES))
    B = np.zeros((N_STATES, N_JOINTS))

    # -------------------------------------------------------------
    # State Jacobian:
    #
    # A = df/dx
    # -------------------------------------------------------------

    for i in range(N_STATES):

        dx = np.zeros(N_STATES)
        dx[i] = epsilon_x

        f_plus = state_derivative(
            x_eq + dx,
            tau_eq
        )

        f_minus = state_derivative(
            x_eq - dx,
            tau_eq
        )

        A[:, i] = (
            f_plus - f_minus
        ) / (2.0 * epsilon_x)

    # -------------------------------------------------------------
    # Input Jacobian:
    #
    # B = df/dtau
    # -------------------------------------------------------------

    for i in range(N_JOINTS):

        du = np.zeros(N_JOINTS)
        du[i] = epsilon_u

        f_plus = state_derivative(
            x_eq,
            tau_eq + du
        )

        f_minus = state_derivative(
            x_eq,
            tau_eq - du
        )

        B[:, i] = (
            f_plus - f_minus
        ) / (2.0 * epsilon_u)

    return A, B