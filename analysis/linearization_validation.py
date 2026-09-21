import numpy as np

from dynamics.state_space import (
    state_derivative,
    equilibrium_input,
    linearize_dynamics,
)


# ---------------------------------------------------------------------
# Operating point
# ---------------------------------------------------------------------

q_eq_deg = np.array([
    0.0,
    -90.0,
    90.0,
    -90.0,
    -90.0,
    0.0
])

q_eq = np.deg2rad(q_eq_deg)

qdot_eq = np.zeros(6)

x_eq = np.concatenate([
    q_eq,
    qdot_eq
])

tau_eq = equilibrium_input(q_eq)


# ---------------------------------------------------------------------
# Linearized model
# ---------------------------------------------------------------------

A, B = linearize_dynamics(
    x_eq,
    tau_eq
)

f_eq = state_derivative(
    x_eq,
    tau_eq
)


# ---------------------------------------------------------------------
# Perturbation tests
# ---------------------------------------------------------------------

perturbation_sizes_deg = [
    0.01,
    0.1,
    1.0,
    5.0,
    10.0,
]


print(
    f"{'perturb [deg]':>14} "
    f"{'||nonlinear||':>16} "
    f"{'||linear||':>16} "
    f"{'||error||':>16} "
    f"{'relative error':>16}"
)

print("-" * 84)


for perturb_deg in perturbation_sizes_deg:

    # Perturb joint 2 only.
    delta_x = np.zeros(12)

    delta_x[1] = np.deg2rad(
        perturb_deg
    )

    # No input perturbation.
    delta_tau = np.zeros(6)

    x_test = x_eq + delta_x
    tau_test = tau_eq + delta_tau


    # -------------------------------------------------------------
    # True nonlinear change in state derivative
    # -------------------------------------------------------------

    nonlinear_delta = (
        state_derivative(
            x_test,
            tau_test
        )
        - f_eq
    )


    # -------------------------------------------------------------
    # Linear prediction
    # -------------------------------------------------------------

    linear_delta = (
        A @ delta_x
        + B @ delta_tau
    )


    # -------------------------------------------------------------
    # Error
    # -------------------------------------------------------------

    error = (
        nonlinear_delta
        - linear_delta
    )

    nonlinear_norm = np.linalg.norm(
        nonlinear_delta
    )

    linear_norm = np.linalg.norm(
        linear_delta
    )

    error_norm = np.linalg.norm(
        error
    )

    if nonlinear_norm > 1e-12:
        relative_error = (
            error_norm
            / nonlinear_norm
        )
    else:
        relative_error = 0.0


    print(
        f"{perturb_deg:14.4f} "
        f"{nonlinear_norm:16.8e} "
        f"{linear_norm:16.8e} "
        f"{error_norm:16.8e} "
        f"{relative_error:16.8e}"
    )