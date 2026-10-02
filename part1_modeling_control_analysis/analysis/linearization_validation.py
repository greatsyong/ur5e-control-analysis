from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dynamics.state_space import (
    state_derivative,
    equilibrium_input,
    linearize_dynamics,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "results" / "data"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


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

perturbation_sizes_deg = np.array([
    0.01,
    0.1,
    1.0,
    5.0,
    10.0,
])

nonlinear_norms = []
linear_norms = []
error_norms = []
relative_errors = []


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


    nonlinear_norms.append(
        nonlinear_norm
    )

    linear_norms.append(
        linear_norm
    )

    error_norms.append(
        error_norm
    )

    relative_errors.append(
        relative_error
    )


    print(
        f"{perturb_deg:14.4f} "
        f"{nonlinear_norm:16.8e} "
        f"{linear_norm:16.8e} "
        f"{error_norm:16.8e} "
        f"{relative_error:16.8e}"
    )


nonlinear_norms = np.asarray(
    nonlinear_norms
)

linear_norms = np.asarray(
    linear_norms
)

error_norms = np.asarray(
    error_norms
)

relative_errors = np.asarray(
    relative_errors
)


# ---------------------------------------------------------------------
# Save numerical data
# ---------------------------------------------------------------------

output_data = np.column_stack([
    perturbation_sizes_deg,
    nonlinear_norms,
    linear_norms,
    error_norms,
    relative_errors,
    100.0 * relative_errors,
])

data_path = (
    DATA_DIR
    / "linearization_validation.csv"
)

np.savetxt(
    data_path,
    output_data,
    delimiter=",",
    header=(
        "joint2_perturbation_deg,"
        "nonlinear_delta_norm,"
        "linear_delta_norm,"
        "error_norm,"
        "relative_error,"
        "relative_error_percent"
    ),
    comments="",
    fmt="%.12e",
)


# ---------------------------------------------------------------------
# Figure: Linearization validity
# ---------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(7.0, 4.5)
)

ax.semilogx(
    perturbation_sizes_deg,
    100.0 * relative_errors,
    marker="o",
    linewidth=2.0,
)

ax.set_xlabel(
    "Joint 2 perturbation magnitude [deg]"
)

ax.set_ylabel(
    "Linearization error [%]"
)

ax.set_title(
    "Local Validity of the Linearized Dynamics"
)

ax.grid(
    True,
    which="both",
    alpha=0.3,
)

fig.tight_layout()

figure_path = (
    FIGURE_DIR
    / "linearization_validity.png"
)

fig.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

print("\nSaved:")
print(data_path)
print(figure_path)