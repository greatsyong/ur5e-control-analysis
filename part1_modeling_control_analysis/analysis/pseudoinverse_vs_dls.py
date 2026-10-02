from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kinematics.jacobian import geometric_jacobian


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "results" / "data"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Configuration definitions
# ---------------------------------------------------------------------

q_safe_deg = np.array([
    0.0, -90.0, 90.0, -90.0, -90.0, 0.0
])

q_singular_deg = np.zeros(6)

q_safe = np.deg2rad(q_safe_deg)
q_singular = np.deg2rad(q_singular_deg)


# Damped least-squares damping factor
damping = 0.05


# ---------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------

s_values = np.linspace(0.0, 0.99, 101)

sigma_min_values = []
pinv_norm_values = []
dls_norm_values = []
pinv_error_values = []
dls_error_values = []


print(
    f"{'s':>6} "
    f"{'sigma_min':>12} "
    f"{'||qdot_pinv||':>16} "
    f"{'||qdot_dls||':>16} "
    f"{'err_pinv':>12} "
    f"{'err_dls':>12}"
)

print("-" * 84)


for s in s_values:

    q = (
        (1.0 - s) * q_safe
        + s * q_singular
    )

    J = geometric_jacobian(q)

    U, singular_values, Vt = np.linalg.svd(J)

    sigma_min = singular_values[-1]

    # Unit task-space command along the weakest instantaneous direction.
    task_command = U[:, -1]


    # -----------------------------------------------------------------
    # Moore-Penrose pseudoinverse
    # -----------------------------------------------------------------

    qdot_pinv = (
        np.linalg.pinv(J)
        @ task_command
    )


    # -----------------------------------------------------------------
    # Damped least-squares inverse
    #
    # qdot = J^T (J J^T + lambda^2 I)^(-1) v
    # -----------------------------------------------------------------

    JJt = J @ J.T

    qdot_dls = (
        J.T
        @ np.linalg.solve(
            JJt + damping**2 * np.eye(6),
            task_command
        )
    )


    # -----------------------------------------------------------------
    # Task-space tracking errors
    # -----------------------------------------------------------------

    pinv_task_error = np.linalg.norm(
        J @ qdot_pinv - task_command
    )

    dls_task_error = np.linalg.norm(
        J @ qdot_dls - task_command
    )


    # -----------------------------------------------------------------
    # Store results
    # -----------------------------------------------------------------

    pinv_norm = np.linalg.norm(qdot_pinv)
    dls_norm = np.linalg.norm(qdot_dls)

    sigma_min_values.append(sigma_min)
    pinv_norm_values.append(pinv_norm)
    dls_norm_values.append(dls_norm)
    pinv_error_values.append(pinv_task_error)
    dls_error_values.append(dls_task_error)


    print(
        f"{s:6.2f} "
        f"{sigma_min:12.4e} "
        f"{pinv_norm:16.6f} "
        f"{dls_norm:16.6f} "
        f"{pinv_task_error:12.6f} "
        f"{dls_task_error:12.6f}"
    )


sigma_min_values = np.asarray(sigma_min_values)
pinv_norm_values = np.asarray(pinv_norm_values)
dls_norm_values = np.asarray(dls_norm_values)
pinv_error_values = np.asarray(pinv_error_values)
dls_error_values = np.asarray(dls_error_values)


# ---------------------------------------------------------------------
# Save numerical data
# ---------------------------------------------------------------------

output_data = np.column_stack([
    s_values,
    sigma_min_values,
    pinv_norm_values,
    dls_norm_values,
    pinv_error_values,
    dls_error_values,
])

data_path = DATA_DIR / "pseudoinverse_vs_dls.csv"

np.savetxt(
    data_path,
    output_data,
    delimiter=",",
    header=(
        "s,sigma_min,"
        "qdot_pinv_norm,qdot_dls_norm,"
        "pinv_task_error,dls_task_error"
    ),
    comments="",
    fmt="%.12e",
)


# ---------------------------------------------------------------------
# Figure 1: Joint-rate amplification
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.0, 4.5))

ax.semilogy(
    s_values,
    pinv_norm_values,
    linewidth=2.0,
    label="Pseudoinverse",
)

ax.semilogy(
    s_values,
    dls_norm_values,
    linewidth=2.0,
    label=f"DLS ($\\lambda$ = {damping:.2f})",
)

ax.set_xlabel("Interpolation parameter, s")
ax.set_ylabel(r"Joint-rate norm, $\|\dot{q}\|_2$")
ax.set_title("Inverse Kinematics Near Singularity")

ax.grid(
    True,
    which="both",
    alpha=0.3,
)

ax.legend()

fig.tight_layout()

rate_path = (
    FIGURE_DIR
    / "pseudoinverse_vs_dls_joint_rate.png"
)

fig.savefig(
    rate_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ---------------------------------------------------------------------
# Figure 2: Task-space error
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.0, 4.5))

ax.plot(
    s_values,
    pinv_error_values,
    linewidth=2.0,
    label="Pseudoinverse",
)

ax.plot(
    s_values,
    dls_error_values,
    linewidth=2.0,
    label=f"DLS ($\\lambda$ = {damping:.2f})",
)

ax.set_xlabel("Interpolation parameter, s")
ax.set_ylabel(r"Task-space error, $\|J\dot{q}-v\|_2$")
ax.set_title("DLS Regularization Trade-off")

ax.grid(
    True,
    alpha=0.3,
)

ax.legend()

fig.tight_layout()

error_path = (
    FIGURE_DIR
    / "pseudoinverse_vs_dls_task_error.png"
)

fig.savefig(
    error_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

peak_dls_index = np.argmax(dls_norm_values)

print("\nDLS peak amplification:")
print(
    f"s                  = "
    f"{s_values[peak_dls_index]:.4f}"
)
print(
    f"sigma_min          = "
    f"{sigma_min_values[peak_dls_index]:.8f}"
)
print(
    f"||qdot_dls||       = "
    f"{dls_norm_values[peak_dls_index]:.8f}"
)
print(
    f"Theoretical maximum= "
    f"{1.0 / (2.0 * damping):.8f}"
)

print("\nSaved:")
print(data_path)
print(rate_path)
print(error_path)