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
# Configurations
# ---------------------------------------------------------------------

q_safe_deg = np.array([
    0.0, -90.0, 90.0, -90.0, -90.0, 0.0
])

q_singular_deg = np.zeros(6)

q_safe = np.deg2rad(q_safe_deg)
q_singular = np.deg2rad(q_singular_deg)


# ---------------------------------------------------------------------
# Singularity sweep
# ---------------------------------------------------------------------

s_values = np.linspace(0.0, 1.0, 101)

sigma_min_values = []
condition_values = []
rank_values = []


print(
    f"{'s':>8} "
    f"{'sigma_min':>14} "
    f"{'condition':>14} "
    f"{'rank':>6}"
)

print("-" * 48)


for s in s_values:

    q = (
        (1.0 - s) * q_safe
        + s * q_singular
    )

    J = geometric_jacobian(q)

    singular_values = np.linalg.svd(
        J,
        compute_uv=False
    )

    sigma_min = singular_values[-1]

    rank = np.linalg.matrix_rank(
        J,
        tol=1e-10
    )

    if sigma_min < 1e-12:
        condition_number = np.inf
    else:
        condition_number = (
            singular_values[0]
            / sigma_min
        )

    sigma_min_values.append(sigma_min)
    condition_values.append(condition_number)
    rank_values.append(rank)

    print(
        f"{s:8.2f} "
        f"{sigma_min:14.8e} "
        f"{condition_number:14.6f} "
        f"{rank:6d}"
    )


sigma_min_values = np.asarray(sigma_min_values)
condition_values = np.asarray(condition_values)
rank_values = np.asarray(rank_values)


# ---------------------------------------------------------------------
# Save numerical data
# ---------------------------------------------------------------------

output_data = np.column_stack([
    s_values,
    sigma_min_values,
    condition_values,
    rank_values
])

data_path = DATA_DIR / "singularity_sweep.csv"

np.savetxt(
    data_path,
    output_data,
    delimiter=",",
    header="s,sigma_min,condition_number,rank",
    comments="",
    fmt=[
        "%.6f",
        "%.12e",
        "%.12e",
        "%d",
    ],
)


# ---------------------------------------------------------------------
# Figure 1: Minimum singular value
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.0, 4.5))

ax.plot(
    s_values,
    sigma_min_values,
    linewidth=2.0,
)

ax.set_xlabel("Interpolation parameter, s")
ax.set_ylabel(r"Minimum singular value, $\sigma_{\min}(J)$")
ax.set_title("Jacobian Singularity Approach")

ax.grid(
    True,
    alpha=0.3,
)

fig.tight_layout()

sigma_path = FIGURE_DIR / "singularity_sigma_min.png"

fig.savefig(
    sigma_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ---------------------------------------------------------------------
# Figure 2: Condition number
# ---------------------------------------------------------------------

finite_condition = np.where(
    np.isfinite(condition_values),
    condition_values,
    np.nan,
)

fig, ax = plt.subplots(figsize=(7.0, 4.5))

ax.semilogy(
    s_values,
    finite_condition,
    linewidth=2.0,
)

ax.set_xlabel("Interpolation parameter, s")
ax.set_ylabel(r"Jacobian condition number, $\kappa(J)$")
ax.set_title("Jacobian Conditioning Near Singularity")

ax.grid(
    True,
    which="both",
    alpha=0.3,
)

fig.tight_layout()

condition_path = (
    FIGURE_DIR
    / "singularity_condition_number.png"
)

fig.savefig(
    condition_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

print("\nSaved:")
print(data_path)
print(sigma_path)
print(condition_path)