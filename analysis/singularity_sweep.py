import numpy as np

from kinematics.jacobian import geometric_jacobian


q_safe_deg = np.array([
    0.0, -90.0, 90.0, -90.0, -90.0, 0.0
])

q_singular_deg = np.zeros(6)

q_safe = np.deg2rad(q_safe_deg)
q_singular = np.deg2rad(q_singular_deg)


print(
    f"{'s':>6} "
    f"{'sigma_min':>14} "
    f"{'condition':>14} "
    f"{'rank':>6}"
)

print("-" * 46)


for s in np.linspace(0.0, 1.0, 21):

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

    print(
        f"{s:6.2f} "
        f"{sigma_min:14.8e} "
        f"{condition_number:14.6f} "
        f"{rank:6d}"
    )