import numpy as np

from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are

from dynamics.state_space import (
    equilibrium_input,
    linearize_dynamics,
)


DT = 0.002

TORQUE_LIMITS = np.array([
    150.0, 150.0, 150.0,
    28.0, 28.0, 28.0,
])

q_eq = np.deg2rad([
    0.0, -90.0, 90.0,
    -90.0, -90.0, 0.0,
])

x_eq = np.concatenate([
    q_eq,
    np.zeros(6),
])

tau_eq = equilibrium_input(q_eq)

A, B = linearize_dynamics(
    x_eq,
    tau_eq,
)

C_dummy = np.eye(12)
D_dummy = np.zeros((12, 6))

Ad, Bd, _, _, _ = cont2discrete(
    (A, B, C_dummy, D_dummy),
    DT,
    method="zoh",
)


# ============================================================
# Bryson normalization
# ============================================================

q_allow = np.deg2rad(
    np.full(6, 2.0)
)

qdot_allow = np.deg2rad(
    np.full(6, 10.0)
)

Q = np.diag(
    np.concatenate([
        1.0 / q_allow**2,
        1.0 / qdot_allow**2,
    ])
)

R_bryson = np.diag(
    1.0 / TORQUE_LIMITS**2
)


# ============================================================
# Global control-penalty sweep
# ============================================================

rho_values = [
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
]


print("\n" + "=" * 100)
print("DISCRETE LQR — GLOBAL CONTROL PENALTY SWEEP")
print("=" * 100)

print(
    "\n"
    f"{'rho_u':>10} "
    f"{'spectral radius':>18} "
    f"{'fastest equiv pole':>22} "
    f"{'||K||_2':>14} "
    f"{'max |K|':>14}"
)

print("-" * 100)


for rho_u in rho_values:

    R = rho_u * R_bryson

    P = solve_discrete_are(
        Ad,
        Bd,
        Q,
        R,
    )

    K = np.linalg.solve(
        R + Bd.T @ P @ Bd,
        Bd.T @ P @ Ad,
    )

    eig_d = np.linalg.eigvals(
        Ad - Bd @ K
    )

    spectral_radius = np.max(
        np.abs(eig_d)
    )

    equivalent_poles = (
        np.log(
            eig_d.astype(complex)
        )
        / DT
    )

    fastest_real = np.min(
        np.real(equivalent_poles)
    )

    k_norm = np.linalg.norm(
        K,
        ord=2,
    )

    max_k = np.max(
        np.abs(K)
    )

    print(
        f"{rho_u:10.1f} "
        f"{spectral_radius:18.10f} "
        f"{fastest_real:22.3f} "
        f"{k_norm:14.3f} "
        f"{max_k:14.3f}"
    )