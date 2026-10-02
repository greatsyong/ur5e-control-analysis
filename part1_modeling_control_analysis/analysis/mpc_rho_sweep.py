import numpy as np
from scipy.signal import cont2discrete

from dynamics.state_space import (
    equilibrium_input,
    linearize_dynamics,
)

from controllers.mpc import UnconstrainedMPC


DT = 0.002
N = 80

RHO_VALUES = [
    1.0,
    10.0,
    100.0,
    1000.0,
]


# ============================================================
# UR5e equilibrium
# ============================================================

q_eq = np.deg2rad([
     0.0,
   -90.0,
    90.0,
   -90.0,
   -90.0,
     0.0,
])

x_eq = np.concatenate([
    q_eq,
    np.zeros(6),
])

tau_limit = np.array([
    150.0,
    150.0,
    150.0,
     28.0,
     28.0,
     28.0,
])


# ============================================================
# Linear model
# ============================================================

tau_eq = equilibrium_input(q_eq)

A, B = linearize_dynamics(
    x_eq,
    tau_eq,
)

Ad, Bd, _, _, _ = cont2discrete(
    (
        A,
        B,
        np.eye(12),
        np.zeros((12, 6)),
    ),
    DT,
    method="zoh",
)


# ============================================================
# Bryson normalization
# ============================================================

q_allow = np.deg2rad(2.0)
qdot_allow = np.deg2rad(10.0)

Q = np.diag(
    np.concatenate([
        np.full(
            6,
            1.0 / q_allow**2,
        ),
        np.full(
            6,
            1.0 / qdot_allow**2,
        ),
    ])
)

R_bryson = np.diag(
    1.0 / tau_limit**2
)


# ============================================================
# Initial perturbation
# ============================================================

dx0 = np.zeros(12)
dx0[1] = np.deg2rad(1.0)

X_ref = np.zeros(
    N * 12
)


# ============================================================
# Sweep
# ============================================================

print()
print("=" * 100)
print("UR5e MPC CONTROL-PENALTY SWEEP")
print("=" * 100)

print(
    f"{'rho_u':>8s} "
    f"{'u0 J2 Nm':>12s} "
    f"{'max J2 Nm':>12s} "
    f"{'J2 util %':>11s} "
    f"{'final J2 deg':>14s} "
    f"{'||U||2':>12s}"
)

print("-" * 100)


for rho_u in RHO_VALUES:

    R = (
        rho_u
        * R_bryson
    )

    mpc = UnconstrainedMPC(
        Ad=Ad,
        Bd=Bd,
        Q=Q,
        R=R,
        horizon=N,
    )

    U = mpc.solve(
        dx0,
        X_ref,
    )

    U_matrix = U.reshape(
        N,
        6,
    )

    DX = (
        mpc.F @ dx0
        + mpc.G @ U
    ).reshape(
        N,
        12,
    )

    u0_j2 = U_matrix[0, 1]

    max_j2 = np.max(
        np.abs(
            U_matrix[:, 1]
        )
    )

    util_j2 = (
        100.0
        * max_j2
        / tau_limit[1]
    )

    final_j2_deg = np.rad2deg(
        DX[-1, 1]
    )

    U_norm = np.linalg.norm(
        U
    )

    print(
        f"{rho_u:8.1f} "
        f"{u0_j2:12.6f} "
        f"{max_j2:12.6f} "
        f"{util_j2:11.4f} "
        f"{final_j2_deg:14.6f} "
        f"{U_norm:12.6f}"
    )