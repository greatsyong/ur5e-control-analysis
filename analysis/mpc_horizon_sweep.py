import time
import numpy as np
from scipy.signal import cont2discrete

from dynamics.state_space import (
    equilibrium_input,
    linearize_dynamics,
)

from controllers.mpc import (
    UnconstrainedMPC,
)


DT = 0.002

HORIZONS = [
    10,
    20,
    40,
    80,
]

RHO_U = 100.0


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
# Bryson weights
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

R = RHO_U * R_bryson


# ============================================================
# Initial perturbation
# ============================================================

dx0 = np.zeros(12)

dx0[1] = np.deg2rad(
    1.0
)


# ============================================================
# Sweep
# ============================================================

print()
print("=" * 96)
print("UR5e MPC HORIZON SWEEP")
print("=" * 96)

print(
    f"{'N':>5s} "
    f"{'window ms':>10s} "
    f"{'vars':>7s} "
    f"{'u0 J2 Nm':>12s} "
    f"{'max J2 Nm':>12s} "
    f"{'final J2 deg':>14s} "
    f"{'solve ms':>12s}"
)

print("-" * 96)


for N in HORIZONS:

    mpc = UnconstrainedMPC(
        Ad=Ad,
        Bd=Bd,
        Q=Q,
        R=R,
        horizon=N,
    )

    X_ref = np.zeros(
        N * 12
    )

    # Warm-up
    mpc.solve(
        dx0,
        X_ref,
    )

    # Timing
    timings = []

    U = None

    for _ in range(20):

        t0 = time.perf_counter()

        U = mpc.solve(
            dx0,
            X_ref,
        )

        t1 = time.perf_counter()

        timings.append(
            (t1 - t0) * 1000.0
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

    final_j2_deg = np.rad2deg(
        DX[-1, 1]
    )

    mean_solve_ms = np.mean(
        timings
    )

    print(
        f"{N:5d} "
        f"{1000.0*N*DT:10.1f} "
        f"{N*6:7d} "
        f"{u0_j2:12.6f} "
        f"{max_j2:12.6f} "
        f"{final_j2_deg:14.6f} "
        f"{mean_solve_ms:12.6f}"
    )


print()
print(
    "Note: solve time above is the dense "
    "unconstrained NumPy solve, not OSQP."
)