import time
import numpy as np
from scipy.signal import cont2discrete

from dynamics.state_space import (
    equilibrium_input,
    linearize_dynamics,
)

from controllers.mpc import OSQPMPC


# ============================================================
# Settings
# ============================================================

DT = 0.002

HORIZONS = [
    40,
    80,
    160,
    320,
]

RHO_U = 10.0

tau_limit = np.array([
    150.0,
    150.0,
    150.0,
     28.0,
     28.0,
     28.0,
])


# ============================================================
# Equilibrium
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

tau_ff = equilibrium_input(q_eq)


# ============================================================
# Linearization and exact ZOH discretization
# ============================================================

A, B = linearize_dynamics(
    x_eq,
    tau_ff,
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

R = (
    RHO_U
    * R_bryson
)


# ============================================================
# Initial deviation
# ============================================================

dx0 = np.zeros(12)

dx0[1] = np.deg2rad(
    1.0
)


# ============================================================
# Correction-torque bounds
#
# total torque:
#
#     tau = tau_ff + u
#
# therefore:
#
#     -tau_limit - tau_ff <= u
#     u <= tau_limit - tau_ff
# ============================================================

u_lower = (
    -tau_limit
    - tau_ff
)

u_upper = (
     tau_limit
    - tau_ff
)


# ============================================================
# Sweep
# ============================================================

print()
print("=" * 122)
print("UR5e CONSTRAINED OSQP MPC HORIZON SWEEP")
print("=" * 122)

print(
    f"{'N':>5s} "
    f"{'window ms':>10s} "
    f"{'vars':>7s} "
    f"{'u0 J2':>11s} "
    f"{'tau0 J2':>11s} "
    f"{'final J2 deg':>14s} "
    f"{'max tau util %':>15s} "
    f"{'iter':>7s} "
    f"{'solve ms':>11s}"
)

print("-" * 122)


for N in HORIZONS:

    mpc = OSQPMPC(
        Ad=Ad,
        Bd=Bd,
        Q=Q,
        R=R,
        horizon=N,
        input_lower=u_lower,
        input_upper=u_upper,
    )

    X_ref = np.zeros(
        N * 12
    )

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    mpc.solve(
        dx0,
        X_ref,
    )

    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    timings = []
    U = None
    result_iterations = None

    for _ in range(50):

        t0 = time.perf_counter()

        U = mpc.solve(
            dx0,
            X_ref,
        )

        t1 = time.perf_counter()

        timings.append(
            (t1 - t0) * 1000.0
        )

        # OSQP iteration count from latest solve
        result_iterations = (
            mpc.last_result.info.iter
        )

    U_matrix = U.reshape(
        N,
        6,
    )

    # --------------------------------------------------------
    # Predicted deviation trajectory
    # --------------------------------------------------------

    DX = (
        mpc.F @ dx0
        + mpc.G @ U
    ).reshape(
        N,
        12,
    )

    # --------------------------------------------------------
    # Total torque over horizon
    # --------------------------------------------------------

    TAU = (
        U_matrix
        + tau_ff.reshape(1, 6)
    )

    utilization = (
        100.0
        * np.abs(TAU)
        / tau_limit.reshape(1, 6)
    )

    max_tau_util = np.max(
        utilization
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    u0_j2 = U_matrix[0, 1]

    tau0_j2 = (
        tau_ff[1]
        + u0_j2
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
        f"{u0_j2:11.4f} "
        f"{tau0_j2:11.4f} "
        f"{final_j2_deg:14.6f} "
        f"{max_tau_util:15.4f} "
        f"{result_iterations:7d} "
        f"{mean_solve_ms:11.4f}"
    )


print()
print("Gravity feedforward [Nm]:")
print(tau_ff)

print()
print("Correction lower bounds [Nm]:")
print(u_lower)

print()
print("Correction upper bounds [Nm]:")
print(u_upper)