import time
import numpy as np
import scipy.sparse as sp
import osqp

from scipy.signal import cont2discrete

from dynamics.state_space import (
    equilibrium_input,
    linearize_dynamics,
)

from controllers.mpc import build_prediction_matrices


# ============================================================
# Settings
# ============================================================

DT = 0.002

NP = 160

CONTROL_HORIZONS = [
    10,
    20,
    40,
    80,
    160,
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
# Linearization
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
# Prediction matrices
# ============================================================

F, G = build_prediction_matrices(
    Ad,
    Bd,
    NP,
)

nx = Ad.shape[0]
nu = Bd.shape[1]


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

R = (
    RHO_U
    * np.diag(
        1.0 / tau_limit**2
    )
)

Qbar = np.kron(
    np.eye(NP),
    Q,
)

Rbar_full = np.kron(
    np.eye(NP),
    R,
)


# ============================================================
# Initial deviation
#
# +1 degree J2 error
# ============================================================

dx0 = np.zeros(12)

dx0[1] = np.deg2rad(
    1.0
)

X_ref = np.zeros(
    NP * nx
)


# ============================================================
# Correction torque bounds
#
# tau = tau_ff + u
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
# Control-horizon expansion matrix
#
# U_full = S @ Uc
#
# First Nc inputs are independent.
# After Nc, the final control move is held constant.
# ============================================================

def build_control_horizon_matrix(
    Np,
    Nc,
    nu,
):

    if Nc <= 0:
        raise ValueError(
            "Nc must be positive."
        )

    if Nc > Np:
        raise ValueError(
            "Nc cannot exceed Np."
        )

    S = np.zeros(
        (
            Np * nu,
            Nc * nu,
        )
    )

    Iu = np.eye(nu)

    for k in range(Np):

        if k < Nc:
            j = k
        else:
            j = Nc - 1

        row = slice(
            k * nu,
            (k + 1) * nu,
        )

        col = slice(
            j * nu,
            (j + 1) * nu,
        )

        S[
            row,
            col,
        ] = Iu

    return S


# ============================================================
# Sweep
# ============================================================

print()
print("=" * 132)
print(
    "UR5e MPC CONTROL-HORIZON SWEEP "
    f"(Np={NP}, {NP * DT * 1000:.0f} ms prediction)"
)
print("=" * 132)

print(
    f"{'Nc':>5s} "
    f"{'control ms':>11s} "
    f"{'vars':>7s} "
    f"{'u0 J2':>11s} "
    f"{'tau0 J2':>11s} "
    f"{'final J2 deg':>14s} "
    f"{'max tau util %':>15s} "
    f"{'iter':>7s} "
    f"{'mean ms':>10s} "
    f"{'p99 ms':>10s} "
    f"{'max ms':>10s}"
)

print("-" * 132)


for NC in CONTROL_HORIZONS:

    # --------------------------------------------------------
    # Control-horizon mapping
    # --------------------------------------------------------

    S = build_control_horizon_matrix(
        NP,
        NC,
        nu,
    )

    # Reduced prediction input matrix
    Gc = G @ S

    # --------------------------------------------------------
    # Reduced QP
    #
    # Cost:
    #
    # (F x + Gc Uc - Xref)' Qbar (...)
    # +
    # (S Uc)' Rbar_full (S Uc)
    # --------------------------------------------------------

    H = (
        Gc.T
        @ Qbar
        @ Gc
        +
        S.T
        @ Rbar_full
        @ S
    )

    prediction_error = (
        F @ dx0
        - X_ref
    )

    q_qp = (
        2.0
        * Gc.T
        @ Qbar
        @ prediction_error
    )

    P = sp.csc_matrix(
        2.0 * H
    )

    # --------------------------------------------------------
    # Bounds on independent control moves
    # --------------------------------------------------------

    lower = np.tile(
        u_lower,
        NC,
    )

    upper = np.tile(
        u_upper,
        NC,
    )

    A_constraint = sp.eye(
        NC * nu,
        format="csc",
    )

    # --------------------------------------------------------
    # OSQP
    # --------------------------------------------------------

    solver = osqp.OSQP()

    solver.setup(
        P=P,
        q=q_qp,
        A=A_constraint,
        l=lower,
        u=upper,
        verbose=False,

        # Disable polishing for timing benchmark
        polishing=False,

        eps_abs=1e-10,
        eps_rel=1e-10,
    )

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    warm = solver.solve()

    if warm.info.status not in (
        "solved",
        "solved inaccurate",
    ):
        raise RuntimeError(
            f"OSQP failed for Nc={NC}: "
            + warm.info.status
        )

    # --------------------------------------------------------
    # Repeated timing
    # --------------------------------------------------------

    timings = []

    result = None

    for _ in range(100):

        t0 = time.perf_counter()

        result = solver.solve()

        t1 = time.perf_counter()

        if result.info.status not in (
            "solved",
            "solved inaccurate",
        ):
            raise RuntimeError(
                f"OSQP failed for Nc={NC}: "
                + result.info.status
            )

        timings.append(
            (t1 - t0) * 1000.0
        )

    Uc = result.x.copy()

    # --------------------------------------------------------
    # Expand reduced control sequence
    # --------------------------------------------------------

    U_full = S @ Uc

    U_matrix = U_full.reshape(
        NP,
        nu,
    )

    # --------------------------------------------------------
    # Predicted state trajectory
    # --------------------------------------------------------

    DX = (
        F @ dx0
        + G @ U_full
    ).reshape(
        NP,
        nx,
    )

    # --------------------------------------------------------
    # Actual total torque
    # --------------------------------------------------------

    TAU = (
        U_matrix
        + tau_ff.reshape(1, nu)
    )

    utilization = (
        100.0
        * np.abs(TAU)
        / tau_limit.reshape(1, nu)
    )

    max_tau_util = np.max(
        utilization
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    u0_j2 = U_matrix[0, 1]

    tau0_j2 = (
        tau_ff[1]
        + u0_j2
    )

    final_j2_deg = np.rad2deg(
        DX[-1, 1]
    )

    timings = np.asarray(
        timings
    )

    mean_ms = np.mean(
        timings
    )

    p99_ms = np.percentile(
        timings,
        99,
    )

    max_ms = np.max(
        timings
    )

    iterations = (
        result.info.iter
    )

    print(
        f"{NC:5d} "
        f"{NC * DT * 1000:11.1f} "
        f"{NC * nu:7d} "
        f"{u0_j2:11.4f} "
        f"{tau0_j2:11.4f} "
        f"{final_j2_deg:14.6f} "
        f"{max_tau_util:15.4f} "
        f"{iterations:7d} "
        f"{mean_ms:10.4f} "
        f"{p99_ms:10.4f} "
        f"{max_ms:10.4f}"
    )


print()
print("Prediction horizon:")
print(
    f"Np = {NP} "
    f"({NP * DT * 1000:.1f} ms)"
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