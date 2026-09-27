import time
import numpy as np
import scipy.sparse as sp
import osqp

from scipy.signal import cont2discrete

from dynamics.state_space import (
    equilibrium_input,
    linearize_dynamics,
)
from trajectories.joint_trajectory import quintic_joint_trajectory
from controllers.mpc import build_prediction_matrices


# ============================================================
# Configuration
# ============================================================

DT = 0.002
DURATION = 4.0

NP = 160
NC = 40
RHO_U = 10.0

TORQUE_LIMITS = np.array([
    150.0, 150.0, 150.0,
    28.0, 28.0, 28.0,
])

q_start = np.deg2rad([
    0.0, -90.0, 90.0,
    -90.0, -90.0, 0.0,
])

q_goal = np.deg2rad([
    20.0, -60.0, 60.0,
    -70.0, -70.0, 20.0,
])


# ============================================================
# Fixed linearization
# ============================================================

x_eq = np.concatenate([
    q_start,
    np.zeros(6),
])

tau_eq = equilibrium_input(
    q_start
)

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

nx = 12
nu = 6


# ============================================================
# Prediction matrices
# ============================================================

F, G = build_prediction_matrices(
    Ad,
    Bd,
    NP,
)


# ============================================================
# Control-horizon mapping
# ============================================================

def build_control_horizon_matrix(
    Np,
    Nc,
    nu,
):
    S = np.zeros(
        (
            Np * nu,
            Nc * nu,
        )
    )

    Iu = np.eye(nu)

    for k in range(Np):

        j = (
            k
            if k < Nc
            else Nc - 1
        )

        S[
            k * nu:(k + 1) * nu,
            j * nu:(j + 1) * nu,
        ] = Iu

    return S


S = build_control_horizon_matrix(
    NP,
    NC,
    nu,
)

Gc = G @ S


# ============================================================
# Weights
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

R = (
    RHO_U
    * np.diag(
        1.0 / TORQUE_LIMITS**2
    )
)

Qbar = np.kron(
    np.eye(NP),
    Q,
)

Rbar = np.kron(
    np.eye(NP),
    R,
)


# ============================================================
# QP Hessian
# ============================================================

H = (
    Gc.T
    @ Qbar
    @ Gc
    +
    S.T
    @ Rbar
    @ S
)

P = sp.csc_matrix(
    2.0 * H
)

# Precompute fixed QP gradient map offline
GRADIENT_MAP = (
    2.0
    * Gc.T
    @ Qbar
)


# ============================================================
# OSQP
# ============================================================

n_decision = NC * nu

solver = osqp.OSQP()

solver.setup(
    P=P,
    q=np.zeros(n_decision),
    A=sp.eye(
        n_decision,
        format="csc",
    ),
    l=np.full(
        n_decision,
        -np.inf,
    ),
    u=np.full(
        n_decision,
        np.inf,
    ),
    verbose=False,
    polishing=False,
    eps_abs=1e-8,
    eps_rel=1e-8,
    warm_starting=True,
)


# ============================================================
# Precompute nominal trajectory
# ============================================================

times = np.arange(
    0.0,
    DURATION + DT,
    DT,
)

n_samples = len(times)

q_ref_all = np.zeros(
    (n_samples, 6)
)

qdot_ref_all = np.zeros(
    (n_samples, 6)
)

tau_ff_all = np.zeros(
    (n_samples, 6)
)

for k, t in enumerate(times):

    q_ref, qdot_ref, _ = (
        quintic_joint_trajectory(
            t,
            DURATION,
            q_start,
            q_goal,
        )
    )

    q_ref_all[k] = q_ref
    qdot_ref_all[k] = qdot_ref

    tau_ff_all[k] = (
        equilibrium_input(
            q_ref
        )
    )


# ============================================================
# Precompute padded reference arrays for vectorized horizon
# extraction
# ============================================================

x_ref_all = np.hstack([
    q_ref_all,
    qdot_ref_all,
])

x_ref_padded = np.vstack([
    x_ref_all,
    np.repeat(
        x_ref_all[-1][None, :],
        NP,
        axis=0,
    ),
])

tau_ff_padded = np.vstack([
    tau_ff_all,
    np.repeat(
        tau_ff_all[-1][None, :],
        NP,
        axis=0,
    ),
])


# ============================================================
# Profiling storage
# ============================================================

t_horizon = []
t_prediction = []
t_gradient = []
t_bounds = []
t_update = []
t_solve = []
t_total = []

iterations = []


# ============================================================
# Profile along nominal reference
#
# For this profiling test, current state is taken as the
# nominal reference state. We are measuring computation,
# not closed-loop tracking performance.
# ============================================================

for k in range(n_samples):

    total_start = time.perf_counter()

    # --------------------------------------------------------
    # 1. Horizon construction
    # --------------------------------------------------------

    tic = time.perf_counter()

    # Vectorized horizon extraction
    Z_ref_matrix = (
        x_ref_padded[
            k + 1:k + 1 + NP
        ]
        - x_eq
    )

    tau_ff_horizon = (
        tau_ff_padded[
            k:k + NP
        ]
    )

    Z_ref = Z_ref_matrix.reshape(-1)

    toc = time.perf_counter()

    t_horizon.append(
        toc - tic
    )


    # --------------------------------------------------------
    # Current nominal state
    # --------------------------------------------------------

    x_current = np.concatenate([
        q_ref_all[k],
        qdot_ref_all[k],
    ])

    z = (
        x_current
        - x_eq
    )


    # --------------------------------------------------------
    # 2. Prediction terms
    # --------------------------------------------------------

    tic = time.perf_counter()

    V_ff = (
        tau_ff_horizon
        - tau_eq.reshape(1, nu)
    ).reshape(-1)

    base_prediction = (
        F @ z
        + G @ V_ff
    )

    prediction_error = (
        base_prediction
        - Z_ref
    )

    toc = time.perf_counter()

    t_prediction.append(
        toc - tic
    )


    # --------------------------------------------------------
    # 3. QP gradient
    # --------------------------------------------------------

    tic = time.perf_counter()

    q_qp = (
        GRADIENT_MAP
        @ prediction_error
    )

    toc = time.perf_counter()

    t_gradient.append(
        toc - tic
    )


    # --------------------------------------------------------
    # 4. Bounds
    # --------------------------------------------------------

    tic = time.perf_counter()

    lower_full = (
        -TORQUE_LIMITS.reshape(1, nu)
        - tau_ff_horizon
    )

    upper_full = (
        TORQUE_LIMITS.reshape(1, nu)
        - tau_ff_horizon
    )

    lower_reduced = np.zeros(
        (NC, nu)
    )

    upper_reduced = np.zeros(
        (NC, nu)
    )

    lower_reduced[:NC - 1] = (
        lower_full[:NC - 1]
    )

    upper_reduced[:NC - 1] = (
        upper_full[:NC - 1]
    )

    lower_reduced[NC - 1] = np.max(
        lower_full[NC - 1:],
        axis=0,
    )

    upper_reduced[NC - 1] = np.min(
        upper_full[NC - 1:],
        axis=0,
    )

    lower = lower_reduced.reshape(-1)
    upper = upper_reduced.reshape(-1)

    toc = time.perf_counter()

    t_bounds.append(
        toc - tic
    )


    # --------------------------------------------------------
    # 5. OSQP update
    # --------------------------------------------------------

    tic = time.perf_counter()

    solver.update(
        q=q_qp,
        l=lower,
        u=upper,
    )

    toc = time.perf_counter()

    t_update.append(
        toc - tic
    )


    # --------------------------------------------------------
    # 6. OSQP solve
    # --------------------------------------------------------

    tic = time.perf_counter()

    result = solver.solve()

    toc = time.perf_counter()

    if result.info.status not in (
        "solved",
        "solved inaccurate",
    ):
        raise RuntimeError(
            result.info.status
        )

    t_solve.append(
        toc - tic
    )

    iterations.append(
        result.info.iter
    )


    # --------------------------------------------------------
    # Total
    # --------------------------------------------------------

    total_end = time.perf_counter()

    t_total.append(
        total_end
        - total_start
    )


# ============================================================
# Summary
# ============================================================

def summarize(
    name,
    values,
):

    ms = (
        1000.0
        * np.asarray(values)
    )

    print(
        f"{name:<22s}"
        f"{np.mean(ms):>12.4f}"
        f"{np.percentile(ms, 99):>12.4f}"
        f"{np.max(ms):>12.4f}"
    )


print()
print("=" * 64)
print("MPC ONLINE COMPUTATION PROFILE")
print("=" * 64)

print(
    f"Np={NP}, "
    f"Nc={NC}, "
    f"rho_u={RHO_U:.1f}, "
    f"Ts={DT * 1000.0:.1f} ms"
)

print()

print(
    f"{'Component':<22s}"
    f"{'Mean [ms]':>12s}"
    f"{'P99 [ms]':>12s}"
    f"{'Max [ms]':>12s}"
)

print("-" * 58)

summarize(
    "Horizon construction",
    t_horizon,
)

summarize(
    "Prediction",
    t_prediction,
)

summarize(
    "QP gradient",
    t_gradient,
)

summarize(
    "Bounds",
    t_bounds,
)

summarize(
    "OSQP update",
    t_update,
)

summarize(
    "OSQP solve",
    t_solve,
)

print("-" * 58)

summarize(
    "TOTAL",
    t_total,
)


iterations = np.asarray(
    iterations
)

print()

print("OSQP iterations:")

print(
    f"Mean : "
    f"{np.mean(iterations):.3f}"
)

print(
    f"P99  : "
    f"{np.percentile(iterations, 99):.1f}"
)

print(
    f"Max  : "
    f"{np.max(iterations)}"
)

print()

print(
    "2 ms control-period utilization "
    "based on total mean:"
)

print(
    f"{100.0 * np.mean(t_total) / DT:.2f} %"
)