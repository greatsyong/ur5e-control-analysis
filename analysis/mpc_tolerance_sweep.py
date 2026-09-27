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

TOLERANCES = [
    1e-8,
    1e-6,
    1e-5,
]

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
# Offline QP matrices
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

GRADIENT_MAP = (
    2.0
    * Gc.T
    @ Qbar
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
# Vectorized horizon data
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
# Sweep
# ============================================================

print()
print("=" * 112)
print("MPC OSQP TOLERANCE SWEEP")
print("=" * 112)

print(
    f"Np={NP}, Nc={NC}, "
    f"rho_u={RHO_U:.1f}, "
    f"Ts={DT * 1000.0:.1f} ms"
)

print()

print(
    f"{'tol':>10s}"
    f"{'mean iter':>12s}"
    f"{'p99 iter':>12s}"
    f"{'mean solve':>14s}"
    f"{'p99 solve':>14s}"
    f"{'max solve':>14s}"
    f"{'mean total':>14s}"
    f"{'p99 total':>14s}"
    f"{'max |du0|':>14s}"
)

print("-" * 112)


baseline_u0 = None


for tolerance in TOLERANCES:

    n_decision = (
        NC * nu
    )

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
        eps_abs=tolerance,
        eps_rel=tolerance,
        warm_starting=True,
    )

    solve_times = []
    total_times = []
    iteration_history = []
    u0_history = []

    for k in range(n_samples):

        total_start = (
            time.perf_counter()
        )

        # ----------------------------------------------------
        # Vectorized horizon
        # ----------------------------------------------------

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

        Z_ref = (
            Z_ref_matrix.reshape(-1)
        )

        # Profiling sweep uses nominal reference state
        x_current = x_ref_all[k]

        z = (
            x_current
            - x_eq
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        V_ff = (
            tau_ff_horizon
            - tau_eq.reshape(1, nu)
        ).reshape(-1)

        prediction_error = (
            F @ z
            + G @ V_ff
            - Z_ref
        )

        # ----------------------------------------------------
        # QP gradient
        # ----------------------------------------------------

        q_qp = (
            GRADIENT_MAP
            @ prediction_error
        )

        # ----------------------------------------------------
        # Bounds
        # ----------------------------------------------------

        lower_full = (
            -TORQUE_LIMITS.reshape(
                1,
                nu,
            )
            - tau_ff_horizon
        )

        upper_full = (
            TORQUE_LIMITS.reshape(
                1,
                nu,
            )
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

        lower_reduced[NC - 1] = (
            np.max(
                lower_full[NC - 1:],
                axis=0,
            )
        )

        upper_reduced[NC - 1] = (
            np.min(
                upper_full[NC - 1:],
                axis=0,
            )
        )

        lower = (
            lower_reduced.reshape(-1)
        )

        upper = (
            upper_reduced.reshape(-1)
        )

        solver.update(
            q=q_qp,
            l=lower,
            u=upper,
        )

        # ----------------------------------------------------
        # Solve timing
        # ----------------------------------------------------

        tic = time.perf_counter()

        result = solver.solve()

        toc = time.perf_counter()

        if result.info.status not in (
            "solved",
            "solved inaccurate",
        ):
            raise RuntimeError(
                f"OSQP failed at "
                f"tol={tolerance}: "
                + result.info.status
            )

        solve_times.append(
            toc - tic
        )

        iteration_history.append(
            result.info.iter
        )

        u0_history.append(
            result.x[:nu].copy()
        )

        total_times.append(
            time.perf_counter()
            - total_start
        )


    solve_ms = (
        1000.0
        * np.asarray(
            solve_times
        )
    )

    total_ms = (
        1000.0
        * np.asarray(
            total_times
        )
    )

    iteration_history = np.asarray(
        iteration_history
    )

    u0_history = np.asarray(
        u0_history
    )


    # --------------------------------------------------------
    # Compare first control move against 1e-8 baseline
    # --------------------------------------------------------

    if baseline_u0 is None:

        baseline_u0 = (
            u0_history.copy()
        )

        max_du0 = 0.0

    else:

        max_du0 = np.max(
            np.abs(
                u0_history
                - baseline_u0
            )
        )


    print(
        f"{tolerance:10.0e}"
        f"{np.mean(iteration_history):12.2f}"
        f"{np.percentile(iteration_history, 99):12.1f}"
        f"{np.mean(solve_ms):14.4f}"
        f"{np.percentile(solve_ms, 99):14.4f}"
        f"{np.max(solve_ms):14.4f}"
        f"{np.mean(total_ms):14.4f}"
        f"{np.percentile(total_ms, 99):14.4f}"
        f"{max_du0:14.6e}"
    )


print()
print(
    "max |du0| is the maximum absolute difference "
    "in the first MPC correction torque [Nm]"
)

print(
    "relative to the 1e-8 solution over the full "
    "nominal reference trajectory."
)