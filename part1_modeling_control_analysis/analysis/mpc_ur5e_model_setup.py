import numpy as np
from scipy.signal import cont2discrete

from dynamics.state_space import (
    equilibrium_input,
    linearize_dynamics,
)

from controllers.mpc import build_prediction_matrices


# ============================================================
# Benchmark / model settings
# ============================================================

DT = 0.002          # 500 Hz
N = 20              # initial prediction horizon
                    # 20 * 2 ms = 40 ms

q_eq = np.deg2rad([
     0.0,
   -90.0,
    90.0,
   -90.0,
   -90.0,
     0.0,
])

qdot_eq = np.zeros(6)

x_eq = np.concatenate([
    q_eq,
    qdot_eq,
])


# ============================================================
# Continuous-time UR5e linearization
# ============================================================

tau_eq = equilibrium_input(q_eq)

A, B = linearize_dynamics(
    x_eq,
    tau_eq,
)


print("\n========================================")
print("UR5e MPC MODEL SETUP")
print("========================================")

print("\nContinuous model:")
print("A shape:", A.shape)
print("B shape:", B.shape)

print("\nEquilibrium gravity torque [Nm]:")
print(tau_eq)


# ============================================================
# Exact ZOH discretization
# ============================================================

C_dummy = np.eye(12)
D_dummy = np.zeros((12, 6))

Ad, Bd, _, _, _ = cont2discrete(
    (
        A,
        B,
        C_dummy,
        D_dummy,
    ),
    DT,
    method="zoh",
)


print("\nDiscrete model:")
print("Ad shape:", Ad.shape)
print("Bd shape:", Bd.shape)

print("\nSample time:")
print(f"{DT:.6f} s")

print("\nPrediction horizon:")
print(f"N = {N}")

print("\nPhysical prediction window:")
print(f"{N * DT:.6f} s")


# ============================================================
# Prediction matrices
# ============================================================

F, G = build_prediction_matrices(
    Ad,
    Bd,
    N,
)


print("\nPrediction matrices:")
print("F shape:", F.shape)
print("G shape:", G.shape)


expected_F_shape = (
    N * 12,
    12,
)

expected_G_shape = (
    N * 12,
    N * 6,
)


assert F.shape == expected_F_shape
assert G.shape == expected_G_shape


# ============================================================
# Direct propagation validation
# ============================================================

rng = np.random.default_rng(
    12345
)

dx0 = rng.normal(
    scale=0.01,
    size=12,
)

U = rng.normal(
    scale=0.1,
    size=N * 6,
)


DX_matrix = (
    F @ dx0
    + G @ U
)


dx = dx0.copy()
DX_direct = []

for k in range(N):

    uk = U[
        k * 6:
        (k + 1) * 6
    ]

    dx = (
        Ad @ dx
        + Bd @ uk
    )

    DX_direct.append(
        dx.copy()
    )


DX_direct = np.concatenate(
    DX_direct
)


prediction_error = (
    DX_matrix
    - DX_direct
)

max_prediction_error = np.max(
    np.abs(
        prediction_error
    )
)


print(
    "\nMaximum stacked/direct "
    "prediction difference:"
)

print(
    f"{max_prediction_error:.3e}"
)


# ============================================================
# Discrete open-loop eigenvalues
# ============================================================

eig_Ad = np.linalg.eigvals(
    Ad
)

spectral_radius = np.max(
    np.abs(
        eig_Ad
    )
)


print(
    "\nOpen-loop discrete spectral radius:"
)

print(
    f"{spectral_radius:.12f}"
)


# ============================================================
# Initial MPC dimensions
# ============================================================

n_state_variables = (
    N * 12
)

n_control_variables = (
    N * 6
)


print("\nMPC stacked dimensions:")

print(
    "Predicted state variables:",
    n_state_variables,
)

print(
    "Decision variables:",
    n_control_variables,
)


# ============================================================
# PASS / FAIL
# ============================================================

TOL = 1e-11

if max_prediction_error >= TOL:

    raise AssertionError(
        "UR5e stacked prediction "
        "does not match direct propagation."
    )


print(
    "\nPASS: UR5e discrete prediction "
    "model is internally consistent."
)