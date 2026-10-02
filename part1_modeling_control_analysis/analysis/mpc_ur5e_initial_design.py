import numpy as np
from scipy.signal import cont2discrete

from dynamics.state_space import (
    equilibrium_input,
    linearize_dynamics,
)

from controllers.mpc import (
    UnconstrainedMPC,
)


# ============================================================
# Settings
# ============================================================

DT = 0.002
N = 20

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
# Linearization
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


# Start from the final LQR control penalty
# for a controlled comparison.
RHO_U = 100.0

R = (
    RHO_U
    * R_bryson
)


# ============================================================
# MPC
# ============================================================

mpc = UnconstrainedMPC(
    Ad=Ad,
    Bd=Bd,
    Q=Q,
    R=R,
    horizon=N,
)


# ============================================================
# Initial perturbation test
#
# 1 deg position error at J2.
# Reference deviation = zero.
# ============================================================

dx = np.zeros(12)

dx[1] = np.deg2rad(
    1.0
)

X_ref = np.zeros(
    N * 12
)


U = mpc.solve(
    dx,
    X_ref,
)

u0 = U[:6]


# ============================================================
# Predicted trajectory
# ============================================================

DX = (
    mpc.F @ dx
    + mpc.G @ U
)

DX = DX.reshape(
    N,
    12,
)


# ============================================================
# Cost / optimality
# ============================================================

prediction_error = (
    mpc.F @ dx
    - X_ref
)

gradient_half = (
    mpc.H @ U
    + mpc.G.T
    @ mpc.Qbar
    @ prediction_error
)

optimality_residual = np.max(
    np.abs(
        gradient_half
    )
)


# ============================================================
# Diagnostics
# ============================================================

np.set_printoptions(
    precision=9,
    suppress=False,
)


print("\n========================================")
print("UR5e INITIAL UNCONSTRAINED MPC DESIGN")
print("========================================")

print("\nSample time:")
print(
    f"{DT:.6f} s"
)

print("\nPrediction horizon:")
print(
    f"N = {N}"
)

print("\nPrediction window:")
print(
    f"{N * DT:.6f} s"
)

print("\nRHO_U:")
print(
    RHO_U
)

print("\nQ diagonal:")
print(
    np.diag(Q)
)

print("\nR diagonal:")
print(
    np.diag(R)
)

print("\nInitial deviation:")
print(
    "J2 position error = +1.0 deg"
)

print("\nFirst optimal correction torque [Nm]:")
print(
    u0
)

print("\nFirst correction torque utilization [%]:")
print(
    100.0
    * np.abs(u0)
    / tau_limit
)

print("\nMaximum |U| over horizon [Nm]:")
print(
    np.max(
        np.abs(
            U.reshape(N, 6)
        ),
        axis=0,
    )
)

print("\nPredicted final state deviation:")
print(
    DX[-1]
)

print(
    "\nPredicted final position deviation [deg]:"
)
print(
    np.rad2deg(
        DX[-1, :6]
    )
)

print(
    "\nPredicted final velocity deviation [deg/s]:"
)
print(
    np.rad2deg(
        DX[-1, 6:]
    )
)

print(
    "\nFirst-order optimality residual:"
)
print(
    f"{optimality_residual:.3e}"
)


# ============================================================
# PASS
# ============================================================

if optimality_residual >= 1e-9:
    raise AssertionError(
        "UR5e unconstrained MPC "
        "optimality check failed."
    )


print(
    "\nPASS: initial UR5e unconstrained MPC "
    "optimization is internally consistent."
)