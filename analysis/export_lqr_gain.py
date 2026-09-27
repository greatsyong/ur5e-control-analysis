import numpy as np
from scipy.signal import cont2discrete

from controllers.lqr import DiscreteJointLQRController
from dynamics.state_space import (
    equilibrium_input,
    linearize_dynamics,
)


# ============================================================
# Final nominal LQR configuration
# ============================================================

DT = 0.002
RHO_U = 100.0

TORQUE_LIMITS = np.array([
    150.0, 150.0, 150.0,
    28.0, 28.0, 28.0,
])

q_start = np.deg2rad([
    0.0, -90.0, 90.0,
    -90.0, -90.0, 0.0,
])


# ============================================================
# Fixed equilibrium linearization
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


# ============================================================
# Exact ZOH discretization
# ============================================================

C_dummy = np.eye(12)
D_dummy = np.zeros((12, 6))

Ad, Bd, _, _, _ = cont2discrete(
    (A, B, C_dummy, D_dummy),
    DT,
    method="zoh",
)


# ============================================================
# Bryson-normalized LQR weights
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

R = RHO_U * R_bryson


# ============================================================
# Final discrete LQR controller
# ============================================================

controller = DiscreteJointLQRController(
    Ad=Ad,
    Bd=Bd,
    Q=Q,
    R=R,
    torque_limits=TORQUE_LIMITS,
)

K = controller.K


# ============================================================
# Verification
# ============================================================

eig_d = np.linalg.eigvals(
    Ad - Bd @ K
)

spectral_radius = np.max(
    np.abs(eig_d)
)

np.set_printoptions(
    precision=15,
    suppress=False,
    linewidth=300,
)

print("\n========================================")
print("FINAL DISCRETE LQR GAIN")
print("========================================")

print("\nK shape:")
print(K.shape)

print("\nSpectral radius:")
print(f"{spectral_radius:.15f}")

print("\n||K||2:")
print(f"{np.linalg.norm(K, 2):.15f}")

print("\nmax |Kij|:")
print(f"{np.max(np.abs(K)):.15f}")

print("\nK =")
print(K)


# ============================================================
# C++ initializer
# ============================================================

print("\n========================================")
print("C++ INITIALIZER")
print("========================================\n")

print("const double K[6][12] = {")

for row in K:

    values = ", ".join(
        f"{value:.17e}"
        for value in row
    )

    print(
        "    {"
        + values
        + "},"
    )

print("};")