import numpy as np

from dynamics.state_space import (
    linearize_dynamics,
    equilibrium_input,
)

from controllers.lqr import JointLQRController


# ============================================================
# Benchmark equilibrium
# ============================================================

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

# Exact equilibrium torque for the nonlinear model
tau_eq = equilibrium_input(q_eq)


# ============================================================
# Linearize nonlinear UR5e dynamics
# ============================================================

A, B = linearize_dynamics(
    x_eq,
    tau_eq,
)

print("=" * 80)
print("UR5e LQR EQUILIBRIUM ANALYSIS")
print("=" * 80)

print("\nA shape:", A.shape)
print("B shape:", B.shape)

print("\nEquilibrium torque [Nm]:")
print(tau_eq)


# ============================================================
# Controllability
# ============================================================

n = A.shape[0]

controllability_blocks = []

AB_power = B.copy()

for _ in range(n):
    controllability_blocks.append(
        AB_power
    )

    AB_power = A @ AB_power

C = np.hstack(
    controllability_blocks
)

rank_C = np.linalg.matrix_rank(C)

print("\nControllability rank:")
print(f"{rank_C} / {n}")


# ============================================================
# Initial LQR weights
#
# State:
#   x = [q1 ... q6, qdot1 ... qdot6]
#
# Start with interpretable uniform weights.
# These are NOT final tuned weights.
# ============================================================

Q = np.diag(
    [100.0] * 6
    + [10.0] * 6
)

R = np.eye(6)

torque_limits = np.array([
    150.0,
    150.0,
    150.0,
    28.0,
    28.0,
    28.0,
])


# ============================================================
# LQR controller
# ============================================================

controller = JointLQRController(
    A=A,
    B=B,
    Q=Q,
    R=R,
    torque_limits=torque_limits,
)

K = controller.K

print("\nLQR gain shape:")
print(K.shape)

print("\nLQR gain K:")
np.set_printoptions(
    precision=6,
    suppress=True,
    linewidth=180,
)

print(K)


# ============================================================
# Closed-loop eigenvalues
# ============================================================

A_cl = A - B @ K

eig_open = np.linalg.eigvals(A)
eig_closed = np.linalg.eigvals(A_cl)

print("\nOpen-loop eigenvalues:")
for value in eig_open:
    print(
        f"{value.real: .8f}"
        f" {value.imag:+.8f}j"
    )

print("\nClosed-loop eigenvalues:")
for value in eig_closed:
    print(
        f"{value.real: .8f}"
        f" {value.imag:+.8f}j"
    )


# ============================================================
# Stability check
# ============================================================

max_real_closed = np.max(
    np.real(eig_closed)
)

print("\nMaximum closed-loop real part:")
print(f"{max_real_closed:.10f}")

if rank_C == n:
    print("\nControllability: PASS")
else:
    print("\nControllability: FAIL")

if max_real_closed < 0.0:
    print("Closed-loop stability: PASS")
else:
    print("Closed-loop stability: FAIL")


# ============================================================
# Equilibrium controller check
# ============================================================

tau_test = controller.compute(
    q=q_eq,
    qdot=qdot_eq,
    q_ref=q_eq,
    qdot_ref=qdot_eq,
    tau_ff=tau_eq,
)

print("\nEquilibrium controller torque:")
print(tau_test)

print("\nDifference from equilibrium torque:")
print(tau_test - tau_eq)

if np.allclose(
    tau_test,
    tau_eq,
    atol=1e-10,
):
    print("\nEquilibrium control check: PASS")
else:
    print("\nEquilibrium control check: FAIL")