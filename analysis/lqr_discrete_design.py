import numpy as np

from scipy.signal import cont2discrete

from controllers.lqr import DiscreteJointLQRController
from dynamics.state_space import (
    equilibrium_input,
    linearize_dynamics,
)


# ============================================================
# Design conditions
# ============================================================

DT = 0.002

TORQUE_LIMITS = np.array([
    150.0, 150.0, 150.0,
    28.0, 28.0, 28.0,
])

q_eq = np.deg2rad([
    0.0, -90.0, 90.0,
    -90.0, -90.0, 0.0,
])

x_eq = np.concatenate([
    q_eq,
    np.zeros(6),
])

tau_eq = equilibrium_input(q_eq)

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
# Bryson normalization
#
# Initial physical design scales:
#
# position error : 2 deg
# velocity error : 10 deg/s
# torque          : benchmark torque limits
#
# These are normalization scales, NOT guaranteed constraints.
# ============================================================

q_allow = np.deg2rad(
    np.full(6, 2.0)
)

qdot_allow = np.deg2rad(
    np.full(6, 10.0)
)

tau_allow = TORQUE_LIMITS.copy()


Q_bryson = np.diag(
    np.concatenate([
        1.0 / q_allow**2,
        1.0 / qdot_allow**2,
    ])
)

R_bryson = np.diag(
    1.0 / tau_allow**2
)


# ============================================================
# Initial normalized design
#
# rho_u controls the global penalty on control effort.
#
# Start at 1.0. We are NOT tuning yet.
# ============================================================

RHO_U = 1.0

Q = Q_bryson.copy()
R = RHO_U * R_bryson


controller = DiscreteJointLQRController(
    Ad=Ad,
    Bd=Bd,
    Q=Q,
    R=R,
    torque_limits=TORQUE_LIMITS,
)


# ============================================================
# Discrete closed-loop analysis
# ============================================================

Acl_d = Ad - Bd @ controller.K

eig_d = np.linalg.eigvals(
    Acl_d
)

spectral_radius = np.max(
    np.abs(eig_d)
)


# Equivalent continuous pole representation
#
# lambda_c = log(lambda_d) / DT
#
# Useful only for interpretation.
# ============================================================

equivalent_continuous_poles = (
    np.log(eig_d.astype(complex))
    / DT
)


# ============================================================
# Print
# ============================================================

np.set_printoptions(
    precision=6,
    suppress=True,
    linewidth=180,
)

print("\n" + "=" * 86)
print("DISCRETE LQR DESIGN — BRYSON NORMALIZATION")
print("=" * 86)

print(f"\nSampling period : {DT:.6f} s")
print(f"Sampling rate   : {1.0 / DT:.1f} Hz")

print("\nPosition normalization [deg]:")
print(
    np.rad2deg(q_allow)
)

print("\nVelocity normalization [deg/s]:")
print(
    np.rad2deg(qdot_allow)
)

print("\nTorque normalization [Nm]:")
print(tau_allow)

print("\nQ diagonal:")
print(np.diag(Q))

print("\nR diagonal:")
print(np.diag(R))

print("\nK shape:")
print(controller.K.shape)

print("\nK:")
print(controller.K)


print("\n" + "=" * 86)
print("DISCRETE CLOSED-LOOP POLES")
print("=" * 86)

for value in eig_d:
    print(
        f"{value.real: .10f}"
        f" {value.imag:+.10f}j"
        f"    |z| = {abs(value):.10f}"
    )


print("\nSpectral radius:")
print(f"{spectral_radius:.10f}")


print("\n" + "=" * 86)
print("EQUIVALENT CONTINUOUS POLES")
print("=" * 86)

for value in equivalent_continuous_poles:
    print(
        f"{value.real: .6f}"
        f" {value.imag:+.6f}j"
    )


# ============================================================
# Equilibrium check
# ============================================================

tau_test = controller.compute(
    q=q_eq,
    qdot=np.zeros(6),
    q_ref=q_eq,
    qdot_ref=np.zeros(6),
    tau_ff=tau_eq,
)

print("\n" + "=" * 86)
print("EQUILIBRIUM CHECK")
print("=" * 86)

print("\nExpected equilibrium torque:")
print(tau_eq)

print("\nController torque:")
print(tau_test)

print("\nDifference:")
print(tau_test - tau_eq)


if spectral_radius < 1.0:
    print("\nDiscrete closed-loop stability: PASS")
else:
    print("\nDiscrete closed-loop stability: FAIL")

if np.allclose(
    tau_test,
    tau_eq,
    atol=1e-10,
):
    print("Equilibrium control check: PASS")
else:
    print("Equilibrium control check: FAIL")