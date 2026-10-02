import numpy as np

from controllers.pid import (
    JointPIDController,
)


DT = 0.002

TORQUE_LIMITS = np.array([
    150.0,
    150.0,
    150.0,
    28.0,
    28.0,
    28.0,
])


controller = JointPIDController(
    kp=np.ones(6) * 100.0,
    ki=np.zeros(6),
    kd=np.ones(6) * 10.0,
    dt=DT,
    torque_limits=TORQUE_LIMITS,
)


# -------------------------------------------------------------
# Test 1
# Zero error -> zero torque
# -------------------------------------------------------------

q = np.zeros(6)
qdot = np.zeros(6)

q_ref = np.zeros(6)
qdot_ref = np.zeros(6)

tau = controller.compute(
    q,
    qdot,
    q_ref,
    qdot_ref,
)

print("=" * 70)
print("TEST 1 — ZERO ERROR")
print("=" * 70)

print("tau:")
print(tau)


# -------------------------------------------------------------
# Test 2
# Position error
# -------------------------------------------------------------

controller.reset()

q_ref = np.deg2rad(
    [1, 1, 1, 1, 1, 1]
)

tau = controller.compute(
    q,
    qdot,
    q_ref,
    qdot_ref,
)

print("\n" + "=" * 70)
print("TEST 2 — POSITION ERROR")
print("=" * 70)

print("tau:")
print(tau)


# -------------------------------------------------------------
# Test 3
# Torque saturation
# -------------------------------------------------------------

controller.reset()

q_ref = np.deg2rad(
    [180, 180, 180, 180, 180, 180]
)

tau = controller.compute(
    q,
    qdot,
    q_ref,
    qdot_ref,
)

print("\n" + "=" * 70)
print("TEST 3 — TORQUE SATURATION")
print("=" * 70)

print("tau:")
print(tau)

# -------------------------------------------------------------
# Test 4
# Conditional-integration anti-windup
# -------------------------------------------------------------

controller = JointPIDController(
    kp=np.ones(6) * 100.0,
    ki=np.ones(6) * 50.0,
    kd=np.zeros(6),
    dt=DT,
    torque_limits=TORQUE_LIMITS,
)

controller.reset()

q = np.zeros(6)
qdot = np.zeros(6)

q_ref = np.deg2rad(
    [180, 180, 180, 180, 180, 180]
)

qdot_ref = np.zeros(6)

for _ in range(1000):
    tau = controller.compute(
        q,
        qdot,
        q_ref,
        qdot_ref,
    )

print("\n" + "=" * 70)
print("TEST 4 — CONDITIONAL ANTI-WINDUP")
print("=" * 70)

print("tau:")
print(tau)

print("\nintegral error:")
print(controller.integral_error)