import numpy as np

from controllers.computed_torque import ComputedTorqueController
from dynamics.rigid_body_dynamics import gravity_vector


TORQUE_LIMITS = np.array([
    150.0, 150.0, 150.0,
    28.0, 28.0, 28.0,
])

q = np.deg2rad([
    0.0, -90.0, 90.0,
    -90.0, -90.0, 0.0,
])

qdot = np.zeros(6)

q_ref = q.copy()
qdot_ref = np.zeros(6)
qddot_ref = np.zeros(6)


# Nominal test gains
WN = 10.0
ZETA = 1.0

kp = np.full(6, WN**2)
kd = np.full(6, 2.0 * ZETA * WN)


controller = ComputedTorqueController(
    kp=kp,
    kd=kd,
    torque_limits=TORQUE_LIMITS,
)


# ------------------------------------------------------------
# Test 1: Zero tracking error
#
# q = q_ref
# qdot = qdot_ref
# qddot_ref = 0
#
# Therefore:
#
# tau = g(q)
# ------------------------------------------------------------

tau = controller.compute(
    q=q,
    qdot=qdot,
    q_ref=q_ref,
    qdot_ref=qdot_ref,
    qddot_ref=qddot_ref,
)

g = gravity_vector(q)

print("\nComputed torque:")
print(tau)

print("\nExpected gravity torque:")
print(g)

print("\nDifference:")
print(tau - g)

assert np.allclose(
    tau,
    g,
    atol=1e-10,
)

print("\nPASS: Zero-error computed torque equals gravity compensation.")


# ------------------------------------------------------------
# Test 2: Small position error
# ------------------------------------------------------------

q_ref_test = q.copy()
q_ref_test[1] += np.deg2rad(1.0)

tau_test = controller.compute(
    q=q,
    qdot=qdot,
    q_ref=q_ref_test,
    qdot_ref=qdot_ref,
    qddot_ref=qddot_ref,
)

print("\nTorque with +1 deg J2 reference error:")
print(tau_test)

print("\nPASS: Computed torque controller executed successfully.")