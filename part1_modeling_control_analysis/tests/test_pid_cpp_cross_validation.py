import numpy as np

from controllers.pid import JointPIDController


DT = 0.002
WN = 20.0
ZETA = 3.0
ALPHA_I = 18.0

effective_inertia = np.array([
    1.05863580,
    2.59135828,
    0.881358542,
    0.0231437242,
    0.00503939441,
    0.00025756,
])

torque_limits = np.array([
    150.0,
    150.0,
    150.0,
    28.0,
    28.0,
    28.0,
])

kp = effective_inertia * WN**2
kd = 2.0 * ZETA * effective_inertia * WN
ki = ALPHA_I * kp


def print_vector(name, v):
    print(name)
    print(
        "["
        + ", ".join(f"{x:.10e}" for x in np.asarray(v))
        + "]\n"
    )


def make_controller():
    return JointPIDController(
        kp=kp,
        ki=ki,
        kd=kd,
        dt=DT,
        torque_limits=torque_limits,
    )


print("=" * 72)
print("UR5e PID PYTHON / C++ CROSS-VALIDATION")
print("=" * 72)

print(f"\nwn      = {WN:g} rad/s")
print(f"zeta    = {ZETA:g}")
print(f"alpha_i = {ALPHA_I:g} 1/s")
print(f"dt      = {DT:g} s\n")

print_vector("Kp:", kp)
print_vector("Kd:", kd)
print_vector("Ki:", ki)


# ============================================================
# Test 1
# Zero tracking error
# ============================================================

controller = make_controller()

q = np.array([
    0.0,
    -np.pi / 2.0,
    np.pi / 2.0,
    -np.pi / 2.0,
    -np.pi / 2.0,
    0.0,
])

qdot = np.zeros(6)
q_ref = q.copy()
qdot_ref = np.zeros(6)

tau_zero = controller.compute(
    q,
    qdot,
    q_ref,
    qdot_ref,
)

print_vector(
    "Test 1 - zero-error torque:",
    tau_zero,
)


# ============================================================
# Test 2
# +1 degree J2 position error
# ============================================================

controller = make_controller()

q_ref = q.copy()
q_ref[1] += np.pi / 180.0

tau_j2 = controller.compute(
    q,
    qdot,
    q_ref,
    qdot_ref,
)

print_vector(
    "Test 2 - +1 deg J2 error torque:",
    tau_j2,
)


# ============================================================
# Test 3
# Mixed position + velocity error
# ============================================================

controller = make_controller()

q_ref = q.copy()

q_ref[0] += 0.01
q_ref[1] -= 0.02
q_ref[2] += 0.015
q_ref[3] -= 0.03
q_ref[4] += 0.025
q_ref[5] -= 0.01

qdot = np.array([
    0.10,
    -0.20,
    0.15,
    -0.10,
    0.05,
    -0.08,
])

qdot_ref = np.array([
    0.15,
    -0.10,
    0.05,
    -0.05,
    0.10,
    -0.02,
])

tau_mixed = controller.compute(
    q,
    qdot,
    q_ref,
    qdot_ref,
)

print_vector(
    "Test 3 - mixed-error torque:",
    tau_mixed,
)


# ============================================================
# Test 4
# Anti-windup under repeated large error
# ============================================================

controller = make_controller()

q = np.zeros(6)
qdot = np.zeros(6)

q_ref = np.full(6, 10.0)
qdot_ref = np.zeros(6)

tau_sat = None

for _ in range(1000):
    tau_sat = controller.compute(
        q,
        qdot,
        q_ref,
        qdot_ref,
    )

print_vector(
    "Test 4 - saturated torque:",
    tau_sat,
)

# The controller stores its integral state internally.
# Current implementation uses the attribute below.
if hasattr(controller, "integral"):
    integral_state = controller.integral
elif hasattr(controller, "integral_error"):
    integral_state = controller.integral_error
elif hasattr(controller, "_integral"):
    integral_state = controller._integral
else:
    integral_state = None

if integral_state is not None:
    print_vector(
        "Test 4 - integral state after 1000 saturated steps:",
        integral_state,
    )
else:
    print(
        "Integral state attribute not found automatically.\n"
        "Torque output can still be compared with C++.\n"
    )


print("Python cross-validation run completed.")