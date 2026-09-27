import numpy as np

from controllers.pid import JointPIDController
from dynamics.rigid_body_dynamics import mass_matrix
from dynamics.state_space import state_derivative
from trajectories.joint_trajectory import quintic_joint_trajectory


# ---------------------------------------------------------------------
# Common benchmark
# ---------------------------------------------------------------------

DT = 0.002
DURATION = 4.0

TORQUE_LIMITS = np.array([
    150.0,
    150.0,
    150.0,
    28.0,
    28.0,
    28.0,
])

q_start = np.deg2rad([
    0.0,
    -90.0,
    90.0,
    -90.0,
    -90.0,
    0.0,
])

q_goal = np.deg2rad([
    20.0,
    -60.0,
    60.0,
    -70.0,
    -70.0,
    20.0,
])


# ---------------------------------------------------------------------
# Gain design
#
# Keep the previous PD gains unchanged:
#
#   Kp = Mii * wn^2
#   Kd = 2*zeta*Mii*wn
#
# Add an integral pole:
#
#   Ki = alpha * Kp
#
# with alpha = 1 rad/s as the initial integral-action scale.
# ---------------------------------------------------------------------

WN = 4.0
ZETA = 1.0
ALPHA_I = 1.0

M0 = mass_matrix(q_start)
effective_inertia = np.diag(M0)

kp = effective_inertia * WN**2

kd = (
    2.0
    * ZETA
    * effective_inertia
    * WN
)

ki = ALPHA_I * kp


print("=" * 70)
print("INITIAL PID GAIN DESIGN")
print("=" * 70)

print("\nKp:")
print(kp)

print("\nKi:")
print(ki)

print("\nKd:")
print(kd)


# ---------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------

controller = JointPIDController(
    kp=kp,
    ki=ki,
    kd=kd,
    dt=DT,
    torque_limits=TORQUE_LIMITS,
)


# ---------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------

x = np.concatenate([
    q_start,
    np.zeros(6),
])

controller.reset()


# ---------------------------------------------------------------------
# History
# ---------------------------------------------------------------------

times = np.arange(
    0.0,
    DURATION + DT,
    DT,
)

q_history = []
q_ref_history = []
tau_history = []
integral_history = []


# ---------------------------------------------------------------------
# Closed-loop nonlinear simulation
# ---------------------------------------------------------------------

for t in times:

    q = x[:6]
    qdot = x[6:]

    q_ref, qdot_ref, _ = (
        quintic_joint_trajectory(
            t,
            DURATION,
            q_start,
            q_goal,
        )
    )

    tau = controller.compute(
        q,
        qdot,
        q_ref,
        qdot_ref,
    )

    q_history.append(q.copy())
    q_ref_history.append(q_ref.copy())
    tau_history.append(tau.copy())
    integral_history.append(
        controller.integral_error.copy()
    )

    if t >= DURATION:
        continue

    # RK4 with zero-order-held controller torque

    k1 = state_derivative(x, tau)

    k2 = state_derivative(
        x + 0.5 * DT * k1,
        tau,
    )

    k3 = state_derivative(
        x + 0.5 * DT * k2,
        tau,
    )

    k4 = state_derivative(
        x + DT * k3,
        tau,
    )

    x = x + (
        DT / 6.0
    ) * (
        k1
        + 2.0 * k2
        + 2.0 * k3
        + k4
    )


q_history = np.asarray(q_history)
q_ref_history = np.asarray(q_ref_history)
tau_history = np.asarray(tau_history)
integral_history = np.asarray(
    integral_history
)


# ---------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------

error_history = (
    q_ref_history
    - q_history
)

rmse = np.sqrt(
    np.mean(
        error_history**2,
        axis=0,
    )
)

max_error = np.max(
    np.abs(error_history),
    axis=0,
)

final_error = error_history[-1]

max_torque = np.max(
    np.abs(tau_history),
    axis=0,
)

torque_utilization = (
    100.0
    * max_torque
    / TORQUE_LIMITS
)


# ---------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("PID BASELINE — NONLINEAR UR5e")
print("=" * 70)

print("\nRMSE [deg]:")
print(np.rad2deg(rmse))

print("\nMaximum absolute tracking error [deg]:")
print(np.rad2deg(max_error))

print("\nFinal tracking error [deg]:")
print(np.rad2deg(final_error))

print("\nMaximum absolute torque [Nm]:")
print(max_torque)

print("\nTorque utilization [%]:")
print(torque_utilization)

print("\nFinal integral state [rad*s]:")
print(integral_history[-1])

print("\nFinal joint position [deg]:")
print(
    np.rad2deg(
        q_history[-1]
    )
)