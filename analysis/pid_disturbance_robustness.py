import os
import time
import numpy as np
import matplotlib.pyplot as plt

from controllers.pid import JointPIDController
from dynamics.rigid_body_dynamics import mass_matrix
from dynamics.state_space import state_derivative
from dynamics.disturbance_dynamics import state_derivative_disturbed
from trajectories.joint_trajectory import quintic_joint_trajectory
from analysis.controller_metrics import summarize_tcp_errors


# ============================================================
# Final tuned PID — nominal nonlinear simulation benchmark
# ============================================================

DT = 0.002
DURATION = 4.0

WN = 20.0
ZETA = 3.0
ALPHA_I = 18.0

TORQUE_LIMITS = np.array([
    150.0, 150.0, 150.0,
    28.0, 28.0, 28.0,
])

q_start = np.deg2rad([
    0.0, -90.0, 90.0, -90.0, -90.0, 0.0,
])

q_goal = np.deg2rad([
    20.0, -60.0, 60.0, -70.0, -70.0, 20.0,
])


# ============================================================
# Gain design
# ============================================================

M0 = mass_matrix(q_start)
effective_inertia = np.diag(M0)

kp = effective_inertia * WN**2
kd = 2.0 * ZETA * effective_inertia * WN
ki = ALPHA_I * kp

controller = JointPIDController(
    kp=kp,
    ki=ki,
    kd=kd,
    dt=DT,
    torque_limits=TORQUE_LIMITS,
)

controller.reset()


# ============================================================
# Simulation
# ============================================================

times = np.arange(
    0.0,
    DURATION + DT,
    DT,
)

x = np.concatenate([
    q_start,
    np.zeros(6),
])

q_history = []
qdot_history = []
q_ref_history = []
qdot_ref_history = []
tau_history = []
controller_time_history = []

for t in times:

    q = x[:6]
    qdot = x[6:]

    q_ref, qdot_ref, _ = quintic_joint_trajectory(
        t,
        DURATION,
        q_start,
        q_goal,
    )

    tic = time.perf_counter()

    tau = controller.compute(
        q,
        qdot,
        q_ref,
        qdot_ref,
    )

    toc = time.perf_counter()

    controller_time_history.append(
        toc - tic
    )

    q_history.append(q.copy())
    qdot_history.append(qdot.copy())
    q_ref_history.append(q_ref.copy())
    qdot_ref_history.append(qdot_ref.copy())
    tau_history.append(tau.copy())

    if t >= DURATION:
        continue

    # RK4 integration with zero-order-held torque
    k1 = state_derivative_disturbed(
        x,
        tau,
        t,
    )

    k2 = state_derivative_disturbed(
        x + 0.5 * DT * k1,
        tau,
        t + 0.5 * DT,
    )

    k3 = state_derivative_disturbed(
        x + 0.5 * DT * k2,
        tau,
        t + 0.5 * DT,
    )

    k4 = state_derivative_disturbed(
        x + DT * k3,
        tau,
        t + DT,
    )

    x = x + (
        DT / 6.0
    ) * (
        k1
        + 2.0 * k2
        + 2.0 * k3
        + k4
    )


# ============================================================
# Convert histories
# ============================================================

q_history = np.asarray(q_history)
qdot_history = np.asarray(qdot_history)

q_ref_history = np.asarray(q_ref_history)
qdot_ref_history = np.asarray(qdot_ref_history)

tau_history = np.asarray(tau_history)

controller_time_history = np.asarray(
    controller_time_history
)


# ============================================================
# Joint-space metrics
# ============================================================

error = q_ref_history - q_history

joint_rmse = np.sqrt(
    np.mean(error**2, axis=0)
)

joint_max_error = np.max(
    np.abs(error),
    axis=0,
)

joint_final_error = error[-1]

overall_joint_rmse = np.sqrt(
    np.mean(error**2)
)


# ============================================================
# TCP metrics
# ============================================================

tcp_metrics = summarize_tcp_errors(
    q_ref_history,
    q_history,
)


# ============================================================
# Control-effort metrics
# ============================================================

max_torque = np.max(
    np.abs(tau_history),
    axis=0,
)

rms_torque = np.sqrt(
    np.mean(tau_history**2, axis=0)
)

torque_utilization = (
    100.0
    * max_torque
    / TORQUE_LIMITS
)


# ============================================================
# Computation metrics
# ============================================================

mean_controller_time_ms = (
    np.mean(controller_time_history) * 1000.0
)

max_controller_time_ms = (
    np.max(controller_time_history) * 1000.0
)


# ============================================================
# Print results
# ============================================================

print("\n" + "=" * 86)
print("FINAL TUNED PID BENCHMARK — NONLINEAR UR5e")
print("=" * 86)

print(f"\nwn      = {WN:.2f} rad/s")
print(f"zeta    = {ZETA:.2f}")
print(f"alpha_i = {ALPHA_I:.2f}")

print("\nKp:")
print(kp)

print("\nKd:")
print(kd)

print("\nKi:")
print(ki)


print("\n" + "=" * 86)
print("JOINT TRACKING")
print("=" * 86)

print("\nRMSE [deg]:")
print(np.rad2deg(joint_rmse))

print("\nMaximum error [deg]:")
print(np.rad2deg(joint_max_error))

print("\nFinal error [deg]:")
print(np.rad2deg(joint_final_error))

print(
    "\nOverall joint RMSE [deg]: "
    f"{np.rad2deg(overall_joint_rmse):.4f}"
)


print("\n" + "=" * 86)
print("TCP TRACKING")
print("=" * 86)

print(
    f"\nPosition RMSE  : "
    f"{tcp_metrics['tcp_position_rmse_mm']:.3f} mm"
)

print(
    f"Position max   : "
    f"{tcp_metrics['tcp_position_max_mm']:.3f} mm"
)

print(
    f"Position final : "
    f"{tcp_metrics['tcp_position_final_mm']:.3f} mm"
)

print(
    f"\nOrientation RMSE  : "
    f"{tcp_metrics['tcp_orientation_rmse_deg']:.3f} deg"
)

print(
    f"Orientation max   : "
    f"{tcp_metrics['tcp_orientation_max_deg']:.3f} deg"
)

print(
    f"Orientation final : "
    f"{tcp_metrics['tcp_orientation_final_deg']:.3f} deg"
)


print("\n" + "=" * 86)
print("CONTROL EFFORT")
print("=" * 86)

print("\nMaximum torque [Nm]:")
print(max_torque)

print("\nRMS torque [Nm]:")
print(rms_torque)

print("\nMaximum torque utilization [%]:")
print(torque_utilization)


print("\n" + "=" * 86)
print("CONTROLLER COMPUTATION")
print("=" * 86)

print(
    f"\nMean controller computation : "
    f"{mean_controller_time_ms:.6f} ms"
)

print(
    f"Maximum controller computation: "
    f"{max_controller_time_ms:.6f} ms"
)


# ============================================================
# Save numerical results
# ============================================================

results_dir = "results/pid"
figures_dir = os.path.join(
    results_dir,
    "figures",
)

os.makedirs(
    figures_dir,
    exist_ok=True,
)

np.savez(
    os.path.join(
        results_dir,
        "pid_final_benchmark.npz",
    ),
    time=times,
    q=q_history,
    qdot=qdot_history,
    q_ref=q_ref_history,
    qdot_ref=qdot_ref_history,
    tau=tau_history,
    joint_error=error,
    kp=kp,
    kd=kd,
    ki=ki,
    joint_rmse=joint_rmse,
    joint_max_error=joint_max_error,
    joint_final_error=joint_final_error,
    overall_joint_rmse=overall_joint_rmse,
    max_torque=max_torque,
    rms_torque=rms_torque,
    torque_utilization=torque_utilization,
    controller_time=controller_time_history,
)


# ============================================================
# Figure 1 — Joint trajectory tracking
# ============================================================

fig, axes = plt.subplots(
    3,
    2,
    figsize=(12, 10),
    sharex=True,
)

axes = axes.flatten()

for i in range(6):

    axes[i].plot(
        times,
        np.rad2deg(q_ref_history[:, i]),
        "--",
        label="Reference",
    )

    axes[i].plot(
        times,
        np.rad2deg(q_history[:, i]),
        label="PID",
    )

    axes[i].set_ylabel(
        f"J{i + 1} [deg]"
    )

    axes[i].grid(True)

axes[-2].set_xlabel("Time [s]")
axes[-1].set_xlabel("Time [s]")

axes[0].legend()

fig.suptitle(
    "Final Tuned PID — Joint Trajectory Tracking"
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        figures_dir,
        "pid_joint_tracking.png",
    ),
    dpi=300,
)

plt.close(fig)


# ============================================================
# Figure 2 — Joint tracking error
# ============================================================

fig, axes = plt.subplots(
    3,
    2,
    figsize=(12, 10),
    sharex=True,
)

axes = axes.flatten()

for i in range(6):

    axes[i].plot(
        times,
        np.rad2deg(error[:, i]),
    )

    axes[i].set_ylabel(
        f"J{i + 1} error [deg]"
    )

    axes[i].grid(True)

axes[-2].set_xlabel("Time [s]")
axes[-1].set_xlabel("Time [s]")

fig.suptitle(
    "Final Tuned PID — Joint Tracking Error"
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        figures_dir,
        "pid_joint_error.png",
    ),
    dpi=300,
)

plt.close(fig)


# ============================================================
# Figure 3 — Joint torque
# ============================================================

fig, axes = plt.subplots(
    3,
    2,
    figsize=(12, 10),
    sharex=True,
)

axes = axes.flatten()

for i in range(6):

    axes[i].plot(
        times,
        tau_history[:, i],
    )

    axes[i].axhline(
        TORQUE_LIMITS[i],
        linestyle="--",
    )

    axes[i].axhline(
        -TORQUE_LIMITS[i],
        linestyle="--",
    )

    axes[i].set_ylabel(
        f"J{i + 1} torque [Nm]"
    )

    axes[i].grid(True)

axes[-2].set_xlabel("Time [s]")
axes[-1].set_xlabel("Time [s]")

fig.suptitle(
    "Final Tuned PID — Control Torque"
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        figures_dir,
        "pid_joint_torque.png",
    ),
    dpi=300,
)

plt.close(fig)


print("\n" + "=" * 86)
print("FILES SAVED")
print("=" * 86)

print(
    "\nNumerical results:"
    "\n  results/pid/pid_final_benchmark.npz"
)

print(
    "\nFigures:"
    "\n  results/pid/figures/pid_joint_tracking.png"
    "\n  results/pid/figures/pid_joint_error.png"
    "\n  results/pid/figures/pid_joint_torque.png"
)