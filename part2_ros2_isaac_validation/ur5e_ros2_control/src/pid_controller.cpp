#include "ur5e_ros2_control/pid_controller.hpp"
#include <algorithm>

namespace ur5e_ros2_control
{
PidController::PidController(
  const JointArray & kp, const JointArray & ki, const JointArray & kd,
  const JointArray & torque_limits, double dt)
: kp_(kp), ki_(ki), kd_(kd), torque_limits_(torque_limits), dt_(dt)
{
  reset();
}

void PidController::reset()
{
  integral_.fill(0.0);
}

PidController::JointArray PidController::compute(
  const JointArray & q, const JointArray & qdot,
  const JointArray & q_ref, const JointArray & qdot_ref)
{
  JointArray e{}, edot{}, tau_current{}, tau{};

  for (std::size_t i = 0; i < N; ++i) {
    e[i] = q_ref[i] - q[i];
    edot[i] = qdot_ref[i] - qdot[i];
    tau_current[i] = kp_[i] * e[i] + kd_[i] * edot[i] + ki_[i] * integral_[i];
  }

  // Same conditional-integration anti-windup as Part 1B.
  for (std::size_t i = 0; i < N; ++i) {
    bool integrate = true;
    if (tau_current[i] >= torque_limits_[i] && e[i] > 0.0) integrate = false;
    if (tau_current[i] <= -torque_limits_[i] && e[i] < 0.0) integrate = false;
    if (integrate) integral_[i] += e[i] * dt_;
  }

  for (std::size_t i = 0; i < N; ++i) {
    const double u = kp_[i] * e[i] + kd_[i] * edot[i] + ki_[i] * integral_[i];
    tau[i] = std::clamp(u, -torque_limits_[i], torque_limits_[i]);
  }
  return tau;
}
}  // namespace ur5e_ros2_control
