#pragma once
#include <array>
#include <cstddef>

namespace ur5e_ros2_control
{
class PidController
{
public:
  static constexpr std::size_t N = 6;
  using JointArray = std::array<double, N>;

  PidController(
    const JointArray & kp,
    const JointArray & ki,
    const JointArray & kd,
    const JointArray & torque_limits,
    double dt);

  void reset();
  JointArray compute(
    const JointArray & q,
    const JointArray & qdot,
    const JointArray & q_ref,
    const JointArray & qdot_ref);

private:
  JointArray kp_, ki_, kd_, torque_limits_, integral_{};
  double dt_;
};
}  // namespace ur5e_ros2_control
