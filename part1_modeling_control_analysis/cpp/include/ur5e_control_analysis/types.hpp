#pragma once

#include <Eigen/Dense>

namespace ur5e
{

constexpr int kNumJoints = 6;

using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;

using Vector3d = Eigen::Vector3d;
using Matrix3d = Eigen::Matrix3d;

using Vector4d = Eigen::Vector4d;
using Matrix4d = Eigen::Matrix4d;

using Jacobian6d = Eigen::Matrix<double, 6, 6>;

}  // namespace ur5e