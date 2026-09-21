#pragma once

#include <array>

#include "ur5e_control_analysis/types.hpp"

namespace ur5e
{

// Standard-DH transform for one joint.
Matrix4d dhTransform(
    double theta,
    double d,
    double a,
    double alpha
);

// Forward kinematics from the analytical model base frame
// to the end-effector frame.
Matrix4d forwardKinematics(
    const Vector6d& q
);

// T_0_i for i = 0 ... 6.
// transforms[0] = Identity
// transforms[6] = end-effector transform
std::array<Matrix4d, 7> forwardTransforms(
    const Vector6d& q
);

// 6x6 geometric Jacobian:
//
// J = [ Jv ]
//     [ Jw ]
//
// expressed in the analytical model base frame.
Jacobian6d geometricJacobian(
    const Vector6d& q
);

}  // namespace ur5e