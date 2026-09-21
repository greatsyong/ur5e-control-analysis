#pragma once

#include <array>

#include "ur5e_control_analysis/types.hpp"

namespace ur5e
{

using TransformArray =
    std::array<Matrix4d, 6>;

using Vector3Array =
    std::array<Vector3d, 6>;

using ComJacobianArray =
    std::array<Eigen::Matrix<double, 3, 6>, 6>;


// Physical-link transforms using the same convention
// as the validated Python/Isaac model.
TransformArray physicalLinkTransforms(
    const Vector6d& q
);


// Base-frame CoM positions of all six moving links.
Vector3Array linkComPositions(
    const Vector6d& q
);


// Joint origins and rotation axes in the base frame.
void jointOriginsAxes(
    const Vector6d& q,
    Vector3Array& origins,
    Vector3Array& axes
);


// Linear and angular Jacobians of each link CoM.
void comJacobians(
    const Vector6d& q,
    ComJacobianArray& Jv,
    ComJacobianArray& Jw
);


// Joint-space mass matrix.
Matrix6d massMatrix(
    const Vector6d& q
);


// Gravity compensation vector.
Vector6d gravityVector(
    const Vector6d& q,
    double gravity = 9.81
);


// Coriolis + centrifugal generalized torque vector.
Vector6d coriolisCentrifugalVector(
    const Vector6d& q,
    const Vector6d& qdot,
    double epsilon = 1.0e-6
);

}  // namespace ur5e