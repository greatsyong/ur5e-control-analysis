#include "ur5e_control_analysis/kinematics.hpp"

#include <cmath>

#include "ur5e_control_analysis/ur5e_parameters.hpp"

namespace ur5e
{

Matrix4d dhTransform(
    const double theta,
    const double d,
    const double a,
    const double alpha)
{
    const double ct = std::cos(theta);
    const double st = std::sin(theta);
    const double ca = std::cos(alpha);
    const double sa = std::sin(alpha);

    Matrix4d T;

    T <<
        ct, -st * ca,  st * sa, a * ct,
        st,  ct * ca, -ct * sa, a * st,
        0.0,      sa,       ca,      d,
        0.0,     0.0,      0.0,    1.0;

    return T;
}


std::array<Matrix4d, 7> forwardTransforms(
    const Vector6d& q)
{
    std::array<Matrix4d, 7> transforms;

    transforms[0] = Matrix4d::Identity();

    Matrix4d T = Matrix4d::Identity();

    for (int i = 0; i < kNumJoints; ++i)
    {
        T = T * dhTransform(
            q(i),
            parameters::d[i],
            parameters::a[i],
            parameters::alpha[i]
        );

        transforms[i + 1] = T;
    }

    return transforms;
}


Matrix4d forwardKinematics(
    const Vector6d& q)
{
    return forwardTransforms(q)[6];
}


Jacobian6d geometricJacobian(
    const Vector6d& q)
{
    const auto transforms = forwardTransforms(q);

    Jacobian6d J = Jacobian6d::Zero();

    const Vector3d p_e =
        transforms[6].block<3, 1>(0, 3);

    for (int i = 0; i < kNumJoints; ++i)
    {
        const Vector3d z_i =
            transforms[i].block<3, 1>(0, 2);

        const Vector3d p_i =
            transforms[i].block<3, 1>(0, 3);

        const Vector3d Jv =
            z_i.cross(p_e - p_i);

        J.block<3, 1>(0, i) = Jv;
        J.block<3, 1>(3, i) = z_i;
    }

    return J;
}

}  // namespace ur5e