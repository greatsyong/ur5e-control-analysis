#include "ur5e_control_analysis/dynamics.hpp"

#include <cmath>

#include "ur5e_control_analysis/ur5e_parameters.hpp"


namespace ur5e
{

namespace
{

Matrix3d rotX(const double theta)
{
    const double c = std::cos(theta);
    const double s = std::sin(theta);

    Matrix3d R;

    R <<
        1.0, 0.0, 0.0,
        0.0,   c,  -s,
        0.0,   s,   c;

    return R;
}


Matrix3d rotZ(const double theta)
{
    const double c = std::cos(theta);
    const double s = std::sin(theta);

    Matrix3d R;

    R <<
          c,  -s, 0.0,
          s,   c, 0.0,
        0.0, 0.0, 1.0;

    return R;
}


Matrix4d makeTransform(
    const Matrix3d& R,
    const Vector3d& p)
{
    Matrix4d T = Matrix4d::Identity();

    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = p;

    return T;
}


Matrix4d rotationTransform(
    const Matrix3d& R)
{
    return makeTransform(
        R,
        Vector3d::Zero()
    );
}


Matrix4d translationTransform(
    const Vector3d& p)
{
    return makeTransform(
        Matrix3d::Identity(),
        p
    );
}

}  // namespace


TransformArray physicalLinkTransforms(
    const Vector6d& q)
{
    TransformArray transforms;

    Matrix4d T =
        rotationTransform(
            rotZ(parameters::kPi)
        );


    // --------------------------------------------------
    // Joint 1: shoulder_pan
    // --------------------------------------------------

    T =
        T
        * translationTransform(
            Vector3d(0.0, 0.0, 0.1625)
        )
        * rotationTransform(
            rotZ(q(0))
        );

    transforms[0] = T;


    // --------------------------------------------------
    // Joint 2: shoulder_lift
    // --------------------------------------------------

    T =
        T
        * rotationTransform(
            rotX(parameters::kPi / 2.0)
        )
        * rotationTransform(
            rotZ(q(1))
        );

    transforms[1] = T;


    // --------------------------------------------------
    // Joint 3: elbow
    // --------------------------------------------------

    T =
        T
        * translationTransform(
            Vector3d(-0.425, 0.0, 0.0)
        )
        * rotationTransform(
            rotZ(q(2))
        );

    transforms[2] = T;


    // --------------------------------------------------
    // Joint 4: wrist_1
    // --------------------------------------------------

    T =
        T
        * translationTransform(
            Vector3d(-0.3922, 0.0, 0.1333)
        )
        * rotationTransform(
            rotZ(q(3))
        );

    transforms[3] = T;


    // --------------------------------------------------
    // Joint 5: wrist_2
    // --------------------------------------------------

    T =
        T
        * translationTransform(
            Vector3d(0.0, -0.0997, 0.0)
        )
        * rotationTransform(
            rotX(parameters::kPi / 2.0)
        )
        * rotationTransform(
            rotZ(q(4))
        );

    transforms[4] = T;


    // --------------------------------------------------
    // Joint 6: wrist_3
    // --------------------------------------------------

    T =
        T
        * translationTransform(
            Vector3d(0.0, 0.0996, 0.0)
        )
        * rotationTransform(
            rotX(-parameters::kPi / 2.0)
        )
        * rotationTransform(
            rotZ(q(5))
        );

    transforms[5] = T;


    return transforms;
}


Vector3Array linkComPositions(
    const Vector6d& q)
{
    const TransformArray transforms =
        physicalLinkTransforms(q);

    Vector3Array positions;

    for (int i = 0; i < kNumJoints; ++i)
    {
        const Matrix3d R =
            transforms[i].block<3, 3>(0, 0);

        const Vector3d p =
            transforms[i].block<3, 1>(0, 3);

        positions[i] =
            p
            + R * parameters::com_positions[i];
    }

    return positions;
}


void jointOriginsAxes(
    const Vector6d& q,
    Vector3Array& origins,
    Vector3Array& axes)
{
    const Vector3d z_axis(
        0.0,
        0.0,
        1.0
    );

    Matrix4d T =
        rotationTransform(
            rotZ(parameters::kPi)
        );


    // --------------------------------------------------
    // Joint 1
    // --------------------------------------------------

    T =
        T
        * translationTransform(
            Vector3d(0.0, 0.0, 0.1625)
        );

    origins[0] =
        T.block<3, 1>(0, 3);

    axes[0] =
        T.block<3, 3>(0, 0)
        * z_axis;

    T =
        T
        * rotationTransform(
            rotZ(q(0))
        );


    // --------------------------------------------------
    // Joint 2
    // --------------------------------------------------

    T =
        T
        * rotationTransform(
            rotX(parameters::kPi / 2.0)
        );

    origins[1] =
        T.block<3, 1>(0, 3);

    axes[1] =
        T.block<3, 3>(0, 0)
        * z_axis;

    T =
        T
        * rotationTransform(
            rotZ(q(1))
        );


    // --------------------------------------------------
    // Joint 3
    // --------------------------------------------------

    T =
        T
        * translationTransform(
            Vector3d(-0.425, 0.0, 0.0)
        );

    origins[2] =
        T.block<3, 1>(0, 3);

    axes[2] =
        T.block<3, 3>(0, 0)
        * z_axis;

    T =
        T
        * rotationTransform(
            rotZ(q(2))
        );


    // --------------------------------------------------
    // Joint 4
    // --------------------------------------------------

    T =
        T
        * translationTransform(
            Vector3d(-0.3922, 0.0, 0.1333)
        );

    origins[3] =
        T.block<3, 1>(0, 3);

    axes[3] =
        T.block<3, 3>(0, 0)
        * z_axis;

    T =
        T
        * rotationTransform(
            rotZ(q(3))
        );


    // --------------------------------------------------
    // Joint 5
    // --------------------------------------------------

    T =
        T
        * translationTransform(
            Vector3d(0.0, -0.0997, 0.0)
        )
        * rotationTransform(
            rotX(parameters::kPi / 2.0)
        );

    origins[4] =
        T.block<3, 1>(0, 3);

    axes[4] =
        T.block<3, 3>(0, 0)
        * z_axis;

    T =
        T
        * rotationTransform(
            rotZ(q(4))
        );


    // --------------------------------------------------
    // Joint 6
    // --------------------------------------------------

    T =
        T
        * translationTransform(
            Vector3d(0.0, 0.0996, 0.0)
        )
        * rotationTransform(
            rotX(-parameters::kPi / 2.0)
        );

    origins[5] =
        T.block<3, 1>(0, 3);

    axes[5] =
        T.block<3, 3>(0, 0)
        * z_axis;
}


void comJacobians(
    const Vector6d& q,
    ComJacobianArray& Jv,
    ComJacobianArray& Jw)
{
    const Vector3Array com_positions =
        linkComPositions(q);

    Vector3Array joint_origins;
    Vector3Array joint_axes;

    jointOriginsAxes(
        q,
        joint_origins,
        joint_axes
    );


    for (int i = 0; i < kNumJoints; ++i)
    {
        Jv[i].setZero();
        Jw[i].setZero();

        const Vector3d& p_com =
            com_positions[i];

        // Link i is affected only by joints 0 ... i.
        for (int j = 0; j <= i; ++j)
        {
            const Vector3d& z =
                joint_axes[j];

            const Vector3d& p_joint =
                joint_origins[j];

            Jv[i].col(j) =
                z.cross(
                    p_com - p_joint
                );

            Jw[i].col(j) = z;
        }
    }
}


Matrix6d massMatrix(
    const Vector6d& q)
{
    const TransformArray transforms =
        physicalLinkTransforms(q);

    ComJacobianArray Jv;
    ComJacobianArray Jw;

    comJacobians(
        q,
        Jv,
        Jw
    );


    Matrix6d M =
        Matrix6d::Zero();


    for (int i = 0; i < kNumJoints; ++i)
    {
        const Matrix3d R =
            transforms[i].block<3, 3>(0, 0);

        const Matrix3d I_world =
            R
            * parameters::inertia_tensors[i]
            * R.transpose();

        M +=
            parameters::mass[i]
            * Jv[i].transpose()
            * Jv[i]
            +
            Jw[i].transpose()
            * I_world
            * Jw[i];
    }


    // Remove floating-point asymmetry.
    M =
        0.5
        * (M + M.transpose());

    return M;
}


Vector6d gravityVector(
    const Vector6d& q,
    const double gravity)
{
    ComJacobianArray Jv;
    ComJacobianArray Jw;

    comJacobians(
        q,
        Jv,
        Jw
    );


    const Vector3d gravity_compensation(
        0.0,
        0.0,
        gravity
    );


    Vector6d g =
        Vector6d::Zero();


    for (int i = 0; i < kNumJoints; ++i)
    {
        g +=
            Jv[i].transpose()
            * (
                parameters::mass[i]
                * gravity_compensation
            );
    }


    return g;
}


Vector6d coriolisCentrifugalVector(
    const Vector6d& q,
    const Vector6d& qdot,
    const double epsilon)
{
    std::array<Matrix6d, 6> dM_dq;


    // --------------------------------------------------
    // Numerical derivatives of M(q)
    //
    // dM_dq[k](i,j) = d M_ij / d q_k
    // --------------------------------------------------

    for (int k = 0; k < kNumJoints; ++k)
    {
        Vector6d dq =
            Vector6d::Zero();

        dq(k) = epsilon;


        const Matrix6d M_plus =
            massMatrix(q + dq);

        const Matrix6d M_minus =
            massMatrix(q - dq);


        dM_dq[k] =
            (M_plus - M_minus)
            / (2.0 * epsilon);
    }


    Vector6d c =
        Vector6d::Zero();


    // --------------------------------------------------
    // Christoffel symbols
    // --------------------------------------------------

    for (int i = 0; i < kNumJoints; ++i)
    {
        for (int j = 0; j < kNumJoints; ++j)
        {
            for (int k = 0; k < kNumJoints; ++k)
            {
                const double gamma_ijk =
                    0.5
                    * (
                        dM_dq[k](i, j)
                        + dM_dq[j](i, k)
                        - dM_dq[i](j, k)
                    );


                c(i) +=
                    gamma_ijk
                    * qdot(j)
                    * qdot(k);
            }
        }
    }


    return c;
}

}  // namespace ur5e