#include <cmath>
#include <iomanip>
#include <iostream>

#include <Eigen/SVD>

#include "ur5e_control_analysis/kinematics.hpp"
#include "ur5e_control_analysis/types.hpp"


int main()
{
    using namespace ur5e;

    constexpr double pi =
        3.14159265358979323846;

    Vector6d q_deg;

    q_deg <<
        10.0,
        -40.0,
        60.0,
        -30.0,
        45.0,
        20.0;

    const Vector6d q =
        q_deg * pi / 180.0;


    const Matrix4d T =
        forwardKinematics(q);

    const Jacobian6d J =
        geometricJacobian(q);


    Eigen::JacobiSVD<Jacobian6d> svd(
        J,
        Eigen::ComputeFullU |
        Eigen::ComputeFullV
    );

    const auto singular_values =
        svd.singularValues();


    std::cout
        << std::fixed
        << std::setprecision(9);


    std::cout
        << "============================================================\n";

    std::cout
        << "UR5e C++ KINEMATICS TEST\n";

    std::cout
        << "============================================================\n";


    std::cout
        << "\nJoint configuration [deg]:\n"
        << q_deg.transpose()
        << "\n";


    std::cout
        << "\nEnd-effector transform:\n"
        << T
        << "\n";


    std::cout
        << "\nTCP position [m]:\n"
        << T.block<3, 1>(0, 3).transpose()
        << "\n";


    std::cout
        << "\nGeometric Jacobian:\n"
        << J
        << "\n";


    std::cout
        << "\nSingular values:\n"
        << singular_values.transpose()
        << "\n";


    const double sigma_max =
        singular_values(0);

    const double sigma_min =
        singular_values(5);

    std::cout
        << "\nSigma max: "
        << sigma_max
        << "\n";

    std::cout
        << "Sigma min: "
        << sigma_min
        << "\n";

    std::cout
        << "Condition number: "
        << sigma_max / sigma_min
        << "\n";


    std::cout
        << "\nExpected Python reference TCP:\n"
        << "-0.733548060 "
        << "-0.336214980 "
        << " 0.215588770\n";


    std::cout
        << "\nExpected Python singular values:\n"
        << "2.030319150 "
        << "1.559960280 "
        << "0.758434110 "
        << "0.467459120 "
        << "0.412384720 "
        << "0.156820140\n";


    std::cout
        << "\nExpected Python condition number:\n"
        << "12.946801\n";


    return 0;
}