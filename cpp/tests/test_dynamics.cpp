#include <cmath>
#include <iomanip>
#include <iostream>

#include <Eigen/Eigenvalues>

#include "ur5e_control_analysis/dynamics.hpp"
#include "ur5e_control_analysis/types.hpp"


int main()
{
    using namespace ur5e;

    constexpr double pi =
        3.14159265358979323846;


    // ========================================================
    // Validated test configuration
    // ========================================================

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


    // Same velocity vector used in Python characterization.
    Vector6d qdot_deg;

    qdot_deg <<
        20.0,
        -30.0,
        40.0,
        -20.0,
        30.0,
        10.0;


    const Vector6d qdot =
        qdot_deg * pi / 180.0;


    // ========================================================
    // Dynamics
    // ========================================================

    const Matrix6d M =
        massMatrix(q);

    const Vector6d g =
        gravityVector(q);

    const Vector6d c =
        coriolisCentrifugalVector(
            q,
            qdot
        );


    Eigen::SelfAdjointEigenSolver<Matrix6d>
        eigen_solver(M);

    const Vector6d eigenvalues =
        eigen_solver.eigenvalues();


    // ========================================================
    // Output
    // ========================================================

    std::cout
        << std::fixed
        << std::setprecision(9);


    std::cout
        << "============================================================\n";

    std::cout
        << "UR5e C++ DYNAMICS TEST\n";

    std::cout
        << "============================================================\n";


    std::cout
        << "\nJoint configuration [deg]:\n"
        << q_deg.transpose()
        << "\n";


    std::cout
        << "\nJoint velocity [deg/s]:\n"
        << qdot_deg.transpose()
        << "\n";


    std::cout
        << "\nMass matrix M(q):\n"
        << M
        << "\n";


    std::cout
        << "\nMass matrix eigenvalues:\n"
        << eigenvalues.transpose()
        << "\n";


    std::cout
        << "\nGravity compensation g(q) [N m]:\n"
        << g.transpose()
        << "\n";


    std::cout
        << "\nCoriolis/centrifugal c(q,qdot) [N m]:\n"
        << c.transpose()
        << "\n";


    std::cout
        << "\n||g||:\n"
        << g.norm()
        << "\n";


    std::cout
        << "\n||c||:\n"
        << c.norm()
        << "\n";


    // ========================================================
    // Python reference
    // ========================================================

    std::cout
        << "\n------------------------------------------------------------\n";

    std::cout
        << "PYTHON REFERENCE\n";

    std::cout
        << "------------------------------------------------------------\n";


    std::cout
        << "\nExpected gravity vector [N m]:\n"
        << "0.000000000 "
        << "-49.685494200 "
        << "-18.034631000 "
        << "-0.707495329 "
        << "0.068867601 "
        << "0.000000000\n";


    std::cout
        << "\nExpected Coriolis/centrifugal vector [N m]:\n"
        << "-0.441316048 "
        << "0.087695139 "
        << "0.278040523 "
        << "0.037945317 "
        << "-0.003085960 "
        << "0.000033387\n";


    return 0;
}