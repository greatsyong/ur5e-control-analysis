#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

#include <Eigen/SVD>

#include "ur5e_control_analysis/dynamics.hpp"
#include "ur5e_control_analysis/kinematics.hpp"
#include "ur5e_control_analysis/trajectory.hpp"
#include "ur5e_control_analysis/types.hpp"


namespace
{

bool checkScalar(
    const std::string& name,
    const double actual,
    const double reference,
    const double tolerance)
{
    const double error =
        std::abs(actual - reference);

    const bool pass =
        error <= tolerance;

    std::cout
        << std::left
        << std::setw(32)
        << name
        << " error = "
        << std::scientific
        << error
        << "   tol = "
        << tolerance
        << "   "
        << (pass ? "PASS" : "FAIL")
        << "\n";

    return pass;
}


template <typename DerivedA, typename DerivedB>
bool checkMatrix(
    const std::string& name,
    const Eigen::MatrixBase<DerivedA>& actual,
    const Eigen::MatrixBase<DerivedB>& reference,
    const double tolerance)
{
    const double max_error =
        (actual - reference)
            .cwiseAbs()
            .maxCoeff();

    const bool pass =
        max_error <= tolerance;

    std::cout
        << std::left
        << std::setw(32)
        << name
        << " max error = "
        << std::scientific
        << max_error
        << "   tol = "
        << tolerance
        << "   "
        << (pass ? "PASS" : "FAIL")
        << "\n";

    return pass;
}

}  // namespace


int main()
{
    using namespace ur5e;

    constexpr double pi =
        3.14159265358979323846;


    std::cout
        << "============================================================\n"
        << "UR5e PYTHON <-> C++ CROSS-VALIDATION\n"
        << "============================================================\n\n";


    bool all_pass = true;


    // ========================================================
    // 1. Validated configuration
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
    // 2. FK
    // ========================================================

    const Matrix4d T =
        forwardKinematics(q);

    const Vector3d tcp =
        T.block<3, 1>(0, 3);


    Vector3d tcp_reference;

    tcp_reference <<
        -0.733548060,
        -0.336214980,
         0.215588770;


    std::cout
        << "[KINEMATICS]\n";

    all_pass &=
        checkMatrix(
            "TCP position",
            tcp,
            tcp_reference,
            1.0e-8
        );


    // ========================================================
    // 3. Jacobian singular values
    // ========================================================

    const Jacobian6d J =
        geometricJacobian(q);


    Eigen::JacobiSVD<Jacobian6d> svd(
        J,
        Eigen::ComputeFullU
        | Eigen::ComputeFullV
    );


    const Vector6d singular_values =
        svd.singularValues();


    Vector6d singular_reference;

    singular_reference <<
        2.030319150,
        1.559960280,
        0.758434110,
        0.467459120,
        0.412384720,
        0.156820140;


    all_pass &=
        checkMatrix(
            "Jacobian singular values",
            singular_values,
            singular_reference,
            1.0e-8
        );


    const double condition_number =
        singular_values(0)
        / singular_values(5);


    all_pass &=
        checkScalar(
            "Jacobian condition number",
            condition_number,
            12.946801,
            1.0e-6
        );


    // ========================================================
    // 4. Mass matrix
    // ========================================================

    std::cout
        << "\n[DYNAMICS]\n";


    const Matrix6d M =
        massMatrix(q);


    Matrix6d M_reference;

    M_reference <<
         3.103412796, -0.173015033,  0.093255847,  0.027184356, -0.039032962,  0.000030698,
        -0.173015033,  3.342034262,  1.208671613,  0.046755469, -0.003372014,  0.000182232,
         0.093255847,  1.208671613,  0.833905721,  0.071410569, -0.011962835,  0.000182232,
         0.027184356,  0.046755469,  0.071410569,  0.021195518, -0.004035023,  0.000182232,
        -0.039032962, -0.003372014, -0.011962835, -0.004035023,  0.005039183,  0.000000923,
         0.000030698,  0.000182232,  0.000182232,  0.000182232,  0.000000923,  0.000257560;


    all_pass &=
        checkMatrix(
            "Mass matrix M(q)",
            M,
            M_reference,
            1.0e-8
        );


    // ========================================================
    // 5. Gravity
    // ========================================================

    const Vector6d g =
        gravityVector(q);


    Vector6d g_reference;

    g_reference <<
         0.000000000,
        -49.685494200,
        -18.034631000,
        -0.707495329,
         0.068867601,
         0.000000000;


    all_pass &=
        checkMatrix(
            "Gravity vector g(q)",
            g,
            g_reference,
            1.0e-6
        );


    // ========================================================
    // 6. Coriolis / centrifugal
    // ========================================================

    const Vector6d c =
        coriolisCentrifugalVector(
            q,
            qdot
        );


    Vector6d c_reference;

    c_reference <<
        -0.441316048,
         0.087695139,
         0.278040523,
         0.037945317,
        -0.003085960,
         0.000033387;


    all_pass &=
        checkMatrix(
            "Coriolis vector c(q,qdot)",
            c,
            c_reference,
            1.0e-6
        );


    // ========================================================
    // 7. Benchmark trajectory
    // ========================================================

    std::cout
        << "\n[TRAJECTORY]\n";


    Vector6d q_start_deg;

    q_start_deg <<
         0.0,
        -90.0,
         90.0,
        -90.0,
        -90.0,
         0.0;


    Vector6d q_goal_deg;

    q_goal_deg <<
         20.0,
        -60.0,
         60.0,
        -70.0,
        -70.0,
         20.0;


    const Vector6d q_start =
        q_start_deg * pi / 180.0;

    const Vector6d q_goal =
        q_goal_deg * pi / 180.0;


    const TrajectoryState midpoint =
        quinticTrajectory(
            2.0,
            q_start,
            q_goal,
            4.0
        );


    // For quintic scaling at s = 0.5:
    // h(0.5) = 0.5
    //
    // Therefore q(2 s) must be exactly halfway
    // between start and goal.

    const Vector6d midpoint_reference =
        0.5 * (q_start + q_goal);


    all_pass &=
        checkMatrix(
            "Trajectory midpoint q",
            midpoint.q,
            midpoint_reference,
            1.0e-12
        );


    // At s = 0.5:
    // dh/ds = 1.875
    //
    // qdot = (1.875 / 4) * delta_q

    const Vector6d midpoint_velocity_reference =
        (1.875 / 4.0)
        * (q_goal - q_start);


    all_pass &=
        checkMatrix(
            "Trajectory midpoint qdot",
            midpoint.qdot,
            midpoint_velocity_reference,
            1.0e-12
        );


    // ========================================================
    // 8. Endpoint conditions
    // ========================================================

    const TrajectoryState start_state =
        quinticTrajectory(
            0.0,
            q_start,
            q_goal,
            4.0
        );


    const TrajectoryState end_state =
        quinticTrajectory(
            4.0,
            q_start,
            q_goal,
            4.0
        );


    all_pass &=
        checkMatrix(
            "Trajectory start q",
            start_state.q,
            q_start,
            1.0e-12
        );


    all_pass &=
        checkMatrix(
            "Trajectory goal q",
            end_state.q,
            q_goal,
            1.0e-12
        );


    all_pass &=
        checkScalar(
            "Start velocity norm",
            start_state.qdot.norm(),
            0.0,
            1.0e-12
        );


    all_pass &=
        checkScalar(
            "Goal velocity norm",
            end_state.qdot.norm(),
            0.0,
            1.0e-12
        );


    all_pass &=
        checkScalar(
            "Start acceleration norm",
            start_state.qddot.norm(),
            0.0,
            1.0e-12
        );


    all_pass &=
        checkScalar(
            "Goal acceleration norm",
            end_state.qddot.norm(),
            0.0,
            1.0e-12
        );


    // ========================================================
    // Final result
    // ========================================================

    std::cout
        << "\n============================================================\n";


    if (all_pass)
    {
        std::cout
            << "OVERALL CROSS-VALIDATION: PASS\n";
    }
    else
    {
        std::cout
            << "OVERALL CROSS-VALIDATION: FAIL\n";
    }


    std::cout
        << "============================================================\n";


    return all_pass ? 0 : 1;
}