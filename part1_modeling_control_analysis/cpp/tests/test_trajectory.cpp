#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "ur5e_control_analysis/dynamics.hpp"
#include "ur5e_control_analysis/trajectory.hpp"
#include "ur5e_control_analysis/types.hpp"
#include "ur5e_control_analysis/ur5e_parameters.hpp"


int main()
{
    using namespace ur5e;

    constexpr double pi =
        3.14159265358979323846;

    constexpr double rad_to_deg =
        180.0 / pi;


    // ========================================================
    // Benchmark definition
    // ========================================================

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


    const double duration =
        parameters::kBenchmarkDuration;

    const double dt =
        parameters::kSampleTime;


    // ========================================================
    // Storage for maximum values
    // ========================================================

    Vector6d max_velocity =
        Vector6d::Zero();

    Vector6d max_acceleration =
        Vector6d::Zero();

    Vector6d max_torque =
        Vector6d::Zero();

    Vector6d peak_torque_time =
        Vector6d::Zero();


    int sample_count = 0;


    // ========================================================
    // Evaluate benchmark trajectory
    // ========================================================

    for (
        double t = 0.0;
        t <= duration + 0.5 * dt;
        t += dt
    )
    {
        const TrajectoryState state =
            quinticTrajectory(
                t,
                q_start,
                q_goal,
                duration
            );


        const Vector6d velocity_abs =
            state.qdot.cwiseAbs();

        const Vector6d acceleration_abs =
            state.qddot.cwiseAbs();


        for (int i = 0; i < kNumJoints; ++i)
        {
            max_velocity(i) =
                std::max(
                    max_velocity(i),
                    velocity_abs(i)
                );

            max_acceleration(i) =
                std::max(
                    max_acceleration(i),
                    acceleration_abs(i)
                );
        }


        // ----------------------------------------------------
        // Inverse-dynamics feedforward torque
        //
        // tau = M(q) qddot + c(q,qdot) + g(q)
        // ----------------------------------------------------

        const Matrix6d M =
            massMatrix(state.q);

        const Vector6d c =
            coriolisCentrifugalVector(
                state.q,
                state.qdot
            );

        const Vector6d g =
            gravityVector(state.q);


        const Vector6d tau =
            M * state.qddot
            + c
            + g;


        for (int i = 0; i < kNumJoints; ++i)
        {
            if (
                std::abs(tau(i))
                > max_torque(i)
            )
            {
                max_torque(i) =
                    std::abs(tau(i));

                peak_torque_time(i) =
                    t;
            }
        }


        ++sample_count;
    }


    // ========================================================
    // Endpoint validation
    // ========================================================

    const TrajectoryState start_state =
        quinticTrajectory(
            0.0,
            q_start,
            q_goal,
            duration
        );

    const TrajectoryState end_state =
        quinticTrajectory(
            duration,
            q_start,
            q_goal,
            duration
        );


    // ========================================================
    // Constraint utilization
    // ========================================================

    const Vector6d velocity_utilization =
        max_velocity
            .cwiseQuotient(
                parameters::velocity_limits
            )
        * 100.0;


    const Vector6d torque_utilization =
        max_torque
            .cwiseQuotient(
                parameters::torque_limits
            )
        * 100.0;


    // ========================================================
    // Output
    // ========================================================

    std::cout
        << std::fixed
        << std::setprecision(9);


    std::cout
        << "============================================================\n"
        << "UR5e C++ BENCHMARK TRAJECTORY TEST\n"
        << "============================================================\n";


    std::cout
        << "\nDuration [s]: "
        << duration
        << "\n";

    std::cout
        << "Sample time [s]: "
        << dt
        << "\n";

    std::cout
        << "Number of samples: "
        << sample_count
        << "\n";


    std::cout
        << "\nStart configuration [deg]:\n"
        << q_start_deg.transpose()
        << "\n";


    std::cout
        << "\nGoal configuration [deg]:\n"
        << q_goal_deg.transpose()
        << "\n";


    std::cout
        << "\nStart qdot [deg/s]:\n"
        << (
            start_state.qdot
            * rad_to_deg
        ).transpose()
        << "\n";


    std::cout
        << "\nEnd qdot [deg/s]:\n"
        << (
            end_state.qdot
            * rad_to_deg
        ).transpose()
        << "\n";


    std::cout
        << "\nStart qddot [deg/s^2]:\n"
        << (
            start_state.qddot
            * rad_to_deg
        ).transpose()
        << "\n";


    std::cout
        << "\nEnd qddot [deg/s^2]:\n"
        << (
            end_state.qddot
            * rad_to_deg
        ).transpose()
        << "\n";


    std::cout
        << "\nMaximum reference velocity [deg/s]:\n"
        << (
            max_velocity
            * rad_to_deg
        ).transpose()
        << "\n";


    std::cout
        << "\nVelocity-limit utilization [%]:\n"
        << velocity_utilization.transpose()
        << "\n";


    std::cout
        << "\nMaximum reference acceleration [deg/s^2]:\n"
        << (
            max_acceleration
            * rad_to_deg
        ).transpose()
        << "\n";


    std::cout
        << "\nMaximum inverse-dynamics torque [N m]:\n"
        << max_torque.transpose()
        << "\n";


    std::cout
        << "\nTorque-limit utilization [%]:\n"
        << torque_utilization.transpose()
        << "\n";


    std::cout
        << "\nPeak-torque times [s]:\n"
        << peak_torque_time.transpose()
        << "\n";


    // ========================================================
    // PASS / FAIL
    // ========================================================

    const bool velocity_pass =
        (
            max_velocity.array()
            <= parameters::velocity_limits.array()
        ).all();


    const bool torque_pass =
        (
            max_torque.array()
            <= parameters::torque_limits.array()
        ).all();


    const bool endpoint_pass =
        start_state.qdot.norm() < 1.0e-12
        &&
        end_state.qdot.norm() < 1.0e-12
        &&
        start_state.qddot.norm() < 1.0e-12
        &&
        end_state.qddot.norm() < 1.0e-12;


    std::cout
        << "\n------------------------------------------------------------\n"
        << "VALIDATION\n"
        << "------------------------------------------------------------\n";


    std::cout
        << "Endpoint conditions: "
        << (
            endpoint_pass
            ? "PASS"
            : "FAIL"
        )
        << "\n";


    std::cout
        << "Velocity constraints: "
        << (
            velocity_pass
            ? "PASS"
            : "FAIL"
        )
        << "\n";


    std::cout
        << "Simulation torque constraints: "
        << (
            torque_pass
            ? "PASS"
            : "FAIL"
        )
        << "\n";


    return (
        endpoint_pass
        && velocity_pass
        && torque_pass
    )
        ? 0
        : 1;
}