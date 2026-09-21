#include "ur5e_control_analysis/trajectory.hpp"

#include <algorithm>
#include <stdexcept>


namespace ur5e
{

TrajectoryState quinticTrajectory(
    const double time,
    const Vector6d& q_start,
    const Vector6d& q_goal,
    const double duration)
{
    if (duration <= 0.0)
    {
        throw std::invalid_argument(
            "Trajectory duration must be positive."
        );
    }


    // Clamp time to the trajectory interval.
    const double t =
        std::clamp(
            time,
            0.0,
            duration
        );


    const double s =
        t / duration;


    const double s2 = s * s;
    const double s3 = s2 * s;
    const double s4 = s3 * s;
    const double s5 = s4 * s;


    // --------------------------------------------------------
    // Quintic position scaling
    // h(s) = 10s^3 - 15s^4 + 6s^5
    // --------------------------------------------------------

    const double h =
        10.0 * s3
        - 15.0 * s4
        + 6.0 * s5;


    // --------------------------------------------------------
    // First derivative with respect to normalized time s
    //
    // dh/ds = 30s^2 - 60s^3 + 30s^4
    //
    // dh/dt = (dh/ds) / duration
    // --------------------------------------------------------

    const double dh_ds =
        30.0 * s2
        - 60.0 * s3
        + 30.0 * s4;

    const double hdot =
        dh_ds / duration;


    // --------------------------------------------------------
    // Second derivative
    //
    // d2h/ds2 = 60s - 180s^2 + 120s^3
    //
    // d2h/dt2 = (d2h/ds2) / duration^2
    // --------------------------------------------------------

    const double d2h_ds2 =
        60.0 * s
        - 180.0 * s2
        + 120.0 * s3;

    const double hddot =
        d2h_ds2
        / (duration * duration);


    const Vector6d delta_q =
        q_goal - q_start;


    TrajectoryState state;

    state.q =
        q_start
        + h * delta_q;

    state.qdot =
        hdot * delta_q;

    state.qddot =
        hddot * delta_q;


    return state;
}

}  // namespace ur5e