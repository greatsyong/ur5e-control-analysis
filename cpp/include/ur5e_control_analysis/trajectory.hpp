#pragma once

#include "ur5e_control_analysis/types.hpp"

namespace ur5e
{

struct TrajectoryState
{
    Vector6d q;
    Vector6d qdot;
    Vector6d qddot;
};


// Quintic time-scaling trajectory:
//
// h(s)   = 10 s^3 - 15 s^4 + 6 s^5
//
// q(t)   = q0 + h(s) (qf - q0)
//
// with:
// s = t / duration
//
// Endpoint velocity and acceleration are zero.
TrajectoryState quinticTrajectory(
    double time,
    const Vector6d& q_start,
    const Vector6d& q_goal,
    double duration
);

}  // namespace ur5e