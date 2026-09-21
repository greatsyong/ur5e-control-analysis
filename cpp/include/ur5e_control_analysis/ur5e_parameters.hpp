#pragma once

#include <array>
#include <cmath>

#include "ur5e_control_analysis/types.hpp"

namespace ur5e
{
namespace parameters
{

constexpr double kPi = 3.14159265358979323846;


// ============================================================
// Standard DH parameters
// ============================================================

inline constexpr std::array<double, 6> a = {
    0.0,
    -0.4250,
    -0.3922,
    0.0,
    0.0,
    0.0
};

inline constexpr std::array<double, 6> d = {
    0.1625,
    0.0,
    0.0,
    0.1333,
    0.0997,
    0.0996
};

inline constexpr std::array<double, 6> alpha = {
    kPi / 2.0,
    0.0,
    0.0,
    kPi / 2.0,
    -kPi / 2.0,
    0.0
};


// ============================================================
// Dynamic parameters
// ============================================================

inline constexpr std::array<double, 6> mass = {
    3.761,
    8.058,
    2.846,
    1.370,
    1.300,
    0.365
};


inline const std::array<Vector3d, 6> com_positions = {

    Vector3d(
         0.0,
        -0.00193,
        -0.02561
    ),

    Vector3d(
        -0.2125,
         0.0,
         0.11336
    ),

    Vector3d(
        -0.2422,
         0.0,
         0.02650
    ),

    Vector3d(
         0.0,
        -0.01634,
        -0.00180
    ),

    Vector3d(
         0.0,
         0.01634,
        -0.00180
    ),

    Vector3d(
         0.0,
         0.0,
        -0.001159
    )
};


inline const std::array<Matrix3d, 6> inertia_tensors = {

    (Matrix3d() <<
         0.00700210,  0.00000073, -0.00001053,
         0.00000073,  0.00648091,  0.00049994,
        -0.00001053,  0.00049994,  0.00657286
    ).finished(),

    (Matrix3d() <<
         0.01505885, -0.00005400,  0.00000563,
        -0.00005400,  0.33388086, -0.00000181,
         0.00000563, -0.00000181,  0.33247207
    ).finished(),

    (Matrix3d() <<
         0.00399632, -0.00001365,  0.00137272,
        -0.00001365,  0.07879254, -0.00000660,
         0.00137272, -0.00000660,  0.07848510
    ).finished(),

    (Matrix3d() <<
         0.00165491, -0.00000282, -0.00000438,
        -0.00000282,  0.00135962,  0.00010157,
        -0.00000438,  0.00010157,  0.00126279
    ).finished(),

    (Matrix3d() <<
         0.00135617, -0.00000274,  0.00000444,
        -0.00000274,  0.00127827, -0.00005048,
         0.00000444, -0.00005048,  0.00096614
    ).finished(),

    (Matrix3d() <<
         0.00018694,  0.00000006, -0.00000017,
         0.00000006,  0.00018908, -0.00000092,
        -0.00000017, -0.00000092,  0.00025756
    ).finished()
};


// ============================================================
// Simulation constraints
// ============================================================

inline const Vector6d torque_limits =
    (Vector6d() <<
        150.0,
        150.0,
        150.0,
        28.0,
        28.0,
        28.0
    ).finished();


inline const Vector6d velocity_limits =
    (Vector6d() <<
        kPi,
        kPi,
        kPi,
        kPi,
        kPi,
        kPi
    ).finished();


// ============================================================
// Benchmark timing
// ============================================================

constexpr double kSampleTime = 0.002;
constexpr double kBenchmarkDuration = 4.0;

}  // namespace parameters
}  // namespace ur5e