cmake_minimum_required(VERSION 3.16)

project(
    ur5e_control_analysis_cpp
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)


# ============================================================
# UR5e core library
# ============================================================

add_library(
    ur5e_core
    src/kinematics.cpp
    src/dynamics.cpp
    src/trajectory.cpp
)

target_include_directories(
    ur5e_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
    ur5e_core
    PUBLIC
        Eigen3::Eigen
)


# ============================================================
# Kinematics test
# ============================================================

add_executable(
    test_kinematics
    tests/test_kinematics.cpp
)

target_link_libraries(
    test_kinematics
    PRIVATE
        ur5e_core
)


# ============================================================
# Dynamics test
# ============================================================

add_executable(
    test_dynamics
    tests/test_dynamics.cpp
)

target_link_libraries(
    test_dynamics
    PRIVATE
        ur5e_core
)


# ============================================================
# Benchmark trajectory test
# ============================================================

add_executable(
    test_trajectory
    tests/test_trajectory.cpp
)

target_link_libraries(
    test_trajectory
    PRIVATE
        ur5e_core
)


# ============================================================
# Python <-> C++ cross-validation
# ============================================================

add_executable(
    cross_validation
    tests/cross_validation.cpp
)

target_link_libraries(
    cross_validation
    PRIVATE
        ur5e_core
)