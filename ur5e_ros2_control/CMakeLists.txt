cmake_minimum_required(VERSION 3.8)

project(ur5e_ros2_control)

if(CMAKE_COMPILER_IS_GNUCXX OR
   CMAKE_CXX_COMPILER_ID MATCHES "Clang")

  add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
  )

endif()


find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Eigen3 REQUIRED)


find_path(
  OSQP_INCLUDE_DIR
  NAMES osqp/osqp.h
  REQUIRED
)

find_library(
  OSQP_LIBRARY
  NAMES osqp
  REQUIRED
)


add_library(
  pid_controller
  src/pid_controller.cpp
)

target_include_directories(
  pid_controller
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)


add_library(
  lqr_controller
  src/lqr_controller.cpp
)

target_include_directories(
  lqr_controller
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)


add_library(
  mpc_controller
  src/mpc_controller.cpp
)

target_include_directories(
  mpc_controller
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OSQP_INCLUDE_DIR}
)

target_link_libraries(
  mpc_controller
  Eigen3::Eigen
  ${OSQP_LIBRARY}
)


add_executable(
  controller_node
  src/controller_node.cpp
)

target_link_libraries(
  controller_node
  pid_controller
  lqr_controller
  mpc_controller
)

ament_target_dependencies(
  controller_node
  ament_index_cpp
  rclcpp
  sensor_msgs
  std_msgs
)


add_executable(
  validation_manager
  src/validation_manager.cpp
)

ament_target_dependencies(
  validation_manager
  rclcpp
  sensor_msgs
  std_msgs
)


add_executable(
  logger_node
  src/logger_node.cpp
)

ament_target_dependencies(
  logger_node
  rclcpp
  sensor_msgs
  std_msgs
)


install(
  TARGETS
    pid_controller
    lqr_controller
    mpc_controller
    controller_node
    validation_manager
    logger_node

  EXPORT export_${PROJECT_NAME}

  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)


install(
  DIRECTORY include/
  DESTINATION include
)


install(
  DIRECTORY
    launch
    config
    scripts

  DESTINATION share/${PROJECT_NAME}
)


ament_export_include_directories(
  include
)


ament_export_libraries(
  pid_controller
  lqr_controller
  mpc_controller
)


ament_package()