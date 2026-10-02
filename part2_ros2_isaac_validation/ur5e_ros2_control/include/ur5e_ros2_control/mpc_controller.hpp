#ifndef UR5E_ROS2_CONTROL__MPC_CONTROLLER_HPP_
#define UR5E_ROS2_CONTROL__MPC_CONTROLLER_HPP_

#include <Eigen/Dense>
#include <osqp/osqp.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ur5e_ros2_control
{

class MpcController
{
public:
  static constexpr std::size_t N = 6;
  static constexpr std::size_t NX = 12;

  using JointArray = std::array<double, N>;

  MpcController(
    const std::string & data_file,
    const JointArray & torque_limits,
    double dt);

  ~MpcController();

  MpcController(const MpcController &) = delete;
  MpcController & operator=(const MpcController &) = delete;

  JointArray compute(
    const JointArray & q,
    const JointArray & qdot,
    std::size_t step);

  void reset();

private:
  using RM =
    Eigen::Matrix<
      double,
      Eigen::Dynamic,
      Eigen::Dynamic,
      Eigen::RowMajor>;

  struct Data
  {
    std::int64_t Np{};
    std::int64_t Nc{};
    std::int64_t nx{};
    std::int64_t nu{};
    std::int64_t nd{};
    std::int64_t ns{};

    double dt{};
    double rho{};
    double eps_abs{};
    double eps_rel{};

    Eigen::VectorXd xeq;
    Eigen::VectorXd teq;
    Eigen::VectorXd lim;

    RM P;
    RM Az;
    RM bref;
    RM tff;

    // Reference trajectories stored in the exported MPC binary.
    // xpy: reference state trajectory, shape = [ns, nx]
    // tpy: reference torque trajectory, shape = [ns, nu]
    RM xpy;
    RM tpy;
  };

  void load_data(
    const std::string & path);

  void setup_solver();

  JointArray solve(
    const JointArray & q,
    const JointArray & qdot,
    std::size_t step);

  Data data_;

  JointArray torque_limits_{};

  double dt_{0.002};

  OSQPSolver * solver_{nullptr};

  OSQPCscMatrix * P_csc_{nullptr};
  OSQPCscMatrix * A_csc_{nullptr};

  std::vector<OSQPFloat> P_x_;
  std::vector<OSQPInt> P_i_;
  std::vector<OSQPInt> P_p_;

  std::vector<OSQPFloat> A_x_;
  std::vector<OSQPInt> A_i_;
  std::vector<OSQPInt> A_p_;

  std::vector<OSQPFloat> q_qp_;
  std::vector<OSQPFloat> lower_;
  std::vector<OSQPFloat> upper_;

  std::size_t last_step_{0};
};

}  // namespace ur5e_ros2_control

#endif  // UR5E_ROS2_CONTROL__MPC_CONTROLLER_HPP_