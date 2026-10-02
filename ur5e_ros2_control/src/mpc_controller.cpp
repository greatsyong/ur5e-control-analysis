#include "ur5e_ros2_control/mpc_controller.hpp"



#include <algorithm>



#include <cmath>



#include <fstream>

#include <iostream>



#include <stdexcept>



namespace ur5e_ros2_control



{



namespace



{



constexpr std::int64_t MPC_MAGIC =



  0x555235454D504331LL;



template<typename T>



void readn(



  std::ifstream & f,



  T * ptr,



  std::size_t n)



{



  f.read(



    reinterpret_cast<char *>(ptr),



    static_cast<std::streamsize>(



      n * sizeof(T)));



  if (!f) {



    throw std::runtime_error(



      "Unexpected end of MPC binary file.");



  }



}



}  // namespace



MpcController::MpcController(



  const std::string & data_file,



  const JointArray & torque_limits,



  double dt)



: torque_limits_(torque_limits),



  dt_(dt)



{



  load_data(data_file);



  if (std::abs(data_.dt - dt_) > 1e-9) {



    throw std::runtime_error(



      "MPC data dt does not match controller dt.");



  }



  if (



    data_.nx != static_cast<std::int64_t>(NX) ||



    data_.nu != static_cast<std::int64_t>(N))



  {



    throw std::runtime_error(



      "MPC data dimensions do not match UR5e.");



  }



  setup_solver();



}



MpcController::~MpcController()



{



  if (solver_ != nullptr) {



    osqp_cleanup(solver_);



    solver_ = nullptr;



  }



  if (P_csc_ != nullptr) {



    OSQPCscMatrix_free(P_csc_);



    P_csc_ = nullptr;



  }



  if (A_csc_ != nullptr) {



    OSQPCscMatrix_free(A_csc_);



    A_csc_ = nullptr;



  }



}



void MpcController::load_data(



  const std::string & path)



{



  std::ifstream f(



    path,



    std::ios::binary);



  if (!f) {



    throw std::runtime_error(



      "Cannot open MPC data file: " + path);



  }



  std::int64_t h[8]{};



  double z[4]{};



  readn(



    f,



    h,



    8);



  readn(



    f,



    z,



    4);



  if (



    h[0] != MPC_MAGIC ||



    h[1] != 1)



  {



    throw std::runtime_error(



      "Invalid MPC binary header.");



  }



  data_.Np = h[2];



  data_.Nc = h[3];



  data_.nx = h[4];



  data_.nu = h[5];



  data_.nd = h[6];



  data_.ns = h[7];



  data_.dt = z[0];



  data_.rho = z[1];



  data_.eps_abs = z[2];



  data_.eps_rel = z[3];



  data_.xeq.resize(



    data_.nx);



  data_.teq.resize(



    data_.nu);



  data_.lim.resize(



    data_.nu);



  data_.P.resize(



    data_.nd,



    data_.nd);



  data_.Az.resize(



    data_.nd,



    data_.nx);



  data_.bref.resize(



    data_.ns,



    data_.nd);



  data_.tff.resize(



    data_.ns + data_.Np,



    data_.nu);



  readn(



    f,



    data_.xeq.data(),



    data_.xeq.size());



  readn(



    f,



    data_.teq.data(),



    data_.teq.size());



  readn(



    f,



    data_.lim.data(),



    data_.lim.size());



  readn(



    f,



    data_.P.data(),



    data_.P.size());



  readn(



    f,



    data_.Az.data(),



    data_.Az.size());



  readn(



    f,



    data_.bref.data(),



    data_.bref.size());



  readn(



    f,



    data_.tff.data(),



    data_.tff.size());



    data_.xpy.resize(



    data_.ns,



    data_.nx);



    data_.tpy.resize(



    data_.ns,



    data_.nu);



    readn(



    f,



    data_.xpy.data(),



    data_.xpy.size());



    readn(



    f,



    data_.tpy.data(),



    data_.tpy.size());



}



void MpcController::setup_solver()



{



  const OSQPInt n =



    static_cast<OSQPInt>(



      data_.nd);



  const OSQPInt m = n;



  /*



   * Match the original validated standalone MPC implementation.



   *



   * Store the complete upper triangular part of P,



   * including numerical zeros.



   */



  P_x_.clear();



  P_i_.clear();



  P_p_.assign(



    n + 1,



    0);



  for (



    OSQPInt j = 0;



    j < n;



    ++j)



  {



    P_p_[j] =



      static_cast<OSQPInt>(



        P_x_.size());



    for (



      OSQPInt i = 0;



      i <= j;



      ++i)



    {



      P_x_.push_back(



        static_cast<OSQPFloat>(



          data_.P(i, j)));



      P_i_.push_back(i);



    }



  }



  P_p_[n] =



    static_cast<OSQPInt>(



      P_x_.size());



  P_csc_ =



    OSQPCscMatrix_new(



      n,



      n,



      static_cast<OSQPInt>(



        P_x_.size()),



      P_x_.data(),



      P_i_.data(),



      P_p_.data());



  /*



   * A = I



   */



  A_x_.resize(



    n,



    1.0);



  A_i_.resize(n);



  A_p_.resize(n + 1);



  for (



    OSQPInt j = 0;



    j < n;



    ++j)



  {



    A_i_[j] = j;



    A_p_[j] = j;



  }



  A_p_[n] = n;



  A_csc_ =



    OSQPCscMatrix_new(



      m,



      n,



      n,



      A_x_.data(),



      A_i_.data(),



      A_p_.data());



  /*



   * Working vectors



   */



  q_qp_.assign(



    n,



    0.0);



  lower_.assign(



    n,



    -OSQP_INFTY);



  upper_.assign(



    n,



    OSQP_INFTY);



  /*



   * Match original validated OSQP settings.



   *



   * IMPORTANT:



   * data_.rho is intentionally NOT applied here.



   * The original standalone MPC loaded rho but used



   * OSQP's default rho setting.



   */



  OSQPSettings settings;



  osqp_set_default_settings(



    &settings);



  settings.verbose = 0;



  settings.polishing = 0;



  settings.warm_starting = 1;



  settings.eps_abs =



    data_.eps_abs;



  settings.eps_rel =



    data_.eps_rel;



  OSQPInt flag =



    osqp_setup(



      &solver_,



      P_csc_,



      q_qp_.data(),



      A_csc_,



      lower_.data(),



      upper_.data(),



      m,



      n,



      &settings);



  if (flag != 0) {



    throw std::runtime_error(



      std::string(



        "OSQP setup failed: ") +



      osqp_error_message(flag));



  }



}



MpcController::JointArray



MpcController::compute(



  const JointArray & q,



  const JointArray & qdot,



  std::size_t step)



{



  if (



    step >=



    static_cast<std::size_t>(



      data_.ns))



  {



    return JointArray{};



  }



  return solve(



    q,



    qdot,



    step);



}



MpcController::JointArray



MpcController::solve(



  const JointArray & q,



  const JointArray & qdot,



  std::size_t step)



{



  Eigen::VectorXd x(



    NX);



  for (



    std::size_t i = 0;



    i < N;



    ++i)



  {



    x[i] = q[i];



    x[i + N] = qdot[i];



  }



  // Compare live Isaac state against nominal MPC state trajectory.

  double q_err_max = 0.0;

  double qdot_err_max = 0.0;



  if (step < static_cast<std::size_t>(data_.ns))

  {

    for (std::size_t i = 0; i < N; ++i)

    {

      q_err_max = std::max(

        q_err_max,

        std::abs(

          x[i] -

          data_.xpy(

            static_cast<Eigen::Index>(step),

            static_cast<Eigen::Index>(i))));



      qdot_err_max = std::max(

        qdot_err_max,

        std::abs(

          x[i + N] -

          data_.xpy(

            static_cast<Eigen::Index>(step),

            static_cast<Eigen::Index>(i + N))));

    }



    if (step < 20 || step % 100 == 0)

    {

      std::cout

        << "[MPC DEBUG] step=" << step

        << " q_err_max=" << q_err_max

        << " qdot_err_max=" << qdot_err_max

        << std::endl;

    }

  }



  /*



   * QP linear term:



   *



   * g = Az * (x - xeq) + bref[k]



   */



  Eigen::VectorXd g =



    data_.Az *



    (x - data_.xeq) +



    data_.bref.row(



      static_cast<Eigen::Index>(



        step)).transpose();



  for (



    std::int64_t i = 0;



    i < data_.nd;



    ++i)



  {



    q_qp_[i] =



      static_cast<OSQPFloat>(



        g[i]);



  }



  /*



   * Torque constraints:



   *



   * tau = tff + delta_tau



   *



   * therefore:



   *



   * -limit - tff



   * <= delta_tau <=



   *  limit - tff



   */



  for (



    std::int64_t j = 0;



    j < data_.Nc - 1;



    ++j)



  {



    for (



      std::int64_t a = 0;



      a < data_.nu;



      ++a)



    {



      const std::int64_t id =



        j * data_.nu + a;



      const std::int64_t r =



        static_cast<std::int64_t>(



          step) + j;



      lower_[id] =



        static_cast<OSQPFloat>(



          -data_.lim[a] -



          data_.tff(r, a));



      upper_[id] =



        static_cast<OSQPFloat>(



          data_.lim[a] -



          data_.tff(r, a));



    }



  }



  /*



   * Last control move is held constant



   * from Nc-1 through the prediction horizon.



   */



  for (



    std::int64_t a = 0;



    a < data_.nu;



    ++a)



  {



    double l =



      -OSQP_INFTY;



    double u =



      OSQP_INFTY;



    for (



      std::int64_t h =



        data_.Nc - 1;



      h < data_.Np;



      ++h)



    {



      const std::int64_t r =



        static_cast<std::int64_t>(



          step) + h;



      l =



        std::max(



          l,



          -data_.lim[a] -



          data_.tff(r, a));



      u =



        std::min(



          u,



          data_.lim[a] -



          data_.tff(r, a));



    }



    const std::int64_t id =



      (data_.Nc - 1) *



      data_.nu + a;



    lower_[id] =



      static_cast<OSQPFloat>(l);



    upper_[id] =



      static_cast<OSQPFloat>(u);



  }



  /*



   * Update QP vectors.



   */



  OSQPInt flag =



    osqp_update_data_vec(



      solver_,



      q_qp_.data(),



      lower_.data(),



      upper_.data());



  if (flag != 0) {



    throw std::runtime_error(



      "OSQP vector update failed.");



  }



  /*



   * Solve QP.



   */



  flag =



    osqp_solve(



      solver_);



  if (step < 20 || step % 100 == 0)

  {

    std::cout

      << "[MPC OSQP] step=" << step

      << " iter=" << solver_->info->iter

      << " status=" << solver_->info->status

      << std::endl;

  }



  if (flag != 0) {



    throw std::runtime_error(



      "OSQP solve failed.");



  }



  if (

    solver_->info->status_val !=

      OSQP_SOLVED &&

    solver_->info->status_val !=

      OSQP_SOLVED_INACCURATE)

  {

    std::cerr

      << "[MPC FAILURE] step=" << step

      << " q_err_max=" << q_err_max

      << " qdot_err_max=" << qdot_err_max

      << " iter=" << solver_->info->iter

      << " status=" << solver_->info->status

      << std::endl;



    throw std::runtime_error(

      std::string(

        "OSQP status: ") +

      solver_->info->status);

  }



  /*



   * Total torque:



   *



   * tau = tau_ff + delta_tau



   */



  JointArray tau{};



  for (



    std::size_t a = 0;



    a < N;



    ++a)



  {



    const double tau_ff =



      data_.tff(



        static_cast<Eigen::Index>(



          step),



        static_cast<Eigen::Index>(



          a));



    const double delta_tau =



      solver_->solution->x[a];



    tau[a] =



      tau_ff +



      delta_tau;



  }

  if (step == 0)
  {
    std::cout << "\n[MPC INITIAL DIAGNOSTIC]\n";

    std::cout << "q_live [rad]      : ";
    for (std::size_t a = 0; a < N; ++a) {
      std::cout << q[a] << (a + 1 < N ? " " : "\n");
    }

    std::cout << "qdot_live [rad/s] : ";
    for (std::size_t a = 0; a < N; ++a) {
      std::cout << qdot[a] << (a + 1 < N ? " " : "\n");
    }

    std::cout << "q_nominal [rad]   : ";
    for (std::size_t a = 0; a < N; ++a) {
      std::cout << data_.xpy(0, static_cast<Eigen::Index>(a))
                << (a + 1 < N ? " " : "\n");
    }

    std::cout << "qdot_nominal      : ";
    for (std::size_t a = 0; a < N; ++a) {
      std::cout << data_.xpy(0, static_cast<Eigen::Index>(a + N))
                << (a + 1 < N ? " " : "\n");
    }

    std::cout << "tff[0] [Nm]       : ";
    for (std::size_t a = 0; a < N; ++a) {
      std::cout << data_.tff(0, static_cast<Eigen::Index>(a))
                << (a + 1 < N ? " " : "\n");
    }

    std::cout << "delta_tau [Nm]    : ";
    for (std::size_t a = 0; a < N; ++a) {
      std::cout << solver_->solution->x[a]
                << (a + 1 < N ? " " : "\n");
    }

    std::cout << "tau_total [Nm]    : ";
    for (std::size_t a = 0; a < N; ++a) {
      std::cout << tau[a] << (a + 1 < N ? " " : "\n");
    }

    std::cout << "torque_limit [Nm] : ";
    for (std::size_t a = 0; a < N; ++a) {
      std::cout << torque_limits_[a]
                << (a + 1 < N ? " " : "\n");
    }

    std::cout << "[END MPC INITIAL DIAGNOSTIC]\n\n";
  }

  last_step_ = step;

  return tau;



}



void MpcController::reset()



{



  last_step_ = 0;



}



}  // namespace ur5e_ros2_control