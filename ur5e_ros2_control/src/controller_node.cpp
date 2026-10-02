#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/joint_state.hpp>

#include <std_msgs/msg/string.hpp>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include "ur5e_ros2_control/pid_controller.hpp"

#include "ur5e_ros2_control/lqr_controller.hpp"

#include "ur5e_ros2_control/mpc_controller.hpp"

#include <algorithm>

#include <array>

#include <chrono>

#include <cmath>

#include <memory>

#include <stdexcept>

#include <string>

#include <unordered_map>

using ur5e_ros2_control::PidController;

using ur5e_ros2_control::LqrController;

using ur5e_ros2_control::MpcController;

class ControllerNode : public rclcpp::Node

{

public:

  static constexpr std::size_t N = 6;

  using JointArray =

    std::array<double, N>;

  ControllerNode()

  : Node("ur5e_controller_node"),

    pid_controller_(

      JointArray{

        423.454320,

        1036.54331,

        352.543417,

        9.25748969,

        2.01575776,

        0.103024

      },

      JointArray{

        7622.17776,

        18657.7796,

        6345.78150,

        166.634814,

        36.2836397,

        1.854432

      },

      JointArray{

        127.036296,

        310.962994,

        105.763025,

        2.77724691,

        0.604727329,

        0.0309072

      },

      JointArray{

        150.0,

        150.0,

        150.0,

        28.0,

        28.0,

        28.0

      },

      0.002),

    lqr_controller_(

      JointArray{

        150.0,

        150.0,

        150.0,

        28.0,

        28.0,

        28.0

      },

      0.002)

  {

    controller_type_ =

      declare_parameter<std::string>(

        "controller",

        "pid");

    if (controller_type_ != "pid" &&
        controller_type_ != "lqr" &&
        controller_type_ != "mpc")

    {

      RCLCPP_ERROR(

        get_logger(),

        "Invalid controller type: %s. "

        "Use 'pid', 'lqr', or 'mpc'.",

        controller_type_.c_str());

      throw std::runtime_error(

        "Invalid controller type");

    }

    const auto package_share =

      ament_index_cpp::get_package_share_directory(

        "ur5e_ros2_control");

const JointArray torque_limits{

      150.0,

      150.0,

      150.0,

      28.0,

      28.0,

      28.0

    };

    /* Construct the MPC controller only when MPC is selected. */

    if (controller_type_ == "mpc")

    {

      const std::string data_file =

        package_share +

        "/config/mpc_cpp_data.bin";

      RCLCPP_INFO(

        get_logger(),

        "Loading raw MPC data: %s",

        data_file.c_str());

      mpc_controller_ =

        std::make_unique<MpcController>(

          data_file,

          torque_limits,

          0.002);

    }

    names_ = {

      "shoulder_pan_joint",

      "shoulder_lift_joint",

      "elbow_joint",

      "wrist_1_joint",

      "wrist_2_joint",

      "wrist_3_joint"

    };

    q0_ =

      deg({

        0,

        -90,

        90,

        -90,

        -90,

        0

      });

    qg_ =

      deg({

        20,

        -60,

        60,

        -70,

        -70,

        20

      });

    cmd_pub_ =

      create_publisher<

        sensor_msgs::msg::JointState>(

          "/joint_command",

          10);

    telemetry_pub_ =

      create_publisher<

        sensor_msgs::msg::JointState>(

          "/validation/telemetry",

          100);

    state_sub_ =

      create_subscription<

        sensor_msgs::msg::JointState>(

          "/joint_states",

          rclcpp::SensorDataQoS(),

          std::bind(

            &ControllerNode::state_cb,

            this,

            std::placeholders::_1));

    manager_sub_ =

      create_subscription<

        std_msgs::msg::String>(

          "/validation/state",

          10,

          std::bind(

            &ControllerNode::manager_cb,

            this,

            std::placeholders::_1));

    ready_pub_ =

      create_publisher<

        std_msgs::msg::String>(

          "/validation/controller_status",

          10);

    timer_ =

      create_wall_timer(

        std::chrono::milliseconds(100),

        [this] {

          std_msgs::msg::String m;

          m.data = "READY";

          ready_pub_->publish(m);

        });

    RCLCPP_INFO(

      get_logger(),

      "Controller READY; type=%s; waiting for RUNNING.",

      controller_type_.c_str());

  }

private:

  static JointArray deg(

    const JointArray & d)

  {

    JointArray r{};

    for (std::size_t i = 0;

         i < N;

         ++i)

    {

      r[i] =

        d[i] *

        M_PI /

        180.0;

    }

    return r;

  }

  static double stamp_sec(

    const builtin_interfaces::msg::Time & s)

  {

    return

      double(s.sec) +

      1e-9 *

      double(s.nanosec);

  }

  bool extract(

    const sensor_msgs::msg::JointState & m,

    JointArray & q,

    JointArray & qd)

  {

    std::unordered_map<

      std::string,

      std::size_t> map;

    for (std::size_t i = 0;

         i < m.name.size();

         ++i)

    {

      map[m.name[i]] = i;

    }

    for (std::size_t j = 0;

         j < N;

         ++j)

    {

      auto it =

        map.find(names_[j]);

      if (it == map.end() ||

          it->second >=

            m.position.size() ||

          it->second >=

            m.velocity.size())

      {

        return false;

      }

      q[j] =

        m.position[it->second];

      qd[j] =

        m.velocity[it->second];

    }

    return true;

  }

  void reference(

    double t,

    JointArray & qr,

    JointArray & qdr)

  {

    constexpr double T = 4.0;

    double s =

      std::clamp(

        t / T,

        0.0,

        1.0);

    double s2 = s * s;

    double s3 = s2 * s;

    double s4 = s3 * s;

    double s5 = s4 * s;

    double h =

      10.0 * s3 -

      15.0 * s4 +

      6.0 * s5;

    double hd =

      (

        30.0 * s2 -

        60.0 * s3 +

        30.0 * s4

      ) / T;

    for (std::size_t i = 0;

         i < N;

         ++i)

    {

      double dq =

        qg_[i] -

        q0_[i];

      qr[i] =

        q0_[i] +

        h * dq;

      qdr[i] =

        hd * dq;

    }

  }

  void manager_cb(

    const std_msgs::msg::String::SharedPtr m)

  {

    if (m->data == "RUNNING")

    {

      active_ = true;

      have_t0_ = false;

    have_last_mpc_step_ = false;

    last_mpc_step_ = 0;

      if (controller_type_ == "pid")

      {

        pid_controller_.reset();

      }

      else if (controller_type_ == "lqr")

      {

        lqr_controller_.reset();

      }

      else if (controller_type_ == "mpc")

      {

        if (!mpc_controller_)

        {

          throw std::runtime_error(

            "Raw MPC controller was not initialized.");

        }

        mpc_controller_->reset();

      }

      RCLCPP_INFO(

        get_logger(),

        "%s armed.",

        controller_type_.c_str());

    }

    else if (

      m->data == "COMPLETE" ||

      m->data == "RELEASED" ||

      m->data == "SHUTDOWN")

    {

      active_ = false;

      publish_zero();

    }

  }

  void state_cb(

    const sensor_msgs::msg::JointState::SharedPtr m)

  {

    if (!active_) {

      return;

    }

    JointArray q{};

    JointArray qd{};

    if (!extract(

          *m,

          q,

          qd))

    {

      return;

    }

    double ts =

      stamp_sec(

        m->header.stamp);

    if (!have_t0_)

    {

      t0_ = ts;

      have_t0_ = true;

    }

    double t =

      ts - t0_;

    if (t >= 4.0)

    {

      publish_zero();

      return;

    }

    JointArray qr{};

    JointArray qdr{};

    reference(

      t,

      qr,

      qdr);

    auto start =

      std::chrono::steady_clock::now();

    JointArray tau{};

    if (controller_type_ == "pid")

    {

      tau =

        pid_controller_.compute(

          q,

          qd,

          qr,

          qdr);

    }

    else if (controller_type_ == "lqr")

    {

      tau =

        lqr_controller_.compute(

          q,

          qd,

          qr,

          qdr);

    }

    else if (controller_type_ == "mpc")

    {

      if (!mpc_controller_)

      {

        throw std::runtime_error(

          "Raw MPC controller is null.");

      }

      const std::size_t step =

        static_cast<std::size_t>(

          std::llround(

            t / 0.002));

      /*

       * MPC is indexed by Isaac simulation time.

       *

       * Duplicate or older discrete samples are ignored.

       * If one or more samples were missed because computation

       * was late, jump directly to the current simulation-time

       * index. Never replay missed MPC steps.

       */

      if (have_last_mpc_step_ &&

          step <= last_mpc_step_)

      {

        return;

      }

      last_mpc_step_ = step;

      have_last_mpc_step_ = true;

      tau =

        mpc_controller_->compute(

          q,

          qd,

          step);

    }

    auto end =

      std::chrono::steady_clock::now();

    double compute_ms =

      std::chrono::duration<

        double,

        std::milli>(

          end - start).count();

    sensor_msgs::msg::JointState cmd;

    cmd.header.stamp =

      m->header.stamp;

    cmd.name.assign(

      names_.begin(),

      names_.end());

    cmd.effort.assign(

      tau.begin(),

      tau.end());

    cmd_pub_->publish(cmd);

    /*

     * Telemetry convention:

     *

     * position = q

     * velocity = qdot

     * effort   = tau

     * frame_id = controller compute time [ms]

     */

    sensor_msgs::msg::JointState tel =

      *m;

    tel.name.assign(

      names_.begin(),

      names_.end());

    tel.position.assign(

      q.begin(),

      q.end());

    tel.velocity.assign(

      qd.begin(),

      qd.end());

    tel.effort.assign(

      tau.begin(),

      tau.end());

    tel.header.frame_id =

      std::to_string(

        compute_ms);

    telemetry_pub_->publish(

      tel);

  }

  void publish_zero()

  {

    sensor_msgs::msg::JointState m;

    m.header.stamp =

      now();

    m.name.assign(

      names_.begin(),

      names_.end());

    m.effort.assign(

      N,

      0.0);

    cmd_pub_->publish(m);

  }

  std::array<

    std::string,

    N> names_;

  JointArray q0_{};

  JointArray qg_{};

  std::string controller_type_;

  PidController

    pid_controller_;

  LqrController

    lqr_controller_;

  std::unique_ptr<MpcController>

    mpc_controller_;

  bool active_{false};

  bool have_t0_{false};

  bool have_last_mpc_step_{false};

  std::size_t last_mpc_step_{0};

  double t0_{0.0};

  rclcpp::Publisher<

    sensor_msgs::msg::JointState>::SharedPtr

    cmd_pub_;

  rclcpp::Publisher<

    sensor_msgs::msg::JointState>::SharedPtr

    telemetry_pub_;

  rclcpp::Publisher<

    std_msgs::msg::String>::SharedPtr

    ready_pub_;

  rclcpp::Subscription<

    sensor_msgs::msg::JointState>::SharedPtr

    state_sub_;

  rclcpp::Subscription<

    std_msgs::msg::String>::SharedPtr

    manager_sub_;

  rclcpp::TimerBase::SharedPtr

    timer_;

};

int main(

  int argc,

  char ** argv)

{

  rclcpp::init(

    argc,

    argv);

  rclcpp::spin(

    std::make_shared<

      ControllerNode>());

  rclcpp::shutdown();

  return 0;

}