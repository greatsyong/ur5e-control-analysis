#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>

#include <array>
#include <cmath>
#include <string>
#include <unordered_map>

class ValidationManager : public rclcpp::Node
{
public:
  static constexpr std::size_t N = 6;
  static constexpr double POSITION_TOL = 0.002;   // rad
  static constexpr double VELOCITY_TOL = 0.01;    // rad/s
  using JointArray = std::array<double, N>;

  ValidationManager() : Node("validation_manager")
  {
    names_ = {"shoulder_pan_joint","shoulder_lift_joint","elbow_joint",
              "wrist_1_joint","wrist_2_joint","wrist_3_joint"};
    q_start_ = {0.0, -M_PI/2.0, M_PI/2.0, -M_PI/2.0, -M_PI/2.0, 0.0};

    state_pub_ = create_publisher<std_msgs::msg::String>("/validation/state", 10);
    hold_pub_ = create_publisher<std_msgs::msg::Bool>("/validation/hold", 10);

    joint_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", rclcpp::SensorDataQoS(),
      std::bind(&ValidationManager::joint_cb, this, std::placeholders::_1));

    controller_sub_ = create_subscription<std_msgs::msg::String>(
      "/validation/controller_status", 10,
      [this](const std_msgs::msg::String::SharedPtr msg) {
        if (msg->data == "READY") controller_ready_ = true;
      });

    timer_ = create_wall_timer(
      std::chrono::milliseconds(20),
      std::bind(&ValidationManager::tick, this));

    publish_state("WAITING");
    publish_hold(true);
    RCLCPP_INFO(get_logger(), "WAITING: HOLD=true");
  }

private:
  enum Phase { WAITING, INITIALIZING, RUNNING, FREE_FALL, DONE };

  void publish_state(const std::string & value)
  {
    std_msgs::msg::String msg;
    msg.data = value;
    state_pub_->publish(msg);
  }

  void publish_hold(bool value)
  {
    std_msgs::msg::Bool msg;
    msg.data = value;
    hold_pub_->publish(msg);
  }

  static double stamp_sec(const builtin_interfaces::msg::Time & stamp)
  {
    return static_cast<double>(stamp.sec) +
           1e-9 * static_cast<double>(stamp.nanosec);
  }

  void joint_cb(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    std::unordered_map<std::string, std::size_t> index;
    for (std::size_t i = 0; i < msg->name.size(); ++i) index[msg->name[i]] = i;

    bool ok = true;
    for (std::size_t j = 0; j < N; ++j) {
      const auto it = index.find(names_[j]);
      if (it == index.end() ||
          it->second >= msg->position.size() ||
          it->second >= msg->velocity.size()) {
        ok = false;
        break;
      }
      const auto k = it->second;
      if (std::abs(msg->position[k] - q_start_[j]) > POSITION_TOL) {
        ok = false;
        break;
      }
    }

    sim_time_ = stamp_sec(msg->header.stamp);
    have_state_ = true;

    if (phase_ == INITIALIZING) {
      stable_samples_ = ok ? stable_samples_ + 1 : 0;
    }
  }

  void tick()
  {
    switch (phase_) {
      case WAITING:
        publish_hold(true);
        if (controller_ready_ && have_state_) {
          phase_ = INITIALIZING;
          stable_samples_ = 0;
          publish_state("INITIALIZING");
          RCLCPP_INFO(get_logger(), "INITIALIZING: verifying q_start");
        }
        break;

      case INITIALIZING:
        publish_hold(true);
        if (stable_samples_ >= 25 && controller_ready_) {
          publish_state("RUNNING");
          publish_hold(false);
          run_start_sim_ = sim_time_;
          phase_ = RUNNING;
          RCLCPP_INFO(get_logger(), "RUNNING: HOLD=false");
        }
        break;

      case RUNNING:
        if (sim_time_ - run_start_sim_ >= 4.0) {
          publish_state("COMPLETE");
          publish_hold(false);
          free_fall_start_wall_ = now();
          phase_ = FREE_FALL;
          RCLCPP_INFO(get_logger(), "COMPLETE: zero effort / free fall");
        }
        break;

      case FREE_FALL:
        publish_hold(false);
        if ((now() - free_fall_start_wall_).seconds() >= 1.5) {
          publish_state("SHUTDOWN");
          phase_ = DONE;
          RCLCPP_INFO(get_logger(), "SHUTDOWN");
          rclcpp::shutdown();
        }
        break;

      case DONE:
        break;
    }
  }

  Phase phase_{WAITING};
  std::array<std::string, N> names_;
  JointArray q_start_{};

  bool controller_ready_{false};
  bool have_state_{false};
  int stable_samples_{0};
  double sim_time_{0.0};
  double run_start_sim_{0.0};
  rclcpp::Time free_fall_start_wall_{0, 0, RCL_ROS_TIME};

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr state_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr hold_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr controller_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ValidationManager>());
  return 0;
}
