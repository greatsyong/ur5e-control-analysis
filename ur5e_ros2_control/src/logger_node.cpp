#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/string.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

/*
 * Validation logger
 * -----------------
 * Records telemetry only while the validation manager is in RUNNING.
 *
 * The controller publishes:
 *   position        -> measured joint position q [rad]
 *   velocity        -> measured joint velocity qdot [rad/s]
 *   effort          -> commanded joint torque tau [Nm]
 *   header.frame_id -> controller compute time [ms]
 *
 * The use of frame_id for compute time is intentionally kept here to avoid
 * introducing a custom ROS2 message solely for the validation experiment.
 * The CSV converts it into an explicitly named compute_ms column.
 *
 * Unlike the original development logger, results are NOT written to /tmp.
 * output_dir is a ROS parameter and defaults to the user's home directory.
 */
class LoggerNode : public rclcpp::Node
{
public:
  static constexpr std::size_t N = 6;
  using JointArray = std::array<double, N>;

  LoggerNode()
  : Node("validation_logger")
  {
    controller_type_ = declare_parameter<std::string>("controller", "unknown");

    const char * home = std::getenv("HOME");
    const std::string default_output_dir = home != nullptr ? std::string(home) : std::string(".");
    output_dir_ = declare_parameter<std::string>("output_dir", default_output_dir);

    q0_ = deg({0.0, -90.0, 90.0, -90.0, -90.0, 0.0});
    qg_ = deg({20.0, -60.0, 60.0, -70.0, -70.0, 20.0});

    telemetry_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      "/validation/telemetry",
      100,
      std::bind(&LoggerNode::telemetry_cb, this, std::placeholders::_1));

    state_sub_ = create_subscription<std_msgs::msg::String>(
      "/validation/state",
      10,
      std::bind(&LoggerNode::state_cb, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(),
      "Validation logger ready; controller=%s; output_dir=%s",
      controller_type_.c_str(),
      output_dir_.c_str());
  }

private:
  static JointArray deg(const JointArray & d)
  {
    constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
    JointArray r{};
    for (std::size_t i = 0; i < N; ++i) {
      r[i] = d[i] * kDegToRad;
    }
    return r;
  }

  static double stamp_sec(const builtin_interfaces::msg::Time & stamp)
  {
    return static_cast<double>(stamp.sec) + 1e-9 * static_cast<double>(stamp.nanosec);
  }

  void reference(double t, JointArray & q_ref, JointArray & qdot_ref) const
  {
    // Same 4 s quintic reference used by controller_node.cpp.
    constexpr double T = 4.0;
    const double s = std::clamp(t / T, 0.0, 1.0);
    const double s2 = s * s;
    const double s3 = s2 * s;
    const double s4 = s3 * s;
    const double s5 = s4 * s;

    const double h = 10.0 * s3 - 15.0 * s4 + 6.0 * s5;
    const double hd = (30.0 * s2 - 60.0 * s3 + 30.0 * s4) / T;

    for (std::size_t i = 0; i < N; ++i) {
      const double dq = qg_[i] - q0_[i];
      q_ref[i] = q0_[i] + h * dq;
      qdot_ref[i] = hd * dq;
    }
  }

  void state_cb(const std_msgs::msg::String::SharedPtr msg)
  {
    if (msg->data == "RUNNING") {
      rows_.clear();
      active_ = true;
      have_t0_ = false;
      return;
    }

    if (msg->data == "COMPLETE" && active_) {
      active_ = false;
      write_csv();
    }
  }

  void telemetry_cb(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    if (!active_) {
      return;
    }

    if (!have_t0_) {
      t0_sim_ = stamp_sec(msg->header.stamp);
      have_t0_ = true;
    }

    rows_.push_back(*msg);
  }

  std::string make_filename() const
  {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};

#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif

    std::ostringstream ss;
    ss << output_dir_;
    if (!output_dir_.empty() && output_dir_.back() != '/') {
      ss << '/';
    }
    ss << "ur5e_" << controller_type_ << "_validation_"
       << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".csv";
    return ss.str();
  }

  void write_csv()
  {
    if (rows_.empty()) {
      RCLCPP_WARN(get_logger(), "No samples to save.");
      return;
    }

    const std::string filename = make_filename();
    std::ofstream f(filename);

    if (!f.is_open()) {
      RCLCPP_ERROR(
        get_logger(),
        "Failed to open output file: %s. Make sure output_dir exists.",
        filename.c_str());
      return;
    }

    // Fixed UR5e order is enforced by controller_node before telemetry publish.
    f << "sample,t_sim,sec,nanosec";
    for (const auto & name : rows_[0].name) f << ',' << name << "_q";
    for (const auto & name : rows_[0].name) f << ',' << name << "_qdot";
    for (const auto & name : rows_[0].name) f << ',' << name << "_q_ref";
    for (const auto & name : rows_[0].name) f << ',' << name << "_qdot_ref";
    for (const auto & name : rows_[0].name) f << ',' << name << "_tau";
    f << ",compute_ms\n";

    f << std::setprecision(12);

    for (std::size_t row = 0; row < rows_.size(); ++row) {
      const auto & msg = rows_[row];
      const double t_sim = stamp_sec(msg.header.stamp) - t0_sim_;

      JointArray q_ref{};
      JointArray qdot_ref{};
      reference(t_sim, q_ref, qdot_ref);

      f << row << ',' << t_sim << ','
        << msg.header.stamp.sec << ',' << msg.header.stamp.nanosec;

      for (double x : msg.position) f << ',' << x;
      for (double x : msg.velocity) f << ',' << x;
      for (double x : q_ref) f << ',' << x;
      for (double x : qdot_ref) f << ',' << x;
      for (double x : msg.effort) f << ',' << x;

      // frame_id carries a numeric compute time string by package convention.
      f << ',' << msg.header.frame_id << '\n';
    }

    f.close();

    RCLCPP_INFO(
      get_logger(),
      "Saved %zu samples to %s",
      rows_.size(),
      filename.c_str());
  }

  bool active_{false};
  bool have_t0_{false};
  double t0_sim_{0.0};

  std::string controller_type_;
  std::string output_dir_;
  JointArray q0_{};
  JointArray qg_{};

  std::vector<sensor_msgs::msg::JointState> rows_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr telemetry_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr state_sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LoggerNode>());
  rclcpp::shutdown();
  return 0;
}
