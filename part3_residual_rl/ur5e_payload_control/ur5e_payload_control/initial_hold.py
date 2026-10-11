
#!/usr/bin/env python3

import sys
import time

import numpy as np
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import JointState


JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

MODEL_DIR = (
    "/home/sooyong/robotics_ws/src/"
    "ur5e_control_analysis/"
    "part1_modeling_control_analysis"
)

MASS = 3.0
COM_TOOL = np.array([0.0, 0.0, 0.025])

KP = 144.0
KD = 24.0

TORQUE_LIMITS = np.array([
    150.0, 150.0, 150.0,
    28.0, 28.0, 28.0,
])


class InitialHold(Node):

    def __init__(self):
        super().__init__("ur5e_initial_hold")

        sys.path.insert(0, MODEL_DIR)

        from dynamics.payload_dynamics import (
            mass_matrix_payload,
            gravity_vector_payload,
        )

        self.mass_matrix = mass_matrix_payload
        self.gravity_vector = gravity_vector_payload

        self.q_target = None
        self.q = None
        self.qd = None
        self.last_state_time = None
        self.last_log_time = 0.0
        self.command_count = 0

        self.publisher = self.create_publisher(
            JointState,
            "/joint_command",
            10,
        )

        self.subscription = self.create_subscription(
            JointState,
            "/joint_states",
            self.state_callback,
            10,
        )

        self.timer = self.create_timer(
            0.02,
            self.control_callback,
        )

        self.get_logger().info(
            "UR5e Initial Hold started | "
            "3 kg | G013 | 50 Hz"
        )

        self.get_logger().warning(
            "Keep Isaac Sim temporary joint drives "
            "enabled until ROS torque output is verified."
        )

    def state_callback(self, msg):

        if not all(j in msg.name for j in JOINTS):
            return

        if (
            len(msg.position) != len(msg.name)
            or len(msg.velocity) != len(msg.name)
        ):
            return

        indices = [
            msg.name.index(j)
            for j in JOINTS
        ]

        q = np.array(
            [msg.position[i] for i in indices],
            dtype=float,
        )

        qd = np.array(
            [msg.velocity[i] for i in indices],
            dtype=float,
        )

        if not (
            np.all(np.isfinite(q))
            and np.all(np.isfinite(qd))
        ):
            return

        self.q = q
        self.qd = qd
        self.last_state_time = time.monotonic()

        if self.q_target is None:

            if np.max(np.abs(qd)) > np.deg2rad(1.0):
                return

            self.q_target = q.copy()

            self.get_logger().info(
                "Initial position locked (deg): "
                + np.array2string(
                    np.rad2deg(self.q_target),
                    precision=3,
                )
            )

    def control_callback(self):

        if (
            self.q_target is None
            or self.q is None
            or self.qd is None
        ):
            return

        if (
            self.last_state_time is None
            or time.monotonic() - self.last_state_time > 0.1
        ):
            self.get_logger().warning(
                "Joint state timeout. No torque published.",
                throttle_duration_sec=2.0,
            )
            return

        q_error = self.q_target - self.q

        M = self.mass_matrix(
            self.q,
            payload_mass=MASS,
            payload_com_tool=COM_TOOL,
        )

        g = self.gravity_vector(
            self.q,
            payload_mass=MASS,
            payload_com_tool=COM_TOOL,
        )

        tau_feedback = M @ (
            KP * q_error - KD * self.qd
        )

        tau_raw = g + tau_feedback

        tau = np.clip(
            tau_raw,
            -TORQUE_LIMITS,
            TORQUE_LIMITS,
        )

        command = JointState()
        command.header.stamp = (
            self.get_clock().now().to_msg()
        )

        command.name = list(JOINTS)

        # Torque-only command.
        command.position = []
        command.velocity = []
        command.effort = tau.tolist()

        self.publisher.publish(command)
        self.command_count += 1

        now = time.monotonic()

        if now - self.last_log_time >= 1.0:

            self.last_log_time = now

            max_error = np.max(
                np.abs(np.rad2deg(q_error))
            )

            max_velocity = np.max(
                np.abs(np.rad2deg(self.qd))
            )

            self.get_logger().info(
                f"commands={self.command_count} | "
                f"max_error={max_error:.4f} deg | "
                f"max_velocity={max_velocity:.4f} deg/s | "
                f"max_torque={np.max(np.abs(tau)):.3f} Nm | "
                f"saturation={np.any(np.abs(tau_raw) > TORQUE_LIMITS)}"
            )


def main(args=None):

    rclpy.init(args=args)
    node = InitialHold()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

