
#!/usr/bin/env python3

import numpy as np
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import JointState


JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


class JointStateMonitor(Node):

    def __init__(self):
        super().__init__("joint_state_monitor")

        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10,
        )

        self.message_count = 0
        self.last_print_time = self.get_clock().now()

        self.get_logger().info(
            "UR5e Joint State Monitor started. "
            "Listening to /joint_states."
        )

    def joint_state_callback(self, msg):

        if not all(name in msg.name for name in JOINT_NAMES):
            self.get_logger().warning(
                "JointState does not contain all six UR5e joints."
            )
            return

        if len(msg.position) < len(msg.name):
            self.get_logger().warning(
                "JointState position array is incomplete."
            )
            return

        if len(msg.velocity) < len(msg.name):
            self.get_logger().warning(
                "JointState velocity array is incomplete."
            )
            return

        index = [msg.name.index(name) for name in JOINT_NAMES]

        q = np.array(
            [msg.position[i] for i in index],
            dtype=float,
        )

        qd = np.array(
            [msg.velocity[i] for i in index],
            dtype=float,
        )

        self.message_count += 1

        now = self.get_clock().now()
        elapsed = (now - self.last_print_time).nanoseconds * 1e-9

        if elapsed < 1.0:
            return

        self.last_print_time = now

        self.get_logger().info(
            f"Received messages: {self.message_count}\n"
            f"Joint position (deg): "
            f"{np.round(np.rad2deg(q), 3)}\n"
            f"Joint velocity (deg/s): "
            f"{np.round(np.rad2deg(qd), 3)}"
        )


def main(args=None):

    rclpy.init(args=args)

    node = JointStateMonitor()

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

