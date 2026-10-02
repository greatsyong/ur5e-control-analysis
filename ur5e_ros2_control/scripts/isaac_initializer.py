# Isaac Sim 6.1 Script Editor initializer.
# Generic ROS2 Subscriber:
# /Graph/ROS_JointStates/ros2_subscriber
# std_msgs/msg/Bool, topic validation/hold

import numpy as np
import omni.graph.core as og

from isaacsim.core.prims import Articulation
from isaacsim.core.simulation_manager import IsaacEvents, SimulationManager

robot = Articulation("/ur5e")
robot.initialize()

q_start = np.deg2rad([0.0, -90.0, 90.0, -90.0, -90.0, 0.0])
qdot_start = np.zeros(6)

hold_attr = og.Controller.attribute(
    "/Graph/ROS_JointStates/ros2_subscriber.outputs:data"
)

print("HOLD attribute valid:", hold_attr.is_valid())
print("Current HOLD:", hold_attr.get())

last_hold = None

def initializer_callback(dt, context):
    global last_hold
    hold = bool(hold_attr.get())

    if hold != last_hold:
        print("Initializer HOLD =", hold)
        last_hold = hold

    if hold:
        robot.set_joint_positions(q_start)
        robot.set_joint_velocities(qdot_start)

callback_id = SimulationManager.register_callback(
    initializer_callback,
    IsaacEvents.PRE_PHYSICS_STEP
)

print("UR5e initializer READY")
print("callback_id =", callback_id)
