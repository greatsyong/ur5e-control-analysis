from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Launch one controller, the validation logger, and the run-state manager."""

    controller_type = DeclareLaunchArgument(
        "controller",
        default_value="pid",
        description="Controller type: pid, lqr, or mpc",
    )

    # Persistent output location. Override this at launch time when results
    # should be written directly into a project-specific results directory.
    output_dir = DeclareLaunchArgument(
        "output_dir",
        default_value=EnvironmentVariable("HOME"),
        description="Existing directory where validation CSV files are saved",
    )

    controller = Node(
        package="ur5e_ros2_control",
        executable="controller_node",
        output="screen",
        parameters=[{
            "controller": LaunchConfiguration("controller"),
        }],
    )

    logger = Node(
        package="ur5e_ros2_control",
        executable="logger_node",
        output="screen",
        parameters=[{
            "controller": LaunchConfiguration("controller"),
            "output_dir": LaunchConfiguration("output_dir"),
        }],
    )

    manager = Node(
        package="ur5e_ros2_control",
        executable="validation_manager",
        output="screen",
    )

    # The validation manager owns the experiment lifecycle. When it exits,
    # shut down the whole launch so the controller/logger do not remain alive.
    stop_launch = RegisterEventHandler(
        OnProcessExit(
            target_action=manager,
            on_exit=[EmitEvent(event=Shutdown(reason="validation complete"))],
        )
    )

    return LaunchDescription([
        controller_type,
        output_dir,
        controller,
        logger,
        manager,
        stop_launch,
    ])
