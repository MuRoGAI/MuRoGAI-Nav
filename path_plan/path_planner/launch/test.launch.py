from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # All robots are now managed inside ONE process (MultiThreadedExecutor).
    # The ROBOTS list lives in one_meter_immediate.py — edit it there.
    return LaunchDescription([
        Node(
            package="path_planner",
            executable="test6",
            name="multi_robot_controller",
            output="screen",
        )
    ])