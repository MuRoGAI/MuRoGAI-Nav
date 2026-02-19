from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    robot1_controller = Node(
        package='burger_robot',
        executable='controller7_diff_drive_pose1',
        name='robot1_controller',
        parameters=[{'robot_name': 'robot1'}],
        output='screen',
    )

    robot2_controller = Node(
        package='burger_robot',
        executable='controller7_diff_drive_pose1',
        name='robot2_controller',
        parameters=[{'robot_name': 'robot2'}],
        output='screen',
    )

    robot3_controller = Node(
        package='burger_robot',
        executable='controller7_diff_drive_pose1',
        name='robot3_controller',
        parameters=[{'robot_name': 'robot3'}],
        output='screen',
    )

    robot4_controller = Node(
        package='burger_robot',
        executable='controller5_holo_pose1',
        name='robot4_controller',
        parameters=[{'robot_name': 'robot4'}],
        output='screen',
    )

    robot5_controller = Node(
        package='burger_robot',
        executable='controller5_holo_pose1',
        name='robot5_controller',
        parameters=[{'robot_name': 'robot5'}],
        output='screen',
    )

    robot6_controller = Node(
        package='burger_robot',
        executable='controller5_holo_pose1',
        name='robot6_controller',
        parameters=[{'robot_name': 'robot6'}],
        output='screen',
    )

    multi_robot_odom_offset = Node(
        package='robot_llm',
        executable='multi_robot_odom_offset',
        name='multi_robot_odom_offset',
        output='screen',
    )

    return LaunchDescription([
        robot1_controller,
        robot2_controller,
        robot3_controller,
        robot4_controller,
        robot5_controller,
        robot6_controller,
        multi_robot_odom_offset,

    ])
