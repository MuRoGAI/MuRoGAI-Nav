from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    robot1_node = Node(
        package='burger_robot',
        executable='burger_robot_llm',
        name='robot1_node',
        parameters=[{'robot_name': 'robot1'}],
        output='screen',
    )

    robot2_node = Node(
        package='burger_robot',
        executable='burger_robot_llm',
        name='robot2_node',
        parameters=[{'robot_name': 'robot2'}],
        output='screen',
    )

    robot3_node = Node(
        package='burger_robot',
        executable='burger_robot_llm',
        name='robot3_node',
        parameters=[{'robot_name': 'robot3'}],
        output='screen',
    )

    robot4_node = Node(
        package='yahboom_llm',
        executable='yahboom_llm',
        name='robot4_node',
        parameters=[{'robot_name': 'robot4'}],
        output='screen',
    )

    robot5_node = Node(
        package='yahboom_llm',
        executable='yahboom_llm',
        name='robot5_node',
        parameters=[{'robot_name': 'robot5'}],
        output='screen',
    )

    robot6_node = Node(
        package='yahboom_llm',
        executable='yahboom_llm',
        name='robot6_node',
        parameters=[{'robot_name': 'robot6'}],
        output='screen',
    )

    drone_llm = Node(
        package='drone',
        executable='drone_llm',
        name='drone_node',
        parameters=[{'robot_name': 'drone'}],
        output='screen',
    )

    team_node = Node(
        package='robot_llm',
        executable='team_llm_node3',
        name='team_node',
        output='screen',
    )

    return LaunchDescription([
        robot1_node,
        robot2_node,
        robot3_node,
        robot4_node,
        robot5_node,
        robot6_node,
        drone_llm,
        team_node,
    ])
