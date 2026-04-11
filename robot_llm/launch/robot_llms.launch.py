from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    delivery_bot1_node = Node(
        package='delivery_bot',
        executable='delivery_bot_llm',
        name='delivery_bot1_node',
        parameters=[{
            'robot_name': 'delivery_bot1',
            'robot_type': 'Differential Drive'
        }],
        output='screen',
    )

    delivery_bot2_node = Node(
        package='delivery_bot',
        executable='delivery_bot_llm',
        name='delivery_bot2_node',
        parameters=[{
            'robot_name': 'delivery_bot2',
            'robot_type': 'Differential Drive'
        }],
        output='screen',
    )

    delivery_bot3_node = Node(
        package='delivery_bot',
        executable='delivery_bot_llm',
        name='delivery_bot3_node',
        parameters=[{
            'robot_name': 'delivery_bot3',
            'robot_type': 'Holonomic Drive'
        }],
        output='screen',
    )

    cleaning_bot_node = Node(
        package='cleaning_bot',
        executable='cleaning_bot_llm',
        name='cleaning_bot_node',
        parameters=[{
            'robot_name': 'cleaning_bot',
            'robot_type': 'Holonomic Drive'
        }],
        output='screen',
    )

    # drone_llm = Node(
    #     package='drone',
    #     executable='drone_llm',
    #     name='drone_node',
    #     parameters=[{
    #         'robot_name': 'drone'
    #     }],
    #     output='screen',
    # )

    # team_node = Node(
    #     package='robot_llm',
    #     executable='team_llm_node3',
    #     name='team_node',
    #     output='screen',
    # )

    return LaunchDescription([
        delivery_bot1_node,
        delivery_bot2_node,
        delivery_bot3_node,
        cleaning_bot_node,
        # drone_llm,
        # team_node,
    ])
