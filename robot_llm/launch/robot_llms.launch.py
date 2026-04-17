from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    config_file = LaunchConfiguration('config_file')

    config_file_arg = DeclareLaunchArgument(
        'config_file', default_value='robot_config_restaurant2', description='Config file name'
    )
    
    delivery_bot1_node = Node(
        package='delivery_bot',
        executable='delivery_bot_llm',
        name='delivery_bot1_node',
        parameters=[{
            'robot_name': 'delivery_bot1',
            'config_file': config_file,
        }],
        output='screen',
    )

    delivery_bot2_node = Node(
        package='delivery_bot',
        executable='delivery_bot_llm',
        name='delivery_bot2_node',
        parameters=[{
            'robot_name': 'delivery_bot2',
            'config_file': config_file,
        }],
        output='screen',
    )

    delivery_bot3_node = Node(
        package='delivery_bot',
        executable='delivery_bot_llm',
        name='delivery_bot3_node',
        parameters=[{
            'robot_name': 'delivery_bot3',
            'config_file': config_file,
        }],
        output='screen',
    )

    cleaning_bot_node = Node(
        package='cleaning_bot',
        executable='cleaning_bot_llm',
        name='cleaning_bot_node',
        parameters=[{
            'robot_name': 'cleaning_bot',
            'config_file': config_file,
        }],
        output='screen',
    )

    drone_llm = Node(
        package='drone',
        executable='drone_llm',
        name='drone_node',
        parameters=[{
            'robot_name': 'drone'
        }],
        output='screen',
    )

    team1_node = Node(
        package='robot_llm',
        executable='team_llm_node',
        name='team1_node',
        parameters=[{
            'team_name': 'team1',
            'robot_names': ['delivery_bot1', 'delivery_bot2', 'delivery_bot3'],
            'config_file': config_file,
        }],
        output='screen',
    )

    return LaunchDescription([
        config_file_arg,

        delivery_bot1_node,
        delivery_bot2_node,
        delivery_bot3_node,
        cleaning_bot_node,
        drone_llm,
        team1_node,
    ])
