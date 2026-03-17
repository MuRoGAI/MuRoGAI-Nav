from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description():


    burger1 = Node(
        package='burger_robot',
        executable='robot_test',
        name='burger1_test',
        output='screen',
        parameters=[{
            'robot': 'burger1',
        }]
    )

    burger2 = Node(
        package='burger_robot',
        executable='robot_test',
        name='burger2_test',
        output='screen',
        parameters=[{
            'robot': 'burger2',
        }]
    )

    burger3 = Node(
        package='burger_robot',
        executable='robot_test',
        name='burger3_test',
        output='screen',
        parameters=[{
            'robot': 'burger3',
        }]
    )

    waffle = Node(
        package='burger_robot',
        executable='robot_test',
        name='waffle_test',
        output='screen',
        parameters=[{
            'robot': 'waffle',
        }]
    )

    tb4_1 = Node(
        package='burger_robot',
        executable='robot_test',
        name='tb4_1_test',
        output='screen',
        parameters=[{
            'robot': 'tb4_1',
        }]
    )

    firebird = Node(
        package='burger_robot',
        executable='robot_test',
        name='firebird_test',
        output='screen',
        parameters=[{
            'robot': 'firebird',
        }]
    )

    return LaunchDescription([
        burger1,
        burger2,
        burger3,
        waffle,
        # tb4_1,
        firebird
    ])