from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    holonomic_robot1 = Node(
        package='yahboom_llm',
        executable='holonomic_position_controller_service',
        name='holonomic_position_controller_1',
        parameters=[{
            'namespace': 'r1',
        }],
        output='screen',
    )


    holonomic_robot2 = Node(
        package='yahboom_llm',
        executable='holonomic_position_controller_service',
        name='holonomic_position_controller_2',
        parameters=[{
            'namespace': 'r2',
        }],
        output='screen',        
    )


    drone_1 = Node(
        package='x3_uav_llm', 
        executable='drone_position_controller_service',
        name='drone_position_controller_service',
        parameters=[{
            'namespace': 'r3',
        }],
        output='screen',
    )

    return LaunchDescription([
        holonomic_robot1,
        holonomic_robot2,
        drone_1,


    ])