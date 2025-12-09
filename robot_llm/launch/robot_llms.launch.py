from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    burger_robot = Node(
        package='turtlebot3_llm',
        executable='burger_llm',
        name='burger_llm_node',
        parameters=[{
            'robot_name': 'burger',
        }],
        output='screen'
    )

    x3_uav_robot = Node(
        package='x3_uav_llm', 
        executable='x3_uav_llm',
        name='x3_uav_llm_node',
        parameters=[{
            'robot_name': 'x3_uav',
        }],
        output='screen'
    )

    yahboom_robot = Node(
        package='yahboom_llm',
        executable='yahboom_llm',
        name='yahboom_llm_node',
        parameters=[{
            'robot_name': 'yahboom',
        }]
    )

    return LaunchDescription([
        burger_robot,
        x3_uav_robot,
        yahboom_robot,

    ])