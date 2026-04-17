from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    
    planner_request_rciever_node = Node(
        package='path_planner',
        executable='path_request_reciever',
        name='planner_request_reciever_node',
        output='screen',
    )

    path_planer_node = Node(
        package='path_planner',
        executable='path_planner_node4',
        name='path_planer_node',
        output='screen',
    )

    path_planer_writer_node = Node(
        package='path_planner',
        executable='path_writer',
        name='planner_output_writer_node',
        output='screen',
    )


    return LaunchDescription([
        planner_request_rciever_node,
        # path_planer_node,
        path_planer_writer_node


    ])
