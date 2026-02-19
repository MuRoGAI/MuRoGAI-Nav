from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    multi_robot_odom_offset = Node(
        package='robot_llm',
        executable='multi_robot_odom_offset',
        name='multi_robot_odom_offset',
        output='screen',
    )
    
    robot1_controller = Node(
        package='burger_robot',
        executable='controller7_diff_drive_pose2',
        name='robot1_controller',
        parameters=[{
            'robot_name': 'robot1',
            'config_file': 'DD1',
            # 'max_linear_vel': 0.7,
            # 'max_angular_vel': 1.8
        }],
        output='screen',
    )
    
    robot2_controller = Node(
        package='burger_robot',
        executable='controller5_holo_pose2',
        name='robot2_controller',
        parameters=[{
            'robot_name': 'robot2',
            'config_file': 'Holo1',
            # 'max_vel_x': 0.9,
            # 'max_vel_y': 0.9,
        }],
        output='log',
    )
    
    robot3_controller = Node(
        package='burger_robot',
        executable='controller7_diff_drive_pose2',
        name='robot3_controller',
        parameters=[{
            'robot_name': 'robot3',
            'config_file': 'DD2',
            # 'max_linear_vel': 0.7,
            # 'max_angular_vel': 1.8
        }],
        output='screen',
    )
    
    robot4_controller = Node(
        package='burger_robot',
        executable='controller7_diff_drive_pose2',
        name='robot4_controller',
        parameters=[{
            'robot_name': 'robot4',
            'config_file': 'HeteroForm_robot0_diff-drive',
            'max_linear_vel': 0.8,
            'max_angular_vel': 2.0
        }],
        output='screen',
    )
    
    robot5_controller = Node(
        package='burger_robot',
        executable='controller5_holo_pose2',
        name='robot5_controller',
        parameters=[{
            'robot_name': 'robot5',
            'config_file': 'HeteroForm_robot1_holonomic',
            'max_vel_x': 1.0,
            'max_vel_y': 1.0
        }],
        output='log',
    )
    
    robot6_controller = Node(
        package='burger_robot',
        executable='controller7_diff_drive_pose2',
        name='robot6_controller',
        parameters=[{
            'robot_name': 'robot6',
            'config_file': 'HeteroForm_robot2_diff-drive',
            'max_linear_vel': 0.8,
            'max_angular_vel': 2.0
        }],
        output='screen',
    )
    
    return LaunchDescription([
        # multi_robot_odom_offset,
        robot1_controller,
        robot2_controller,
        robot3_controller,
        # robot4_controller,
        # robot5_controller,
        # robot6_controller,
    ])