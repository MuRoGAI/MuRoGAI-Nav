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
            'robot_name': 'burger1',
            'config_file': 'TeamBurger_robot0_diff-drive',
            'kp_linear': 7.3,
            'kd_linear': 1.9,
            'kp_angular': 3.9,
            'kd_angular': 0.9,
            'max_linear_vel': 0.22,
            'max_angular_vel': 2.84,

        }],
        output='screen',
    )
    
    robot2_controller = Node(
        package='burger_robot',
        executable='controller7_diff_drive_pose2',
        name='robot2_controller',
        parameters=[{
            'robot_name': 'burger2',
            'config_file': 'TeamBurger_robot1_diff-drive',
            'kp_linear': 7.3,
            'kd_linear': 0.9,
            'kp_angular': 0.9,
            'kd_angular': 0.9,
            'max_linear_vel': 0.22,
            'max_angular_vel': 2.84,
        }],
        output='log',
    )
    
    robot3_controller = Node(
        package='burger_robot',
        executable='controller7_diff_drive_pose2',
        name='robot3_controller',
        parameters=[{
            'robot_name': 'burger3',
            'config_file': 'TeamBurger_robot2_diff-drive',
            'kp_linear': 7.3,
            'kd_linear': 0.9,
            'kp_angular': 0.9,
            'kd_angular': 0.9,
            'max_linear_vel': 0.22,
            'max_angular_vel': 2.84,
        }],
        output='screen',
    )
    
    robot4_controller = Node(
        package='burger_robot',
        executable='controller7_diff_drive_pose2',
        name='robot4_controller',
        parameters=[{
            'robot_name': 'waffle',
            'config_file': 'Waffle',
            'kp_linear': 7.3,
            'kd_linear': 0.9,
            'kp_angular': 0.9,
            'kd_angular': 0.9,
            'max_linear_vel': 0.22,
            'max_angular_vel': 2.84,
        }],
        output='screen',
    )
    
    robot5_controller = Node(
        package='burger_robot',
        executable='controller7_diff_drive_pose2',
        name='robot5_controller',
        parameters=[{
            'robot_name': 'tb4_1',
            'config_file': 'TB4',
            'max_vel_x': 1.0,
            'max_vel_y': 1.0,

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
        # robot2_controller,
        # robot3_controller,
        # robot4_controller,
        # robot5_controller,
        # robot6_controller,
    ])