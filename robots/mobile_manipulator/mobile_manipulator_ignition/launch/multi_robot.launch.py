#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, 
    IncludeLaunchDescription, 
    TimerAction
)
from launch.substitutions import LaunchConfiguration, Command, PythonExpression, PathJoinSubstitution
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import AndSubstitution, NotSubstitution
from ament_index_python.packages import get_package_share_directory




def generate_launch_description():

    # ------------------------------------------------------------------
    # Package paths
    # ------------------------------------------------------------------
    descrip_pkg_share = get_package_share_directory('mobile_manipulator_description')
    control_pkg_share = get_package_share_directory('mobile_manipulator_control')
    ignition_pkg_share = get_package_share_directory('mobile_manipulator_ignition')

    xacro_file = os.path.join(descrip_pkg_share, 'urdf', 'mobile_manipulator.urdf.xacro')

    # ------------------------------------------------------------------
    # Launch configurations
    # ------------------------------------------------------------------
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_ignition = LaunchConfiguration('use_ignition')
    headless = LaunchConfiguration('headless')
    use_single_rviz = LaunchConfiguration('use_single_rviz')
    use_multi_rviz = LaunchConfiguration('use_multi_rviz')
    world_file = LaunchConfiguration('world_file')


    # ------------------------------------------------------------------
    # Default world file path
    # ------------------------------------------------------------------
    default_world_file = PathJoinSubstitution([
        ignition_pkg_share,
        'worlds',
        'empty2.world'
    ])

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )

    use_ignition_arg = DeclareLaunchArgument(
        'use_ignition',
        default_value='true',
        description='Use Gazebo Ignition simulation'
    )

    declare_use_single_rviz_arg = DeclareLaunchArgument(
        'use_single_rviz',
        default_value='true',
        description='Launch RViz'
    )

    declare_use_multi_rviz_arg = DeclareLaunchArgument(
        'use_multi_rviz',
        default_value='false',
        description='Launch RViz'
    )
        
    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run Gazebo in headless mode (no GUI)'
    )
    
    declare_world_file = DeclareLaunchArgument(
        'world_file',
        default_value=default_world_file,
        description='Path to world file'
    )




    # ------------------------------------------------------------------
    # Gazebo simulation launch
    # ------------------------------------------------------------------
    pkg_ros_gz_sim = FindPackageShare(package='ros_gz_sim').find('ros_gz_sim')

    start_gz_server_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': ['-r -s -v4 ', world_file],
            'on_exit_shutdown': 'true'
        }.items(),
        condition=IfCondition(use_ignition)
    )

    start_gz_client_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': '-g -v4'
        }.items(),
        condition=IfCondition(
            AndSubstitution(
                NotSubstitution(headless),
                use_ignition
            )
        )
    )


    # ------------------------------------------------------------------
    # Gazebo bridge for global topics (/clock, /tf)
    # ------------------------------------------------------------------
    ignition_no_namespace_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='clock_tf_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
        condition=IfCondition(use_ignition),
    )


    mobile_manipulator_launch = PathJoinSubstitution([
        ignition_pkg_share, 'launch', 'robot.launch.py'
    ])

    robot1_launch = TimerAction(
        period=5.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(mobile_manipulator_launch),
                launch_arguments={
                    'use_sim_time': use_sim_time,
                    'robot_name': 'mobile_manipulator',
                    'prefix': 'robot1',
                    'use_ignition': use_ignition,
                    'use_hardware': 'False',
                    'use_mock_hardware': 'False',
                    'use_plugin': 'False',
                    'use_ros2_control': 'False',
                    'use_plugin_control': 'True',
                    'spawn_x': '0.0',
                    'spawn_y': '1.0',
                    'spawn_z': '0.05',
                    'spawn_roll': '0.0',
                    'spawn_pitch': '0.0',
                    'spawn_yaw': '0.0',
                    'use_rviz': use_multi_rviz,
                }.items()
            )
        ]
    )


    robot2_launch = TimerAction(
        period=10.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(mobile_manipulator_launch),
                launch_arguments={
                    'use_sim_time': use_sim_time,
                    'robot_name': 'mobile_manipulator',
                    'prefix': 'robot2',
                    'use_ignition': use_ignition,
                    'use_hardware': 'False',
                    'use_mock_hardware': 'False',
                    'use_plugin': 'False',
                    'use_ros2_control': 'False',
                    'use_plugin_control': 'True',
                    'spawn_x': '0.0',
                    'spawn_y': '-1.0',
                    'spawn_z': '0.05',
                    'spawn_roll': '0.0',
                    'spawn_pitch': '0.0',
                    'spawn_yaw': '0.0',
                    'use_rviz': use_multi_rviz,
                }.items()
            )
        ]
    )

    # ------------------------------------------------------------------
    # RViz
    # ------------------------------------------------------------------

    rviz_config_file = PathJoinSubstitution([
        descrip_pkg_share,
        'rviz',
        'multi_robot.rviz'
    ])

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file],
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        condition=IfCondition(use_single_rviz)
    )

    # ------------------------------------------------------------------
    # Launch description
    # ------------------------------------------------------------------
    return LaunchDescription([

        declare_use_single_rviz_arg,
        declare_use_multi_rviz_arg,
        declare_use_sim_time,
        use_ignition_arg,
        headless_arg,
        declare_world_file,


        # Gazebo simulation
        start_gz_server_cmd,
        start_gz_client_cmd,

        # Gazebo bridges
        ignition_no_namespace_bridge,

        # Spawn robot
        robot1_launch,
        robot2_launch,


        # RViz
        rviz
    ])