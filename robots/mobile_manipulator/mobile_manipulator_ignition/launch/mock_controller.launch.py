#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    pkg_share   = get_package_share_directory('lekiwi_description')
    control_pkg_share   = get_package_share_directory('lekiwi_control')

    xacro_file  = os.path.join(pkg_share, 'urdf', 'lekiwi.urdf.xacro')
    rviz_cfg    = os.path.join(pkg_share, 'rviz', 'display.rviz')

    controller_config = os.path.join(control_pkg_share, 'config', 'lekiwi_controllers.yaml')

    # ------------------------------------------------------------------
    # Launch arguments
    # ------------------------------------------------------------------
    declare_robot_name = DeclareLaunchArgument(
        'robot_name',
        default_value='',
        description='Robot namespace (empty → no namespace)'
    )
    robot_name = LaunchConfiguration('robot_name')

    # ------------------------------------------------------------------
    # Helpers (prefixes, namespace)
    # ------------------------------------------------------------------
    ns = robot_name  # ROS namespace for nodes

    # ------------------------------------------------------------------
    # robot_description (xacro → string)
    # ------------------------------------------------------------------
    robot_description_content = Command([
        'xacro ', xacro_file,
        ' robot_name:=', robot_name,
        ' use_ros2_control:=true',
        ' use_ignition:=false',
        ' use_hardware:=false'
    ])
    robot_description = ParameterValue(robot_description_content, value_type=str)

    # ------------------------------------------------------------------
    # 1. robot_state_publisher
    # ------------------------------------------------------------------
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace=ns,
        output='screen',
        parameters=[{
            'robot_description': robot_description,
        }]
    )

    # ------------------------------------------------------------------
    # 2. controller_manager (mock hardware)
    # ------------------------------------------------------------------
    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        namespace=ns,
        parameters=[
            {'robot_description': robot_description},
            controller_config
        ],
        output='both'
    )

    # ------------------------------------------------------------------
    # 3. spawn joint_state_broadcaster
    # ------------------------------------------------------------------
    # Use PythonExpression to build the controller_manager path dynamically
    controller_manager_name = PythonExpression([
        "'/controller_manager' if '", robot_name, "' == '' else '/", robot_name, "/controller_manager'"
    ])
    
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        namespace=ns,
        arguments=[
            'joint_state_broadcaster',
            '-c', controller_manager_name
        ],
        output='screen'
    )

    # ------------------------------------------------------------------
    # 4. spawn omni_wheel_controller (optional, uncomment to auto-start)
    # ------------------------------------------------------------------
    omni_wheel_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        namespace=ns,
        arguments=[
            'omni_wheel_controller',
            '-c', controller_manager_name
        ],
        output='screen'
    )

    # ------------------------------------------------------------------
    # 5. spawn arm_controller (optional, uncomment to auto-start)
    # ------------------------------------------------------------------
    arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        namespace=ns,
        arguments=[
            'arm_controller',
            '-c', controller_manager_name
        ],
        output='screen'
    )

    # ------------------------------------------------------------------
    # 6. RViz (optional)
    # ------------------------------------------------------------------
    rviz_args = ['-d', rviz_cfg] if os.path.exists(rviz_cfg) else []
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        namespace=ns,
        name='rviz2',
        output='screen',
        arguments=rviz_args,
        parameters=[{'use_sim_time': False}]
    )

    # ------------------------------------------------------------------
    # Launch description
    # ------------------------------------------------------------------
    return LaunchDescription([
        declare_robot_name,
        rsp,
        controller_manager,
        joint_state_broadcaster_spawner,
        omni_wheel_controller_spawner,
        arm_controller_spawner,
        rviz  
    ])