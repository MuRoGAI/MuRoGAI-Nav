#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetLaunchConfiguration
from launch.substitutions import LaunchConfiguration, Command, PythonExpression, PathJoinSubstitution
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import AndSubstitution, NotSubstitution
from ament_index_python.packages import get_package_share_directory


def controller_config(context, *args, **kwargs):

    control_pkg_share = kwargs['control_pkg_share']
    prefix = LaunchConfiguration('prefix').perform(context) or ""
    # Sane normalized string for replacement (no trailing slash)
    name = prefix.rstrip('/')

    template_path = os.path.join(control_pkg_share, 'config', 'ros2_controllers_template.yaml')

    # Decide output file name first
    file_name = f'{name}_ros2_controllers.yaml' if name else 'ros2_controllers.yaml'
    modified_config_path = os.path.join(control_pkg_share, 'config', file_name)

    # Read template as text and do a simple string replace
    with open(template_path, 'r', encoding='utf-8') as f:
        txt = f.read()

    txt = txt.replace('${prefix}', name + '/' if name else '')

    os.makedirs(os.path.dirname(modified_config_path), exist_ok=True)
    with open(modified_config_path, 'w', encoding='utf-8') as f:
        f.write(txt)

    return [SetLaunchConfiguration('controller_config_file', modified_config_path)]




def generate_launch_description():

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    descrip_pkg_share   = get_package_share_directory('mobile_manipulator_description')
    control_pkg_share   = get_package_share_directory('mobile_manipulator_control')

    xacro_file  = os.path.join(descrip_pkg_share, 'urdf', 'mobile_manipulator.urdf.xacro')

    # ------------------------------------------------------------------
    # Launch arguments
    # ------------------------------------------------------------------
    robot_name             = LaunchConfiguration('robot_name') 
    prefix                 = LaunchConfiguration('prefix')
    use_sim_time           = LaunchConfiguration('use_sim_time')
    use_rviz               = LaunchConfiguration('use_rviz')
    use_ros2_control       = LaunchConfiguration('use_ros2_control')
    use_ignition           = LaunchConfiguration('use_ignition')
    use_mock_hardware      = LaunchConfiguration('use_mock_hardware')
    controller_config_file = LaunchConfiguration('controller_config_file')

    declare_robot_name = DeclareLaunchArgument(
        'robot_name',
        default_value='mobile_manipulator',
        description='Robot namespace (empty → no namespace)'
    )

    declare_prefix_arg = DeclareLaunchArgument(
        'prefix',
        default_value='',
        description='namespace'
    )

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='whether are we using simulation or not'
    )

    declare_use_rviz_arg = DeclareLaunchArgument('use_rviz', default_value='true')

    use_ros2_control_arg = DeclareLaunchArgument('use_ros2_control', default_value='true')

    use_ignition_arg = DeclareLaunchArgument('use_ignition', default_value='false')

    use_mock_hardware_arg = DeclareLaunchArgument('use_mock_hardware', default_value='true')

    # ------------------------------------------------------------------
    # robot_description (xacro → string)
    # ------------------------------------------------------------------
    robot_description_content = Command([
        'xacro ', xacro_file,
        ' robot_name:=', robot_name,
        ' prefix:=', prefix,
        ' use_ignition:=', use_ignition,
        ' use_ros2_control:=', use_ros2_control,
        ' use_mock_hardware:=', use_mock_hardware,
        ' use_hardware:=', 'False',
    ])

    robot_description = ParameterValue(robot_description_content, value_type=str)

    # ------------------------------------------------------------------
    # 1. robot_state_publisher
    # ------------------------------------------------------------------
    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace=prefix,
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': use_sim_time
        }],
    )


    # ------------------------------------------------------------------
    # Generate controller config with prefix
    # ------------------------------------------------------------------
    bridge_config_action = OpaqueFunction(
        function=controller_config, 
        kwargs={'control_pkg_share': control_pkg_share}
    )

    # ------------------------------------------------------------------
    # 2. controller_manager (mock hardware)
    # ------------------------------------------------------------------
    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        namespace=prefix,
        parameters=[
            controller_config_file,                            # your ros2_controllers.yaml
            { 'robot_description': robot_description }    # inline (must be a dict)
        ],
        condition=IfCondition(
            AndSubstitution(
                NotSubstitution(use_ignition),
                use_ros2_control
            )
        ),
        output='both',
    )

    # ------------------------------------------------------------------
    # 3. spawn joint_state_broadcaster
    # ------------------------------------------------------------------
    # Use PythonExpression to build the controller_manager path dynamically
    controller_manager_name = PythonExpression([
        "'/controller_manager' if '", prefix, "' == '' else '/", prefix, "/controller_manager'"
    ])
    
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        namespace=prefix,
        arguments=[
            'joint_state_broadcaster',
            '-c', controller_manager_name
        ],
        output='screen'
    )

    # ------------------------------------------------------------------
    # 4. spawn diff_drive_controller (optional, uncomment to auto-start)
    # ------------------------------------------------------------------
    diff_drive_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        namespace=prefix,
        arguments=[
            'diff_drive_controller',
            '-c', controller_manager_name
        ],
        output='screen'
    )

    # ------------------------------------------------------------------
    # 5. spawn arm_controller (optional, uncomment to auto-start)
    # ------------------------------------------------------------------
    # arm_controller_spawner = Node(
    #     package='controller_manager',
    #     executable='spawner',
    #     namespace=prefix,
    #     arguments=[
    #         'arm_controller',
    #         '-c', controller_manager_name
    #     ],
    #     output='screen'
    # )

    # ------------------------------------------------------------------
    # 6) front_steering_controller
    # ------------------------------------------------------------------
    # front_steering_spawner = Node(
    #     package='controller_manager',
    #     executable='spawner',
    #     namespace=prefix,
    #     arguments=[
    #         'front_steering_controller',
    #         '-c', controller_manager_name
    #     ],
    #     output='screen',
    # )

    # ------------------------------------------------------------------
    # 7. RViz (optional)
    # ------------------------------------------------------------------
    rviz_file = PythonExpression([
        "'display.rviz' if ", "'", prefix, "'", " == '' else ", "'", prefix, "'", " + '.rviz'"
    ])

    rviz_config = PathJoinSubstitution([
        descrip_pkg_share,
        'rviz',
        rviz_file
    ])
  
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        namespace=prefix,
        output='log',
        arguments=['-d', rviz_config],
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        condition=IfCondition(use_rviz)
    )

    # ------------------------------------------------------------------
    # Launch description
    # ------------------------------------------------------------------
    return LaunchDescription([
        declare_robot_name,
        declare_prefix_arg,
        declare_use_sim_time,
        declare_use_rviz_arg,
        use_ros2_control_arg,
        use_ignition_arg,
        use_mock_hardware_arg,

        rsp_node,

        bridge_config_action,

        controller_manager,

        joint_state_broadcaster_spawner,
        diff_drive_controller_spawner,
        # arm_controller_spawner,
        # front_steering_spawner,

        rviz  
    ])