#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetLaunchConfiguration
from launch.substitutions import LaunchConfiguration, Command, PythonExpression, PathJoinSubstitution
from launch.actions import DeclareLaunchArgument, AppendEnvironmentVariable, IncludeLaunchDescription, OpaqueFunction, SetLaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
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


def bridge_topics(context, *args, **kwargs):
    """
    Render ros_gz_bridge_template.yaml by replacing {prefix} placeholders.
    - ROS names: leave relative (no leading '/'); node namespace will add <prefix>/...
      Keep global ones absolute ('/tf', '/clock') in the template so they stay global.
    - GZ names: keep absolute and include '{prefix}' where you really want it.
    """
    pkg_share_gazebo = kwargs['pkg_share_gazebo']
    prefix = LaunchConfiguration('prefix').perform(context) or ""
    # Sane normalized string for replacement (no trailing slash)
    name = prefix.rstrip('/')

    template_path = os.path.join(pkg_share_gazebo, 'config', 'ros_gz_bridge_template.yaml')

    # Decide output file name first
    file_name = f'ros_gz_bridge_{name}.yaml' if name else 'ros_gz_bridge.yaml'
    modified_config_path = os.path.join(pkg_share_gazebo, 'config', file_name)

    # Read template as text and do a simple string replace
    with open(template_path, 'r', encoding='utf-8') as f:
        txt = f.read()

    # Replace {prefix} placeholders.
    # NOTE: We DO NOT add slashes here. Put them in the template where needed.
    txt = txt.replace('{prefix}', name)
    # txt = txt.replace('{n_prefix}', 'x3')

    os.makedirs(os.path.dirname(modified_config_path), exist_ok=True)
    with open(modified_config_path, 'w', encoding='utf-8') as f:
        f.write(txt)

    # Return a LaunchConfiguration setter so downstream Nodes can use it
    return [SetLaunchConfiguration('bridge_config_file', modified_config_path)]


def generate_launch_description():

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    descrip_pkg_share   = get_package_share_directory('mobile_manipulator_description')
    control_pkg_share   = get_package_share_directory('mobile_manipulator_control')
    ignition_pkg_share  = get_package_share_directory('mobile_manipulator_ignition')

    xacro_file  = os.path.join(descrip_pkg_share, 'urdf', 'mobile_manipulator.urdf.xacro')

    # ------------------------------------------------------------------
    # Launch arguments
    # ------------------------------------------------------------------
    use_sim_time     = LaunchConfiguration('use_sim_time')
    robot_name       = LaunchConfiguration('robot_name') 
    prefix           = LaunchConfiguration('prefix')
    use_ignition     = LaunchConfiguration('use_ignition')
    use_hardware     = LaunchConfiguration('use_hardware')
    use_mock_hardware     = LaunchConfiguration('use_mock_hardware')
    use_plugin = LaunchConfiguration('use_plugin')
    use_ros2_control = LaunchConfiguration('use_ros2_control')
    controller_config_file = LaunchConfiguration('controller_config_file')
    use_rviz         = LaunchConfiguration('use_rviz')
    headless = LaunchConfiguration('headless')
    world_file = LaunchConfiguration('world_file')
    spawn_x = LaunchConfiguration('spawn_x')
    spawn_y = LaunchConfiguration('spawn_y')
    spawn_z = LaunchConfiguration('spawn_z')
    spawn_roll = LaunchConfiguration('spawn_roll')
    spawn_pitch = LaunchConfiguration('spawn_pitch')
    spawn_yaw = LaunchConfiguration('spawn_yaw')

    default_world_file = PathJoinSubstitution([
        ignition_pkg_share,
        'worlds',
        'empty.world'
    ])

    declare_use_sim_time = DeclareLaunchArgument('use_sim_time', default_value='true', description='whether are we using simulation or not' )
    declare_robot_name = DeclareLaunchArgument('robot_name', default_value='mobile_manipulator', description='Robot namespace (empty → no namespace)' )
    declare_prefix_arg = DeclareLaunchArgument('prefix', default_value='r1', description='')
    use_ignition_arg = DeclareLaunchArgument('use_ignition', default_value='true')
    use_hardware_arg = DeclareLaunchArgument('use_hardware', default_value='false')
    use_mock_hardware_arg = DeclareLaunchArgument('use_mock_hardware', default_value='false')
    use_plugin_arg = DeclareLaunchArgument('use_plugin', default_value='false')
    use_ros2_control_arg = DeclareLaunchArgument('use_ros2_control', default_value='true')
    declare_use_rviz_arg = DeclareLaunchArgument('use_rviz', default_value='true')
    headless_arg = DeclareLaunchArgument('headless', default_value='false')
    declare_world_file = DeclareLaunchArgument(name='world_file', default_value=default_world_file, description='World file name (e.g., empty.world, house.world, pick_and_place_demo.world)' )
    declare_x_cmd = DeclareLaunchArgument('spawn_x', default_value='0.0', description='x component of initial position, meters')
    declare_y_cmd = DeclareLaunchArgument('spawn_y', default_value='1.5', description='y component of initial position, meters')
    declare_z_cmd = DeclareLaunchArgument('spawn_z', default_value='0.05', description='z component of initial position, meters')
    declare_roll_cmd = DeclareLaunchArgument('spawn_roll', default_value='0.0', description='roll angle of initial orientation, radians')
    declare_pitch_cmd = DeclareLaunchArgument('spawn_pitch', default_value='0.0', description='pitch angle of initial orientation, radians')
    declare_yaw_cmd = DeclareLaunchArgument('spawn_yaw', default_value='0.0', description='yaw angle of initial orientation, radians')



    # ------------------------------------------------------------------
    # robot_description (xacro → string)
    # ------------------------------------------------------------------
    robot_description_content = Command([
        'xacro ', xacro_file,
        ' robot_name:=', robot_name,
        ' prefix:=', prefix,
        ' use_ignition:=', use_ignition,
        ' use_ros2_control:=', use_ros2_control,
        ' use_hardware:=', use_hardware,
        ' use_mock_hardware:=', use_mock_hardware,
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


    model_name = PythonExpression([
        "'", robot_name, "' if '", prefix, "' == '' else '", prefix, "'"
    ])

    robot_description_topic = PythonExpression([
        "'/robot_description' if '", prefix, "' == '' else '/", prefix, "/robot_description'"
    ])


    pkg_ros_gz_sim = FindPackageShare(package='ros_gz_sim').find('ros_gz_sim')

    start_gz_server_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': [ '-r -s -v2 ', world_file ],
            'on_exit_shutdown': 'true'
        }.items(),
        condition=IfCondition(use_ignition)
    )

    start_gz_client_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': '-g -v2'
        }.items(),
        condition=IfCondition(
            AndSubstitution(
                NotSubstitution(headless),
                use_ignition
            )
        )
    )


    start_gazebo_ros_spawner_cmd = Node(
        package='ros_gz_sim',
        executable='create',
        namespace=prefix,
        output='screen',
        arguments=[
            '-topic', robot_description_topic,
            '-name', model_name,
            '-allow_renaming', 'true',
            '-x', spawn_x,
            '-y', spawn_y,
            '-z', spawn_z,
            '-R', spawn_roll,
            '-P', spawn_pitch,
            '-Y', spawn_yaw
        ],
        condition=IfCondition(use_ignition),
    )

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

    bridge_config_action_gazebo = OpaqueFunction(
        function=bridge_topics,
        kwargs={'pkg_share_gazebo': ignition_pkg_share}
    )

    start_gazebo_ros_bridge_cmd = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        namespace=prefix,
        parameters=[{
            'config_file': LaunchConfiguration('bridge_config_file'),
            'use_sim_time': use_sim_time,
        }],
        output='screen',
        condition=IfCondition(use_ignition),
    )

    # ------------------------------------------------------------------
    # Generate controller config with prefix
    # ------------------------------------------------------------------
    bridge_config_action_controller = OpaqueFunction(
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
        condition=IfCondition(use_ros2_control),
        output='screen',
    )

    # ------------------------------------------------------------------
    # 4. spawn omni_wheel_controller (optional, uncomment to auto-start)
    # ------------------------------------------------------------------
    diff_drive_wheel_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        namespace=prefix,
        arguments=[
            'diff_drive_controller',
            '-c', controller_manager_name
        ],
        condition=IfCondition(use_ros2_control),
        output='screen',
    )

    # ------------------------------------------------------------------
    # 5. spawn arm_controller (optional, uncomment to auto-start)
    # ------------------------------------------------------------------
    arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        namespace=prefix,
        arguments=[
            'arm_controller',
            '-c', controller_manager_name
        ],
        condition=IfCondition(use_ros2_control),
        output='screen',
    )

    # ------------------------------------------------------------------
    # 6) front_steering_controller
    # ------------------------------------------------------------------
    front_steering_spawner = Node(
        package='controller_manager',
        executable='spawner',
        namespace=prefix,
        arguments=[
            'front_steering_controller',
            '-c', controller_manager_name
        ],
        condition=IfCondition(use_ros2_control),
        output='screen',
    )

    child_frame = PythonExpression([
        "'", prefix, "' + '/odom' if '", prefix, "' != '' else 'odom'"
    ])

    static_tf_world_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_world_to_prefix_odom',
        namespace=prefix,
        arguments=[
            # '--x', spawn_x, '--y', spawn_y, '--z', spawn_z,
            # '--roll', spawn_roll, '--pitch', spawn_pitch, '--yaw', spawn_yaw,
            '--x', '0.0',
            '--y', '0.0',
            '--z', '0.0',
            '--roll', '0.0',
            '--pitch', '0.0',
            '--yaw', '0.0',
            '--frame-id', 'world',
            '--child-frame-id', child_frame
        ],
        output='screen'
    )  

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
        use_hardware_arg,
        use_mock_hardware_arg,
        use_plugin_arg,
        headless_arg,
        declare_x_cmd,
        declare_y_cmd,
        declare_z_cmd,
        declare_roll_cmd,
        declare_pitch_cmd,
        declare_yaw_cmd,
        declare_world_file,

        rsp_node,

        bridge_config_action_gazebo,
        bridge_config_action_controller,

        start_gz_server_cmd,
        start_gz_client_cmd,
        start_gazebo_ros_spawner_cmd,

        ignition_no_namespace_bridge,
        start_gazebo_ros_bridge_cmd,

        controller_manager,

        joint_state_broadcaster_spawner,
        diff_drive_wheel_controller_spawner,
        arm_controller_spawner,
        front_steering_spawner,

        static_tf_world_to_odom,

        rviz  
    ])