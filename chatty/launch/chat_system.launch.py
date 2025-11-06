from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # -------------------- Launch Arguments --------------------
    model_arg = DeclareLaunchArgument(
        'model', default_value='4', description='Model number'
    )

    student_arg = DeclareLaunchArgument(
        'student', default_value='0', description='Student number'
    )

    config_file_arg = DeclareLaunchArgument(
        'config_file', default_value='robot_config_assmble_help', description='Config file name'
    )
    # use_audio_arg = DeclareLaunchArgument(
    #     'use_audio',
    #     default_value='false',
    #     description="Whether to use audio input (true/false)"
    # )

    # -------------------- Substitutions --------------------
    model = LaunchConfiguration('model')
    student = LaunchConfiguration('student')
    config_file = LaunchConfiguration('config_file')
    # use_audio = LaunchConfiguration('use_audio')

    # -------------------- Node Definitions --------------------
    chat_gui = Node(
        package='chatty',
        executable='chat_gui',
        name='chat_gui',
        output='screen'
    )

    chat_manager = Node(
        package='chatty',
        executable='chat_manager',
        name='chat_manager',
        parameters=[
            {'config_file': config_file},
            {'student': student}
        ],

        output='screen'
    )

    task_manager = Node(
        package='chatty',
        executable='task_manager',
        name='task_manager',
        parameters=[
            {'model': model},
            {'config_file': config_file}
        ],
        output='screen'
    )

    speak_ = Node(
        package='chatty',
        executable='speak',
        name='tts_speaker',
        output='screen'
    )

    time_ = Node(
        package='chatty',
        executable='time',
        name='time_publisher',
        output='screen'  
    )

    # -------------------- Nested Launch (audio, conditional) --------------------
    # chatty_share = FindPackageShare('chatty')
    # audio_whisper_launch = PathJoinSubstitution([
    #     chatty_share, 'launch', 'audio_convertor.launch.py'
    # ])

    # include_audio_convertor = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(audio_whisper_launch),
    #     # Run only if use_audio == "true"
    #     condition=IfCondition(
    #         PythonExpression(["'", use_audio, "' == 'true'"])
    #     )
    # )

    # -------------------- Return LaunchDescription --------------------
    return LaunchDescription([
        model_arg,
        student_arg,
        config_file_arg,
        # use_audio_arg,

        chat_gui,
        chat_manager,
        task_manager,
        # speak_,
        # include_audio_convertor,
        time_
    ])

