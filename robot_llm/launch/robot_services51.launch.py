from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    delivery_bot1_controller_node = Node(
        package="burger_robot",
        executable="controller9",
        name="delivery_bot1_controller",
        output="screen",
        parameters=[{
            "robot_name": 'delivery_bot1',
            "kp_linear": 13.2,
            "kp_angular": 4.3,
            "kd_linear": 0.5,
            "kd_angular": 0.3,
            "ky": 14.4,
            "max_lin_x": 0.25,
            "max_lin_y": 0.0,
            "max_ang_z": 0.4,
            "max_lin_acc": 2.5,
            "max_ang_acc": 2.5,
        }]
    )

    delivery_bot2_controller_node = Node(
        package="burger_robot",
        executable="controller9",
        name="delivery_bot2_controller",
        output="screen",
        parameters=[{
            "robot_name": 'delivery_bot2',
            "kp_linear": 13.2,
            "kp_angular": 4.3,
            "kd_linear": 0.5,
            "kd_angular": 0.3,
            "ky": 14.4,
            "max_lin_x": 0.25,
            "max_lin_y": 0.0,
            "max_ang_z": 0.4,
            "max_lin_acc": 2.5,
            "max_ang_acc": 2.5,
        }]
    )

    delivery_bot3_controller_node = Node(
        package="burger_robot",
        executable="controller9",
        name="delivery_bot3_controller",
        output="screen",
        parameters=[{
            "robot_name": 'delivery_bot3',
            "kp_linear": 32.3,
            "kp_angular": 27.6,
            "kd_linear": 3.53,
            "kd_angular": 2.21,
            "ky": 23.3,
            "max_lin_x": 0.25,
            "max_lin_y": 0.25,
            "max_ang_z": 0.4,
            "max_lin_acc": 2.5,
            "max_ang_acc": 2.5,
        }]
    )

    cleaning_bot_controller_node = Node(
        package="burger_robot",
        executable="controller9",
        name="cleaning_bot_controller",
        output="screen",
        parameters=[{
            "robot_name": 'cleaning_bot',
            "kp_linear": 32.3,
            "kp_angular": 27.6,
            "kd_linear": 3.53,
            "kd_angular": 2.21,
            "ky": 23.3,
            "max_lin_x": 0.25,
            "max_lin_y": 0.25,
            "max_ang_z": 0.4,
            "max_lin_acc": 2.5,
            "max_ang_acc": 2.5,
        }]
    )
 
    return LaunchDescription([
        delivery_bot1_controller_node,
        delivery_bot2_controller_node,
        delivery_bot3_controller_node,
        cleaning_bot_controller_node,
    ])
