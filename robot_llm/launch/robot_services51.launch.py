from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    delivery_bot1_controller_node = Node(
        package="burger_robot",
        executable="controller9",
        name=f"delivery_bot1_controller",
        output="screen",
        parameters=[{
            "robot_name": 'delivery_bot1',
            "kp_linear": 7.2,
            "kp_angular": 3.9,
            "kd_linear": 0.5,
            "kd_angular": 0.3,
            "ky": 11.4,
            "max_lin_x": 0.22,
            "max_lin_y": 0.0,
            "max_ang_z": 2.84,
            "max_lin_acc": 2.5,
            "max_ang_acc": 3.2,
        }]
    )

    delivery_bot2_controller_node = Node(
        package="burger_robot",
        executable="controller9",
        name=f"delivery_bot2_controller",
        output="screen",
        parameters=[{
            "robot_name": 'delivery_bot2',
            "kp_linear": 2.7,
            "kp_angular": 3.6,
            "kd_linear": 0.5,
            "kd_angular": 0.5,
            "ky": 2.4,
            "max_lin_x": 0.22,
            "max_lin_y": 0.0,
            "max_ang_z": 2.84,
            "max_lin_acc": 2.5,
            "max_ang_acc": 3.2,
        }]
    )

    delivery_bot3_controller_node = Node(
        package="burger_robot",
        executable="controller9",
        name=f"delivery_bot3_controller",
        output="screen",
        parameters=[{
            "robot_name": 'delivery_bot3',
            "kp_linear": 2.7,
            "kp_angular": 3.6,
            "kd_linear": 0.5,
            "kd_angular": 0.5,
            "ky": 2.4,
            "max_lin_x": 0.22,
            "max_lin_y": 0.0,
            "max_ang_z": 2.84,
            "max_lin_acc": 2.5,
            "max_ang_acc": 3.2,
        }]
    )

    cleaning_bot_controller_node = Node(
        package="burger_robot",
        executable="controller9",
        name=f"cleaning_bot_controller",
        output="screen",
        parameters=[{
            "robot_name": 'cleaning_bot',
            "kp_linear": 2.7,
            "kp_angular": 3.6,
            "kd_linear": 0.5,
            "kd_angular": 0.5,
            "ky": 2.4,
            "max_lin_x": 0.22,
            "max_lin_y": 0.22,
            "max_ang_z": 2.84,
            "max_lin_acc": 2.5,
            "max_ang_acc": 3.2,
        }]
    )
 
    return LaunchDescription([
        delivery_bot1_controller_node,
        delivery_bot2_controller_node,
        delivery_bot3_controller_node,
        cleaning_bot_controller_node,
    ])
