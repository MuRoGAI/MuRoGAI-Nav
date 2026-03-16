from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    # robot_names = [
    #     "burger1",
    #     "burger2",
    #     "burger3",
    #     "waffle",
    #     "tb4_1",
    # ]

    # nodes = []

    # for name in robot_names:
    #     nodes.append(
    #         Node(
    #             package="burger_robot",
    #             executable="controller9",
    #             name=f"{name}_controller",
    #             output="screen",
    #             parameters=[{
    #                 "robot_name": name,
    #                 "kp_linear": 2.0,
    #                 "kp_angular": 3.0,
    #                 "kd_linear": 0.5,
    #                 "kd_angular": 0.3,
    #                 "max_lin_x": 0.22,
    #                 "max_lin_y": 0.0,
    #                 "max_ang_z": 2.84,
    #                 "max_lin_acc": 0.5,
    #                 "max_ang_acc": 3.0,
    #             }]
    #         )
    #     )

    burger1 = Node(
        package="burger_robot",
        executable="controller9",
        name=f"burger1_controller",
        output="screen",
        parameters=[{
            "robot_name": 'burger1',
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

    burger2 = Node(
        package="burger_robot",
        executable="controller9",
        name=f"burger2_controller",
        output="screen",
        parameters=[{
            "robot_name": 'burger2',
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

    burger3 = Node(
        package="burger_robot",
        executable="controller9",
        name=f"burger3_controller",
        output="screen",
        parameters=[{
            "robot_name": 'burger3',
            "kp_linear": 1.8,
            "kp_angular": 2.5,
            "kd_linear": 0.5,
            "kd_angular": 0.3,
            "ky": 3.5,
            "max_lin_x": 0.22,
            "max_lin_y": 0.0,
            "max_ang_z": 2.84,
            "max_lin_acc": 2.5,
            "max_ang_acc": 3.2,
        }]
    )


    waffle = Node(
        package="burger_robot",
        executable="controller9",
        name=f"waffle_controller",
        output="screen",
        parameters=[{
            "robot_name": 'waffle',
            "kp_linear": 2.3,
            "kp_angular": 1.8,
            "kd_linear": 0.5,
            "kd_angular": 0.3,
            "ky": 3.2,
            "max_lin_x": 0.26,
            "max_lin_y": 0.0,
            "max_ang_z": 1.82,
            "max_lin_acc": 2.5,
            "max_ang_acc": 3.2,
        }]
    )

    tb4_1 = Node(
        package="burger_robot",
        executable="controller9",
        name=f"tb4_1_controller",
        output="screen",
        parameters=[{
            "robot_name": 'tb4_1',
            "kp_linear": 2.6,
            "kp_angular": 3.0,
            "kd_linear": 0.5,
            "kd_angular": 0.3,
            "ky": 3.9,
            "max_lin_x": 0.31,
            "max_lin_y": 0.0,
            "max_ang_z": 1.9,
            "max_lin_acc": 0.9,
            "max_ang_acc": 2.0,
        }]
    )
    
    firebird = Node(
        package="burger_robot",
        executable="controller9",
        name=f"burger1_controller",
        output="screen",
        parameters=[{
            "robot_name": 'firebird',
            "kp_linear": 1.3,
            "kp_angular": 3.0,
            "kd_linear": 0.5,
            "kd_angular": 0.5,
            "ky": 3.3,
            "max_lin_x": 0.4,
            "max_lin_y": 0.0,
            "max_ang_z": 2.84,
            "max_lin_acc": 2.5,
            "max_ang_acc": 3.2,
        }]
    )
    
    return LaunchDescription([
        # burger1,
        burger2,
        # burger3,
        waffle,
        tb4_1,
        # firebird,
    ])
