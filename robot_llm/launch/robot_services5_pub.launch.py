from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    robots = [
        {
            "name": "burger1",
            "type": "diff-drive",
            "file": "HeteroForm1_robot0_diff-drive.csv"
        },
        {
            "name": "burger2",
            "type": "diff-drive",
            "file": "HeteroForm1_robot1_diff-drive.csv"
        },
        {
            "name": "burger3",
            "type": "diff-drive",
            "file": "HeteroForm1_robot2_diff-drive.csv"
        },
        {
            "name": "waffle",
            "type": "diff-drive",
            "file": "waffle.csv"
        },
        {
            "name": "tb4_1",
            "type": "diff-drive",
            "file": "HeteroForm2_robot0_diff-drive.csv"
        },
        {
            "name": "firebird",
            "type": "diff-drive",
            "file": "HeteroForm2_robot1_diff-drive.csv"
        },
    ]

    nodes = []

    for robot in robots:

        nodes.append(
            Node(
                package="burger_robot",
                executable="controller9_pub",
                name=f"{robot['name']}_publisher",
                output="screen",
                parameters=[{
                    "robot_name": robot["name"],
                    "robot_type": robot["type"],
                    "package_name": "burger_robot",
                    "dir_name": "trajectory_logs_1",
                    "file_name": robot["file"]
                }]
            )
        )

    node1 = Node(
        package="burger_robot",
        executable="controller9_pub1",
        name="go2_path_publisher",
        output="screen",
        parameters=[{    
            "robot_name": "go2",
            "robot_type": "holonomic",
            "package_name": "burger_robot",
            "dir_name": "trajectory_logs",
            "file_name": "DD3.csv",
            "tcp_host": "0.0.0.0",
            "tcp_port": 5001,
        }]
    )

    # nodes.append(node1)

    return LaunchDescription(nodes)
