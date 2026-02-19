#!/usr/bin/env python3

import os
import csv
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from path_planner_interface.msg import (
    RobotTrajectoryArray,
    RobotTrajectory,
    DiffDriveTrajectory,
    HoloTrajectory
)


class TrajectoryCSVPublisher(Node):

    def __init__(self):
        super().__init__('trajectory_csv_publisher')

        # --------------------------------------------------
        # Declare Parameters
        # --------------------------------------------------
        self.declare_parameter("robot_name", "tb4_1")
        self.declare_parameter("robot_type", "diff-drive")
        self.declare_parameter("package_name", "burger_robot")
        self.declare_parameter("dir_name", "trajectory_logs2")
        self.declare_parameter("file_name", "TeamBurger_robot0_diff-drive.csv")

        self.robot_name = self.get_parameter("robot_name").value
        self.robot_type = self.get_parameter("robot_type").value
        self.package_name = self.get_parameter("package_name").value
        self.dir_name = self.get_parameter("dir_name").value
        self.file_name = self.get_parameter("file_name").value

        # --------------------------------------------------
        # Publisher
        # --------------------------------------------------
        self.publisher = self.create_publisher(
            RobotTrajectoryArray,
            "/path_planner/paths",
            10
        )

        # Publish once after short delay
        self.timer = self.create_timer(1.0, self.publish_trajectory)

        self.get_logger().info("Trajectory CSV Publisher started.")

    # ============================================================
    # Main Publisher Function
    # ============================================================

    def publish_trajectory(self):

        # Stop timer after first call
        self.timer.cancel()

        # --------------------------------------------------
        # Locate CSV File
        # --------------------------------------------------
        pkg_share = get_package_share_directory(self.package_name)
        file_path = os.path.join(pkg_share, self.dir_name, self.file_name)

        if not os.path.exists(file_path):
            self.get_logger().error(f"CSV file not found: {file_path}")
            return

        self.get_logger().info(f"Reading CSV: {file_path}")

        time_list = []
        x_list = []
        y_list = []
        theta_list = []

        # --------------------------------------------------
        # Read CSV
        # --------------------------------------------------
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)

            for row in reader:
                time_list.append(float(row["time"]))
                x_list.append(float(row["x"]))
                y_list.append(float(row["y"]))

                if self.robot_type == "diff-drive":
                    theta_list.append(float(row["theta"]))

        # --------------------------------------------------
        # Create RobotTrajectory Message
        # --------------------------------------------------
        robot_traj_msg = RobotTrajectory()
        robot_traj_msg.robot_name = self.robot_name
        robot_traj_msg.robot_type = self.robot_type

        if self.robot_type == "diff-drive":

            diff_msg = DiffDriveTrajectory()
            diff_msg.time = time_list
            diff_msg.x = x_list
            diff_msg.y = y_list
            diff_msg.theta = theta_list

            robot_traj_msg.diff_drive_trajectories.append(diff_msg)

        elif self.robot_type == "holonomic":

            holo_msg = HoloTrajectory()
            holo_msg.time = time_list
            holo_msg.x = x_list
            holo_msg.y = y_list

            robot_traj_msg.holo_trajectories.append(holo_msg)

        else:
            self.get_logger().error("Invalid robot_type parameter.")
            return

        # --------------------------------------------------
        # Wrap in RobotTrajectoryArray
        # --------------------------------------------------
        array_msg = RobotTrajectoryArray()
        array_msg.robot_trajectories.append(robot_traj_msg)

        # --------------------------------------------------
        # Publish
        # --------------------------------------------------
        self.publisher.publish(array_msg)

        self.get_logger().info(
            f"Published trajectory for {self.robot_name} "
            f"({self.robot_type}) with {len(time_list)} points."
        )


# ============================================================
# Main
# ============================================================

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryCSVPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
