#!/usr/bin/env python3

import os
import csv
import json
import socket
import struct
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory


class TrajectoryTCPPublisher(Node):

    def __init__(self):
        super().__init__('trajectory_tcp_publisher')

        # --------------------------------------------------
        # Declare Parameters
        # --------------------------------------------------
        self.declare_parameter("robot_name", "tb4_1")
        self.declare_parameter("robot_type", "diff-drive")
        self.declare_parameter("package_name", "burger_robot")
        self.declare_parameter("dir_name", "trajectory_logs2")
        self.declare_parameter("file_name", "TeamBurger_robot0_diff-drive.csv")
        self.declare_parameter("tcp_host", "0.0.0.0")   # listen on all interfaces
        self.declare_parameter("tcp_port", 5000)

        self.robot_name = self.get_parameter("robot_name").value
        self.robot_type = self.get_parameter("robot_type").value
        self.package_name = self.get_parameter("package_name").value
        self.dir_name = self.get_parameter("dir_name").value
        self.file_name = self.get_parameter("file_name").value
        self.tcp_host = self.get_parameter("tcp_host").value
        self.tcp_port = self.get_parameter("tcp_port").value

        # Publish once after short delay
        self.timer = self.create_timer(1.0, self.publish_trajectory)

        self.get_logger().info("Trajectory TCP Publisher started.")

    # ============================================================
    # Build Payload
    # ============================================================

    def build_payload(self, time_list, x_list, y_list, theta_list):
        """Serialize trajectory to JSON bytes."""
        data = {
            "robot_name": self.robot_name,
            "robot_type": self.robot_type,
            "trajectory": {
                "time":  time_list,
                "x":     x_list,
                "y":     y_list,
            }
        }
        if self.robot_type == "diff-drive":
            data["trajectory"]["theta"] = theta_list

        return json.dumps(data).encode("utf-8")

    # ============================================================
    # Main Publisher Function
    # ============================================================

    def publish_trajectory(self):

        self.timer.cancel()

        # --------------------------------------------------
        # Locate & Read CSV
        # --------------------------------------------------
        pkg_share = get_package_share_directory(self.package_name)
        file_path = os.path.join(pkg_share, self.dir_name, self.file_name)

        if not os.path.exists(file_path):
            self.get_logger().error(f"CSV file not found: {file_path}")
            return

        self.get_logger().info(f"Reading CSV: {file_path}")

        time_list, x_list, y_list, theta_list = [], [], [], []

        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                time_list.append(float(row["time"]))
                x_list.append(float(row["x"]))
                y_list.append(float(row["y"]))
                if self.robot_type == "diff-drive":
                    theta_list.append(float(row["theta"]))

        if self.robot_type not in ("diff-drive", "holonomic"):
            self.get_logger().error("Invalid robot_type parameter.")
            return

        # --------------------------------------------------
        # Serialize
        # --------------------------------------------------
        payload = self.build_payload(time_list, x_list, y_list, theta_list)

        # --------------------------------------------------
        # Send over TCP
        #
        # Protocol:
        #   [4 bytes big-endian uint32 = payload length] [payload bytes]
        #
        # The length prefix lets the receiver know exactly how many
        # bytes to read before trying to decode JSON.
        # --------------------------------------------------
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((self.tcp_host, self.tcp_port))
        server_sock.listen(1)

        self.get_logger().info(
            f"Waiting for a TCP client on {self.tcp_host}:{self.tcp_port} ..."
        )

        conn, addr = server_sock.accept()
        self.get_logger().info(f"Client connected: {addr}")

        with conn:
            # Send 4-byte length header then payload
            header = struct.pack(">I", len(payload))
            conn.sendall(header + payload)

        server_sock.close()

        self.get_logger().info(
            f"Sent trajectory for '{self.robot_name}' "
            f"({self.robot_type}) with {len(time_list)} points to {addr}."
        )


# ============================================================
# Main
# ============================================================

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryTCPPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()