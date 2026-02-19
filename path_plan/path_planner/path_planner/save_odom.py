#!/usr/bin/env python3

import csv
import math
from pathlib import Path

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, ReliabilityPolicy


def quaternion_to_yaw(qx, qy, qz, qw):
    """Convert quaternion to yaw"""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class MultiRobotLogger(Node):

    def __init__(self):
        super().__init__('multi_robot_logger')

        self.robots = [
            "burger1",
            "burger2",
            "burger3",
            "waffle",
            "tb4_1",
        ]

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        # Store latest cmd_vel for each robot
        self.last_cmd = {robot: (0.0, 0.0, 0.0) for robot in self.robots}

        self.files = {}
        self.writers = {}

        save_dir = Path("odom_logs_1")
        save_dir.mkdir(exist_ok=True)

        for robot in self.robots:

            file_path = save_dir / f"{robot}_log.csv"
            f = open(file_path, "w", newline="")
            writer = csv.writer(f)

            #  Required CSV Format
            writer.writerow([
                "time",
                "x",
                "y",
                "yaw",
                "linx",
                "liny",
                "angz"
            ])

            self.files[robot] = f
            self.writers[robot] = writer

            # -------- ODOM SUB --------
            if robot == 'tb4_1':
                self.create_subscription(
                    Odometry,
                    f"/{robot}/odom_world",
                    lambda msg, r=robot: self.odom_callback(msg, r),
                    qos_profile
                )

                self.create_subscription(
                    Twist,
                    f"/{robot}/cmd_vel",
                    lambda msg, r=robot: self.cmd_callback(msg, r),
                    qos_profile
                )
            else:
                self.create_subscription(
                    Odometry,
                    f"/{robot}/odom_world",
                    lambda msg, r=robot: self.odom_callback(msg, r),
                    10
                )

                self.create_subscription(
                    Twist,
                    f"/{robot}/cmd_vel",
                    lambda msg, r=robot: self.cmd_callback(msg, r),
                    10
                )

            self.get_logger().info(f"Subscribed to /{robot}/odom and /{robot}/cmd_vel")

    # -------------------------------------------------
    def cmd_callback(self, msg, robot):
        """Store latest cmd_vel"""
        self.last_cmd[robot] = (
            msg.linear.x,
            msg.linear.y,
            msg.angular.z
        )

    # -------------------------------------------------
    def odom_callback(self, msg, robot):

        # Time
        t = self.get_clock().now().seconds_nanoseconds()


        # Position
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Orientation → yaw
        q = msg.pose.pose.orientation
        yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)

        # Latest cmd_vel
        linx, liny, angz = self.last_cmd[robot]

        # Save row
        self.writers[robot].writerow([
            t,
            x,
            y,
            yaw,
            linx,
            liny,
            angz
        ])

    # -------------------------------------------------
    def destroy_node(self):
        for f in self.files.values():
            f.close()

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = MultiRobotLogger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
