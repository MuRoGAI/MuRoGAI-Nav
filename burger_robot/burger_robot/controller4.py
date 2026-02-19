#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String

import json


class VelocityController(Node):

    def __init__(self):
        super().__init__('velocity_sequence_controller')

        # ----- Storage -----
        self.velocities = []
        self.index = 0
        self.start_motion = False

        # ----- Step duration -----
        self.step_time = 0.11   # seconds per velocity command
        self.elapsed_time = 0.0
        self.dt = 0.1

        # ----- Subscriber -----
        self.create_subscription(
            String,
            '/burger1/targets',
            self.vel_callback,
            10
        )

        # ----- Publisher -----
        self.cmd_pub = self.create_publisher(Twist, '/burger1/cmd_vel', 10)

        # ----- Timer -----
        self.create_timer(self.dt, self.control_loop)

        self.get_logger().info("Velocity Sequence Controller Ready")


    # ---------- Receive Velocity Arrays ----------
    def vel_callback(self, msg):

        data = json.loads(msg.data)

        vx = data["vx"]
        vy = data["vy"]

        self.velocities = list(zip(vx, vy))
        self.index = 0
        self.elapsed_time = 0.0
        self.start_motion = True

        self.get_logger().info(f"Received {len(self.velocities)} velocity steps")


    # ---------- Control Loop ----------
    def control_loop(self):

        if not self.start_motion:
            return

        if self.index >= len(self.velocities):
            self.stop_robot()
            return

        vx, vy = self.velocities[self.index]

        cmd = Twist()
        cmd.linear.x = vx
        cmd.linear.y = vy

        self.cmd_pub.publish(cmd)

        # ---- Time based switching ----
        self.elapsed_time += self.dt

        if self.elapsed_time >= self.step_time:
            self.index += 1
            self.elapsed_time = 0.0
            self.get_logger().info(f"Switching to step {self.index}")


    def stop_robot(self):
        self.cmd_pub.publish(Twist())


def main():
    rclpy.init()
    node = VelocityController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
