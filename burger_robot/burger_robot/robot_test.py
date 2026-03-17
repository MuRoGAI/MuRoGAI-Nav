#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
from rclpy.qos import QoSProfile, ReliabilityPolicy


class SingleRobotMover(Node):

    def __init__(self):
        super().__init__('single_robot_mover')

        # Parameter
        self.declare_parameter('robot', 'burger1')
        self.declare_parameter('forward_distance', 1.0)
        self.declare_parameter('backward_distance', 1.0)
        self.declare_parameter('speed', 0.2)
        self.robot = self.get_parameter('robot').value

        # Goals
        self.forward_distance = self.get_parameter('forward_distance').value
        self.backward_distance = self.get_parameter('backward_distance').value
        self.speed = self.get_parameter('speed').value

        # State
        self.start_x = None
        self.start_y = None
        self.current_x = None
        self.current_y = None
        self.stage = 0

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        # QoS handling
        if self.robot == 'tb4_1':
            self.cmd_pub = self.create_publisher(
                Twist,
                f'/{self.robot}/cmd_vel',
                qos_profile
            )

            self.sub = self.create_subscription(
                Odometry,
                f'/{self.robot}/odom',
                self.odom_callback,
                qos_profile
            )
        else:
            self.cmd_pub = self.create_publisher(
                Twist,
                f'/{self.robot}/cmd_vel',
                10
            )

            self.sub = self.create_subscription(
                Odometry,
                f'/{self.robot}/odom',
                self.odom_callback,
                10
            )

        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info(f"Controlling robot: {self.robot}")

    def odom_callback(self, msg):

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        if self.start_x is None:
            self.start_x = x
            self.start_y = y

        self.current_x = x
        self.current_y = y

    def distance(self):

        dx = self.current_x - self.start_x
        dy = self.current_y - self.start_y

        return math.sqrt(dx*dx + dy*dy)

    def control_loop(self):

        if self.current_x is None:
            return

        twist = Twist()
        d = self.distance()

        # Forward
        if self.stage == 0:

            if d < self.forward_distance:
                twist.linear.x = self.speed
            else:
                self.stage = 1
                self.start_x = self.current_x
                self.start_y = self.current_y

        # Backward
        elif self.stage == 1:

            if d < self.backward_distance:
                twist.linear.x = -self.speed
            else:
                self.stage = 2
                twist.linear.x = 0.0

        self.cmd_pub.publish(twist)


def main():
    rclpy.init()
    node = SingleRobotMover()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()