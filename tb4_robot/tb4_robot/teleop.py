#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

import sys
import termios
import tty
import threading

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


class TeleopNode(Node):

    def __init__(self):
        super().__init__('tb4_teleop')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            durability=DurabilityPolicy.VOLATILE
        )

        self.publisher_ = self.create_publisher(
            Twist,
            '/tb4_1/cmd_vel',
            qos
        )

        self.speed = 0.3
        self.turn = 0.8
        self.running = True

        self.get_logger().info("Teleop Started")
        self.get_logger().info("WASD to move | q to quit | Ctrl+C to kill")

        # Start keyboard thread
        self.thread = threading.Thread(target=self.keyboard_loop)
        self.thread.daemon = True
        self.thread.start()


    def get_key(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setraw(fd)
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        return key


    def keyboard_loop(self):
        while self.running and rclpy.ok():

            key = self.get_key()
            twist = Twist()

            if key == 'w':
                twist.linear.x = self.speed
            elif key == 's':
                twist.linear.x = -self.speed
            elif key == 'a':
                twist.angular.z = self.turn
            elif key == 'd':
                twist.angular.z = -self.turn
            elif key == 'q':
                self.running = False
                break

            self.publisher_.publish(twist)


def main(args=None):
    rclpy.init(args=args)

    node = TeleopNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.running = False
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()  