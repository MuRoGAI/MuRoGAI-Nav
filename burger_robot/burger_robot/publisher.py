#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json


class TargetPublisher(Node):

    def __init__(self):
        super().__init__('target_publisher')

        self.pub = self.create_publisher(String, '/burger1/targets', 10)

        # ---- Define Targets Here ----
        self.targets = {
            "x"   : [1.0, 2.0, 3.0],
            "y"   : [0.5, 1.5, 2.5],
            "yaw" : [0.0, 1.57, 3.14],
            
            "vx"  : [0.4, 0.5, 0.6, 0.2, 0.32, 0.45],
            "vy"  : [0.0, 0.9, 4.0, 0.8, 0.00, 0.4],
        }

        # Publish once after startup
        self.timer = self.create_timer(1.0, self.publish_targets)
        self.published = False


    def publish_targets(self):

        if self.published:
            return

        msg = String()
        msg.data = json.dumps(self.targets)

        self.pub.publish(msg)
        self.get_logger().info("Targets Published")

        self.published = True


def main():
    rclpy.init()
    node = TargetPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
