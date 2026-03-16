#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
from rclpy.qos import QoSProfile, ReliabilityPolicy


class MultiRobotMover(Node):

    def __init__(self):
        super().__init__('multi_robot_mover')

        # Robot names
        self.robots = ['burger1', 'burger2', 'burger3', 'waffle', 'firebird', 'tb4_1']

        # Distance goals
        self.forward_distance = 1.0
        self.backward_distance = -1.0

        # Velocity
        self.speed = 0.2

        self.robot_data = {}
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        for r in self.robots:
            if not r == 'tb4_1': 
                cmd_pub = self.create_publisher(
                    Twist,
                    f'/{r}/cmd_vel',
                    10
                )

                sub = self.create_subscription(
                    Odometry,
                    f'/{r}/odom',
                    lambda msg, robot=r: self.odom_callback(msg, robot),
                    10
                )
            else:
                cmd_pub = self.create_publisher(
                    Twist,
                    f'/{r}/cmd_vel',
                    qos_profile
                )
                sub = self.create_subscription(
                    Odometry,
                    f'/{r}/odom',
                    self.odom_callback,
                    qos_profile
                )

            self.robot_data[r] = {
                'cmd_pub': cmd_pub,
                'start_x': None,
                'start_y': None,
                'stage': 0
            }

        self.timer = self.create_timer(0.1, self.control_loop)

    def odom_callback(self, msg, robot):

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        if self.robot_data[robot]['start_x'] is None:
            self.robot_data[robot]['start_x'] = x
            self.robot_data[robot]['start_y'] = y

        self.robot_data[robot]['current_x'] = x
        self.robot_data[robot]['current_y'] = y

    def distance(self, r):

        dx = self.robot_data[r]['current_x'] - self.robot_data[r]['start_x']
        dy = self.robot_data[r]['current_y'] - self.robot_data[r]['start_y']

        return math.sqrt(dx*dx + dy*dy)

    def control_loop(self):

        for r in self.robots:

            data = self.robot_data[r]

            if 'current_x' not in data:
                continue

            twist = Twist()

            d = self.distance(r)

            # Stage 0: move forward
            if data['stage'] == 0:

                if d < self.forward_distance:
                    twist.linear.x = self.speed
                else:
                    data['stage'] = 1
                    data['start_x'] = data['current_x']
                    data['start_y'] = data['current_y']

            # Stage 1: move backward
            elif data['stage'] == 1:

                if d < abs(self.backward_distance):
                    twist.linear.x = -self.speed
                else:
                    data['stage'] = 2
                    twist.linear.x = 0.0

            data['cmd_pub'].publish(twist)


def main():

    rclpy.init()

    node = MultiRobotMover()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()