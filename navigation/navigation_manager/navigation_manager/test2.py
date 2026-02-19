#!/usr/bin/env python3

"""
Single Navigation Test Node

Hardcoded request
Publishes once
Waits for planner output
Then shuts down
"""

import json
import rclpy
from rclpy.node import Node

from navigation_manager_interface.msg import NavigationRobotRequest
from path_planner_interface.msg import PathPlannerRequest


class NavigationSingleTester(Node):

    def __init__(self):
        super().__init__("navigation_single_tester")

        self.get_logger().info("Single Navigation Tester Started")

        # --------------------------------------------------
        # 🔥 HARD CODE YOUR TEST HERE
        # --------------------------------------------------
        self.robot_name = "robot1"   # Leave empty if formation
        self.goal = "assist service at table2"
        # self.formation_robots = ["robot1", "robot2", "robot5", "robot6"]
        self.formation_robots = []

        # --------------------------------------------------
        # Publisher
        # --------------------------------------------------
        self.req_pub = self.create_publisher(
            NavigationRobotRequest,
            "/navigation/request",
            10
        )

        # --------------------------------------------------
        # Subscriber
        # --------------------------------------------------
        self.plan_sub = self.create_subscription(
            PathPlannerRequest,
            "/path_planner/request",
            self.plan_callback,
            10
        )

        # Send request once
        self.send_request()

    # ==================================================
    # Send Request
    # ==================================================

    def send_request(self):

        msg = NavigationRobotRequest()

        msg.robot_name = self.robot_name
        msg.goal = self.goal
        msg.formation_robots = self.formation_robots

        self.req_pub.publish(msg)

        self.get_logger().info(
            f"Published Request → robot={self.robot_name or self.formation_robots}, goal={self.goal}"
        )

    # ==================================================
    # Receive Planner Output
    # ==================================================

    def plan_callback(self, msg: PathPlannerRequest):

        self.get_logger().info("Received Planner Output:")
        self.get_logger().info(msg.plan_json)

        # Shutdown after receiving response
        self.get_logger().info("Test complete. Shutting down.")

        rclpy.shutdown()


# ==================================================
# Main
# ==================================================

def main(args=None):

    rclpy.init(args=args)

    node = NavigationSingleTester()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
