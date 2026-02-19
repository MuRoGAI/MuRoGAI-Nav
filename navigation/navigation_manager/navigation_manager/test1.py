#!/usr/bin/env python3

"""
Navigation Manager Test Node

Reads Navigation requests from Excel
Publishes them one-by-one
Waits for PathPlanner output
Stores results in new Excel file
"""

import os
import json
import pandas as pd
from datetime import datetime

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from navigation_manager_interface.msg import NavigationRobotRequest
from path_planner_interface.msg import PathPlannerRequest


class NavigationManagerTester(Node):

    def __init__(self):
        super().__init__("navigation_manager_tester")

        self.get_logger().info("Navigation Manager Tester Started")

        # --------------------------------------------------
        # Parameters
        # --------------------------------------------------
        self.declare_parameter("input_excel_file", "")
        self.declare_parameter("output_excel_file", "")
        self.declare_parameter("map_metadata_file", "")

        # --------------------------------------------------
        # Resolve file paths (same style as NavigationManager)
        # --------------------------------------------------
        self.input_excel_path = self.resolve_file(
            "input_excel_file",
            "navigation_manager",
            "data1/navigation_test_formation_20_cases.xlsx"
        )

        self.output_excel_path = self.resolve_file(
            "output_excel_file",
            "navigation_manager",
            "data1/navigation_test_formation_20_cases_output.xlsx"
        )

        # --------------------------------------------------
        # Load Excel input
        # --------------------------------------------------
        self.test_df = pd.read_excel(self.input_excel_path)

        self.test_index = 0
        self.results = []

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

        # Start first test
        self.send_next_test()

    # ==================================================
    # File Resolver
    # ==================================================

    def resolve_file(self, param_name, default_pkg, relative_path):
        param_value = self.get_parameter(param_name).value

        if param_value:
            return param_value

        pkg_path = get_package_share_directory(default_pkg)
        return os.path.join(pkg_path, relative_path)

    # ==================================================
    # Send test
    # ==================================================

    def send_next_test(self):

        if self.test_index >= len(self.test_df):
            self.get_logger().info("All tests completed.")
            self.save_results()
            return

        row = self.test_df.iloc[self.test_index]

        msg = NavigationRobotRequest()

        msg.robot_name = str(row["robot_namespace"]) if not pd.isna(row["robot_namespace"]) else ""
        msg.goal = str(row["target_goal"])

        formation_value = row.get("formation_robots", "")

        if pd.isna(formation_value) or str(formation_value).strip() == "":
            msg.formation_robots = []
        else:
            text = str(formation_value).strip()

            # If already JSON format
            if text.startswith("["):
                msg.formation_robots = json.loads(text)
            else:
                # Comma separated format from Excel
                msg.formation_robots = [
                    r.strip() for r in text.split(",") if r.strip()
                ]


        self.current_request = {
            "robot_namespace": msg.robot_name,
            "goal": msg.goal,
            "formation_robots": msg.formation_robots
        }

        self.req_pub.publish(msg)

        self.get_logger().info(
            f"Sent test #{self.test_index + 1}: {self.current_request}"
        )

    # ==================================================
    # Receive Planner Output
    # ==================================================

    def plan_callback(self, msg: PathPlannerRequest):

        plan_json = msg.plan_json

        # Save result
        result = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "robot_namespace": self.current_request["robot_namespace"],
            "goal": self.current_request["goal"],
            "formation_robots": json.dumps(self.current_request["formation_robots"]),
            "planner_output": plan_json
        }

        self.results.append(result)
        self.save_results()

        self.get_logger().info(
            f"Received planner output for test #{self.test_index + 1}"
        )

        # Move to next test
        self.test_index += 1
        self.send_next_test()

    # ==================================================
    # Save results
    # ==================================================

    def save_results(self):

        os.makedirs(os.path.dirname(self.output_excel_path), exist_ok=True)

        df = pd.DataFrame(self.results)
        df.to_excel(self.output_excel_path, index=False)

        self.get_logger().info(
            f"Saved results to {self.output_excel_path}"
        )


# ==================================================
# Main
# ==================================================

def main(args=None):

    rclpy.init(args=args)

    node = NavigationManagerTester()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
