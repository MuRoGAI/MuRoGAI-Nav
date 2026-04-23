#!/usr/bin/env python3
"""
Example client — call the MoveObjects service.

Usage:
    ros2 run object_mover_srv move_objects_client
"""

import rclpy
from rclpy.node import Node
from object_mover_interface.srv import MoveObjects


class MoveObjectsClient(Node):
    def __init__(self):
        super().__init__("move_objects_client")
        self._client = self.create_client(MoveObjects, "move_objects")
        while not self._client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /move_objects service …")

    def send(self, object_names: list[str], goal_names: list[str]):
        req = MoveObjects.Request()
        req.object_names = object_names
        req.goal_names   = goal_names

        future = self._client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()
        self.get_logger().info(f"success={resp.success}\n{resp.message}")
        return resp


def main(args=None):
    rclpy.init(args=args)
    client = MoveObjectsClient()

    # ── Example 1: goals are robot names → use their odom poses ──────────
    # client.send(
    #     object_names = ["cake", "speaker", "gift"],
    #     goal_names   = ["delivery_bot3", "delivery_bot2", "delivery_bot1"],
    # )

    # # ── Example 2: goals are location names ──────────────────────────────
    # client.send(
    #     object_names = ["cake", "speaker", "gift"],
    #     goal_names   = ["table2", "table2", "table2"],   # spread to named slots
    # )

    # # ── Example 3: mixed — robot names and location names together ────────
    # client.send(
    #     object_names = ["cake", "speaker"],
    #     goal_names   = ["sink1", "delivery_bot2"],  # sink1=location, bot2=robot
    # )

    # client.send(
    #     object_names = ["cake"],
    #     goal_names   = ["salad_stall"],  # sink1=location, bot2=robot
    # )

    client.send(
        object_names = ["juice"],
        goal_names   = ["cleaning_bot"],  # sink1=location, bot2=robot
    )

    client.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()