#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from robot_interface.srv import Find
import time


class FindService(Node):
    """
    Simple server:
      - Provides '/find' (Find.srv)
      - Simulates a find/search process for a few seconds
      - Subscribes to '/find/cancel' to stop mid-way
      - Responds with success/failure + message
    """

    def __init__(self):
        super().__init__('find_server')

        self.cb_group = ReentrantCallbackGroup()
        self._srv = self.create_service(
            Find, 'find', self.handle_find_request, callback_group=self.cb_group
        )

        self._cancel_sub = self.create_subscription(
            String, '/find/cancel', self._on_cancel_msg, 10, callback_group=self.cb_group
        )

        self._cancel_requested = False
        self._busy = False

        self.get_logger().info("Find service ready at /find")
        self.get_logger().info("Publish String to /find/cancel to interrupt")

    # ---------------------------------------------------------
    def _on_cancel_msg(self, msg: String):
        """Triggered on cancel topic."""
        if self._busy:
            self._cancel_requested = True
            self.get_logger().warn("Cancel request received! Stopping current find operation…")
        else:
            self.get_logger().info("Cancel received, but no active find in progress.")

    # ---------------------------------------------------------
    def handle_find_request(self, request: Find.Request, response: Find.Response):
        """Main service callback."""
        if self._busy:
            response.success = False
            response.message = "Server busy with another find operation"
            return response

        target = request.name.strip()
        if not target:
            response.success = False
            response.message = "Target name is empty"
            return response

        self.get_logger().info(f"Starting find operation for '{target}'")
        self._busy = True
        self._cancel_requested = False

        loop = 20  # simulate 10 seconds
        for i in range(loop):
            if self._cancel_requested:
                self.get_logger().warn(f"Find canceled at step {i+1}/{loop} for '{target}'")
                response.success = False
                response.message = f"Find canceled for '{target}'"
                self._busy = False
                return response

            self.get_logger().info(f"Finding '{target}'... step {i+1}/{loop}")
            time.sleep(1.0)

        # Finished successfully
        self.get_logger().info(f"Successfully found '{target}'")
        response.success = True
        response.message = f"Successfully found '{target}'"
        self._busy = False
        return response


def main(args=None):
    rclpy.init(args=args)
    node = FindService()

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Find service")
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
