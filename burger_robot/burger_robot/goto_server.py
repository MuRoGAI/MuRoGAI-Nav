#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from robot_interface.srv import GoTo
import time


class GoToService(Node):
    """
    Simple server:
      - Provides 'goto' service (robot_interface/srv/GoTo.srv)
      - Simulates a navigation/pick process for a few seconds
      - Subscribes to '/goto/cancel' to stop it mid-way
      - Publishes final result as a log and service response
    """

    def __init__(self):
        super().__init__('goto_server')

        self.cb_group = ReentrantCallbackGroup()
        self._srv = self.create_service(
            GoTo, 'goto', self.handle_goto_request, callback_group=self.cb_group
        )

        # Cancel subscriber — any message on this topic stops current goto
        self._cancel_sub = self.create_subscription(
            String, '/goto/cancel', self._on_cancel_msg, 10, callback_group=self.cb_group
        )

        self._cancel_requested = False
        self._busy = False

        self.get_logger().info("GoTo service ready at /goto")
        self.get_logger().info("Send cancel message to /goto/cancel to interrupt")

    # --------------------------------------------------------
    def _on_cancel_msg(self, msg: String):
        """Called whenever a message is published to /goto/cancel."""
        if self._busy:
            self._cancel_requested = True
            self.get_logger().info("Cancel request received! Stopping current goal...")
        else:
            self.get_logger().info("Cancel received, but no active goal in progress.")

    # --------------------------------------------------------
    def handle_goto_request(self, request: GoTo.Request, response: GoTo.Response):
        """Main service callback."""
        if self._busy:
            response.success = False
            response.message = "Server busy with another navigation"
            return response

        # NOTE: srv defines the field as `string name`
        goal_name = request.name.strip()
        if not goal_name:
            response.success = False
            response.message = "place/object name is empty"
            return response

        self.get_logger().info(f"Starting navigation operation for '{goal_name}'")
        self._busy = True
        self._cancel_requested = False

        # Simulate a task taking `loop` seconds (1 second per step)
        loop = 20
        for i in range(loop):
            if self._cancel_requested:
                self.get_logger().warn(f"Operation canceled at step {i+1}/{loop} for '{goal_name}'")
                response.success = False
                response.message = f"Operation canceled for '{goal_name}'"
                self._busy = False
                return response

            self.get_logger().info(f"Processing '{goal_name}'... step {i+1}/{loop}")
            time.sleep(1.0)  # simulate work

        # Finished successfully
        self.get_logger().info(f"Successfully completed operation for '{goal_name}'")
        response.success = True
        response.message = f"Successfully completed operation for '{goal_name}'"
        self._busy = False
        return response


def main(args=None):
    rclpy.init(args=args)
    node = GoToService()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down GoTo server")
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
