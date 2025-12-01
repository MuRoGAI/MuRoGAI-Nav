#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from pick_object_interface.srv import StartPick
import time


class StartPickServer(Node):
    """
    Simple server:
      - Provides `/start_pick` (StartPick.srv)
      - Simulates a pick process for few seconds
      - Subscribes to `/start_pick/cancel` to stop it mid-way
      - Publishes final result as a log and service response
    """

    def __init__(self):
        super().__init__('start_pick_server')

        self.cb_group = ReentrantCallbackGroup()
        self._srv = self.create_service(
            StartPick, 'start_pick', self.handle_start_pick, callback_group=self.cb_group
        )

        # Cancel subscriber — any message on this topic stops current pick
        self._cancel_sub = self.create_subscription(
            String, '/start_pick/cancel', self._on_cancel_msg, 10, callback_group=self.cb_group
        )

        self._cancel_requested = False
        self._busy = False

        self.get_logger().info("StartPick service ready at /start_pick")
        self.get_logger().info("Send cancel message to /start_pick/cancel to interrupt")

    # --------------------------------------------------------
    def _on_cancel_msg(self, msg: String):
        """Called whenever a message is published to /start_pick/cancel."""
        if self._busy:
            self._cancel_requested = True
            self.get_logger().info("Cancel request received! Stopping current pick...")
        else:
            self.get_logger().info("Cancel received, but no active pick in progress.")

    # --------------------------------------------------------
    def handle_start_pick(self, request: StartPick.Request, response: StartPick.Response):
        """Main service callback."""
        if self._busy:
            response.success = False
            response.message = "Server busy with another pick"
            return response

        object_name = request.object_name.strip()
        if not object_name:
            response.success = False
            response.message = "object_name is empty"
            return response

        self.get_logger().info(f"Starting pick operation for '{object_name}'")
        self._busy = True
        self._cancel_requested = False

        # Simulate a pick task taking 10 seconds
        loop = 20
        for i in range(loop):
            if self._cancel_requested:
                self.get_logger().warn(f"Pick canceled at step {i+1}/10 for '{object_name}'")
                response.success = False
                response.message = f"Pick canceled for '{object_name}'"
                self._busy = False
                return response

            self.get_logger().info(f"Picking '{object_name}'... step {i+1}/{loop}")
            time.sleep(1.0)  # simulate work

        # Finished successfully
        self.get_logger().info(f"Successfully picked '{object_name}'")
        response.success = True
        response.message = f"Successfully picked '{object_name}'"
        self._busy = False
        return response


def main(args=None):
    rclpy.init(args=args)
    node = StartPickServer()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down StartPick server")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
