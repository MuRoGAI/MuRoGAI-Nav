#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from pick_object_interface.action import PickObject
import sys
import json

class PickObjectClient(Node):
    def __init__(self):
        super().__init__('pick_object_client')
        self._action_client = ActionClient(self, PickObject, 'pick_object')

    def pick_object_send_goal(self, object_name="red_gear"):
        # Wait for server
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Action server not available!")
            return

        goal_msg = PickObject.Goal()
        goal_msg.object_name = object_name

        self.get_logger().info(f"Sending goal: pick '{object_name}'")

        # Send goal async
        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected")
            return

        self.get_logger().info("Goal accepted, waiting for result...")
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        fb = feedback_msg.feedback
        try:
            data = json.loads(fb.json_state)
            self.get_logger().info(f"Feedback → {data}")
        except json.JSONDecodeError:
            self.get_logger().info(f"Feedback → {fb.json_state}")

    def get_result_callback(self, future):
        result = future.result().result
        if result.success:
            self.get_logger().info(f"SUCCESS: {result.status_msg}")
        else:
            self.get_logger().warn(f"FAILED: {result.status_msg}")
        


def main(args=None):
    rclpy.init(args=args)
    client = PickObjectClient()

    obj = 'grey gear'
    client.pick_object_send_goal(obj)

    try:
        rclpy.spin(client)
    except KeyboardInterrupt:
        client.get_logger().info("Client interrupted")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()