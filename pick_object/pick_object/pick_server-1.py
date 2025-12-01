#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse
from pick_object_interface.action import PickObject
import time
import json


class PickObjectServer(Node):
    def __init__(self):
        super().__init__('pick_object_server')

        # Accept all goals, accept cancel requests
        self._action_server = ActionServer(
            self,
            PickObject,
            'pick_object',
            execute_callback=self.execute_callback,
            cancel_callback=self.cancel_callback,      # <-- accept cancels
            goal_callback=self.goal_callback           # <-- accept (or filter) goals
        )
        self.get_logger().info("PickObject server started")

    # Accept or reject an incoming goal (optional but explicit = clearer)
    def goal_callback(self, goal_request):
        object_name = goal_request.object_name.strip()
        if not object_name or object_name.lower() == "bomb":
            # Reject by returning rclpy.action.GoalResponse.REJECT (import if you want to be explicit)
            from rclpy.action import GoalResponse
            return GoalResponse.REJECT
        from rclpy.action import GoalResponse
        return GoalResponse.ACCEPT

    # Explicitly accept cancellation
    def cancel_callback(self, goal_handle):
        self.get_logger().info(f"Cancel requested for '{goal_handle.request.object_name}'")
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        object_name = goal_handle.request.object_name.strip()
        self.get_logger().info(f"Executing goal: pick '{object_name}'")

        # If goal was rejected in goal_callback(), execute_callback won't be called.
        # Main work loop (simulate 20 steps ~ 20s)
        total_steps = 20
        for i in range(total_steps):
            # Cooperate with cancel
            if goal_handle.is_cancel_requested:
                self.get_logger().info(f"Canceled picking '{object_name}'")
                goal_handle.canceled()
                result = PickObject.Result()
                result.success = False
                result.status_msg = f"Canceled picking '{object_name}'"
                return result

            # Send feedback
            feedback = PickObject.Feedback()
            feedback.json_state = json.dumps({
                "phase": "picking",
                "object": object_name,
                "progress": (i + 1) / float(total_steps),
                "step": i + 1
            })
            goal_handle.publish_feedback(feedback)

            time.sleep(1)

        # Success path — mark succeed *after* finishing work
        self.get_logger().info(f"Successfully picked '{object_name}'")
        goal_handle.succeed()
        result = PickObject.Result()
        result.success = True
        result.status_msg = f"Successfully picked '{object_name}'"
        return result


def main(args=None):
    rclpy.init(args=args)
    server = PickObjectServer()
    try:
        rclpy.spin(server)
    except KeyboardInterrupt:
        server.get_logger().info("Shutting down")
    finally:
        server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
