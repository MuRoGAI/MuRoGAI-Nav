import threading
import json
from pick_object_interface.action import PickObject


class YourNode(Node):
    def __init__(self):
        super().__init__('your_node_name')
        self._action_client = ActionClient(self, PickObject, 'pick_object')

        # These make blocking + return value possible
        self._pick_success = False
        self._pick_done = threading.Event()   # ← this is the key

    def pick_object(self, obect_name: str) -> bool:
        """Blocking call: sends pick goal and returns True only if succeeded."""
        self.get_logger().info(f"Picking {obect_name} object...")

        # Reset state for new call
        self._pick_success = False
        self._pick_done.clear()

        # Send the goal
        success = self.pick_object_send_goal(obect_name)
        if not success:
            self.get_logger().error("Failed to send goal")
            return False

        # BLOCK until result arrives (or timeout)
        if not self._pick_done.wait(timeout=20.0):  # 20 sec max
            self.get_logger().error("Pick action timed out!")
            return False

        self.get_logger().info('after the action call')
        return self._pick_success

    def pick_object_send_goal(self, object_name="red_gear") -> bool:
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Action server not available!")
            return False

        goal_msg = PickObject.Goal()
        goal_msg.object_name = object_name

        self.get_logger().info(f"Sending goal: pick '{object_name}'")

        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)
        return True  # Goal sent successfully

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected")
            self._pick_done.set()        # ← unblock pick_object()
            return

        self.get_logger().info("Goal accepted, waiting for result...")
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        fb = feedback_msg.feedback
        try:
            json_data = json.loads(fb.json_state)
            self.get_logger().info(f"Feedback → {json_data}")
            self.update_robot_state(json_data)
        except json.JSONDecodeError:
            self.get_logger().info(f"Feedback → {fb.json_state}")

    def get_result_callback(self, future):
        result = future.result().result
        self._pick_success = result.success   # ← this is the real answer

        if result.success:
            self.get_logger().info(f"SUCCESS: {result.status_msg}")
        else:
            self.get_logger().warn(f"FAILED: {result.status_msg}")

        self._pick_done.set()  # ← THIS unblocks pick_object() and returns the real result