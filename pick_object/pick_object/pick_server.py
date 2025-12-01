#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from pick_object_interface.action import PickObject
import json

SLICE_SEC = 0.05  # small spin slice; 20 slices ~= 1s

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionServer, CancelResponse, GoalResponse

SLICE = 0.05  # seconds

class PickObjectServer(Node):
    def __init__(self):
        super().__init__('pick_object_server')
        self._cb = ReentrantCallbackGroup()
        self._srv = ActionServer(
            self, PickObject, 'pick_object',
            execute_callback=self.execute_callback,
            cancel_callback=self.cancel_callback,
            goal_callback=self.goal_callback,
            callback_group=self._cb,
        )
        self.get_logger().info("PickObject server started")

    def goal_callback(self, req):
        name = req.object_name.strip()
        return GoalResponse.ACCEPT if name and name.lower() != "bomb" else GoalResponse.REJECT

    def cancel_callback(self, handle):
        self.get_logger().info(f"Cancel requested for '{handle.request.object_name}'")
        return CancelResponse.ACCEPT

    def _sleep_coop(self, handle, sec: float) -> bool:
        remaining = sec
        while remaining > 0.0:
            rclpy.spin_once(self, timeout_sec=min(SLICE, remaining))
            if handle.is_cancel_requested:
                return True
            remaining -= SLICE
        return False

    def execute_callback(self, handle):
        name = handle.request.object_name.strip()
        self.get_logger().info(f"Executing goal: pick '{name}'")

        total = 20
        for i in range(total):
            if handle.is_cancel_requested:
                self.get_logger().info(f"Canceled picking '{name}' (pre-sleep)")
                handle.canceled()
                out = PickObject.Result(); out.success = False; out.status_msg = f"Canceled picking '{name}'"
                return out

            fb = PickObject.Feedback()
            fb.json_state = json.dumps({"phase":"picking","object":name,"progress":(i+1)/float(total),"step":i+1})
            handle.publish_feedback(fb)

            if self._sleep_coop(handle, 1.0):
                self.get_logger().info(f"Canceled picking '{name}' (during sleep)")
                handle.canceled()
                out = PickObject.Result(); out.success = False; out.status_msg = f"Canceled picking '{name}'"
                return out

        self.get_logger().info(f"Successfully picked '{name}'")
        handle.succeed()
        out = PickObject.Result(); out.success = True; out.status_msg = f"Successfully picked '{name}'"
        return out

def main():
    rclpy.init()
    node = PickObjectServer()
    exec = MultiThreadedExecutor(num_threads=4)
    exec.add_node(node)
    try: exec.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()
