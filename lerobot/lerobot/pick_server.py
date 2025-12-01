#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from robot_interface.srv import StartPick
import time
import json


class StartPickServer(Node):
    """
    Simple server:
      - Provides `/start_pick` (StartPick.srv)
      - Simulates a pick process for few seconds
      - Subscribes to `/start_pick/cancel` to stop it mid-way
      - Publishes final result as a log and service response
    """

    def __init__(self):
        super().__init__('pick_server')

        self.robot_states = {}

        self.cb_group = ReentrantCallbackGroup()
        self._srv = self.create_service(
            StartPick, '/start_pick', self.handle_start_pick, callback_group=self.cb_group
        )

        # Cancel subscriber — any message on this topic stops current pick
        self._cancel_sub = self.create_subscription(
            String, '/start_pick/cancel', self._on_cancel_msg, 10, callback_group=self.cb_group
        )

        self.pub_robot_states = self.create_publisher(String, '/robot_states', 10)
        self.sub_robot_states = self.create_subscription(
            String, '/robot_states', self.on_robot_states, 10,
        )

        self._cancel_requested = False
        self._busy = False

        self.get_logger().info("StartPick service ready at /start_pick")
        self.get_logger().info("Send cancel message to /start_pick/cancel to interrupt")

    # --------------------------------------------------------
    def on_robot_states(self, msg: String):
        """Handle robot state updates (expects JSON string in msg.data)."""
        try:
            data = json.loads(msg.data)
            rs = data.get("robot_states")
            if isinstance(rs, dict):
                self.robot_states = rs
                self.get_logger().debug(f"robot_states keys: {list(self.robot_states.keys())}")
            else:
                self.get_logger().warning("/robot_states missing or not a dict; ignoring payload.")
        except json.JSONDecodeError:
            self.get_logger().error(f"/robot_states not JSON: {msg.data}")


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
        updated_state = {
            "arm_state": "moving",
            "gripper_state": "closed",
            "carry_state": 'filled',
        }
        self.update_robot_state(updated_state, robot_name='lerobot')
        for i in range(loop):
            if self._cancel_requested:
                self.get_logger().warn(f"Pick canceled at step {i+1}/10 for '{object_name}'")
                response.success = False
                response.message = f"Pick canceled for '{object_name}'"
                self._busy = False
                updated_state = {
                    "arm_state": "stowed",
                    "gripper_state": "closed",
                    "carry_state": 'filled',
                }
                self.update_robot_state(updated_state, robot_name='lerobot')
                return response

            self.get_logger().info(f"Picking '{object_name}'... step {i+1}/{loop}")
            time.sleep(1.0)  # simulate work

        # Finished successfully
        self.get_logger().info(f"Successfully picked '{object_name}'")
        response.success = True
        response.message = f"Successfully picked '{object_name}'"
        self._busy = False
        updated_state = {
            "arm_state": "stowed",
            "gripper_state": "closed",
            "carry_state": 'empty',
        }
        self.update_robot_state(updated_state, robot_name='lerobot')
        return response

    def update_robot_state(self, incoming: dict, robot_name: str | None = None) -> None:
        """
        Merge a partial robot-state update into self.robot_states.

        - self.robot_states may be:
            * None
            * a dict without "robot_states" (e.g., only meta fields like date)
            * a dict that already has "robot_states"
            * (optionally) a bare robot-states dict from older code; we wrap it.

        - Ensures top-level "robot_states" key exists.
        - Ensures per-robot dict exists.
        - Adds new keys and updates changed ones.
        """

        # Default robot_name to this node's robot name if not given
        if robot_name is None:
            robot_name = self.robot_name

        # 1) Ensure top-level container exists
        if self.robot_states is None:
            self.robot_states = {}
            self.get_logger().info("Created top-level robot_states container dict")

        # 2) Make sure we have a "robot_states" key at top-level
        if "robot_states" not in self.robot_states:
            # Heuristic: if current dict already *looks* like it only contains robots,
            # then wrap it under "robot_states" instead of losing it.
            if all(isinstance(v, dict) for v in self.robot_states.values()) and len(self.robot_states) > 0:
                # Wrap existing as robot_states
                self.robot_states = {"robot_states": self.robot_states}
                self.get_logger().debug("Wrapped existing dict under 'robot_states'")
            else:
                # Start fresh robot_states dict, keep other meta fields as-is
                self.robot_states["robot_states"] = {}
                self.get_logger().info("Created 'robot_states' key in top-level dict")

        robots_dict = self.robot_states["robot_states"]

        # 3) Ensure the robot entry exists
        if robot_name not in robots_dict or not isinstance(robots_dict[robot_name], dict):
            robots_dict[robot_name] = {}
            self.get_logger().info(f"Created new robot entry '{robot_name}'")

        saved_state = robots_dict[robot_name]

        # 4) Merge incoming keys into that robot's state
        for key, new_val in incoming.items():
            if key not in saved_state:
                saved_state[key] = new_val
                self.get_logger().debug(f"{robot_name}: Added '{key}' = '{new_val}'")
            else:
                old_val = saved_state[key]
                if old_val != new_val:
                    saved_state[key] = new_val
                    self.get_logger().debug(f"{robot_name}: Updated '{key}' from '{old_val}' → '{new_val}'")

        # 5) Publish the updated robot_states
        msg = String()
        msg.data = json.dumps(self.robot_states)
        self.pub_robot_states.publish(msg)

        # 6) Log the updated root structure (just the robot_states part to keep it readable)
        self.get_logger().debug(f"robot_states now: {self.robot_states['robot_states']}")

        return


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
