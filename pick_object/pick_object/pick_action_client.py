#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from pick_object_interface.action import PickObject

import json


class PickObjectActionClient(Node):
    def __init__(self):
        super().__init__('pick_object_action_client')
        
        # Create action client
        self._action_client = ActionClient(
            self,
            PickObject,
            'pick_object'
        )
        
        self.get_logger().info('PickObject Action Client has been started')

    def send_goal(self, object_name):
        """
        Send a goal to the action server.
        """
        self.get_logger().info(f'Waiting for action server...')
        
        # Wait for server to be available
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server not available!')
            return
        
        # Create goal message
        goal_msg = PickObject.Goal()
        goal_msg.object_name = object_name
        
        self.get_logger().info(f'Sending goal to pick: {object_name}')
        
        # Send goal with callbacks
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        
        # Register callback when goal response is received
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """
        Called when the server accepts or rejects the goal.
        """
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().error('Goal was REJECTED by server!')
            return
        
        self.get_logger().info('Goal was ACCEPTED by server')
        
        # Store goal handle for potential cancellation
        self._goal_handle = goal_handle
        
        # Get result
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        """
        Called periodically when feedback is received from the server.
        """
        feedback = feedback_msg.feedback
        
        # Parse JSON state
        try:
            state = json.loads(feedback.json_state)
            self.get_logger().info(
                f'FEEDBACK: Progress={state["progress_percent"]}%, '
                f'Step={state["current_step"]}/{state["total_steps"]}, '
                f'Gripper={state["gripper_status"]}'
            )
        except json.JSONDecodeError:
            self.get_logger().warn(f'Could not parse feedback JSON: {feedback.json_state}')

    def get_result_callback(self, future):
        """
        Called when the action completes (succeeded, aborted, or canceled).
        """
        result = future.result().result
        status = future.result().status
        
        # Check status
        if status == 4:  # SUCCEEDED
            self.get_logger().info(f'Action SUCCEEDED: {result.status_msg}')
        elif status == 5:  # CANCELED
            self.get_logger().warn(f'Action CANCELED: {result.status_msg}')
        elif status == 6:  # ABORTED
            self.get_logger().error(f'Action ABORTED: {result.status_msg}')
        else:
            self.get_logger().error(f'Action failed with status: {status}')
        
        self.get_logger().info(f'Success flag: {result.success}')

    def cancel_goal(self):
        """
        Cancel the current goal.
        """
        if hasattr(self, '_goal_handle'):
            self.get_logger().info('Sending cancel request...')
            cancel_future = self._goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self.cancel_done_callback)
        else:
            self.get_logger().warn('No active goal to cancel')

    def cancel_done_callback(self, future):
        """
        Called when cancel request is processed.
        """
        cancel_response = future.result()
        if len(cancel_response.goals_canceling) > 0:
            self.get_logger().info('Goal successfully canceled')
        else:
            self.get_logger().warn('Goal failed to cancel')


def main(args=None):
    rclpy.init(args=args)
    
    action_client = PickObjectActionClient()
    
    # Example 1: Send a goal
    object_to_pick = 'red_cup'  # Change this to test different objects
    action_client.send_goal(object_to_pick)
    
    # Example 2: To cancel after 5 seconds, uncomment below:
    # import threading
    # def cancel_after_delay():
    #     import time
    #     time.sleep(5)
    #     action_client.cancel_goal()
    # threading.Thread(target=cancel_after_delay).start()
    
    # Spin to process callbacks
    try:
        rclpy.spin(action_client)
    except KeyboardInterrupt:
        pass
    finally:
        action_client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()