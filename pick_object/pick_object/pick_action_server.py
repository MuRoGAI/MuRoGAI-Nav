#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from pick_object_interface.action import PickObject

import time
import json


class PickObjectActionServer(Node):
    def __init__(self):
        super().__init__('pick_object_action_server')
        
        # Create action server with callback group for concurrent execution
        self._action_server = ActionServer(
            self,
            PickObject,
            'pick_object',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            handle_accepted_callback=self.handle_accepted_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup()
        )
        
        self.get_logger().info('PickObject Action Server has been started')

    def goal_callback(self, goal_request):
        """
        Called when a new goal is received.
        Decide whether to accept or reject the goal.
        """
        self.get_logger().info(f'Received goal request to pick object: {goal_request.object_name}')
        
        # You can add validation logic here
        if not goal_request.object_name or goal_request.object_name.strip() == '':
            self.get_logger().warn('Rejecting goal: object_name is empty')
            return GoalResponse.REJECT
        
        self.get_logger().info(f'Accepting goal to pick: {goal_request.object_name}')
        return GoalResponse.ACCEPT

    def handle_accepted_callback(self, goal_handle):
        """
        Called after a goal has been accepted.
        This is where you can spawn a thread or start execution.
        """
        self.get_logger().info(f'Goal accepted for object: {goal_handle.request.object_name}')
        # Execute immediately (in a separate thread due to ReentrantCallbackGroup)
        goal_handle.execute()

    def cancel_callback(self, goal_handle):
        """
        Called when a cancel request is received.
        Decide whether to accept or reject the cancellation.
        """
        self.get_logger().info(f'Received cancel request for object: {goal_handle.request.object_name}')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """
        Main execution logic for the action.
        This simulates picking an object over 10 seconds.
        """
        self.get_logger().info(f'Executing goal for object: {goal_handle.request.object_name}')
        
        # Prepare feedback message
        feedback_msg = PickObject.Feedback()
        
        # Simulate 10-second pick operation with feedback every second
        total_duration = 10
        for i in range(total_duration):
            # Check if goal has been canceled
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result = PickObject.Result()
                result.status_msg = f'Picking {goal_handle.request.object_name} was canceled'
                result.success = False
                self.get_logger().info(result.status_msg)
                return result
            
            # Create robot state as JSON
            robot_state = {
                'object_name': goal_handle.request.object_name,
                'progress_percent': int((i + 1) / total_duration * 100),
                'current_step': i + 1,
                'total_steps': total_duration,
                'gripper_status': 'approaching' if i < 5 else 'grasping',
                'arm_position': [0.1 * i, 0.2 * i, 0.3 * i],
                'timestamp': time.time()
            }
            
            # Convert to JSON string for feedback
            feedback_msg.json_state = json.dumps(robot_state)
            
            # Publish feedback
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(
                f'[{i+1}/{total_duration}] Picking {goal_handle.request.object_name}... '
                f'{robot_state["progress_percent"]}% complete'
            )
            
            # Wait 1 second
            time.sleep(1)
        
        # Mark goal as succeeded
        goal_handle.succeed()
        
        # Prepare result
        result = PickObject.Result()
        result.success = True
        result.status_msg = f'Picked {goal_handle.request.object_name} successfully'
        
        self.get_logger().info(f'SUCCESS: {result.status_msg}')
        return result


def main(args=None):
    rclpy.init(args=args)
    
    action_server = PickObjectActionServer()
    
    # Use MultiThreadedExecutor to handle concurrent callbacks
    executor = MultiThreadedExecutor()
    executor.add_node(action_server)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        action_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()