#!/usr/bin/env python3
"""
Feedforward Controller for Robot HoloTrajectory Tracking
Subscribes to target trajectory and directly publishes vx, vy commands
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from path_planner_interface.msg import HoloTrajectory


class FeedforwardController(Node):
    def __init__(self):
        super().__init__('feedforward_controller')
        
        # Declare parameters
        self.declare_parameter('max_vel_x', 2.2)
        self.declare_parameter('max_vel_y', 1.5)
        
        # Get parameters
        self.max_vel_x = self.get_parameter('max_vel_x').value
        self.max_vel_y = self.get_parameter('max_vel_y').value
        
        # Target trajectory
        self.target_times = []
        self.target_vx = []
        self.target_vy = []
        self.trajectory_received = False
        self.start_time = None
        self.trajectory_finished = False
        
        # Subscriber
        self.target_sub = self.create_subscription(
            HoloTrajectory,
            '/target',
            self.target_callback,
            10
        )
        
        # Publisher
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/burger1/cmd_vel',
            10
        )
        
        # Control loop timer (50 Hz)
        self.timer = self.create_timer(0.02, self.control_loop)
        
        self.get_logger().info('Feedforward Controller initialized')
    
    def target_callback(self, msg):
        """Receive target trajectory"""
        self.target_times = list(msg.time)
        self.target_vx = list(msg.vx)
        self.target_vy = list(msg.vy)
        
        # Validate arrays have equal length
        if not (len(msg.time) == len(msg.vx) == len(msg.vy)):
            self.get_logger().error('HoloTrajectory time, vx, vy arrays must have equal length')
            return
        
        self.trajectory_received = True
        self.trajectory_finished = False
        self.start_time = self.get_clock().now()
        self.get_logger().info(f'New trajectory received with {len(self.target_times)} waypoints')
        self.get_logger().info(f'HoloTrajectory duration: {self.target_times[-1]:.2f} seconds')
    
    def get_target_velocity(self):
        """Get target velocity based on current time"""
        if not self.trajectory_received or len(self.target_times) == 0:
            return None, None
        
        # Calculate elapsed time since trajectory start
        current_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        
        # Check if trajectory time has ended
        if current_time > self.target_times[-1]:
            if not self.trajectory_finished:
                self.trajectory_finished = True
                self.get_logger().info('HoloTrajectory time ended - stopping robot')
            return None, None
        
        # Find the appropriate velocity based on time
        # Use the velocity from the current segment (hold until next waypoint)
        if current_time <= self.target_times[0]:
            return self.target_vx[0], self.target_vy[0]
        
        # Find which time segment we're in
        for i in range(len(self.target_times) - 1):
            if self.target_times[i] <= current_time < self.target_times[i + 1]:
                # Use velocity from current waypoint (hold constant until next waypoint)
                return self.target_vx[i], self.target_vy[i]
        
        # If we're at or past the last waypoint time
        return self.target_vx[-1], self.target_vy[-1]
    
    def control_loop(self):
        """Main control loop - Direct feedforward velocity publishing"""
        cmd = Twist()
        
        if not self.trajectory_received:
            # No trajectory, stop the robot
            self.cmd_vel_pub.publish(cmd)
            return
        
        # Get current target velocity
        target_vx, target_vy = self.get_target_velocity()
        
        if target_vx is None:
            # HoloTrajectory finished or not available, stop the robot
            self.cmd_vel_pub.publish(cmd)
            return
        
        # Apply velocity limits
        vel_x = max(min(target_vx, self.max_vel_x), -self.max_vel_x)
        vel_y = max(min(target_vy, self.max_vel_y), -self.max_vel_y)
        
        # Directly publish vx, vy (no transformation needed)
        cmd.linear.x = vel_x
        cmd.linear.y = vel_y
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)
        
        # Log status periodically
        current_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        self.get_logger().info(
            f'Time: {current_time:.2f}s | '
            f'Cmd Vel: ({vel_x:.2f}, {vel_y:.2f})',
            throttle_duration_sec=0.5
        )


def main(args=None):
    rclpy.init(args=args)
    controller = FeedforwardController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the robot
        stop_cmd = Twist()
        controller.cmd_vel_pub.publish(stop_cmd)
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()