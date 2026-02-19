#!/usr/bin/env python3
"""
Differential Drive Feedforward Controller
Directly publishes v and omega from trajectory
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from path_planner_interface.msg import DiffDriveTrajectory


class DiffDriveFeedforwardController(Node):
    def __init__(self):
        super().__init__('diff_drive_feedforward_controller')
        
        # Declare parameters
        self.declare_parameter('max_linear_vel', 0.5)
        self.declare_parameter('max_angular_vel', 1.0)
        
        # Get parameters
        self.max_linear_vel = self.get_parameter('max_linear_vel').value
        self.max_angular_vel = self.get_parameter('max_angular_vel').value
        
        # Target trajectory
        self.target_times = []
        self.target_v = []
        self.target_omega = []
        self.trajectory_received = False
        self.start_time = None
        self.trajectory_finished = False
        
        # Subscriber
        self.target_sub = self.create_subscription(
            DiffDriveTrajectory,
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
        
        self.get_logger().info('Differential Drive Feedforward Controller initialized')
    
    def target_callback(self, msg):
        """Receive target trajectory"""
        self.target_times = list(msg.time)
        self.target_v = list(msg.v)
        self.target_omega = list(msg.omega)
        
        # Validate arrays
        if not (len(msg.time) == len(msg.v) == len(msg.omega)):
            self.get_logger().error('DiffDriveTrajectory time, v, omega arrays must have equal length')
            return
        
        self.trajectory_received = True
        self.trajectory_finished = False
        self.start_time = self.get_clock().now()
        self.get_logger().info(f'New trajectory received with {len(self.target_times)} waypoints')
        self.get_logger().info(f'DiffDriveTrajectory duration: {self.target_times[-1]:.2f} seconds')
    
    def get_target_velocity(self):
        """Get target velocities based on current time"""
        if not self.trajectory_received or len(self.target_times) == 0:
            return None, None
        
        # Calculate elapsed time
        current_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        
        # Check if trajectory ended
        if current_time > self.target_times[-1]:
            if not self.trajectory_finished:
                self.trajectory_finished = True
                self.get_logger().info('DiffDriveTrajectory time ended - stopping robot')
            return None, None
        
        # Find velocity for current time segment
        if current_time <= self.target_times[0]:
            return self.target_v[0], self.target_omega[0]
        
        # Use velocity from current segment (hold until next waypoint)
        for i in range(len(self.target_times) - 1):
            if self.target_times[i] <= current_time < self.target_times[i + 1]:
                return self.target_v[i], self.target_omega[i]
        
        return self.target_v[-1], self.target_omega[-1]
    
    def control_loop(self):
        """Main control loop - Direct velocity publishing"""
        cmd = Twist()
        
        if not self.trajectory_received:
            self.cmd_vel_pub.publish(cmd)
            return
        
        # Get target velocities
        target_v, target_omega = self.get_target_velocity()
        
        if target_v is None:
            self.cmd_vel_pub.publish(cmd)
            return
        
        # Apply velocity limits
        linear_vel = max(min(target_v, self.max_linear_vel), -self.max_linear_vel)
        angular_vel = max(min(target_omega, self.max_angular_vel), -self.max_angular_vel)
        
        # Publish command
        cmd.linear.x = linear_vel
        cmd.angular.z = angular_vel
        self.cmd_vel_pub.publish(cmd)
        
        # Log status
        current_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        self.get_logger().info(
            f'Time: {current_time:.2f}s | '
            f'Cmd: v={linear_vel:.2f}, omega={angular_vel:.2f}',
            throttle_duration_sec=0.5
        )


def main(args=None):
    rclpy.init(args=args)
    controller = DiffDriveFeedforwardController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        stop_cmd = Twist()
        controller.cmd_vel_pub.publish(stop_cmd)
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()