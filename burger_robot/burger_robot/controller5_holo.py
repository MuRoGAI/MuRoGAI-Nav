#!/usr/bin/env python3
"""
Holonomic Time-Synchronized Controller for Robot HoloTrajectory Tracking
Combines feedforward velocities with feedback correction
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from path_planner_interface.msg import HoloTrajectory
import math


def quaternion_to_euler(x, y, z, w):
    """
    Convert quaternion to euler angles (roll, pitch, yaw)
    
    Args:
        x, y, z, w: Quaternion components
    
    Returns:
        roll, pitch, yaw in radians
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


class TimeSyncController(Node):
    def __init__(self):
        super().__init__('time_sync_controller')
        
        # Declare parameters
        self.declare_parameter('kp_x', 2.5)
        self.declare_parameter('kp_y', 2.5)
        self.declare_parameter('max_vel_x', 2.2)
        self.declare_parameter('max_vel_y', 1.5)
        self.declare_parameter('use_feedforward', True)  # Use vx, vy from trajectory
        
        # Get parameters
        self.kp_x = self.get_parameter('kp_x').value
        self.kp_y = self.get_parameter('kp_y').value
        self.max_vel_x = self.get_parameter('max_vel_x').value
        self.max_vel_y = self.get_parameter('max_vel_y').value
        self.use_feedforward = self.get_parameter('use_feedforward').value
        
        # Robot state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        
        # Target trajectory
        self.target_times = []
        self.target_x = []
        self.target_y = []
        self.target_vx = []
        self.target_vy = []
        self.trajectory_received = False
        self.start_time = None
        self.trajectory_finished = False
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            '/burger1/odom',
            self.odom_callback,
            10
        )
        
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
        
        self.get_logger().info('Holonomic Time-Synchronized Controller initialized')
        self.get_logger().info(f'Kp_x: {self.kp_x}, Kp_y: {self.kp_y}')
        self.get_logger().info(f'Use feedforward: {self.use_feedforward}')
    
    def odom_callback(self, msg):
        """Update current robot position from odometry"""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        # Extract yaw from quaternion using custom function
        orientation_q = msg.pose.pose.orientation
        _, _, self.current_yaw = quaternion_to_euler(
            orientation_q.x, 
            orientation_q.y,
            orientation_q.z, 
            orientation_q.w
        )
    
    def target_callback(self, msg):
        """Receive target trajectory"""
        self.target_times = list(msg.time)
        self.target_x = list(msg.x)
        self.target_y = list(msg.y)
        self.target_vx = list(msg.vx)
        self.target_vy = list(msg.vy)
        
        # Validate arrays
        if not (len(msg.time) == len(msg.x) == len(msg.y) == len(msg.vx) == len(msg.vy)):
            self.get_logger().error('HoloTrajectory arrays must have equal length')
            return
        
        self.trajectory_received = True
        self.trajectory_finished = False
        self.start_time = self.get_clock().now()
        self.get_logger().info(f'New trajectory received with {len(self.target_times)} waypoints')
        self.get_logger().info(f'Trajectory duration: {self.target_times[-1]:.2f} seconds')
    
    def get_target_state(self):
        """Get target position and velocity based on current time"""
        if not self.trajectory_received or len(self.target_times) == 0:
            return None, None, None, None
        
        # Calculate elapsed time since trajectory start
        current_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        
        # Check if trajectory ended
        if current_time > self.target_times[-1]:
            if not self.trajectory_finished:
                self.trajectory_finished = True
                self.get_logger().info('Trajectory time ended - stopping robot')
            return None, None, None, None
        
        # Find the appropriate target based on time
        if current_time <= self.target_times[0]:
            return self.target_x[0], self.target_y[0], self.target_vx[0], self.target_vy[0]
        
        # Linear interpolation between waypoints
        for i in range(len(self.target_times) - 1):
            if self.target_times[i] <= current_time <= self.target_times[i + 1]:
                # Interpolation factor
                t = (current_time - self.target_times[i]) / \
                    (self.target_times[i + 1] - self.target_times[i])
                
                # Interpolate position
                target_x = self.target_x[i] + t * (self.target_x[i + 1] - self.target_x[i])
                target_y = self.target_y[i] + t * (self.target_y[i + 1] - self.target_y[i])
                
                # Interpolate velocity (or use current segment velocity)
                target_vx = self.target_vx[i] + t * (self.target_vx[i + 1] - self.target_vx[i])
                target_vy = self.target_vy[i] + t * (self.target_vy[i + 1] - self.target_vy[i])
                
                return target_x, target_y, target_vx, target_vy
        
        return self.target_x[-1], self.target_y[-1], self.target_vx[-1], self.target_vy[-1]
    
    def control_loop(self):
        """Main control loop - Feedforward + Feedback controller"""
        cmd = Twist()
        
        if not self.trajectory_received:
            # No trajectory, stop the robot
            self.cmd_vel_pub.publish(cmd)
            return
        
        # Get current target state
        target_x, target_y, target_vx, target_vy = self.get_target_state()
        
        if target_x is None:
            # Trajectory finished, stop the robot
            self.cmd_vel_pub.publish(cmd)
            return
        
        # Calculate elapsed time since trajectory start
        current_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        
        # === NEW: Find which trajectory segment we're in ===
        trajectory_time_point = current_time  # The time point in the trajectory we're querying
        current_segment = -1
        for i in range(len(self.target_times) - 1):
            if self.target_times[i] <= current_time <= self.target_times[i + 1]:
                current_segment = i
                break
        
        # Calculate position error in global frame
        error_x = target_x - self.current_x
        error_y = target_y - self.current_y
        distance_error = math.sqrt(error_x**2 + error_y**2)
        
        # Feedforward + Feedback control in global frame
        if self.use_feedforward:
            # Use feedforward velocities from trajectory + feedback correction
            vel_x_global = target_vx + self.kp_x * error_x
            vel_y_global = target_vy + self.kp_y * error_y
        else:
            # Pure feedback control (original P controller)
            vel_x_global = self.kp_x * error_x
            vel_y_global = self.kp_y * error_y
        
        # Transform global velocities to robot's local frame
        cos_yaw = math.cos(self.current_yaw)
        sin_yaw = math.sin(self.current_yaw)
        
        vel_x_local = cos_yaw * vel_x_global + sin_yaw * vel_y_global
        vel_y_local = -sin_yaw * vel_x_global + cos_yaw * vel_y_global
        
        # Apply velocity limits
        vel_x_local = max(min(vel_x_local, self.max_vel_x), -self.max_vel_x)
        vel_y_local = max(min(vel_y_local, self.max_vel_y), -self.max_vel_y)
        
        # Publish command
        cmd.linear.x = vel_x_local
        cmd.linear.y = vel_y_local
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)
        
        # === ENHANCED LOGGING ===
        segment_info = f'Segment: {current_segment}' if current_segment >= 0 else 'Before start'
        
        self.get_logger().info(
            f'Time: {current_time:.3f}s | TrajTime: {trajectory_time_point:.3f}s | {segment_info} | '
            f'Pos: ({self.current_x:.2f}, {self.current_y:.2f}) | Target: ({target_x:.2f}, {target_y:.2f}) | '
            f'Error: {distance_error:.3f}m | FF_Vel: ({target_vx:.2f}, {target_vy:.2f}) | '
            f'Cmd: ({vel_x_local:.2f}, {vel_y_local:.2f})',
            throttle_duration_sec=0.5
        )


def main(args=None):
    rclpy.init(args=args)
    controller = TimeSyncController()
    
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