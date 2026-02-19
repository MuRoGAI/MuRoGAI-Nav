#!/usr/bin/env python3
"""
Improved Differential Drive Time-Synchronized Controller
With PROPER REVERSE DRIVING - rotates 180° when needed
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from path_planner_interface.msg import DiffDriveTrajectory
import math


def quaternion_to_euler(x, y, z, w):
    """Convert quaternion to euler angles (roll, pitch, yaw)"""
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


def normalize_angle(angle):
    """Normalize angle to [-pi, pi]"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


class ImprovedDiffDriveController(Node):
    def __init__(self):
        super().__init__('diff_drive_controller')
        
        # Declare parameters
        self.declare_parameter('kp_linear', 6.3)
        self.declare_parameter('kd_linear', 1.2)
        self.declare_parameter('kp_angular', 8.0)
        self.declare_parameter('kd_angular', 0.8)
        self.declare_parameter('max_linear_vel', 1.2)
        self.declare_parameter('max_angular_vel', 4.0)
        self.declare_parameter('use_feedforward', False)
        self.declare_parameter('orientation_blend_distance', 2.0)
        self.declare_parameter("robot_name", "robot")

        
        # NEW: Reverse driving parameters
        self.declare_parameter('enable_reverse', True)  # Allow reverse driving
        self.declare_parameter('reverse_threshold_angle', 100.0)  # degrees - when to use reverse
        
        # Get parameters
        self.kp_linear = self.get_parameter('kp_linear').value
        self.kd_linear = self.get_parameter('kd_linear').value
        self.kp_angular = self.get_parameter('kp_angular').value
        self.kd_angular = self.get_parameter('kd_angular').value
        self.max_linear_vel = self.get_parameter('max_linear_vel').value
        self.max_angular_vel = self.get_parameter('max_angular_vel').value
        self.use_feedforward = self.get_parameter('use_feedforward').value
        self.orientation_blend_distance = self.get_parameter('orientation_blend_distance').value
        self.enable_reverse = self.get_parameter('enable_reverse').value
        self.reverse_threshold_angle = math.radians(self.get_parameter('reverse_threshold_angle').value)
        self.robot_name = self.get_parameter("robot_name").value

        
        # Robot state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0
        
        # Target trajectory
        self.target_times = []
        self.target_x = []
        self.target_y = []
        self.target_theta = []
        self.trajectory_received = False
        self.start_time = None
        self.trajectory_finished = False
        
        # PD control variables
        self.prev_error_distance = 0.0
        self.prev_error_theta = 0.0
        self.prev_time = None
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            f'/{self.robot_name}/odom_world',
            self.odom_callback,
            10
        )
        
        self.target_sub = self.create_subscription(
            DiffDriveTrajectory,
            '/diff_target',
            self.target_callback,
            10
        )
        
        # Publisher
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            f'/{self.robot_name}/diff_drive_controller/cmd_vel',
            10
        )
        
        # Control loop timer (50 Hz)
        self.control_dt = 0.02
        self.timer = self.create_timer(self.control_dt, self.control_loop)
        
        self.get_logger().info('Improved Differential Drive Controller initialized')
        self.get_logger().info(f'Kp_linear: {self.kp_linear}, Kd_linear: {self.kd_linear}')
        self.get_logger().info(f'Kp_angular: {self.kp_angular}, Kd_angular: {self.kd_angular}')
        self.get_logger().info(f'Max velocities: linear={self.max_linear_vel} m/s, angular={self.max_angular_vel} rad/s')
        self.get_logger().info(f'Reverse enabled: {self.enable_reverse}, threshold: {math.degrees(self.reverse_threshold_angle):.0f}°')
    
    def odom_callback(self, msg):
        """Update current robot position and orientation from odometry"""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        orientation_q = msg.pose.pose.orientation
        _, _, self.current_theta = quaternion_to_euler(
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
        self.target_theta = list(msg.theta)
        
        if not (len(msg.time) == len(msg.x) == len(msg.y) == len(msg.theta)):
            self.get_logger().error('DiffDriveTrajectory arrays must have equal length')
            return
        
        self.trajectory_received = True
        self.trajectory_finished = False
        self.start_time = self.get_clock().now()
        
        self.prev_error_distance = 0.0
        self.prev_error_theta = 0.0
        self.prev_time = None
        
        self.get_logger().info(f'New trajectory received with {len(self.target_times)} waypoints')
        self.get_logger().info(f'Trajectory duration: {self.target_times[-1]:.2f} seconds')
    
    def get_target_state(self):
        """Get target position, orientation, and segment index"""
        if not self.trajectory_received or len(self.target_times) == 0:
            return None, None, None, -1
        
        current_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        
        if current_time > self.target_times[-1]:
            if not self.trajectory_finished:
                self.trajectory_finished = True
                self.get_logger().info('Trajectory time ended - stopping robot')
            return None, None, None, -1
        
        if current_time <= self.target_times[0]:
            return self.target_x[0], self.target_y[0], self.target_theta[0], 0
        
        # Find segment and interpolate
        for i in range(len(self.target_times) - 1):
            if self.target_times[i] <= current_time <= self.target_times[i + 1]:
                t = (current_time - self.target_times[i]) / \
                    (self.target_times[i + 1] - self.target_times[i])
                
                target_x = self.target_x[i] + t * (self.target_x[i + 1] - self.target_x[i])
                target_y = self.target_y[i] + t * (self.target_y[i + 1] - self.target_y[i])
                
                theta_diff = normalize_angle(self.target_theta[i + 1] - self.target_theta[i])
                target_theta = normalize_angle(self.target_theta[i] + t * theta_diff)
                
                return target_x, target_y, target_theta, i
        
        return self.target_x[-1], self.target_y[-1], self.target_theta[-1], len(self.target_times) - 2
    
    def get_target_velocity(self, segment_idx):
        """Calculate target velocity from trajectory (for feedforward)"""
        if segment_idx < 0 or segment_idx >= len(self.target_times) - 1:
            return 0.0, 0.0
        
        dt = self.target_times[segment_idx + 1] - self.target_times[segment_idx]
        if dt <= 0:
            return 0.0, 0.0
        
        vx = (self.target_x[segment_idx + 1] - self.target_x[segment_idx]) / dt
        vy = (self.target_y[segment_idx + 1] - self.target_y[segment_idx]) / dt
        
        target_speed = math.sqrt(vx**2 + vy**2)
        target_angle = math.atan2(vy, vx)
        
        return target_speed, target_angle
    
    def control_loop(self):
        """FIXED control loop with proper reverse driving"""
        cmd = Twist()
        
        if not self.trajectory_received:
            self.cmd_vel_pub.publish(cmd)
            return
        
        target_x, target_y, target_theta, segment_idx = self.get_target_state()
        
        if target_x is None:
            self.cmd_vel_pub.publish(cmd)
            return
        
        current_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        
        # Calculate position error
        error_x = target_x - self.current_x
        error_y = target_y - self.current_y
        distance_error = math.sqrt(error_x**2 + error_y**2)
        
        # Calculate angle to target
        angle_to_target = math.atan2(error_y, error_x)
        
        # Orientation error
        error_theta = normalize_angle(target_theta - self.current_theta)
        
        # CRITICAL FIX: Determine if we should drive in reverse
        heading_error = normalize_angle(angle_to_target - self.current_theta)
        
        # Decide forward or reverse based on heading error
        drive_reverse = False
        if self.enable_reverse and abs(heading_error) > self.reverse_threshold_angle:
            drive_reverse = True
            # When reversing, flip the heading by 180°
            angle_to_target = normalize_angle(angle_to_target + math.pi)
            heading_error = normalize_angle(angle_to_target - self.current_theta)
        
        # Calculate derivatives
        if self.prev_time is not None:
            dt = current_time - self.prev_time
            if dt > 0:
                d_error_distance = (distance_error - self.prev_error_distance) / dt
                d_error_theta = (error_theta - self.prev_error_theta) / dt
            else:
                d_error_distance = 0.0
                d_error_theta = 0.0
        else:
            d_error_distance = 0.0
            d_error_theta = 0.0
        
        # Smooth blending based on distance
        orientation_weight = max(0.0, min(1.0, distance_error / self.orientation_blend_distance))
        heading_weight = 1.0 - orientation_weight
        
        # Blended angular error
        blended_angular_error = heading_weight * heading_error + orientation_weight * error_theta
        
        # Get feedforward velocity if enabled
        if self.use_feedforward:
            target_speed, _ = self.get_target_velocity(segment_idx)
        else:
            target_speed = 0.0
        
        # LINEAR VELOCITY CONTROL
        linear_vel = target_speed + \
                     self.kp_linear * distance_error * math.cos(heading_error) + \
                     self.kd_linear * d_error_distance
        
        # If driving in reverse, flip the linear velocity sign
        if drive_reverse:
            linear_vel = -abs(linear_vel)
        
        # ANGULAR VELOCITY CONTROL
        angular_vel = self.kp_angular * blended_angular_error + \
                      self.kd_angular * d_error_theta
        
        # Apply velocity limits
        linear_vel = max(min(linear_vel, self.max_linear_vel), -self.max_linear_vel)
        angular_vel = max(min(angular_vel, self.max_angular_vel), -self.max_angular_vel)
        
        # Publish command
        cmd.linear.x = linear_vel
        cmd.linear.y = 0.0
        cmd.angular.z = angular_vel
        self.cmd_vel_pub.publish(cmd)
        
        # Update previous values
        self.prev_error_distance = distance_error
        self.prev_error_theta = error_theta
        self.prev_time = current_time
        
        # Logging
        lin_saturated = "SAT" if abs(linear_vel) >= self.max_linear_vel * 0.99 else ""
        ang_saturated = "SAT" if abs(angular_vel) >= self.max_angular_vel * 0.99 else ""
        direction = "REV" if drive_reverse else "FWD"
        
        self.get_logger().info(
            f'T:{current_time:.2f}s | Seg:{segment_idx} | {direction} | '
            f'Pos:({self.current_x:.2f},{self.current_y:.2f},{math.degrees(self.current_theta):.0f}°) | '
            f'Tgt:({target_x:.2f},{target_y:.2f},{math.degrees(target_theta):.0f}°) | '
            f'Err_d:{distance_error:.3f}m | Err_ψ:{math.degrees(error_theta):.0f}° | '
            f'Head_err:{math.degrees(heading_error):.0f}° | '
            f'Cmd:v={linear_vel:.2f}{lin_saturated} ω={math.degrees(angular_vel):.0f}°/s{ang_saturated}',
            throttle_duration_sec=0.5
        )


def main(args=None):
    rclpy.init(args=args)
    controller = ImprovedDiffDriveController()
    
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