#!/usr/bin/env python3
"""
Holonomic Time-Synchronized Position-Only Controller with PD Control
Uses only position targets (no velocity feedforward)
PD controller based on position error and its derivative
UPDATED: Receives RobotTrajectoryArray and filters by robot_name
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from path_planner_interface.msg import HoloTrajectory, RobotTrajectoryArray
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


class PositionOnlyController(Node):
    def __init__(self):
        super().__init__('holonomic_controller')
        
        # Declare parameters
        self.declare_parameter('kp_x', 4.0)
        self.declare_parameter('kp_y', 4.0)
        self.declare_parameter('kd_x', 0.6)  # Derivative gain for x
        self.declare_parameter('kd_y', 0.6)  # Derivative gain for y
        self.declare_parameter('max_vel_x', 0.25)
        self.declare_parameter('max_vel_y', 0.25)
        self.declare_parameter("robot_name", "robot")

        
        # Get parameters
        self.kp_x = self.get_parameter('kp_x').value
        self.kp_y = self.get_parameter('kp_y').value
        self.kd_x = self.get_parameter('kd_x').value
        self.kd_y = self.get_parameter('kd_y').value
        self.max_vel_x = self.get_parameter('max_vel_x').value
        self.max_vel_y = self.get_parameter('max_vel_y').value
        self.robot_name = self.get_parameter("robot_name").value

        # Waypoint logging (event-based)
        self.next_wp_idx = 0

        # Continuous tracking log (while moving)
        self.last_track_log_time = 0.0
        
        # Robot state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        
        # Target trajectory (position only)
        self.target_times = []
        self.target_x = []
        self.target_y = []
        self.trajectory_received = False
        self.start_time = None
        self.trajectory_finished = False
        
        # PD control variables
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.prev_time = None
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            f'/{self.robot_name}/odom_world',
            self.odom_callback,
            10
        )
        
        # UPDATED: Subscribe to RobotTrajectoryArray instead of HoloTrajectory
        self.target_sub = self.create_subscription(
            RobotTrajectoryArray,
            '/path_planner/paths',
            self.trajectory_array_callback,
            10
        )
        
        # Publisher
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            f'/{self.robot_name}/cmd_vel',
            10
        )
        
        # Control loop timer (50 Hz)
        self.control_dt = 0.02
        self.timer = self.create_timer(self.control_dt, self.control_loop)
        
        self.get_logger().info('Position-Only Time-Synchronized PD Controller initialized')
        self.get_logger().info(f'Robot name: {self.robot_name}')
        self.get_logger().info(f'Kp_x: {self.kp_x}, Kp_y: {self.kp_y}')
        self.get_logger().info(f'Kd_x: {self.kd_x}, Kd_y: {self.kd_y}')
        self.get_logger().info('Mode: PD controller (no feedforward)')
    
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
    
    def trajectory_array_callback(self, msg):
        """
        Receive RobotTrajectoryArray and extract trajectory for this robot.
        Filters by robot_name to get only the relevant trajectory.
        """
        # Search for this robot's trajectory in the array
        for robot_traj in msg.robot_trajectories:
            if robot_traj.robot_name == self.robot_name:
                # Check if robot type is holonomic
                if robot_traj.robot_type != 'holonomic':
                    self.get_logger().warn(
                        f'Received trajectory for {self.robot_name} but type is {robot_traj.robot_type}, '
                        f'expected holonomic'
                    )
                    continue
                
                # Extract HoloTrajectory (should have at least one)
                if len(robot_traj.holo_trajectories) == 0:
                    self.get_logger().warn(
                        f'No holonomic trajectories found for {self.robot_name}'
                    )
                    continue
                
                # Take the first (and typically only) holonomic trajectory
                holo_traj = robot_traj.holo_trajectories[0]
                
                # Process the trajectory
                self.process_holo_trajectory(holo_traj)
                
                self.get_logger().info(
                    f'Received trajectory for {self.robot_name}: '
                    f'{len(holo_traj.time)} waypoints, '
                    f'duration: {holo_traj.time[-1] if holo_traj.time else 0.0:.2f}s'
                )
                
                return  # Found our robot, no need to continue
        
        # If we get here, no trajectory was found for this robot
        self.get_logger().debug(
            f'No trajectory found for robot {self.robot_name} in received array'
        )
    
    def process_holo_trajectory(self, traj_msg):
        """Process a HoloTrajectory message - only using time, x, y"""
        self.target_times = list(traj_msg.time)
        self.target_x = list(traj_msg.x)
        self.target_y = list(traj_msg.y)
        self.next_wp_idx = 0

        # Ignoring vx and vy from the message
        
        # Validate arrays
        if not (len(traj_msg.time) == len(traj_msg.x) == len(traj_msg.y)):
            self.get_logger().error('HoloTrajectory time, x, y arrays must have equal length')
            return
        
        self.trajectory_received = True
        self.trajectory_finished = False
        self.start_time = self.get_clock().now()
        
        # Reset PD control variables
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.prev_time = None
        
        self.get_logger().info(f'New trajectory received with {len(self.target_times)} waypoints')
        self.get_logger().info(f'Trajectory duration: {self.target_times[-1]:.2f} seconds')
    
    def get_target_position(self):
        """Get target position based on current time (no velocity)"""
        if not self.trajectory_received or len(self.target_times) == 0:
            return None, None
        
        # Calculate elapsed time since trajectory start
        current_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        
        # Check if trajectory ended
        if current_time > self.target_times[-1]:
            if not self.trajectory_finished:
                self.trajectory_finished = True
                self.get_logger().info('Trajectory time ended - stopping robot')
            return None, None
        
        # Find the appropriate target based on time
        if current_time <= self.target_times[0]:
            return self.target_x[0], self.target_y[0]
        
        # Linear interpolation between waypoints
        for i in range(len(self.target_times) - 1):
            if self.target_times[i] <= current_time <= self.target_times[i + 1]:
                # Interpolation factor
                t = (current_time - self.target_times[i]) / \
                    (self.target_times[i + 1] - self.target_times[i])
                
                # Interpolate position only
                target_x = self.target_x[i] + t * (self.target_x[i + 1] - self.target_x[i])
                target_y = self.target_y[i] + t * (self.target_y[i + 1] - self.target_y[i])
                
                return target_x, target_y
        
        return self.target_x[-1], self.target_y[-1]
    
    def control_loop(self):
        """Main control loop - PD controller (position error + derivative)"""
        cmd = Twist()
        
        if not self.trajectory_received:
            # No trajectory, stop the robot
            self.cmd_vel_pub.publish(cmd)
            return
        
        # Get current target position
        target_x, target_y = self.get_target_position()
        
        if target_x is None:
            # Trajectory finished, stop the robot
            self.cmd_vel_pub.publish(cmd)
            return
        
        # Calculate elapsed time since trajectory start
        current_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        # ================== WAYPOINT TIME-CROSSING LOG ==================
        if self.next_wp_idx < len(self.target_times):

            wp_time = self.target_times[self.next_wp_idx]

            if (self.prev_time is None and current_time >= wp_time) or \
            (self.prev_time is not None and self.prev_time < wp_time <= current_time):

                wp_x = self.target_x[self.next_wp_idx]
                wp_y = self.target_y[self.next_wp_idx]

                wp_pos_err = math.hypot(
                    self.current_x - wp_x,
                    self.current_y - wp_y
                )

                wp_time_err = current_time - wp_time

                self.get_logger().info(
                    f'[WP {self.next_wp_idx}] '
                    f'T_plan={wp_time:.2f}s | '
                    f'T_act={current_time:.2f}s | '
                    f'ΔT={wp_time_err:+.3f}s | '
                    f'PosErr={wp_pos_err:.3f}m'
                )

                self.next_wp_idx += 1
        # ===============================================================

        
        # Find which trajectory segment we're in
        current_segment = -1
        for i in range(len(self.target_times) - 1):
            if self.target_times[i] <= current_time <= self.target_times[i + 1]:
                current_segment = i
                break
        
        # ========== CONTINUOUS TRACKING LOG (WHILE MOVING) ==========
        if current_segment >= 0:

            wp_time = self.target_times[current_segment]
            wp_x = self.target_x[current_segment]
            wp_y = self.target_y[current_segment]

            cont_time_err = current_time - wp_time
            cont_pos_err = math.hypot(
                self.current_x - wp_x,
                self.current_y - wp_y
            )

            # Throttle manually (independent of ROS throttle)
            if current_time - self.last_track_log_time > 0.5:
                self.last_track_log_time = current_time

                self.get_logger().info(
                    f'[TRACK] WP~{current_segment} | '
                    f'ΔT={cont_time_err:+.3f}s | '
                    f'PosErr={cont_pos_err:.3f}m'
                )
        # ============================================================


        # Calculate position error in global frame
        error_x = target_x - self.current_x
        error_y = target_y - self.current_y
        distance_error = math.sqrt(error_x**2 + error_y**2)
        
        # Calculate derivative of error
        if self.prev_time is not None:
            dt = current_time - self.prev_time
            if dt > 0:
                d_error_x = (error_x - self.prev_error_x) / dt
                d_error_y = (error_y - self.prev_error_y) / dt
            else:
                d_error_x = 0.0
                d_error_y = 0.0
        else:
            # First iteration - use control loop dt
            d_error_x = 0.0
            d_error_y = 0.0
        
        # PD control in global frame (no feedforward)
        vel_x_global = self.kp_x * error_x + self.kd_x * d_error_x
        vel_y_global = self.kp_y * error_y + self.kd_y * d_error_y
        
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
        
        # Update previous values for next iteration
        self.prev_error_x = error_x
        self.prev_error_y = error_y
        self.prev_time = current_time
        
        # Logging
        segment_info = f'Segment: {current_segment}' if current_segment >= 0 else 'Before start'
        
        self.get_logger().info(
            f'Time: {current_time:.3f}s | {segment_info} | '
            f'Pos: ({self.current_x:.2f}, {self.current_y:.2f}) | '
            f'Target: ({target_x:.2f}, {target_y:.2f}) | '
            f'Error: {distance_error:.3f}m | '
            f'D_err: ({d_error_x:.2f}, {d_error_y:.2f}) | '
            f'Cmd: ({vel_x_local:.2f}, {vel_y_local:.2f})',
            throttle_duration_sec=0.5
        )


def main(args=None):
    rclpy.init(args=args)
    controller = PositionOnlyController()
    
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