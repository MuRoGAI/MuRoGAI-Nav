#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import json
import math

class PController(Node):
    def __init__(self):
        super().__init__('p_controller_holonomic')
        
        # ----- Gains -----
        self.kp_x = 1.5
        self.kp_y = 1.5
        self.kp_yaw = 2.0
        
        # ----- Current Pose -----
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        
        # ----- Waypoints -----
        self.waypoints = []
        self.current_waypoint = 0
        self.mission_complete = False
        
        # ----- Subscribers -----
        self.create_subscription(
            String,
            '/burger1/targets',
            self.targets_callback,
            10
        )
        
        self.create_subscription(
            Odometry,
            '/burger1/odom',
            self.odom_callback,
            10
        )
        
        # ----- Publisher -----
        self.cmd_pub = self.create_publisher(Twist, '/burger1/cmd_vel', 10)
        
        # ----- Control Timer -----
        self.create_timer(0.11, self.control_loop)
        
        self.get_logger().info("P Controller Ready")
    
    def targets_callback(self, msg):
        data = json.loads(msg.data)
        xs = data["x"]
        ys = data["y"]
        yaws = data["yaw"]
        
        # Store waypoints as list of tuples
        self.waypoints = [(xs[i], ys[i], yaws[i]) for i in range(len(xs))]
        self.current_waypoint = 0
        self.mission_complete = False
        
        self.get_logger().info(f"Received {len(self.waypoints)} waypoints")
    
    def odom_callback(self, msg):
        # Get position
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        
        # Convert quaternion to yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)
    
    def control_loop(self):
        # If mission is complete, do nothing
        if self.mission_complete:
            return
        
        # If no waypoints, do nothing
        if len(self.waypoints) == 0:
            return
        
        # If all waypoints visited, stop and end mission
        if self.current_waypoint >= len(self.waypoints):
            cmd = Twist()  # Zero velocity
            self.cmd_pub.publish(cmd)
            self.mission_complete = True
            self.get_logger().info("Mission Complete - Robot Stopped")
            return
        
        # Get current target
        target_x, target_y, target_yaw = self.waypoints[self.current_waypoint]
        
        # Calculate errors
        error_x = target_x - self.x
        error_y = target_y - self.y
        error_yaw = target_yaw - self.yaw
        
        # Normalize yaw error to [-pi, pi]
        error_yaw = math.atan2(math.sin(error_yaw), math.cos(error_yaw))
        
        # Thresholds
        position_threshold = 0.05  # 5 cm
        yaw_threshold = 0.05       # ~3 degrees
        
        # Check if waypoint reached
        distance = math.sqrt(error_x**2 + error_y**2)
        
        if distance < position_threshold and abs(error_yaw) < yaw_threshold:
            # Waypoint reached, move to next
            self.get_logger().info(f"Reached waypoint {self.current_waypoint}")
            self.current_waypoint += 1
            return
        
        # Create velocity command
        cmd = Twist()
        
        # P-control for position
        cmd.linear.x = self.kp_x * error_x
        cmd.linear.y = self.kp_y * error_y
        
        # P-control for orientation
        cmd.angular.z = self.kp_yaw * error_yaw
        
        # Publish command
        self.cmd_pub.publish(cmd)

def main():
    rclpy.init()
    controller = PController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()