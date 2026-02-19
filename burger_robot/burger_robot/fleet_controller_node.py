#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from path_planner_interface.msg import RobotFleetCmd, RobotCmd

# Global variable for the robot name this node controls
ROBOT_NAME = "burger1"


class RobotFleetController(Node):
    """
    ROS2 Node that subscribes to fleet commands and publishes cmd_vel
    for a specific robot based on the global ROBOT_NAME variable.
    Publishes all velocities from the arrays one by one using a timer.
    """

    def __init__(self):
        super().__init__('robot_fleet_controller')
        
        # Declare and get the robot name parameter (can override global ROBOT_NAME)
        self.declare_parameter('robot_name', ROBOT_NAME)
        self.robot_name = self.get_parameter('robot_name').get_parameter_value().string_value
        
        # Create subscriber for fleet commands
        self.subscription = self.create_subscription(
            RobotFleetCmd,
            'robot_fleet_cmd',
            self.fleet_cmd_callback,
            10
        )
        
        # Create publisher for this robot's cmd_vel
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            f'/{self.robot_name}/cmd_vel',
            10
        )
        
        # Storage for current velocity arrays
        self.current_velocities = None
        self.velocity_index = 0
        self.num_velocities = 0
        
        # Timer to publish velocities at 1 Hz (every 1 second)
        self.publish_timer = self.create_timer(1.0, self.publish_next_velocity)
        
        self.get_logger().info(f'Robot Fleet Controller initialized for robot: {self.robot_name}')
        self.get_logger().info(f'Subscribing to: robot_fleet_cmd')
        self.get_logger().info(f'Publishing to: /{self.robot_name}/cmd_vel')
        self.get_logger().info(f'Publishing rate: 1 Hz (every 1 second)')

    def fleet_cmd_callback(self, msg):
        """
        Callback function that processes fleet commands.
        Searches for this robot's name in the command and stores velocity arrays.
        
        Args:
            msg (RobotFleetCmd): The fleet command message containing
                                 an array of RobotCmd messages
        """
        # Search for this robot in the command
        for robot_cmd in msg.robot_cmds:
            if robot_cmd.robot_name == self.robot_name:
                # Found matching robot, store velocity arrays
                self.store_velocity_arrays(robot_cmd)
                return
        
        # Robot name not found in this command
        self.get_logger().debug(f'Robot {self.robot_name} not found in fleet command')

    def store_velocity_arrays(self, robot_cmd):
        """
        Store the velocity arrays for sequential publishing.
        
        Args:
            robot_cmd (RobotCmd): Robot command with velocity component arrays
        """
        # Get the length of velocity arrays
        self.num_velocities = max(
            len(robot_cmd.lin_x) if robot_cmd.lin_x else 0,
            len(robot_cmd.lin_y) if robot_cmd.lin_y else 0,
            len(robot_cmd.lin_z) if robot_cmd.lin_z else 0,
            len(robot_cmd.ang_x) if robot_cmd.ang_x else 0,
            len(robot_cmd.ang_y) if robot_cmd.ang_y else 0,
            len(robot_cmd.ang_z) if robot_cmd.ang_z else 0
        )
        
        if self.num_velocities == 0:
            self.get_logger().warn(f'No velocity data for robot {self.robot_name}')
            self.current_velocities = None
            return
        
        # Store the velocity arrays
        self.current_velocities = robot_cmd
        self.velocity_index = 0  # Reset to start from beginning
        
        self.get_logger().info(
            f'Received {self.num_velocities} velocities for {self.robot_name}. '
            f'Will publish one every second.'
        )

    def publish_next_velocity(self):
        """
        Timer callback that publishes the next velocity from the arrays.
        Called every second.
        """
        # Check if we have velocities to publish
        if self.current_velocities is None or self.num_velocities == 0:
            return
        
        # Check if we've published all velocities
        if self.velocity_index >= self.num_velocities:
            self.get_logger().info(f'Finished publishing all {self.num_velocities} velocities')
            self.velocity_index = 0  # Reset to loop again
            return
        
        # Create Twist message from current index
        twist = Twist()
        
        idx = self.velocity_index
        
        # Safely get values at current index (use 0.0 if array is shorter)
        if self.current_velocities.lin_x and idx < len(self.current_velocities.lin_x):
            twist.linear.x = self.current_velocities.lin_x[idx]
        
        if self.current_velocities.lin_y and idx < len(self.current_velocities.lin_y):
            twist.linear.y = self.current_velocities.lin_y[idx]
        
        if self.current_velocities.lin_z and idx < len(self.current_velocities.lin_z):
            twist.linear.z = self.current_velocities.lin_z[idx]
        
        if self.current_velocities.ang_x and idx < len(self.current_velocities.ang_x):
            twist.angular.x = self.current_velocities.ang_x[idx]
        
        if self.current_velocities.ang_y and idx < len(self.current_velocities.ang_y):
            twist.angular.y = self.current_velocities.ang_y[idx]
        
        if self.current_velocities.ang_z and idx < len(self.current_velocities.ang_z):
            twist.angular.z = self.current_velocities.ang_z[idx]
        
        # Publish the velocity
        self.cmd_vel_publisher.publish(twist)
        
        self.get_logger().info(
            f'[{self.velocity_index + 1}/{self.num_velocities}] Publishing cmd_vel for {self.robot_name}: '
            f'linear=({twist.linear.x:.2f}, {twist.linear.y:.2f}, {twist.linear.z:.2f}), '
            f'angular=({twist.angular.x:.2f}, {twist.angular.y:.2f}, {twist.angular.z:.2f})'
        )
        
        # Move to next velocity
        self.velocity_index += 1


def main(args=None):
    rclpy.init(args=args)
    
    node = RobotFleetController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()