#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from path_planner_interface.msg import RobotFleetCmd, RobotCmd


class BurgerFleetPublisher(Node):
    """
    Publisher node that sends commands to burger1, burger2, and burger3.
    Each robot gets MANY velocity values in arrays.
    """

    def __init__(self):
        super().__init__('burger_fleet_publisher')
        
        self.publisher = self.create_publisher(
            RobotFleetCmd,
            'robot_fleet_cmd',
            10
        )
        
        # Publish commands at 2 Hz
        # self.timer = self.create_timer(0.5, self.publish_burger_commands)
        self.publish_burger_commands()
        
        # self.counter = 0
        self.get_logger().info('Burger Fleet Publisher started')
        self.get_logger().info('Publishing commands for: burger1, burger2, burger3')

    def publish_burger_commands(self):
        """
        Publishes commands with LOTS of velocity values for each robot.
        """
        msg = RobotFleetCmd()
        
        # ========== BURGER1 - 20 velocity values ==========
        burger1_cmd = RobotCmd()
        burger1_cmd.robot_name = 'burger1'
        burger1_cmd.lin_x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        burger1_cmd.lin_y = [0.0, 0.00, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        burger1_cmd.lin_z = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        burger1_cmd.ang_x = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        burger1_cmd.ang_y = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        burger1_cmd.ang_z = [0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.5]
        
        # ========== BURGER2 - 25 velocity values ==========
        burger2_cmd = RobotCmd()
        burger2_cmd.robot_name = 'burger2'
        burger2_cmd.lin_x = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4]
        burger2_cmd.lin_y = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48]
        burger2_cmd.lin_z = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        burger2_cmd.ang_x = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        burger2_cmd.ang_y = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        burger2_cmd.ang_z = [0.5, 0.48, 0.46, 0.44, 0.42, 0.4, 0.38, 0.36, 0.34, 0.32, 0.3, 0.28, 0.26, 0.24, 0.22, 0.2, 0.18, 0.16, 0.14, 0.12, 0.1, 0.08, 0.06, 0.04, 0.02]
        
        # ========== BURGER3 - 30 velocity values ==========
        burger3_cmd = RobotCmd()
        burger3_cmd.robot_name = 'burger3'
        burger3_cmd.lin_x = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6]
        burger3_cmd.lin_y = [0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.3, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48, 0.51, 0.54, 0.57, 0.6, 0.63, 0.66, 0.69, 0.72, 0.75, 0.78, 0.81, 0.84, 0.87]
        burger3_cmd.lin_z = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        burger3_cmd.ang_x = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        burger3_cmd.ang_y = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        burger3_cmd.ang_z = [-0.3, -0.28, -0.26, -0.24, -0.22, -0.2, -0.18, -0.16, -0.14, -0.12, -0.1, -0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28]
        
        # Add all burger commands to fleet message
        msg.robot_cmds = [burger1_cmd, burger2_cmd, burger3_cmd]
        
        # Publish the fleet command
        self.publisher.publish(msg)
        
        # Log the published commands
        self.get_logger().info(
            # f'[{self.counter}] Published commands: '
            f'burger1({len(burger1_cmd.lin_x)} values, lin_x={burger1_cmd.lin_x[0]:.2f}, ang_z={burger1_cmd.ang_z[0]:.2f}), '
            f'burger2({len(burger2_cmd.lin_x)} values, lin_x={burger2_cmd.lin_x[0]:.2f}, ang_z={burger2_cmd.ang_z[0]:.2f}), '
            f'burger3({len(burger3_cmd.lin_x)} values, lin_x={burger3_cmd.lin_x[0]:.2f}, ang_z={burger3_cmd.ang_z[0]:.2f})'
        )
        
        # self.counter += 1


def main(args=None):
    rclpy.init(args=args)
    
    node = BurgerFleetPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()