#!/usr/bin/env python3
"""
Fake Differential Drive DiffDriveTrajectory Publisher
Reads trajectory from CSV file and publishes to /target topic
"""

import rclpy
from rclpy.node import Node
from path_planner_interface.msg import DiffDriveTrajectory
import csv
import os
from ament_index_python.packages import get_package_share_directory


class FakeDiffDrivePublisher(Node):
    def __init__(self):
        super().__init__('fake_diff_traj_publisher')
        
        # Publisher
        self.publisher = self.create_publisher(DiffDriveTrajectory, '/diff_target', 10)
        
        # Timer to publish once after delay
        self.timer = self.create_timer(1.0, self.publish_trajectory)
        self.published = False
        
        self.get_logger().info('Fake Differential Drive DiffDriveTrajectory Publisher initialized')
    
    def read_csv_trajectory(self, csv_file_path):
        """
        Read trajectory from CSV file
        
        Args:
            csv_file_path: Path to the CSV file
            
        Returns:
            tuple: (time_list, x_list, y_list, theta_list, v_list, omega_list)
        """
        time_list = []
        x_list = []
        y_list = []
        theta_list = []
        v_list = []
        omega_list = []
        
        try:
            with open(csv_file_path, 'r') as file:
                csv_reader = csv.DictReader(file)
                
                for row in csv_reader:
                    time_list.append(float(row['time']))
                    x_list.append(float(row['x']))
                    y_list.append(float(row['y']))
                    theta_list.append(float(row['theta']))
                    v_list.append(float(row['v']))
                    omega_list.append(float(row['omega']))
            
            self.get_logger().info(f'Successfully read {len(time_list)} waypoints from CSV')
            return time_list, x_list, y_list, theta_list, v_list, omega_list
            
        except FileNotFoundError:
            self.get_logger().error(f'CSV file not found: {csv_file_path}')
            return None, None, None, None, None, None
        except KeyError as e:
            self.get_logger().error(f'Missing column in CSV file: {str(e)}')
            return None, None, None, None, None, None
        except Exception as e:
            self.get_logger().error(f'Error reading CSV file: {str(e)}')
            return None, None, None, None, None, None
    
    def publish_trajectory(self):
        """Publish trajectory once"""
        if self.published:
            return
        
        # Get package share directory
        try:
            package_share_dir = get_package_share_directory('burger_robot')
            csv_file_path = os.path.join(package_share_dir, 'data', 'DD_controls.csv')
        except Exception as e:
            self.get_logger().error(f'Could not find package: {str(e)}')
            self.get_logger().info('Trying relative path...')
            csv_file_path = 'data/DiffDrive_controls.csv'
        
        self.get_logger().info(f'Reading trajectory from: {csv_file_path}')
        
        # Read CSV file
        time_list, x_list, y_list, theta_list, v_list, omega_list = self.read_csv_trajectory(csv_file_path)
        
        if time_list is None:
            self.get_logger().error('Failed to read trajectory, shutting down')
            self.timer.cancel()
            return
        
        # Create and publish message
        msg = DiffDriveTrajectory()
        msg.time = time_list
        msg.x = x_list
        msg.y = y_list
        msg.theta = theta_list
        msg.v = v_list
        msg.omega = omega_list
        
        self.publisher.publish(msg)
        self.get_logger().info(f'Published trajectory with {len(time_list)} waypoints')
        self.get_logger().info(f'Time range: {time_list[0]:.2f}s to {time_list[-1]:.2f}s')
        self.get_logger().info(f'Start pose: ({x_list[0]:.2f}, {y_list[0]:.2f}, {theta_list[0]:.2f})')
        self.get_logger().info(f'End pose: ({x_list[-1]:.2f}, {y_list[-1]:.2f}, {theta_list[-1]:.2f})')
        self.get_logger().info(f'Start velocity: (v={v_list[0]:.2f}, omega={omega_list[0]:.2f})')
        self.get_logger().info(f'End velocity: (v={v_list[-1]:.2f}, omega={omega_list[-1]:.2f})')
        
        # Mark as published
        self.published = True
        self.timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    publisher = FakeDiffDrivePublisher()
    
    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        pass
    finally:
        publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()