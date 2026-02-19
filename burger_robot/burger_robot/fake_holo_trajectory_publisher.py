#!/usr/bin/env python3
"""
Fake HoloTrajectory Publisher
Reads trajectory from CSV file and publishes to /target topic
"""
import rclpy
from rclpy.node import Node
from path_planner_interface.msg import HoloTrajectory
import csv
import os
from ament_index_python.packages import get_package_share_directory


class FakeTrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('fake_holo_traj_publisher')
        
        # Publisher
        self.publisher = self.create_publisher(HoloTrajectory, '/holo_target', 10)
        
        # Timer to publish once after a short delay (to ensure subscriber is ready)
        self.timer = self.create_timer(1.0, self.publish_trajectory)
        self.published = False
        
        self.get_logger().info('Fake HoloTrajectory Publisher initialized')
    
    def read_csv_trajectory(self, csv_file_path):
        """
        Read trajectory from CSV file
        
        Args:
            csv_file_path: Path to the CSV file
            
        Returns:
            tuple: (time_list, x_list, y_list, vx_list, vy_list)
        """
        time_list = []
        x_list = []
        y_list = []
        vx_list = []
        vy_list = []
        
        try:
            with open(csv_file_path, 'r') as file:
                csv_reader = csv.DictReader(file)
                
                for row in csv_reader:
                    time_list.append(float(row['time']))
                    x_list.append(float(row['x']))
                    y_list.append(float(row['y']))
                    vx_list.append(float(row['vx']))
                    vy_list.append(float(row['vy']))
            
            self.get_logger().info(f'Successfully read {len(time_list)} waypoints from CSV')
            return time_list, x_list, y_list, vx_list, vy_list
            
        except FileNotFoundError:
            self.get_logger().error(f'CSV file not found: {csv_file_path}')
            return None, None, None, None, None
        except KeyError as e:
            self.get_logger().error(f'Missing column in CSV file: {str(e)}')
            return None, None, None, None, None
        except Exception as e:
            self.get_logger().error(f'Error reading CSV file: {str(e)}')
            return None, None, None, None, None
    
    def publish_trajectory(self):
        """Publish trajectory once"""
        if self.published:
            return
        
        # Get package share directory
        try:
            package_share_dir = get_package_share_directory('burger_robot')
            csv_file_path = os.path.join(package_share_dir, 'data', 'Holo_controls1.csv')
        except Exception as e:
            self.get_logger().error(f'Could not find package: {str(e)}')
            self.get_logger().info('Trying relative path...')
            csv_file_path = 'data/Holo_controls.csv'
        
        self.get_logger().info(f'Reading trajectory from: {csv_file_path}')
        
        # Read CSV file
        time_list, x_list, y_list, vx_list, vy_list = self.read_csv_trajectory(csv_file_path)
        
        if time_list is None:
            self.get_logger().error('Failed to read trajectory, shutting down')
            self.timer.cancel()
            return
        
        # Create and publish message
        msg = HoloTrajectory()
        msg.time = time_list
        msg.x = x_list
        msg.y = y_list
        msg.vx = vx_list
        msg.vy = vy_list
        
        self.publisher.publish(msg)
        self.get_logger().info(f'Published trajectory with {len(time_list)} waypoints')
        self.get_logger().info(f'Time range: {time_list[0]:.2f}s to {time_list[-1]:.2f}s')
        self.get_logger().info(f'Start position: ({x_list[0]:.2f}, {y_list[0]:.2f})')
        self.get_logger().info(f'End position: ({x_list[-1]:.2f}, {y_list[-1]:.2f})')
        self.get_logger().info(f'Start velocity: ({vx_list[0]:.2f}, {vy_list[0]:.2f})')
        self.get_logger().info(f'End velocity: ({vx_list[-1]:.2f}, {vy_list[-1]:.2f})')
        
        # Mark as published and cancel timer
        self.published = True
        self.timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    publisher = FakeTrajectoryPublisher()
    
    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        pass
    finally:
        publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()