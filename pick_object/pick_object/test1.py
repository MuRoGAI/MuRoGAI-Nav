#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time

class DummyTaskPublisher(Node):
    def __init__(self):
        super().__init__('dummy_task_publisher')
        self.pub = self.create_publisher(String, '/task_manager/tasks_json', 10)
        
        # Send the two messages with 5-second delay
        self.timer = self.create_timer(1.0, self.send_messages)
        self.step = 0

    def send_messages(self):
        if self.step == 0:
            msg = String()
            payload = {
                "robot_tasks": {
                    "lerobot1": "pick green gear",
                    "lerobot2": ""
                },
                "sequence": ""
            }
            msg.data = json.dumps(payload)
            self.pub.publish(msg)
            self.get_logger().info('SENT → pick green gear')
            self.step += 1

        elif self.step == 1:
            # Wait 7 seconds then send stop
            time.sleep(7.0)
            msg = String()
            payload = {
                "robot_tasks": {
                    "lerobot1": "stop",
                    "lerobot2": ""
                },
                "sequence": ""
            }
            msg.data = json.dumps(payload)
            self.pub.publish(msg)
            self.get_logger().info('SENT → stop')
            
            self.get_logger().info('Demo finished – shutting down in 2s')
            self.step += 1

        elif self.step == 2:
            time.sleep(2.0)
            rclpy.shutdown()


def main():
    rclpy.init()
    node = DummyTaskPublisher()
    rclpy.spin(node)

if __name__ == '__main__':
    main()