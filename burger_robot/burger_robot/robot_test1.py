#!/usr/bin/env python3
import rclpy, math, csv
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

class DiagNode(Node):
    def __init__(self):
        super().__init__('cleaning_bot_diag')
        self.csv_file = open('cleaning_bot_odom.csv', 'w', newline='')
        self.writer   = csv.writer(self.csv_file)
        self.writer.writerow(['t','x','y','yaw','vx','vy','wz','cmd_vx','cmd_vy'])
        self.start    = self.get_clock().now().nanoseconds * 1e-9
        self.cmd_vx   = 0.0
        self.cmd_vy   = 0.0
        self.done     = False

        self.cmd_pub = self.create_publisher(Twist, '/cleaning_bot/cmd_vel', 10)
        self.create_subscription(Odometry, '/cleaning_bot/odom', self.odom_cb, 10)
        self.create_timer(0.1, self.publish_cmd)

    def publish_cmd(self):
        if self.done:
            return
        now = self.get_clock().now().nanoseconds * 1e-9 - self.start
        cmd = Twist()
        if now < 5.0:                        # pure Y
            cmd.linear.y = 0.25
        elif now < 10.0:                     # pure X
            cmd.linear.x = 0.25
        elif now < 15.0:                     # diagonal
            cmd.linear.x = 0.25
            cmd.linear.y = 0.25
        # else: zero twist → stop

        self.cmd_vx = cmd.linear.x
        self.cmd_vy = cmd.linear.y
        self.cmd_pub.publish(cmd)

        if now > 16.0:
            self.get_logger().info("Done. Saved cleaning_bot_odom.csv")
            self.done = True

    def odom_cb(self, msg):
        if self.done:
            return
        now = self.get_clock().now().nanoseconds * 1e-9 - self.start
        p   = msg.pose.pose.position
        q   = msg.pose.pose.orientation
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        self.writer.writerow([
            f'{now:.3f}', f'{p.x:.4f}', f'{p.y:.4f}', f'{yaw:.4f}',
            f'{msg.twist.twist.linear.x:.4f}',
            f'{msg.twist.twist.linear.y:.4f}',
            f'{msg.twist.twist.angular.z:.4f}',
            self.cmd_vx, self.cmd_vy
        ])

def main():
    rclpy.init()
    node = DiagNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())   # stop robot
        node.csv_file.close()

if __name__ == '__main__':
    main()