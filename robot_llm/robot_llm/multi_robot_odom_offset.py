#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import math
from rclpy.qos import QoSProfile, ReliabilityPolicy


class MultiRobotOdomOffset(Node):

    def __init__(self):
        super().__init__('multi_robot_odom_offset')

        qos = QoSProfile(depth=10)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        # self.offsets = {
        #     "burger1": (1.71, 1.14, 1.57),
        #     # "burger1": (2.47, 0.76, 1.57),
        #     "burger2": (2.85, 1.14, 1.57),
        #     "burger3": (2.28, 1.71, 1.57),
        #     # "waffle": (3.42, 7.41, -1.57),
        #     "waffle": (3.42, 7.41, -1.57),
        #     "tb4_1": (0.57, 4.56, 0.0),
        #     # "tb4_1": (2.47, 0.76, 1.57),
        #     # "tb4_1": (0.0, 0.0, 0.0),
        #     # "robot6": (0.0, 0.0, 0.0),
        # }

        # self.offsets = {
        #     "burger1": (1.71, 1.14, 1.57),
        #     "burger2": (2.85, 1.14, 1.57),
        #     "burger3": (2.28, 1.71, 1.57),
        #     "waffle": (2.85, 7.41, -1.57),
        #     "tb4_1": (0.57, 4.56, 0.0),
        # }

        # self.offsets = {
        #     "waffle": (1.71, 1.14, 1.57),
        #     "tb4_1": (2.85, 1.14, 1.57),
        #     "burger2": (2.85, 7.41, -1.57),
        #     "burger1": (0.57, 4.56, 0.0),
        #     "firebird": (0.57, 4.56, 0.0),
        #     # "burger1": (0.57, 4.56, -1.57),
        # }
        self.offsets = {
            "burger1" : (2.567768, 0.875452,  1.57),
            "burger2" : (3.704768,0.875546,1.57),
            "burger3" : (3.135464,1.750000,1.57),
            "waffle"  : (5.130000,4.375000,3.1415),
            "tb4_1"   : (3.704768,7.874454,-1.57),
            "firebird": (2.565768,7.874547,-1.57),
        }
        # self.offsets = {
        #     "burger1" : (2.565, 0.875,  1.57),
        #     "burger2" : (3.705, 0.875,  1.57),
        #     "burger3" : (3.135, 1.750,  1.57),
        #     "waffle"  : (5.130, 4.375,  3.14),
        #     "tb4_1"   : (3.705, 7.875, -1.57),
        #     "firebird": (2.565, 7.875,  -1.57),
        # }
        # self.offsets = {
        #     "burger1" : (2.565, 0.875,  1.57),
        #     "burger2" : (3.705, 0.875,  1.57),
        #     "burger3" : (3.705, 1.750,  1.57),
        #     "waffle"  : (2.565, 1.750,  3.14),
        #     "tb4_1"   : (3.705, 7.875, -1.57),
        #     "firebird": (2.565, 7.875,  -1.57),
        # }
        # self.offsets = {
        #     "burger1": (0.0, 0.0, 0.0),
        #     "burger2": (0.0, 0.0, 0.0),
        #     "burger3": (0.0, 0.0, 0.0),
        #     "waffle": (0.0, 0.0, 0.0),
        #     "tb4_1": (0.0, 0.0, 0.0),
        # }
        # =====================================================
        #  SET INITIAL WORLD POSE FOR EACH ROBOT HERE
        # (x, y, yaw in radians)
        # =====================================================
        # self.offsets = {
        #     "robot1": (3.0, 2.0, 0.0),
        #     "robot2": (3.0, 1.0, 0.0),
        #     "robot3": (3.0, 0.0, 0.0),
        #     "robot4": (9.9, -1.17, 0.0),
        #     "robot5": (9.9, -2.17, 0.0),
        #     "robot6": (9.9, -3.17, 0.0),
        # }
        # self.offsets = {
        #     "robot1": (0.0, 0.0, 0.0),
        #     "robot2": (0.0, 0.0, 0.0),
        #     "robot3": (0.0, 0.0, 0.0),
        #     "robot4": (0.0, 0.0, 0.0),
        #     "robot5": (0.0, 0.0, 0.0),
        #     "robot6": (0.0, 0.0, 0.0),
        # }

        # self.offsets = {
        #     "robot1": (4.0, 6.0, 0.0),
        #     "robot2": (4.0, 4.0, 0.0),
        #     "robot3": (5.0, 5.0, 0.0),
        #     "robot4": (16.0, 8.0, 0.0),
        #     "robot5": (16.0, 10.0, 0.0),
        #     "robot6": (15.0, 9.0, 0.0),
        # }
        # self.offsets = {
        #     "delivery_bot1": (0.0, 0.0, 0.0),
        #     "delivery_bot2": (0.0, 0.0, 0.0),
        # }
        self.subs = {}
        self.pubs = {}

        for ns in self.offsets.keys():

            if ns == 'tb4_1':
                self.subs[ns] = self.create_subscription(
                    Odometry,
                    f'/{ns}/odom',
                    lambda msg, n=ns: self.odom_cb(msg, n),
                    qos_profile
                )

                self.pubs[ns] = self.create_publisher(
                    Odometry,
                    f'/{ns}/odom_world',
                    qos_profile 
                )
            else:

                self.subs[ns] = self.create_subscription(
                    Odometry,
                    f'/{ns}/odom',
                    lambda msg, n=ns: self.odom_cb(msg, n),
                    qos
                )

                self.pubs[ns] = self.create_publisher(
                    Odometry,
                    f'/{ns}/odom_world',
                    qos
                )

            self.get_logger().info(f"{ns}: /odom → /odom_world")

    # =====================================================

    def odom_cb(self, msg, ns):

        ox, oy, oyaw = self.offsets[ns]

        # local odom pose
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        yaw = self.quat_to_yaw(q)

        # rotate + translate into world frame
        X = ox + math.cos(oyaw)*x - math.sin(oyaw)*y
        Y = oy + math.sin(oyaw)*x + math.cos(oyaw)*y
        YAW = oyaw + yaw

        # build new odom
        new = Odometry()
        new.header.stamp = self.get_clock().now().to_msg()
        new.header.frame_id = "world"
        new.child_frame_id = f"{ns}/base_footprint"

        new.pose.pose.position.x = X
        new.pose.pose.position.y = Y

        new.pose.pose.orientation = self.yaw_to_quat(YAW)

        # keep velocity
        new.twist = msg.twist

        self.pubs[ns].publish(new)

    # =====================================================

    def quat_to_yaw(self, q):
        return math.atan2(
            2*(q.w*q.z + q.x*q.y),
            1 - 2*(q.y*q.y + q.z*q.z)
        )

    def yaw_to_quat(self, yaw):
        from geometry_msgs.msg import Quaternion
        q = Quaternion()
        q.w = math.cos(yaw/2)
        q.z = math.sin(yaw/2)
        q.x = 0.0
        q.y = 0.0
        return q


# =====================================================

def main():
    rclpy.init()
    node = MultiRobotOdomOffset()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
