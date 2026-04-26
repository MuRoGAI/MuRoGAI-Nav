#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from path_planner_interface.msg import PathPlannerRequest
import os
import json
from nav_msgs.msg import Odometry
import math
from ament_index_python.packages import get_package_share_directory

USER  = os.environ.get("USER")

INPUT_FILE = os.environ.get(
    "PLANNER_INPUT",
    f"/home/{USER}/murogai_nav/src/MuRoGAI-Nav/"
    "path_plan/path_planner/data/formation_input.json"
)

def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Convert quaternion to yaw (rotation around Z)."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)

class PathRequestReciever(Node):
    def __init__(self, node_name):
        super().__init__(node_name)

        self.declare_parameter("config_file_path", "")


        self.nav_req_sub = self.create_subscription(
            PathPlannerRequest,
            "/path_planner/request",
            self.path_req_receiver_callback,
            10
        )

        self.default_pose = {
            # "burger1" : (2.565, 0.875,  1.57),
            # "burger2" : (3.705, 0.875,  1.57),
            # "burger3" : (3.135, 1.75,   1.57),
            # "waffle"  : (5.13,  4.375,  3.14),
            # "tb4_1"   : (3.705, 7.875, -1.57),
            # "firebird": (2.565, 7.875, -1.57),
            # "go2"     : (3.135, 7.0,   -1.57),

            "delivery_bot1"  : (5.0, 7.5, 0.0),
            "delivery_bot2"  : (5.0, 4.5, 0.0),
            "delivery_bot3"  : (7.0, 6.0, 0.0),
            "cleaning_bot" : (19.0, 15.0,  3.14),

        }
        self.robot_odom_subs     = {}
        self.robot_odom_received = {}
        self.robot_current_poses = {}

        self.config_file_path = self._resolve_file(
            "config_file_path",
            default_pkg="chatty",
            default_relative_path="config/robot_config_restaurant2.json",
        )
        self.config_data      = self._load_json(self.config_file_path)
        self.robot_names      = self.config_data.get('robot_names', [])
        for robot_name in self.robot_names:
            topic_name = f"/{robot_name}/odom"

            sub = self.create_subscription(
                Odometry,
                topic_name,
                lambda msg, rn=robot_name: self._odom_callback(msg, rn),
                10,
            )
            self.robot_odom_subs[robot_name]     = sub
            self.robot_odom_received[robot_name] = False

            # seed with fallback pose until real odom arrives
            default = self.default_pose.get(robot_name, (0.0, 0.0, 0.0))
            self.robot_current_poses[robot_name] = {
                'x':         default[0],
                'y':         default[1],
                'yaw':       default[2],
            }
            self.get_logger().info(f"Subscribed to {topic_name}")



    def path_req_receiver_callback(self, msg: PathPlannerRequest):

        # Convert JSON string to dictionary
        req = json.loads(msg.plan_json)

        # Build current_pose from live odom (falls back to defaults if odom not yet received)
        current_pose = {}
        for robot_name, pose in self.robot_current_poses.items():
            current_pose[robot_name] = [
                float(pose['x']),
                float(pose['y']),
                float(pose['yaw']),
            ]

        # Wrap into the expected structure
        payload = {
            "current_pose": current_pose,
            "goal_pose":    req,          # req is whatever came in on the topic
        }

        # Write atomically to avoid planner reading a half-written file
        try:
            os.makedirs(os.path.dirname(INPUT_FILE), exist_ok=True)
            tmp_path = INPUT_FILE + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(payload, f, indent=4)
            os.replace(tmp_path, INPUT_FILE)   # atomic swap
            self.get_logger().info(
                f"Planner input written — "
                f"{len(current_pose)} pose(s), "
                f"{len(req)} goal(s) → {INPUT_FILE}"
            )

        except Exception as e:
            self.get_logger().error(f"Failed to write planner input: {e}")

    def _load_json(self, path: str) -> dict:
        if not os.path.exists(path):
            self.get_logger().warn(f"JSON file not found: {path}")
            return {}
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.get_logger().info(f"Loaded JSON config from: {path}")
            return data
        except Exception as e:
            self.get_logger().error(f"Failed to load JSON {path}: {e}")
            return {}

    def _resolve_file(self, param_name: str, default_pkg: str,
                      default_relative_path: str) -> str:
        param_value = self.get_parameter(param_name).value
        if param_value:
            if os.path.exists(param_value):
                self.get_logger().info(f"Using {param_name} from parameter: {param_value}")
                return param_value
            self.get_logger().warn(
                f"{param_name} file not found: {param_value}, using default"
            )
        try:
            pkg_path     = get_package_share_directory(default_pkg)
            default_path = os.path.join(pkg_path, default_relative_path)
            if os.path.exists(default_path):
                self.get_logger().info(f"Using default {param_name}: {default_path}")
                return default_path
            raise FileNotFoundError(
                f"Default {param_name} file not found: {default_path}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to resolve {param_name}: {e}")
            raise


    def _odom_callback(self, msg: Odometry, robot_name: str):
        position    = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        yaw = quaternion_to_yaw(
            orientation.x, orientation.y,
            orientation.z, orientation.w,
        )
        self.robot_current_poses[robot_name] = {
            'x':         float(position.x),
            'y':         float(position.y),
            'yaw':       float(yaw),
            'timestamp': self.get_clock().now(),
        }
        if not self.robot_odom_received.get(robot_name, False):
            self.robot_odom_received[robot_name] = True
            self.get_logger().info(
                f"[ODOM] First message received from {robot_name}: "
                f"x={position.x:.3f} y={position.y:.3f} yaw={yaw:.3f}"
            )
        self.get_logger().debug(
            f"[ODOM] {robot_name}: "
            f"x={position.x:.3f} y={position.y:.3f} yaw={yaw:.3f}"
        )


def main(args=None):
    rclpy.init(args=args)

    node = PathRequestReciever('path_request_reciever')

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()