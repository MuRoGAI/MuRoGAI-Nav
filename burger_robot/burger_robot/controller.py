#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Differential-drive time-parameterized trajectory tracking
using quintic polynomials and Kanayama controller.

Frame conventions:
- Planner outputs (x_d, y_d, t) in GLOBAL frame
- Robot odometry gives pose in ROBOT frame
- At t=0:
    robot global pose = (X_OFFSET, Y_OFFSET, YAW_OFFSET)
"""

import math
import time
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


# ============================================================
# Constants
# ============================================================

PI = math.pi


# ============================================================
# USER SETTINGS
# ============================================================

# package_name = "turtlebot3_control"
# directory = "data"
# script_name = "all_agents_paths.npy"
# PATH_INDEX = 0        # 0 = formation F1, 1 = individual R1
# ROBOT_INDEX = 1       # which robot inside formation (0,1,2...)

# package_path = get_package_share_directory(package_name)
# PATH_FILE = os.path.join(package_path, directory, script_name)

# # Robot initial pose in GLOBAL frame at t=0
# X_OFFSET = 1.71
# Y_OFFSET = 1.14
# YAW_OFFSET_DEG = 90.0

# PI = math.pi
# YAW_OFFSET = YAW_OFFSET_DEG * PI / 180.0

# ROBOT_NAME = 'burger1'
# NODE_NAME= ROBOT_NAME + '_trajectory_controller'

# ============================================================
# Helpers
# ============================================================

def normalize_angle(a):
    while a > PI:
        a -= 2.0 * PI
    while a < -PI:
        a += 2.0 * PI
    return a


def yaw_from_quat(q):
    t3 = 2.0 * (q.w * q.z + q.x * q.y)
    t4 = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(t3, t4)


# ============================================================
# Load waypoints  ✅ WITH PATH PRINTING
# ============================================================

def load_waypoints_from_npy(path_file, path_index, robot_index=0):
    """
    Reads your NPY file which has structure:

    data = {
      "F1": {
          "times": [...],
          "robots": {
              "robot_0": array(N,3),
              "robot_1": array(N,3),
              ...
          }
      },
      "F2": {...}
    }

    traj array shape = (N,3)  → columns [x, y, time]
    """

    # Load pickled dict
    data = np.load(path_file, allow_pickle=True)
    data = data.item()   # unwrap dict

    formation_keys = list(data.keys())

    if path_index >= len(formation_keys):
        raise ValueError(
            f"PATH_INDEX {path_index} out of range. "
            f"Available formations: {formation_keys}"
        )

    # ---------------- Select formation ----------------
    formation_name = formation_keys[path_index]
    formation = data[formation_name]

    print(f"\n Using formation: {formation_name}")

    robots_dict = formation["robots"]
    robot_keys = list(robots_dict.keys())

    if robot_index >= len(robot_keys):
        raise ValueError(
            f"ROBOT_INDEX {robot_index} out of range. "
            f"Available robots: {robot_keys}"
        )

    # ---------------- Select robot ----------------
    robot_name = robot_keys[robot_index]
    traj = np.array(robots_dict[robot_name])

    print(f"✅ Using robot: {robot_name}")

    # traj shape must be (N,3)
    if traj.ndim != 2 or traj.shape[1] != 3:
        raise RuntimeError(
            f"Invalid trajectory shape {traj.shape}. Expected (N,3)"
        )

    # Split columns
    xs = traj[:, 0]
    ys = traj[:, 1]
    ts = traj[:, 2]

    waypoints = [(float(x), float(y), float(t)) for x, y, t in zip(xs, ys, ts)]

    # ---- Remove exact duplicate timestamps ----
    cleaned = []
    last_t = None

    for p in waypoints:
        t = p[2]
        if last_t is not None and t == last_t:
            print(" Skipping exact duplicate timestamp:", t)
            continue
        cleaned.append(p)
        last_t = t

    # -------------------------------------------------
    #  PRINT PATH HERE
    # -------------------------------------------------
    print("\n📍 Loaded Waypoints (x, y, t):")
    for i, (x, y, t) in enumerate(cleaned):
        print(f"  WP[{i:02d}] →  x={x:.3f} , y={y:.3f} , t={t:.3f}")

    print(f"✅ Total waypoints loaded: {len(cleaned)}\n")

    return cleaned


# ============================================================
# Quintic utilities
# ============================================================

def compute_quintic_coeffs(t0, tf, p0, v0, a0, pf, vf, af):
    A = np.array([
        [1, t0, t0**2, t0**3, t0**4, t0**5],
        [0, 1, 2*t0, 3*t0**2, 4*t0**3, 5*t0**4],
        [0, 0, 2, 6*t0, 12*t0**2, 20*t0**3],
        [1, tf, tf**2, tf**3, tf**4, tf**5],
        [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
        [0, 0, 2, 6*tf, 12*tf**2, 20*tf**3],
    ], dtype=float)

    b = np.array([p0, v0, a0, pf, vf, af], dtype=float)
    return np.linalg.solve(A, b)


def eval_quintic(t, c):
    p = c[0] + c[1]*t + c[2]*t**2 + c[3]*t**3 + c[4]*t**4 + c[5]*t**5
    v = c[1] + 2*c[2]*t + 3*c[3]*t**2 + 4*c[4]*t**3 + 5*c[5]*t**4
    return p, v


# ============================================================
# Trajectory Tracker
# ============================================================

class TimeWaypointTrajectoryTracker(Node):

    def __init__(self):
        # super().__init__(NODE_NAME)
        super().__init__('trajectory_controller')



        # --------------------------------------------------
        # Declare ROS parameters
        # --------------------------------------------------
        self.declare_parameter("path_file", "")
        self.declare_parameter("path_index", 0)
        self.declare_parameter("robot_index", 0)
        self.declare_parameter("x_offset", 0.0)
        self.declare_parameter("y_offset", 0.0)
        self.declare_parameter("yaw_offset_deg", 0.0)
        self.declare_parameter("robot_name", "burger1")

        # --------------------------------------------------
        # Get parameter values
        # --------------------------------------------------
        # path_file = self.get_parameter("path_file").value
        path_index = self.get_parameter("path_index").value
        robot_index = self.get_parameter("robot_index").value
        self.x_offset = self.get_parameter("x_offset").value
        self.y_offset = self.get_parameter("y_offset").value
        yaw_offset_deg = self.get_parameter("yaw_offset_deg").value
        self.robot_name = self.get_parameter("robot_name").value

        # --------------------------------------------------
        # Convert yaw offset to radians
        # --------------------------------------------------
        self.yaw_offset = yaw_offset_deg * PI / 180.0

        # --------------------------------------------------
        # Resolve path file using the same pattern as navigation_manager
        # --------------------------------------------------
        path_file = self.resolve_file(
            "path_file", "turtlebot3_control", "data/all_agents_paths.npy"
        )

        # --------------------------------------------------
        # Update node name based on robot name
        # --------------------------------------------------
        self.get_logger().info(f"Starting trajectory controller for robot: {self.robot_name}")


        self.waypoints = load_waypoints_from_npy(
            path_file,
            path_index,
            robot_index
        )

        # ---- Waypoint error logs ----
        self.wp_indices = []
        self.wp_pos_errors = []
        self.wp_time_errors = []

        # Kanayama gains
        self.kx = 1.2
        self.ky = 1.0
        self.kth = 3.0

        self.max_v = 0.22
        self.max_w = 1.5

        self.t0_wall = None
        self.started = False
        self.finished = False

        # ---- Waypoint logging state ----
        self.wp_log_index = 0
        self.wp_logged = [False] * len(self.waypoints)

        # Tolerances
        self.pose_tol = 0.10   # meters
        self.time_tol = 0.10   # seconds

        self._build_segments()

        self.cmd_pub = self.create_publisher(Twist, f"/{self.robot_name}/cmd_vel", 10)
        self.odom_sub = self.create_subscription(
            Odometry, f"/{self.robot_name}/odom", self.odom_callback, 10
        )

        self.get_logger().info("Trajectory tracker started.")

    # ==================================================
    # Helper Methods
    # ==================================================

    def resolve_file(self, param_name: str, default_pkg: str, relative_path: str) -> str:
        """
        Resolve file path from parameter or use default package path.
        
        Args:
            param_name: Name of the ROS parameter
            default_pkg: Default package name if parameter is not set
            relative_path: Relative path within the package
            
        Returns:
            Absolute path to the file
        """
        param_value = self.get_parameter(param_name).value
        if param_value:
            self.get_logger().info(f"Using {param_name} from parameter: {param_value}")
            return param_value

        pkg_path = get_package_share_directory(default_pkg)
        default_path = os.path.join(pkg_path, relative_path)

        self.get_logger().info(f"Using default {param_name}: {default_path}")
        return default_path

    def _build_segments(self):
        self.segments = []
        for i in range(len(self.waypoints) - 1):
            x0, y0, t0 = self.waypoints[i]
            x1, y1, tf = self.waypoints[i + 1]

            cx = compute_quintic_coeffs(t0, tf, x0, 0, 0, x1, 0, 0)
            cy = compute_quintic_coeffs(t0, tf, y0, 0, 0, y1, 0, 0)

            heading = math.atan2(y1 - y0, x1 - x0)

            self.segments.append({
                "t0": t0, "tf": tf,
                "cx": cx, "cy": cy,
                "heading": heading
            })

    def desired_state(self, t):
        # Handle time beyond trajectory
        if t >= self.waypoints[-1][2]:
            x, y, _ = self.waypoints[-1]
            return x, y, self.segments[-1]["heading"], 0.0

        # Find correct segment
        for seg in self.segments:
            if seg["t0"] <= t <= seg["tf"]:
                x, xd = eval_quintic(t, seg["cx"])
                y, yd = eval_quintic(t, seg["cy"])
                v = math.hypot(xd, yd)
                th = math.atan2(yd, xd) if v > 1e-4 else seg["heading"]
                return x, y, normalize_angle(th), v

        # Handle time before trajectory starts
        x, y, _ = self.waypoints[0]
        return x, y, 0.0, 0.0
    
    def check_and_log_waypoints(self, t, x, y):
        """Check and log waypoint passage"""
        if self.wp_log_index >= len(self.waypoints):
            return

        x_wp, y_wp, t_wp = self.waypoints[self.wp_log_index]

        # Trigger once when time crosses waypoint time
        if (t >= t_wp) and (not self.wp_logged[self.wp_log_index]):
            self.wp_logged[self.wp_log_index] = True

            # ---- Errors ----
            ex = x - x_wp
            ey = y - y_wp
            pos_err = math.hypot(ex, ey)
            time_err = t - t_wp

            # ---- Store for plotting ----
            self.wp_indices.append(self.wp_log_index)
            self.wp_pos_errors.append(pos_err)
            self.wp_time_errors.append(time_err)

            # ---- Logging ----
            pose_status = "POSE_OK" if pos_err <= self.pose_tol else "POSE_ERROR"
            if abs(time_err) <= self.time_tol:
                time_status = "ON_TIME"
            elif time_err > 0.0:
                time_status = "LATE"
            else:
                time_status = "EARLY"

            self.get_logger().info(
                "\n"
                f"   Waypoint[{self.wp_log_index}]\n"
                f"   Target Pose : x={x_wp:.3f}, y={y_wp:.3f}\n"
                f"   Actual Pose : x={x:.3f}, y={y:.3f}\n"
                f"   Position Error : {pos_err:.3f} m  --> {pose_status}\n"
                f"   Target Time : {t_wp:.3f} s\n"
                f"   Actual Time : {t:.3f} s\n"
                f"   Time Error : {time_err:.3f} s  --> {time_status}\n"
            )

            self.wp_log_index += 1

    def odom_callback(self, msg):
        """Main control callback"""
        # Stop if already finished
        if self.finished:
            self.cmd_pub.publish(Twist())
            return

        # Initialize timer on first callback
        if not self.started:
            self.t0_wall = time.time()
            self.started = True
            self.get_logger().info("Trajectory time synchronized.")
            return

        # Current trajectory time
        t = time.time() - self.t0_wall

        # -------- SE(2) transform: ROBOT -> GLOBAL --------
        xo = msg.pose.pose.position.x
        yo = msg.pose.pose.position.y
        tho = yaw_from_quat(msg.pose.pose.orientation)

        c = math.cos(self.yaw_offset)
        s = math.sin(self.yaw_offset)

        x = self.x_offset + c * xo - s * yo
        y = self.y_offset + s * xo + c * yo
        th = normalize_angle(tho + self.yaw_offset)

        # ---- Log waypoints before checking if finished ----
        self.check_and_log_waypoints(t, x, y)

        # ---- Check if trajectory is complete ----
        if t >= self.waypoints[-1][2]:
            self.finished = True
            self.cmd_pub.publish(Twist())
            self.get_logger().info(
                f"Trajectory complete. Logged "
                f"{len(self.wp_indices)} / {len(self.waypoints)} waypoints."
            )
            return

        # -------- Desired state (GLOBAL) --------
        xd, yd, thd, vd = self.desired_state(t)

        # -------- Global error --------
        exg = xd - x
        eyg = yd - y

        # -------- Body-frame error --------
        ex =  math.cos(th) * exg + math.sin(th) * eyg
        ey = -math.sin(th) * exg + math.cos(th) * eyg
        eth = normalize_angle(thd - th)

        # -------- Kanayama control --------
        v = vd * math.cos(eth) + self.kx * ex
        w = vd * (self.ky * ey + self.kth * math.sin(eth))

        # Apply velocity limits
        v = max(min(v, self.max_v), -self.max_v)
        w = max(min(w, self.max_w), -self.max_w)

        # Publish command
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.cmd_pub.publish(cmd)

    
# ============================================================
# Main
# ============================================================

def main(args=None):
    rclpy.init(args=args)
    node = TimeWaypointTrajectoryTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()