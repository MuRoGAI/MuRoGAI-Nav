#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Differential-drive time-parameterized trajectory tracking
using quintic polynomials and Kanayama controller.
"""

import math
import time
import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from path_planner_interface.msg import RobotPathArray
from navigation_manager_interface.msg import StopRobotsRequest, RobotGoalStatus


# ============================================================
# Constants
# ============================================================

PI = math.pi


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
# Trajectory Tracker Node
# ============================================================

class TimeWaypointTrajectoryTracker(Node):

    def __init__(self):
        super().__init__('trajectory_controller')

        # --------------------------------------------------
        # Parameters
        # --------------------------------------------------
        self.declare_parameter("x_offset", 0.0)
        self.declare_parameter("y_offset", 0.0)
        self.declare_parameter("yaw_offset_deg", 0.0)
        self.declare_parameter("robot_name", "burger1")

        self.x_offset = self.get_parameter("x_offset").value
        self.y_offset = self.get_parameter("y_offset").value
        yaw_offset_deg = self.get_parameter("yaw_offset_deg").value
        self.robot_name = self.get_parameter("robot_name").value

        self.yaw_offset = yaw_offset_deg * PI / 180.0

        self.get_logger().info(
            f"Starting trajectory controller for robot: {self.robot_name}"
        )

        # --------------------------------------------------
        # Internal state
        # --------------------------------------------------
        self.waypoints = []
        self.segments = []

        self.has_path = False
        self.stop_requested = False

        self.wp_indices = []
        self.wp_pos_errors = []
        self.wp_time_errors = []

        self.kx = 1.2
        self.ky = 1.0
        self.kth = 3.0

        self.max_v = 0.22
        self.max_w = 1.5

        self.t0_wall = None
        self.started = False
        self.finished = False

        self.wp_log_index = 0
        self.wp_logged = []

        self.pose_tol = 0.10   # meters
        self.time_tol = 0.10   # seconds

        # --------------------------------------------------
        # Subscribers
        # --------------------------------------------------
        self.path_sub = self.create_subscription(
            RobotPathArray,
            "/path_planner/paths",
            self.path_callback,
            10
        )

        self.stop_sub = self.create_subscription(
            StopRobotsRequest,
            "/navigation/stop_robots",
            self.stop_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            f"/{self.robot_name}/odom_world",
            self.odom_callback,
            10
        )

        # --------------------------------------------------
        # Publishers
        # --------------------------------------------------
        self.cmd_pub = self.create_publisher(
            Twist,
            f"/{self.robot_name}/cmd_vel",
            10
        )

        self.goal_status_pub_ = self.create_publisher(
            RobotGoalStatus,
            "/controller/goal_status",
            10
        )

        self.get_logger().info("Trajectory tracker ready (waiting for path).")

    # ==================================================
    # Callbacks
    # ==================================================

    def path_callback(self, msg: RobotPathArray):
        """
        Receive paths and extract the path for this robot.
        """
        for robot_path in msg.paths:
            if robot_path.robot_name == self.robot_name:

                waypoints = list(
                    zip(
                        robot_path.x_positions,
                        robot_path.y_positions,
                        robot_path.times,
                    )
                )

                if len(waypoints) < 2:
                    self.get_logger().error(
                        "Received path has less than 2 points — ignoring."
                    )
                    return

                self.get_logger().info(
                    f" Received new path for {self.robot_name} "
                    f"with {len(waypoints)} points"
                )

                self.waypoints = waypoints

                # Reset controller state
                self._build_segments()

                self.wp_indices = []
                self.wp_pos_errors = []
                self.wp_time_errors = []
                self.wp_log_index = 0
                self.wp_logged = [False] * len(self.waypoints)

                self.t0_wall = None
                self.started = False
                self.finished = False
                self.stop_requested = False
                self.has_path = True

                return

    def stop_callback(self, msg: StopRobotsRequest):
        """
        Stop robot if its name appears in stop request.
        """
        if self.robot_name in msg.robot_names:
            self.get_logger().warn(
                f" Stop requested for {self.robot_name}. Reason: {msg.reason}"
            )
            self.stop_requested = True

    # ==================================================
    # Trajectory helpers
    # ==================================================

    # def _build_segments(self):
    #     self.segments = []

    #     for i in range(len(self.waypoints) - 1):
    #         x0, y0, t0 = self.waypoints[i]
    #         x1, y1, tf = self.waypoints[i + 1]

    #         cx = compute_quintic_coeffs(t0, tf, x0, 0, 0, x1, 0, 0)
    #         cy = compute_quintic_coeffs(t0, tf, y0, 0, 0, y1, 0, 0)

    #         heading = math.atan2(y1 - y0, x1 - x0)

    #         self.segments.append({
    #             "t0": t0,
    #             "tf": tf,
    #             "cx": cx,
    #             "cy": cy,
    #             "heading": heading
    #         })

    def _build_segments(self):
        self.segments = []

        for i in range(len(self.waypoints) - 1):
            x0, y0, t0 = self.waypoints[i]
            x1, y1, tf = self.waypoints[i + 1]

            # -------------------------------
            # Safety: skip invalid time segments
            # -------------------------------
            if tf <= t0:
                self.get_logger().warn(
                    f" Skipping invalid segment {i}: "
                    f"t0={t0:.3f}, tf={tf:.3f}"
                )
                continue

            try:
                cx = compute_quintic_coeffs(t0, tf, x0, 0, 0, x1, 0, 0)
                cy = compute_quintic_coeffs(t0, tf, y0, 0, 0, y1, 0, 0)
            except np.linalg.LinAlgError:
                self.get_logger().error(
                    f" Singular matrix while building segment {i}. Skipping."
                )
                continue

            heading = math.atan2(y1 - y0, x1 - x0)

            self.segments.append({
                "t0": t0,
                "tf": tf,
                "cx": cx,
                "cy": cy,
                "heading": heading
            })

        if len(self.segments) == 0:
            self.get_logger().error("❌ No valid trajectory segments created!")
            self.has_path = False


    def desired_state(self, t):
        if t >= self.waypoints[-1][2]:
            x, y, _ = self.waypoints[-1]
            return x, y, self.segments[-1]["heading"], 0.0

        for seg in self.segments:
            if seg["t0"] <= t <= seg["tf"]:
                x, xd = eval_quintic(t, seg["cx"])
                y, yd = eval_quintic(t, seg["cy"])
                v = math.hypot(xd, yd)
                th = math.atan2(yd, xd) if v > 1e-4 else seg["heading"]
                return x, y, normalize_angle(th), v

        x, y, _ = self.waypoints[0]
        return x, y, 0.0, 0.0

    def check_and_log_waypoints(self, t, x, y):
        if self.wp_log_index >= len(self.waypoints):
            return

        x_wp, y_wp, t_wp = self.waypoints[self.wp_log_index]

        if (t >= t_wp) and (not self.wp_logged[self.wp_log_index]):
            self.wp_logged[self.wp_log_index] = True

            ex = x - x_wp
            ey = y - y_wp
            pos_err = math.hypot(ex, ey)
            time_err = t - t_wp

            self.wp_indices.append(self.wp_log_index)
            self.wp_pos_errors.append(pos_err)
            self.wp_time_errors.append(time_err)

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

    # ==================================================
    # Main control loop
    # ==================================================

    def odom_callback(self, msg):

        # No path yet
        if not self.has_path:
            self.cmd_pub.publish(Twist())
            return

        # Stop requested
        if self.stop_requested:
            self.cmd_pub.publish(Twist())
            return

        # Finished trajectory
        if self.finished:
            self.cmd_pub.publish(Twist())
            return

        # Initialize timing
        if not self.started:
            self.t0_wall = time.time()
            self.started = True
            self.get_logger().info("Trajectory time synchronized.")
            return

        t = time.time() - self.t0_wall

        # -------- ROBOT → GLOBAL transform --------
        xo = msg.pose.pose.position.x
        yo = msg.pose.pose.position.y
        tho = yaw_from_quat(msg.pose.pose.orientation)

        c = math.cos(self.yaw_offset)
        s = math.sin(self.yaw_offset)

        # x = self.x_offset + c * xo - s * yo
        # y = self.y_offset + s * xo + c * yo
        th = normalize_angle(tho + self.yaw_offset)
        x = xo
        y= yo
        th = normalize_angle(tho)

        self.check_and_log_waypoints(t, x, y)

        # End condition
        if t >= self.waypoints[-1][2]:
            self.finished = True
            self.cmd_pub.publish(Twist())

            # -------------------------------
            # Publish goal reached status
            # -------------------------------
            status_msg = RobotGoalStatus()
            status_msg.robot_name = self.robot_name
            status_msg.goal_reached = True
            self.goal_status_pub_.publish(status_msg)

            self.get_logger().info(
                f" Goal reached for {self.robot_name}. "
                f"Published goal status."
            )

            return


        # Desired state
        xd, yd, thd, vd = self.desired_state(t)

        exg = xd - x
        eyg = yd - y

        ex =  math.cos(th) * exg + math.sin(th) * eyg
        ey = -math.sin(th) * exg + math.cos(th) * eyg
        eth = normalize_angle(thd - th)

        v = vd * math.cos(eth) + self.kx * ex
        w = vd * (self.ky * ey + self.kth * math.sin(eth))

        v = max(min(v, self.max_v), -self.max_v)
        w = max(min(w, self.max_w), -self.max_w)

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
