#!/usr/bin/env python3
import math
import os
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

# Controller 9 — single-file multi-robot launcher
# Pattern : robot_services51.launch.py  (file 1)
# Controller: controller9               (file 2)

# ═════════════════════════════════════════════════════════════════════════════
#  PER-ROBOT CONFIG
#  robot_type = "diff-drive"  → Kanayama + cubic Hermite x,y,theta tracking
#  robot_type = "holonomic"   → proportional x,y control (no theta)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class RobotConfig:
    robot_name:   str   = ""
    robot_type:   str   = "diff-drive"   # "diff-drive" | "holonomic"
    package:      str   = "burger_robot"
    executable:   str   = "controller9"

    kp_linear:    float = 1.5
    kp_angular:   float = 2.5
    kd_linear:    float = 0.5
    kd_angular:   float = 0.3
    ky:           float = 3.5
    max_lin_x:    float = 0.22
    max_lin_y:    float = 0.0
    max_ang_z:    float = 2.84
    max_lin_acc:  float = 1.2
    max_ang_acc:  float = 4.5

    # QoS — set True for robots that need BEST_EFFORT (e.g. tb4)
    best_effort_qos: bool = False

ROBOT_CONFIG: dict[str, RobotConfig] = {
    "delivery_bot1": RobotConfig(
        robot_name  = "delivery_bot1",
        robot_type  = "diff-drive",
        kp_linear   = 13.2,
        kp_angular  = 4.3,
        kd_linear   = 0.5,
        kd_angular  = 0.3,
        ky          = 14.4,
        max_lin_x   = 0.25,
        max_lin_y   = 0.0,
        max_ang_z   = 0.4,
        max_lin_acc = 2.5,
        max_ang_acc = 2.5,
    ),
    "delivery_bot2": RobotConfig(
        robot_name  = "delivery_bot2",
        robot_type  = "diff-drive",
        kp_linear   = 13.2,
        kp_angular  = 4.3,
        kd_linear   = 0.5,
        kd_angular  = 0.3,
        ky          = 14.4,
        max_lin_x   = 0.25,
        max_lin_y   = 0.0,
        max_ang_z   = 0.4,
        max_lin_acc = 2.5,
        max_ang_acc = 2.5,
    ),
    "delivery_bot3": RobotConfig(
        robot_name  = "delivery_bot3",
        robot_type  = "holonomic",
        kp_linear   = 32.3,
        kp_angular  = 27.6,
        kd_linear   = 3.53,
        kd_angular  = 2.21,
        ky          = 23.3,
        max_lin_x   = 0.25,
        max_lin_y   = 0.25,
        max_ang_z   = 0.4,
        max_lin_acc = 2.5,
        max_ang_acc = 2.5,
    ),
    "cleaning_bot": RobotConfig(
        robot_name  = "cleaning_bot",
        robot_type  = "holonomic",
        kp_linear   = 32.3,
        kp_angular  = 27.6,
        kd_linear   = 3.53,
        kd_angular  = 2.21,
        ky          = 23.3,
        max_lin_x   = 0.25,
        max_lin_y   = 0.25,
        max_ang_z   = 0.4,
        max_lin_acc = 2.5,
        max_ang_acc = 2.5,
    ),
}

ROBOTS         = list(ROBOT_CONFIG.keys())
SYNC_DELAY     = 1.0       # seconds all robots wait after last one is ready
GOAL_TOLERANCE = 0.08      # metres — final-seek stop threshold

QOS_RELIABLE    = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,    depth=10)
QOS_BEST_EFFORT = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)

# ═════════════════════════════════════════════════════════════════════════════
#  CUBIC HERMITE SPLINE  (controller9's "compute_quintic_coeffs" / "eval_quintic"
#  renamed to reflect what they actually compute)
# ═════════════════════════════════════════════════════════════════════════════
def cubic_hermite_coeffs(t0: float, tf: float,
                         p0: float, v0: float,
                         pf: float, vf: float) -> np.ndarray:
    """
    Returns [a0, a1, a2, a3, t0] for the cubic Hermite spline defined by
    boundary positions p0, pf and velocities v0, vf on [t0, tf].
    Evaluation uses local time tau = t - t0.
    """
    T = tf - t0
    if T <= 1e-9:
        return np.array([p0, 0.0, 0.0, 0.0, t0])
    a2 = (3.0*(pf - p0) / T**2) - (2.0*v0 + vf) / T
    a3 = (-2.0*(pf - p0) / T**3) + (v0 + vf) / T**2
    return np.array([p0, v0, a2, a3, t0])


def eval_cubic_hermite(t: float, c: np.ndarray):
    """Return (position, velocity) at global time t."""
    a0, a1, a2, a3, t0 = c
    tau = max(0.0, t - t0)
    pos = a0 + a1*tau + a2*tau**2 + a3*tau**3
    vel = a1 + 2.0*a2*tau + 3.0*a3*tau**2
    return pos, vel


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def _norm(a: float) -> float:
    """Wrap angle to (-π, π]."""
    return math.atan2(math.sin(a), math.cos(a))


# ═════════════════════════════════════════════════════════════════════════════
#  COORDINATOR  — synchronised start across all robots
# ═════════════════════════════════════════════════════════════════════════════
class Coordinator:
    def __init__(self, robot_names, sync_delay: float = SYNC_DELAY):
        self._expected   = set(robot_names)
        self._ready      = set()
        self._start_time: Optional[float] = None
        self._lock       = threading.Lock()
        self._delay      = sync_delay

    def report_ready(self, robot_name: str, now_sec: float) -> Optional[float]:
        with self._lock:
            self._ready.add(robot_name)
            if self._ready >= self._expected and self._start_time is None:
                self._start_time = now_sec + self._delay
        return self._start_time


# ═════════════════════════════════════════════════════════════════════════════
#  ROBOT NODE  — subscribes to /path_planner/paths, publishes /{robot}/cmd_vel
# ═════════════════════════════════════════════════════════════════════════════
class PathFollower(Node):
    def __init__(self, robot_name: str, coordinator: Coordinator):
        super().__init__(f'{robot_name}_controller')
        self.robot = robot_name
        self.coord = coordinator
        self.cfg   = ROBOT_CONFIG[robot_name]

        # ── State ────────────────────────────────────────────────────────────
        self.current_pose:    Optional[tuple]  = None   # (x, y, yaw)
        self.segments:        list             = []
        self.robot_type:      Optional[str]    = None
        self.traj_received:   bool             = False
        self.start_time:      Optional[float]  = None
        self.active:          bool             = False
        self.stop_requested:  bool             = False
        self.logged_segments: set              = set()
        self.last_debug_time: float            = 0.0
        self.prev_cmd:        Twist            = Twist()
        self.prev_time:       float            = 0.0

        # ── QoS ──────────────────────────────────────────────────────────────
        sub_qos = QOS_BEST_EFFORT if self.cfg.best_effort_qos else QOS_RELIABLE

        self.get_logger().info(
            f"[INIT] {self.robot}  type={self.cfg.robot_type}  "
            f"sub_qos={'BE' if self.cfg.best_effort_qos else 'RE'}  "
            f"kp_lin={self.cfg.kp_linear}  kp_ang={self.cfg.kp_angular}  "
            f"ky={self.cfg.ky}  "
            f"v_max=({self.cfg.max_lin_x},{self.cfg.max_lin_y})  "
            f"w_max={self.cfg.max_ang_z}"
        )

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            Odometry,
            f'/{self.robot}/odom',
            self.odom_callback,
            sub_qos
        )

        # Path-planner trajectory topic (shared across all robots)
        from path_planner_interface.msg import RobotTrajectoryArray
        self.create_subscription(
            RobotTrajectoryArray,
            '/path_planner/paths',
            self.trajectory_callback,
            QOS_RELIABLE
        )

        # Stop signal
        from navigation_manager_interface.msg import StopRobotsRequest
        self.create_subscription(
            StopRobotsRequest,
            '/navigation/stop_robots',
            self.stop_callback,
            QOS_RELIABLE
        )

        # ── Publishers ────────────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(
            Twist, f'/{self.robot}/cmd_vel', QOS_RELIABLE)

        from navigation_manager_interface.msg import RobotGoalStatus
        self.goal_status_pub = self.create_publisher(
            RobotGoalStatus, '/controller/goal_status', QOS_RELIABLE)

        # ── 50 Hz control timer ───────────────────────────────────────────────
        self.timer = self.create_timer(0.02, self.control_loop)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        q   = msg.pose.pose.orientation
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y),
                         1.0 - 2.0*(q.y*q.y + q.z*q.z))
        self.current_pose = (pos.x, pos.y, yaw)

    def trajectory_callback(self, msg):
        for robot_traj in msg.robot_trajectories:
            if robot_traj.robot_name != self.robot:
                continue

            self.robot_type = self.cfg.robot_type
            self.segments.clear()
            self.logged_segments.clear()

            if self.robot_type == "diff-drive":
                traj = robot_traj.diff_drive_trajectories[0]
                self._build_segments(traj.time, traj.x, traj.y, traj.theta)

            elif self.robot_type == "holonomic":
                if robot_traj.holo_trajectories:
                    traj = robot_traj.holo_trajectories[0]
                else:
                    traj = robot_traj.diff_drive_trajectories[0]
                self._build_segments(traj.time, traj.x, traj.y)  # no theta

            else:
                self.get_logger().warn(
                    f"[{self.robot}] Unknown robot_type='{self.robot_type}' in config, skipping.")
                return

            t_offset        = float(traj.time[0])
            self.start_time = self.get_clock().now().nanoseconds * 1e-9 - t_offset
            self.traj_received  = True
            self.active         = True
            self.stop_requested = False
            self.get_logger().info(
                f"[TRAJ] {self.robot}: {len(self.segments)} segments  "
                f"type={self.robot_type}  "
                f"duration={self.segments[-1]['tf']:.2f}s"
            )
            
    def stop_callback(self, msg):
        if self.robot in msg.robot_names:
            self.stop_requested = True
            self.active         = False
            self.cmd_pub.publish(Twist())
            self.get_logger().warn(f"[STOP] {self.robot}: stop requested.")

    # ── Segment builder (controller9's build_segments) ────────────────────────
    def _build_segments(self, time_list, x_list, y_list, theta_list=None):
        n  = len(time_list)
        vx = [0.0] * n
        vy = [0.0] * n
        vt = ([0.0] * n) if (theta_list is not None) else None

        # Central-difference interior knot velocities
        for i in range(1, n - 1):
            dt_prev = time_list[i]     - time_list[i - 1]
            dt_next = time_list[i + 1] - time_list[i]
            if dt_prev < 1e-9 or dt_next < 1e-9:
                continue
            vx[i] = 0.5 * ((x_list[i]   - x_list[i-1]) / dt_prev +
                            (x_list[i+1] - x_list[i])   / dt_next)
            vy[i] = 0.5 * ((y_list[i]   - y_list[i-1]) / dt_prev +
                            (y_list[i+1] - y_list[i])   / dt_next)
            if theta_list is not None:
                dth_p = _norm(theta_list[i]   - theta_list[i-1]) / dt_prev
                dth_n = _norm(theta_list[i+1] - theta_list[i])   / dt_next
                vt[i] = 0.5 * (dth_p + dth_n)

        for i in range(n - 1):
            t0 = float(time_list[i])
            tf = float(time_list[i + 1])
            if abs(tf - t0) < 1e-9:
                self.get_logger().warn(
                    f"[{self.robot}] Duplicate time at index {i} (t={t0:.3f}), skipping.")
                continue

            seg = {
                "t0": t0, "tf": tf,
                "cx": cubic_hermite_coeffs(t0, tf, x_list[i], vx[i], x_list[i+1], vx[i+1]),
                "cy": cubic_hermite_coeffs(t0, tf, y_list[i], vy[i], y_list[i+1], vy[i+1]),
            }
            if theta_list is not None:
                seg["ct"] = cubic_hermite_coeffs(
                    t0, tf, theta_list[i], vt[i], theta_list[i+1], vt[i+1])
            self.segments.append(seg)

    # ── Control loop ──────────────────────────────────────────────────────────
    def control_loop(self):
        if self.current_pose is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        # ── Phase 1: wait until trajectory arrives AND coordinator fires ──────
        if not self.traj_received:
            shared_start = self.coord.report_ready(self.robot, now)
            if shared_start is not None and not self.active:
                # Trajectory not yet received — keep waiting
                if now - self.last_debug_time > 2.0:
                    self.last_debug_time = now
                    self.get_logger().info(
                        f"[WAIT] {self.robot}: waiting for /path_planner/paths …")
            return

        if not self.active or self.stop_requested:
            return

        if not self.segments:
            return

        t = now - self.start_time

        # ── Waypoint crossing log ─────────────────────────────────────────────
        self._log_waypoint_crossings(t)

        # ── Phase 2: trajectory finished → yaw-align then stop ───────────────
        if t > self.segments[-1]["tf"]:
            self._phase_finished(t)
            return

        # ── Phase 3: find active segment and track ───────────────────────────
        seg = None
        for s in self.segments:
            if s["t0"] <= t <= s["tf"]:
                seg = s
                break
        if seg is None:
            return

        if self.robot_type == "diff-drive":
            self._control_diff_drive(t, seg, now)
        elif self.robot_type == "holonomic":
            self._control_holonomic(t, seg, now)

    # ── Waypoint crossing logger ──────────────────────────────────────────────
    def _log_waypoint_crossings(self, t: float):
        x, y, yaw = self.current_pose
        for i, seg in enumerate(self.segments):
            if i in self.logged_segments or t < seg["tf"]:
                continue
            self.logged_segments.add(i)

            xd_e, _ = eval_cubic_hermite(seg["tf"], seg["cx"])
            yd_e, _ = eval_cubic_hermite(seg["tf"], seg["cy"])
            td_e    = 0.0
            ang_str = "N/A"
            if "ct" in seg:
                td_e, _ = eval_cubic_hermite(seg["tf"], seg["ct"])
                td_e    = _norm(td_e)
                ang_str = f"{_norm(td_e - yaw):.4f} rad"

            pos_err = math.hypot(xd_e - x, yd_e - y)
            self.get_logger().info(
                f"\n========== [{self.robot}] WAYPOINT {i+1}/{len(self.segments)} "
                f"==========\n"
                f"  Time error    : {t - seg['tf']:+.4f} s\n"
                f"  Desired pose  : ({xd_e:.3f}, {yd_e:.3f}, {td_e:.3f})\n"
                f"  Actual  pose  : ({x:.3f}, {y:.3f}, {yaw:.3f})\n"
                f"  Position error: {pos_err:.4f} m\n"
                f"  Angular error : {ang_str}\n"
                f"==========================================="
            )

    # ── Phase: trajectory finished ────────────────────────────────────────────
    def _phase_finished(self, t: float):
        last_seg   = self.segments[-1]
        final_theta = 0.0
        if "ct" in last_seg:
            final_theta, _ = eval_cubic_hermite(last_seg["tf"], last_seg["ct"])

        x, y, yaw = self.current_pose
        yaw_err   = _norm(final_theta - yaw)

        cmd = Twist()
        if abs(yaw_err) > 0.02:                   # ~1° tolerance — keep rotating
            cmd.angular.z = float(np.clip(
                self.cfg.kp_angular * yaw_err, -self.cfg.max_ang_z, self.cfg.max_ang_z))
            self.cmd_pub.publish(cmd)
            return

        # Aligned — publish zero and signal goal reached
        self.cmd_pub.publish(Twist())
        self._publish_goal_reached()
        self.active = False

        xf, _ = eval_cubic_hermite(last_seg["tf"], last_seg["cx"])
        yf, _ = eval_cubic_hermite(last_seg["tf"], last_seg["cy"])
        self.get_logger().info(
            f"[DONE] {self.robot}  "
            f"final_err={math.hypot(xf - x, yf - y):.3f}m  "
            f"yaw_err={math.degrees(yaw_err):.2f}°"
        )

    # ── Diff-drive controller (controller9) ───────────────────────────────────
    def _control_diff_drive(self, t: float, seg: dict, now: float):
        xd,  xd_dot = eval_cubic_hermite(t, seg["cx"])
        yd,  yd_dot = eval_cubic_hermite(t, seg["cy"])
        x, y, yaw   = self.current_pose
        cfg         = self.cfg

        # Desired heading from path tangent
        if math.hypot(xd_dot, yd_dot) > 1e-6:
            desired_theta     = math.atan2(yd_dot, xd_dot)
            desired_theta_dot = 0.0
        else:
            desired_theta     = yaw
            desired_theta_dot = 0.0

        # Near final goal → enforce final waypoint heading
        last_seg = self.segments[-1]
        if seg is last_seg:
            xf, _ = eval_cubic_hermite(last_seg["tf"], last_seg["cx"])
            yf, _ = eval_cubic_hermite(last_seg["tf"], last_seg["cy"])
            if math.hypot(xf - x, yf - y) < 0.05 and "ct" in last_seg:
                desired_theta, _ = eval_cubic_hermite(last_seg["tf"], last_seg["ct"])
                desired_theta_dot = 0.0

        dx = xd - x
        dy = yd - y
        ex =  math.cos(yaw)*dx + math.sin(yaw)*dy   # longitudinal error
        ey = -math.sin(yaw)*dx + math.cos(yaw)*dy   # lateral error
        e_theta = _norm(desired_theta - yaw)

        v_d = xd_dot*math.cos(desired_theta) + yd_dot*math.sin(desired_theta)
        w_d = desired_theta_dot

        pos_err    = math.hypot(dx, dy)
        bearing    = math.atan2(dy, dx)
        hdg_err    = _norm(bearing - yaw)

        # Augmented angular correction (Kanayama lateral + bearing-to-desired)
        lateral_correction = (
            cfg.ky          * v_d * ey
            + cfg.kp_angular * pos_err * math.sin(hdg_err)
        )

        cmd = Twist()
        cmd.linear.x  = float(np.clip(
            v_d * math.cos(e_theta) + cfg.kp_linear * ex,
            -cfg.max_lin_x, cfg.max_lin_x))
        cmd.angular.z = float(np.clip(
            w_d + lateral_correction,
            -cfg.max_ang_z, cfg.max_ang_z))
        self.cmd_pub.publish(cmd)

        if now - self.last_debug_time > 0.5:
            self.last_debug_time = now
            self.get_logger().info(
                f"[CTRL] {self.robot} | t={t:.2f}s "
                f"seg=[{seg['t0']:.1f},{seg['tf']:.1f}] | "
                f"des=({xd:.2f},{yd:.2f},θ={math.degrees(desired_theta):.1f}°) "
                f"act=({x:.2f},{y:.2f},{math.degrees(yaw):.1f}°) | "
                f"v={cmd.linear.x:.3f} w={cmd.angular.z:.3f}"
            )

    # ── Holonomic controller (controller9) ───────────────────────────────────
    def _control_holonomic(self, t: float, seg: dict, now: float):
        xd, _ = eval_cubic_hermite(t, seg["cx"])
        yd, _ = eval_cubic_hermite(t, seg["cy"])
        x, y, yaw = self.current_pose
        cfg = self.cfg

        ex = xd - x
        ey = yd - y

        cmd = Twist()
        cmd.linear.x = float(np.clip(cfg.kp_linear * ex, -cfg.max_lin_x, cfg.max_lin_x))
        cmd.linear.y = (
            float(np.clip(cfg.kp_linear * ey, -cfg.max_lin_y, cfg.max_lin_y))
            if cfg.max_lin_y > 1e-6 else 0.0
        )
        self.cmd_pub.publish(cmd)

        if now - self.last_debug_time > 0.5:
            self.last_debug_time = now
            self.get_logger().info(
                f"[CTRL] {self.robot} | t={t:.2f}s "
                f"seg=[{seg['t0']:.1f},{seg['tf']:.1f}] | "
                f"des=({xd:.2f},{yd:.2f}) act=({x:.2f},{y:.2f}) | "
                f"vx={cmd.linear.x:.3f} vy={cmd.linear.y:.3f}"
            )

    # ── Utilities ─────────────────────────────────────────────────────────────
    def _publish_goal_reached(self):
        from navigation_manager_interface.msg import RobotGoalStatus
        msg = RobotGoalStatus()
        msg.robot_name   = self.robot
        msg.goal_reached = True
        self.goal_status_pub.publish(msg)


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
def main():
    rclpy.init()
    coordinator = Coordinator(ROBOTS)
    nodes       = [PathFollower(r, coordinator) for r in ROBOTS]
    executor    = MultiThreadedExecutor(num_threads=len(nodes))
    for node in nodes:
        executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        for node in nodes:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()