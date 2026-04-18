#!/usr/bin/env python3
import math
import os
import threading
from dataclasses import dataclass
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
# ═════════════════════════════════════════════════════════════════════════════
#  PATH DIRECTORY
# ═════════════════════════════════════════════════════════════════════════════
PATH_DIR = (
"/home/suraj/murogai_nav/src/MuRoGAI-Nav/"
"path_plan/path_planner/path_planner/trajectory_logs2"
)
# ═════════════════════════════════════════════════════════════════════════════
#  PER-ROBOT CONFIG  —  edit gains here
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class RobotConfig:
    csv_file: str = ""
    # ── Control gains ────────────────────────────────────────────────────────
    kp_linear:  float = 1.5   # forward position error gain       (ex  → v)
    kp_angular: float = 2.5   # bearing recovery gain             (pos_err·sin(hdg_err) → ω)
    ky:         float = 3.5   # Kanayama lateral gain             (v_d·ey → ω)
    # ── Velocity limits ───────────────────────────────────────────────────────
    v_max: float = 0.22   # m/s
    w_max: float = 2.84   # rad/s

ROBOT_CONFIG: dict[str, RobotConfig] = {
    "burger1": RobotConfig(
        csv_file="burger1.csv",
        kp_linear=3.2, kp_angular=3.7, ky=4.2,
        v_max=0.3, w_max=2.84,
    ),
    "burger2": RobotConfig(
        csv_file="burger2.csv",
        kp_linear=3.2, kp_angular=3.7, ky=4.2,
        v_max=0.3, w_max=2.84,
    ),
    "burger3": RobotConfig(
        csv_file="burger3.csv",
        kp_linear=3.2, kp_angular=3.7, ky=4.2,
        v_max=0.3, w_max=2.84,
    ),
    # "waffle": RobotConfig(
    #     csv_file="waffle.csv",
    #     kp_linear=1.5, kp_angular=2.5, ky=3.5,
    #     v_max=0.22, w_max=2.84,
    # ),
    # "firebird": RobotConfig(
    #     csv_file="firebird.csv",
    #     kp_linear=1.5, kp_angular=2.5, ky=3.5,
    #     v_max=0.22, w_max=2.84,
    # ),
    # "tb4_1": RobotConfig(
    #     csv_file="tb4_1.csv",
    #     kp_linear=1.5, kp_angular=2.5, ky=3.5,
    #     v_max=0.26, w_max=1.90,
    # )
}

ROBOTS     = list(ROBOT_CONFIG.keys())
SYNC_DELAY = 1.0   # seconds after ALL robots have odom → all start together
GOAL_TOLERANCE = 0.05   # metres — robot considered "done" within this radius
# Minimum forward velocity injected in Phase 4 when v_d ≈ 0 but the robot
# still has significant position error and is roughly facing the target.
V_FLOOR_FRAC   = 0.12   # fraction of v_max  (≈ 0.036 m/s for burger1)

QOS_RELIABLE    = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,    depth=10)
QOS_BEST_EFFORT = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
# ═════════════════════════════════════════════════════════════════════════════
#  CSV LOADER
# ═════════════════════════════════════════════════════════════════════════════
def load_waypoints(csv_path: str) -> np.ndarray:
    """
    Load CSV with columns: time, x, y, theta
    Returns (N, 4) float array sorted by time, with time shifted so t[0] = 0.
    """
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    data = data[np.argsort(data[:, 0])]
    data[:, 0] -= data[0, 0]
    return data   # (N, 4): t, x, y, theta
# ═════════════════════════════════════════════════════════════════════════════
#  CUBIC POLYNOMIAL  —  one axis, one segment
# ═════════════════════════════════════════════════════════════════════════════
#
#  p(tau) = a0 + a1·tau + a2·tau² + a3·tau³,   tau = t - t0
#  Boundary conditions: p(t0)=p0, p'(t0)=v0, p(tf)=pf, p'(tf)=vf
def cubic_coeffs(t0: float, tf: float,
                 p0: float, v0: float,
                 pf: float, vf: float) -> np.ndarray:
    T = tf - t0
    if T <= 1e-9:
        return np.array([p0, 0.0, 0.0, 0.0, t0])
    h    = pf - p0
    a0_c = p0
    a1_c = v0
    a2_c = (3*h / T**2) - (2*v0 + vf) / T
    a3_c = (-2*h / T**3) + (v0 + vf)  / T**2
    return np.array([a0_c, a1_c, a2_c, a3_c, t0])

def eval_cubic(t: float, c: np.ndarray):
    """Returns (position, velocity) at absolute time t."""
    a0_c, a1_c, a2_c, a3_c, t0 = c
    tau = max(0.0, t - t0)
    p   = a0_c + a1_c*tau   + a2_c*tau**2 + a3_c*tau**3
    v   = a1_c + 2*a2_c*tau + 3*a3_c*tau**2
    return p, v
# ═════════════════════════════════════════════════════════════════════════════
#  SPLINE BUILDER
# ═════════════════════════════════════════════════════════════════════════════
def build_spline(waypoints: np.ndarray, pose_offset: tuple):
    """
    Build cubic splines for x, y, and theta from waypoints.
    pose_offset = (dx, dy) shifts the path to start at the robot's actual
    position rather than the logged position.
    Interior knot velocities are estimated with the same weighted central
    finite-difference scheme used in Doc-3 so the robot does NOT decelerate
    to zero at every waypoint.  Endpoint velocities are clamped to zero.
    Returns:
        segments  — list of dicts {t0, tf, cx, cy, ct}
        duration  — total trajectory time (seconds)
    """
    dx, dy = pose_offset
    N       = len(waypoints)
    times   = waypoints[:, 0]
    xs      = waypoints[:, 1] + dx
    ys      = waypoints[:, 2] + dy
    thetas  = waypoints[:, 3]

    # ── Central finite-difference velocity estimation (Doc-3 method) ──────────
    vx = np.zeros(N)
    vy = np.zeros(N)
    vt = np.zeros(N)
    for i in range(1, N - 1):
        dt_prev = times[i]     - times[i - 1]
        dt_next = times[i + 1] - times[i]
        if dt_prev < 1e-9 or dt_next < 1e-9:
            continue
        vx[i] = 0.5 * ((xs[i]  - xs[i-1])  / dt_prev + (xs[i+1]  - xs[i])  / dt_next)
        vy[i] = 0.5 * ((ys[i]  - ys[i-1])  / dt_prev + (ys[i+1]  - ys[i])  / dt_next)
        dth_p = _norm(thetas[i]   - thetas[i-1]) / dt_prev
        dth_n = _norm(thetas[i+1] - thetas[i])   / dt_next
        vt[i] = 0.5 * (dth_p + dth_n)

    # Endpoints: zero velocity → smooth start / stop
    vx[0] = vy[0] = vt[0] = 0.0
    vx[-1] = vy[-1] = vt[-1] = 0.0

    # ── Build one cubic segment per consecutive waypoint pair ─────────────────
    segments = []
    for i in range(N - 1):
        t0, tf = float(times[i]), float(times[i + 1])
        if abs(tf - t0) < 1e-9:
            continue
        segments.append({
            "t0": t0,
            "tf": tf,
            "cx": cubic_coeffs(t0, tf, xs[i],     vx[i], xs[i+1],     vx[i+1]),
            "cy": cubic_coeffs(t0, tf, ys[i],     vy[i], ys[i+1],     vy[i+1]),
            "ct": cubic_coeffs(t0, tf, thetas[i], vt[i], thetas[i+1], vt[i+1]),
        })
    return segments, float(times[-1])

def _norm(a: float) -> float:
    """Wrap angle to [-π, π]."""
    return math.atan2(math.sin(a), math.cos(a))
# ═════════════════════════════════════════════════════════════════════════════
#  COORDINATOR  —  synchronised start gate
# ═════════════════════════════════════════════════════════════════════════════
class Coordinator:
    """
    Thread-safe.  Every robot calls report_ready() on its first odom.
    Once ALL robots have checked in, releases a shared absolute start_time.
    """
    def __init__(self, robot_names, sync_delay=SYNC_DELAY):
        self._expected   = set(robot_names)
        self._ready      = set()
        self._start_time = None
        self._lock       = threading.Lock()
        self._delay      = sync_delay

    def report_ready(self, robot_name: str, now_sec: float):
        with self._lock:
            self._ready.add(robot_name)
            if self._ready >= self._expected and self._start_time is None:
                self._start_time = now_sec + self._delay
        return self._start_time
# ═════════════════════════════════════════════════════════════════════════════
#  ROBOT NODE
# ═════════════════════════════════════════════════════════════════════════════
class PathFollower(Node):
    def __init__(self, robot_name: str, coordinator: Coordinator):
        super().__init__(f'{robot_name}_controller')
        self.robot = robot_name
        self.coord = coordinator
        self.cfg   = ROBOT_CONFIG[robot_name]

        self.current_pose    = None
        self.segments        = []
        self.traj_duration   = 0.0
        self.traj_created    = False
        self.start_time      = None
        self.last_debug_time = 0.0
        self.logged_segments = set()   # tracks which waypoints have been logged

        # ── Load waypoints from CSV ───────────────────────────────────────────
        csv_path = os.path.join(PATH_DIR, self.cfg.csv_file)
        try:
            self.waypoints = load_waypoints(csv_path)
            self.get_logger().info(
                f"[CSV]  {self.robot}: {len(self.waypoints)} waypoints  "
                f"total_time={self.waypoints[-1, 0]:.2f}s  ({csv_path})"
            )
        except Exception as e:
            self.get_logger().error(
                f"[CSV]  {self.robot}: FAILED to load {csv_path} — {e}"
            )
            self.waypoints = None

        # ── QoS ───────────────────────────────────────────────────────────────
        # tb4_1 publishes odom with BEST_EFFORT → sub must match.
        # tb4_1 cmd_vel subscriber is RELIABLE  → pub must match.
        if robot_name == "tb4_1":
            sub_qos = QOS_BEST_EFFORT
            pub_qos = QOS_RELIABLE
        else:
            sub_qos = QOS_RELIABLE
            pub_qos = QOS_RELIABLE

        self.get_logger().info(
            f"[INIT] {self.robot}  "
            f"sub_qos={'BE' if sub_qos is QOS_BEST_EFFORT else 'RE'}  "
            f"pub_qos={'BE' if pub_qos is QOS_BEST_EFFORT else 'RE'}  "
            f"kp_lin={self.cfg.kp_linear} kp_ang={self.cfg.kp_angular} ky={self.cfg.ky}  "
            f"v_max={self.cfg.v_max} w_max={self.cfg.w_max}"
        )

        # ── ROS interfaces ────────────────────────────────────────────────────
        self.create_subscription(
            Odometry, f'/{self.robot}/odom_world', self.odom_callback, sub_qos
        )
        self.cmd_pub = self.create_publisher(
            Twist, f'/{self.robot}/cmd_vel', pub_qos
        )
        self.timer = self.create_timer(0.02, self.control_loop)

    # ── Odom ─────────────────────────────────────────────────────────────────
    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        q   = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0*(q.w*q.z + q.x*q.y),
            1.0 - 2.0*(q.y*q.y + q.z*q.z)
        )
        self.current_pose = (pos.x, pos.y, yaw)
        self.get_logger().debug(
            f"[ODOM] {self.robot} x={pos.x:.3f} y={pos.y:.3f} yaw={yaw:.3f}"
        )

    # ── Build spline once all robots are ready ────────────────────────────────
    def create_trajectory(self, shared_start_time: float):
        if self.waypoints is None:
            self.get_logger().error(f"[TRAJ] {self.robot}: no waypoints — skipping")
            self.traj_created = True
            return
        x0, y0, _ = self.current_pose
        dx = x0 - float(self.waypoints[0, 1])
        dy = y0 - float(self.waypoints[0, 2])
        self.segments, self.traj_duration = build_spline(self.waypoints, (dx, dy))
        self.start_time   = shared_start_time
        self.traj_created = True
        self.get_logger().info(
            f"[TRAJ] {self.robot}: {len(self.segments)} cubic segments  "
            f"total={self.traj_duration:.2f}s  "
            f"offset=({dx:.3f},{dy:.3f})  "
            f"start_at={shared_start_time:.3f}s"
        )

    # ── Find active segment ───────────────────────────────────────────────────
    def _active_segment(self, t: float):
        for seg in self.segments:
            if seg["t0"] <= t <= seg["tf"]:
                return seg
        # t is slightly past the last segment's tf — clamp to it
        last = self.segments[-1]
        if t > last["tf"] and t - last["tf"] < 0.5:   # within 0.5 s grace window
            return last
        return None

    # ── Waypoint crossing logger (Doc-3 style) ────────────────────────────────
    def _log_waypoint_crossings(self, t: float):
        x, y, yaw = self.current_pose
        for i, seg in enumerate(self.segments):
            if i in self.logged_segments:
                continue
            if t < seg["tf"]:
                break
            self.logged_segments.add(i)
            xd_end, _ = eval_cubic(seg["tf"], seg["cx"])
            yd_end, _ = eval_cubic(seg["tf"], seg["cy"])
            td_end, _ = eval_cubic(seg["tf"], seg["ct"])
            pos_error = math.hypot(xd_end - x, yd_end - y)
            ang_error = _norm(td_end - yaw)
            time_err  = t - seg["tf"]
            self.get_logger().info(
                f"\n========== [{self.robot}] WAYPOINT {i+1}/{len(self.segments)} "
                f"==========\n"
                f"  Time error    : {time_err:+.4f} s\n"
                f"  Desired pose  : ({xd_end:.3f}, {yd_end:.3f}, {td_end:.3f})\n"
                f"  Actual  pose  : ({x:.3f}, {y:.3f}, {yaw:.3f})\n"
                f"  Position error: {pos_error:.4f} m\n"
                f"  Angular error : {ang_error:.4f} rad\n"
                f"==========================================="
            )

    # ── Control loop (50 Hz) ──────────────────────────────────────────────────
    def control_loop(self):
        if self.current_pose is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        # Phase 1 — wait until every robot has odom
        if not self.traj_created:
            shared_start = self.coord.report_ready(self.robot, now)
            if shared_start is not None:
                self.create_trajectory(shared_start)
            return

        if not self.segments:
            return

        t = now - self.start_time

        # Phase 2 — countdown
        if t < 0:
            if now - self.last_debug_time > 1.0:
                self.last_debug_time = now
                self.get_logger().info(f"[WAIT] {self.robot} → starts in {-t:.2f}s")
            return

        # Log waypoint crossings
        self._log_waypoint_crossings(t)

        # Phase 3 — trajectory time expired: switch to goal-seeking.
        # Linear velocity uses pos_err * cos(heading_err_g) — NOT ex_g * cos(e_theta_g).
        # The old formula used the desired *final heading* error (e_theta_g) which can
        # be large and drive cos() to zero or negative, stalling the robot.
        # The new formula uses only the bearing to the goal, which is always correct
        # regardless of what the trajectory's final heading happens to be.
        if t > self.segments[-1]["tf"]:
            last_seg = self.segments[-1]
            xd_g, _  = eval_cubic(last_seg["tf"], last_seg["cx"])
            yd_g, _  = eval_cubic(last_seg["tf"], last_seg["cy"])

            x, y, yaw = self.current_pose
            pos_err   = math.hypot(xd_g - x, yd_g - y)

            if pos_err < GOAL_TOLERANCE:
                self.cmd_pub.publish(Twist())
                self.get_logger().info(
                    f"[DONE] {self.robot}  final_err={pos_err:.3f}m"
                )
                self.segments = []
                return

            dx_g          = xd_g - x
            dy_g          = yd_g - y
            bearing_g     = math.atan2(dy_g, dx_g)
            heading_err_g = _norm(bearing_g - yaw)

            cfg = self.cfg
            cmd = Twist()
            cmd.linear.x  = float(np.clip(
                cfg.kp_linear * pos_err * math.cos(heading_err_g),
                0.0, cfg.v_max          # clamp to [0, v_max] — never reverse
            ))
            cmd.angular.z = float(np.clip(
                cfg.kp_angular * pos_err * math.sin(heading_err_g),
                -cfg.w_max, cfg.w_max
            ))
            self.cmd_pub.publish(cmd)

            if now - self.last_debug_time > 0.5:
                self.last_debug_time = now
                self.get_logger().info(
                    f"[SEEK] {self.robot}  pos_err={pos_err:.3f}m  "
                    f"des=({xd_g:.2f},{yd_g:.2f})  act=({x:.2f},{y:.2f})"
                )
            return

        # Phase 4 — find active segment
        seg = self._active_segment(t)
        if seg is None:
            return

        # ── Desired state from cubic splines ──────────────────────────────────
        xd,            xd_dot           = eval_cubic(t, seg["cx"])
        yd,            yd_dot           = eval_cubic(t, seg["cy"])
        desired_theta, desired_theta_dot = eval_cubic(t, seg["ct"])

        x, y, yaw = self.current_pose
        dx = xd - x
        dy = yd - y

        # ── Errors in robot body frame ────────────────────────────────────────
        ex      =  math.cos(yaw)*dx + math.sin(yaw)*dy
        ey      = -math.sin(yaw)*dx + math.cos(yaw)*dy
        e_theta =  _norm(desired_theta - yaw)

        # ── Feedforward velocities (Doc-3 method) ─────────────────────────────
        # Project 2D world-frame velocity onto the desired heading direction
        v_d = xd_dot * math.cos(desired_theta) + yd_dot * math.sin(desired_theta)
        w_d = desired_theta_dot

        # ── Lateral correction (Doc-3 augmented law) ──────────────────────────
        #   Kanayama term:  ky · v_d · ey
        #       Corrects lateral drift proportional to forward speed.
        #   Bearing recovery: kp_angular · pos_err · sin(bearing_err)
        #       Steers toward the desired point when moving.  sin() keeps it
        #       smooth and bounded at ±1.
        #   Orientation correction: kp_angular · sin(e_theta)
        #       Takes over when v_d ≈ 0 so the robot corrects heading without
        #       spinning uncontrollably (bearing term is singular at low speed).
        #   v_blend smoothly transitions between the two angular strategies:
        #       1.0 when v_d ≥ 0.08 m/s (bearing recovery active)
        #       0.0 when v_d = 0       (orientation correction only)
        pos_error          = math.hypot(dx, dy)
        bearing_to_desired = math.atan2(dy, dx)
        heading_error      = _norm(bearing_to_desired - yaw)
        v_blend            = min(1.0, abs(v_d) / 0.08)

        cfg = self.cfg
        lateral_correction = (
            cfg.ky          * v_d * ey
            + v_blend       * cfg.kp_angular * pos_error * math.sin(heading_error)
            + (1.0-v_blend) * cfg.kp_angular * math.sin(e_theta)
        )

        # ── Control commands ──────────────────────────────────────────────────
        v_cmd = v_d * math.cos(e_theta) + cfg.kp_linear * ex
        # Velocity floor: the spline forces v_d → 0 in the final segments.
        # When feedforward dies but the robot still has significant position
        # error and is roughly facing the target, inject a minimum forward
        # velocity so the robot does not stall before Phase 3 kicks in.
        if pos_error > GOAL_TOLERANCE and math.cos(heading_error) > 0.5:
            v_cmd = max(v_cmd, cfg.v_max * V_FLOOR_FRAC)

        cmd = Twist()
        cmd.linear.x  = float(np.clip(v_cmd, -cfg.v_max, cfg.v_max))
        cmd.angular.z = float(np.clip(
            w_d + lateral_correction,
            -cfg.w_max, cfg.w_max
        ))
        self.cmd_pub.publish(cmd)

        # ── Throttled log ─────────────────────────────────────────────────────
        if now - self.last_debug_time > 0.5:
            self.last_debug_time = now
            self.get_logger().info(
                f"[CTRL] {self.robot} | t={t:.2f}s "
                f"seg=[{seg['t0']:.1f},{seg['tf']:.1f}] | "
                f"des=({xd:.2f},{yd:.2f},{desired_theta:.2f}) "
                f"act=({x:.2f},{y:.2f},{yaw:.2f}) | "
                f"v={cmd.linear.x:.2f} w={cmd.angular.z:.2f}"
            )
# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
def main():
    rclpy.init()
    coordinator = Coordinator(ROBOTS)
    nodes = [PathFollower(r, coordinator) for r in ROBOTS]
    executor = MultiThreadedExecutor(num_threads=len(nodes))
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