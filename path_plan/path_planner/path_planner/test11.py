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
    "/home/multi-robot/murogai_nav/src/MuRoGAI-Nav/"
    "path_plan/path_planner/path_planner/trajectory_logs2"
)

# ═════════════════════════════════════════════════════════════════════════════
#  PER-ROBOT CONFIG
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class RobotConfig:
    csv_file:   str   = ""
    kp_linear:  float = 1.5
    kp_angular: float = 2.5
    ky:         float = 3.5
    v_max:      float = 0.22
    w_max:      float = 2.84

ROBOT_CONFIG: dict[str, RobotConfig] = {
    # "burger1": RobotConfig(
    #     csv_file="burger1.csv",
    #     kp_linear=3.2, kp_angular=3.7, ky=4.2,
    #     v_max=0.22, w_max=2.84
    # ),
    # "burger2": RobotConfig(
    #     csv_file="burger2.csv",
    #     kp_linear=3.2, kp_angular=3.7, ky=4.2,
    #     v_max=0.22, w_max=2.84
    # ),
    # "burger3": RobotConfig(
    #     csv_file="burger3.csv",
    #     kp_linear=3.2, kp_angular=3.7, ky=4.2,
    #     v_max=0.22, w_max=2.84
    # ),
    # "waffle": RobotConfig(
    #     csv_file="waffle.csv",
    #     kp_linear=3.2, kp_angular=1.0, ky=1.35,
    #     v_max=0.26, w_max=1.82,
    # ),
    # "firebird": RobotConfig(
    #     csv_file="firebird.csv",
    #     kp_linear=1.5, kp_angular=0.73, ky=2.32,
    #     v_max=0.26, w_max=1.9
    # ),
    # "tb4_1": RobotConfig(
    #     csv_file="tb4_1.csv",
    #     kp_linear=1.5, kp_angular=2.5, ky=3.5,
    #     v_max=0.26, w_max=1.9
    # ),
}

ROBOTS         = list(ROBOT_CONFIG.keys())
SYNC_DELAY     = 1.0    # s — wait after all robots have odom before starting
GOAL_TOLERANCE = 0.08   # m — SEEK done within this radius

# Skip angles / distances too small to bother with a dedicated sub-segment
MIN_ROT_RAD = 0.05   # rad
MIN_DIST_M  = 0.02   # m
# Drop near-duplicate CSV rows (e.g. the 9 ms final heading-only row)
MIN_WP_DT   = 0.10   # s

QOS_RELIABLE    = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,    depth=10)
QOS_BEST_EFFORT = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)

# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def _norm(a: float) -> float:
    """Wrap angle to [-π, π]."""
    return math.atan2(math.sin(a), math.cos(a))

# ═════════════════════════════════════════════════════════════════════════════
#  CSV LOADER
# ═════════════════════════════════════════════════════════════════════════════
def load_waypoints(csv_path: str) -> np.ndarray:
    """
    Load CSV (time, x, y, theta[, ...]).
    Returns (N, 4) sorted by time, t[0]=0, near-duplicate rows removed.
    """
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    data = data[np.argsort(data[:, 0])]
    data[:, 0] -= data[0, 0]
    keep = [0]
    for i in range(1, len(data)):
        if data[i, 0] - data[keep[-1], 0] >= MIN_WP_DT:
            keep.append(i)
    return data[np.ix_(keep, [0, 1, 2, 3])]   # (N,4): t,x,y,theta

# ═════════════════════════════════════════════════════════════════════════════
#  3-STEP TRAJECTORY BUILDER
# ═════════════════════════════════════════════════════════════════════════════
#
#  For each consecutive waypoint pair (p0 → p1) we produce up to 3 segments:
#
#   STEP 1  [rot]   Rotate in place to face p1.
#                   Duration = |bearing_to_p1 - current_heading| / w_max
#
#   STEP 2  [tra]   Drive straight to p1 at v_max.
#                   Duration = distance(p0, p1) / v_max
#
#   STEP 3  [rot]   Rotate in place to desired heading θ1.
#                   Duration = |θ1 - bearing_to_p1| / w_max
#
#  Timing comes purely from robot limits (v_max, w_max), so each step is
#  guaranteed achievable and the robot arrives at each waypoint on time.

def build_trajectory(waypoints: np.ndarray,
                     pose_offset: tuple,
                     v_max: float,
                     w_max: float) -> tuple[list, float]:
    """
    Returns (segments, total_duration_s).

    Each segment dict:
      type        : 'rot' | 'tra'
      t0, tf      : absolute start/end time from trajectory start
      x0,y0,th0   : start pose
      x1,y1,th1   : end pose
      v_ff        : feedforward linear  velocity (0 for rot, ±v_max for tra)
      w_ff        : feedforward angular velocity (±w_max for rot, 0 for tra)
      wp_idx      : which original waypoint gap this belongs to
      is_wp_end   : True on the last sub-segment of each waypoint gap
    """
    dx_off, dy_off = pose_offset
    poses = [
        (float(wp[1]) + dx_off,
         float(wp[2]) + dy_off,
         float(wp[3]))
        for wp in waypoints
    ]

    segments = []
    t        = 0.0

    for i in range(len(poses) - 1):
        x0, y0, _ = poses[i]
        x1, y1, th1 = poses[i + 1]

        # Current heading = end heading of the last sub-segment, or CSV value
        if segments:
            th0 = segments[-1]['th1']
        else:
            th0 = poses[0][2]

        ddx     = x1 - x0
        ddy     = y1 - y0
        dist    = math.hypot(ddx, ddy)
        bearing = math.atan2(ddy, ddx)   # direction from p0 to p1

        wp_segs = []

        if dist < MIN_DIST_M:
            # ── Pure heading correction (positions identical) ─────────────
            dth = _norm(th1 - th0)
            if abs(dth) > MIN_ROT_RAD:
                dt = abs(dth) / w_max
                wp_segs.append(_make_rot(t, dt, x0, y0, th0, th1, i))
                t += dt

        else:
            # ── STEP 1: rotate in place to face the next waypoint ─────────
            turn1 = _norm(bearing - th0)
            if abs(turn1) > MIN_ROT_RAD:
                dt = abs(turn1) / w_max
                wp_segs.append(_make_rot(t, dt, x0, y0, th0, bearing, i))
                t  += dt
                th0 = bearing   # heading after step 1

            # ── STEP 2: drive straight to the waypoint ────────────────────
            dt = dist / v_max
            wp_segs.append({
                'type':       'tra',
                'wp_idx':     i,
                'is_wp_end':  False,
                't0': t, 'tf': t + dt,
                'x0': x0, 'y0': y0, 'th0': th0,
                'x1': x1, 'y1': y1, 'th1': th0,   # heading unchanged
                'v_ff': v_max,
                'w_ff': 0.0,
            })
            t  += dt
            th0 = th0   # same heading after step 2

            # ── STEP 3: rotate in place to desired heading at waypoint ─────
            turn2 = _norm(th1 - th0)
            if abs(turn2) > MIN_ROT_RAD:
                dt = abs(turn2) / w_max
                wp_segs.append(_make_rot(t, dt, x1, y1, th0, th1, i))
                t += dt

        # Mark the last sub-segment of this waypoint gap
        if wp_segs:
            wp_segs[-1]['is_wp_end'] = True
        segments.extend(wp_segs)

    return segments, t


def _make_rot(t0, dt, x, y, th0, th1, wp_idx):
    """Create a rotate-in-place sub-segment."""
    dth  = _norm(th1 - th0)
    w_ff = math.copysign(abs(dth) / max(dt, 1e-9), dth)
    return {
        'type':      'rot',
        'wp_idx':    wp_idx,
        'is_wp_end': False,
        't0': t0, 'tf': t0 + dt,
        'x0': x,  'y0': y,  'th0': th0,
        'x1': x,  'y1': y,  'th1': th1,
        'v_ff': 0.0,
        'w_ff': w_ff,
    }

# ═════════════════════════════════════════════════════════════════════════════
#  SEGMENT EVALUATOR
# ═════════════════════════════════════════════════════════════════════════════
def eval_seg(t: float, seg: dict):
    """
    Returns desired pose + feedforward (xd, yd, th_d, v_d, w_d) at time t.
    """
    t0  = seg['t0']
    tf  = seg['tf']
    T   = max(tf - t0, 1e-9)
    tau = max(0.0, min(t - t0, T))

    if seg['type'] == 'rot':
        xd  = seg['x0']
        yd  = seg['y0']
        th_d = _norm(seg['th0'] + seg['w_ff'] * tau)
        v_d  = 0.0
        w_d  = seg['w_ff']
    else:   # 'tra'
        frac = tau / T
        xd   = seg['x0'] + (seg['x1'] - seg['x0']) * frac
        yd   = seg['y0'] + (seg['y1'] - seg['y0']) * frac
        th_d = seg['th0']
        v_d  = seg['v_ff']
        w_d  = 0.0

    return xd, yd, th_d, v_d, w_d

# ═════════════════════════════════════════════════════════════════════════════
#  COORDINATOR
# ═════════════════════════════════════════════════════════════════════════════
class Coordinator:
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
        self.logged_wps      = set()

        csv_path = os.path.join(PATH_DIR, self.cfg.csv_file)
        try:
            self.waypoints = load_waypoints(csv_path)
            self.get_logger().info(
                f"[CSV]  {self.robot}: {len(self.waypoints)} waypoints  "
                f"total_time={self.waypoints[-1,0]:.2f}s  ({csv_path})"
            )
        except Exception as e:
            self.get_logger().error(f"[CSV]  {self.robot}: FAILED — {e}")
            self.waypoints = None

        sub_qos = QOS_BEST_EFFORT if robot_name == "tb4_1" else QOS_RELIABLE
        pub_qos = QOS_RELIABLE
        self.get_logger().info(
            f"[INIT] {self.robot}  "
            f"sub_qos={'BE' if sub_qos is QOS_BEST_EFFORT else 'RE'}  "
            f"pub_qos=RE  "
            f"kp_lin={self.cfg.kp_linear} kp_ang={self.cfg.kp_angular} "
            f"ky={self.cfg.ky}  v_max={self.cfg.v_max} w_max={self.cfg.w_max}"
        )

        self.create_subscription(
            Odometry, f'/{self.robot}/odom_world', self.odom_callback, sub_qos)
        self.cmd_pub = self.create_publisher(
            Twist, f'/{self.robot}/cmd_vel', pub_qos)
        self.timer = self.create_timer(0.02, self.control_loop)

    # ── Odom ─────────────────────────────────────────────────────────────────
    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        q   = msg.pose.pose.orientation
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y),
                         1.0 - 2.0*(q.y*q.y + q.z*q.z))
        self.current_pose = (pos.x, pos.y, yaw)

    # ── Build trajectory ──────────────────────────────────────────────────────
    def create_trajectory(self, shared_start_time: float):
        if self.waypoints is None:
            self.get_logger().error(f"[TRAJ] {self.robot}: no waypoints")
            self.traj_created = True
            return
        x0, y0, _ = self.current_pose
        dx = x0 - float(self.waypoints[0, 1])
        dy = y0 - float(self.waypoints[0, 2])
        self.segments, self.traj_duration = build_trajectory(
            self.waypoints, (dx, dy), self.cfg.v_max, self.cfg.w_max)
        self.start_time   = shared_start_time
        self.traj_created = True
        n_wp  = len(self.waypoints) - 1
        n_seg = len(self.segments)
        self.get_logger().info(
            f"[TRAJ] {self.robot}: {n_wp} waypoints → {n_seg} sub-segments  "
            f"total={self.traj_duration:.2f}s  "
            f"offset=({dx:.3f},{dy:.3f})  start_at={shared_start_time:.3f}s"
        )

    # ── Find active segment ───────────────────────────────────────────────────
    def _active_segment(self, t: float):
        for seg in self.segments:
            if seg['t0'] <= t <= seg['tf']:
                return seg
        last = self.segments[-1]
        if t > last['tf'] and t - last['tf'] < 0.5:
            return last
        return None

    # ── Waypoint crossing logger ──────────────────────────────────────────────
    def _log_waypoint_crossings(self, t: float):
        x, y, yaw = self.current_pose
        for seg in self.segments:
            if not seg.get('is_wp_end', False):
                continue
            wi = seg['wp_idx']
            if wi in self.logged_wps:
                continue
            if t < seg['tf']:
                break
            self.logged_wps.add(wi)
            xd_end  = seg['x1']
            yd_end  = seg['y1']
            td_end  = _norm(seg['th1'])
            pos_err = math.hypot(xd_end - x, yd_end - y)
            ang_err = _norm(td_end - yaw)
            self.get_logger().info(
                f"\n========== [{self.robot}] WAYPOINT {wi+1}/"
                f"{len(self.waypoints)-1} ==========\n"
                f"  Time error    : {t - seg['tf']:+.4f} s\n"
                f"  Desired pose  : ({xd_end:.3f}, {yd_end:.3f}, {td_end:.3f})\n"
                f"  Actual  pose  : ({x:.3f}, {y:.3f}, {yaw:.3f})\n"
                f"  Position error: {pos_err:.4f} m\n"
                f"  Angular error : {ang_err:.4f} rad\n"
                f"==========================================="
            )

    # ── Control loop (50 Hz) ──────────────────────────────────────────────────
    def control_loop(self):
        if self.current_pose is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        # Phase 1 — wait for all robots
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

        self._log_waypoint_crossings(t)

        # ── Phase 3 — SEEK after trajectory ends ─────────────────────────────
        if t > self.segments[-1]['tf']:
            last      = self.segments[-1]
            xd_g, yd_g = last['x1'], last['y1']
            x, y, yaw  = self.current_pose
            pos_err    = math.hypot(xd_g - x, yd_g - y)

            if pos_err < GOAL_TOLERANCE:
                self.cmd_pub.publish(Twist())
                self.get_logger().info(f"[DONE] {self.robot}  final_err={pos_err:.3f}m")
                self.segments = []
                return

            bearing_g     = math.atan2(yd_g - y, xd_g - x)
            heading_err_g = _norm(bearing_g - yaw)
            cfg = self.cfg
            cmd = Twist()
            cmd.linear.x  = float(np.clip(
                cfg.kp_linear * pos_err * math.cos(heading_err_g), 0.0, cfg.v_max))
            cmd.angular.z = float(np.clip(
                cfg.kp_angular * math.sin(heading_err_g), -cfg.w_max, cfg.w_max))
            self.cmd_pub.publish(cmd)
            if now - self.last_debug_time > 0.5:
                self.last_debug_time = now
                self.get_logger().info(
                    f"[SEEK] {self.robot}  pos_err={pos_err:.3f}m  "
                    f"des=({xd_g:.2f},{yd_g:.2f})  act=({x:.2f},{y:.2f})"
                )
            return

        # ── Phase 4 — follow the 3-step trajectory ───────────────────────────
        seg = self._active_segment(t)
        if seg is None:
            return

        xd, yd, th_d, v_d, w_d = eval_seg(t, seg)

        x, y, yaw = self.current_pose
        dx = xd - x
        dy = yd - y

        # Errors in robot body frame
        ex      =  math.cos(yaw)*dx + math.sin(yaw)*dy
        ey      = -math.sin(yaw)*dx + math.cos(yaw)*dy
        e_theta =  _norm(th_d - yaw)

        pos_error     = math.hypot(dx, dy)
        bearing_to_d  = math.atan2(dy, dx)
        heading_error = _norm(bearing_to_d - yaw)

        # v_blend: 1 when moving (bearing correction), 0 when rotating (heading correction)
        v_blend = min(1.0, abs(v_d) / 0.08)

        cfg = self.cfg
        lateral_correction = (
            cfg.ky          * v_d * ey
            + v_blend       * cfg.kp_angular * pos_error * math.sin(heading_error)
            + (1.0-v_blend) * cfg.kp_angular * math.sin(e_theta)
        )

        if seg['type'] == 'rot':
            # Strictly rotate in place — no forward/backward motion at all.
            # kp_linear * ex would drive the robot off the spot; suppress it.
            v_cmd = 0.0
        else:
            v_cmd = v_d * math.cos(e_theta) + cfg.kp_linear * ex
            # Never reverse on a forward segment. Position error behind the
            # robot (ex < 0) at segment transitions causes kp_linear*ex to
            # go negative, making the robot drive backward. Clamp it out.
            if seg['v_ff'] > 0:
                v_cmd = max(v_cmd, 0.0)
            # Velocity floor so robot doesn't stall near end of segment
            if pos_error > GOAL_TOLERANCE and math.cos(heading_error) > 0.5:
                v_cmd = max(v_cmd, cfg.v_max * 0.12)

        cmd = Twist()
        cmd.linear.x  = float(np.clip(v_cmd, -cfg.v_max, cfg.v_max))
        cmd.angular.z = float(np.clip(
            w_d + lateral_correction, -cfg.w_max, cfg.w_max))
        self.cmd_pub.publish(cmd)

        if now - self.last_debug_time > 0.5:
            self.last_debug_time = now
            self.get_logger().info(
                f"[CTRL] {self.robot} | t={t:.2f}s "
                f"seg=[{seg['t0']:.1f},{seg['tf']:.1f}]"
                f"({'rot' if seg['type']=='rot' else 'tra'}) | "
                f"des=({xd:.2f},{yd:.2f},{th_d:.2f}) "
                f"act=({x:.2f},{y:.2f},{yaw:.2f}) | "
                f"v={cmd.linear.x:.2f} w={cmd.angular.z:.2f}"
            )

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