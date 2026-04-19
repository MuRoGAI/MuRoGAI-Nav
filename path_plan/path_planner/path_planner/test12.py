#!/usr/bin/env python3
import math
import os
import threading
from dataclasses import dataclass, field
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
#  use_rsr=True  → rot+tra 2-step planner  (waffle)
#  use_rsr=False → cubic spline planner    (all others)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class RobotConfig:
    csv_file:   str   = ""
    kp_linear:  float = 1.5
    kp_angular: float = 2.5
    ky:         float = 3.5
    v_max:      float = 0.22
    w_max:      float = 2.84
    use_rsr:    bool  = False   # True → rot+tra planner, False → cubic spline

ROBOT_CONFIG: dict[str, RobotConfig] = {
    # "burger1": RobotConfig(
    #     csv_file="burger1.csv",
    #     kp_linear=3.0, kp_angular=3.5, ky=4.0,
    #     v_max=0.3, w_max=2.84, use_rsr=False,
    # ),
    # "burger2": RobotConfig(
    #     csv_file="burger2.csv",
    #     kp_linear=1.5, kp_angular=2.5, ky=3.5,
    #     v_max=0.4, w_max=2.84, use_rsr=False,
    # ),
    # "burger3": RobotConfig(
    #     csv_file="burger3.csv",
    #     kp_linear=1.5, kp_angular=2.5, ky=3.5,
    #     v_max=0.22, w_max=2.84, use_rsr=False,
    # ),
    # "waffle": RobotConfig(
    #     csv_file="waffle.csv",
    #     kp_linear=1.2, kp_angular=0.5, ky=1.5,
    #     v_max=0.26, w_max=1.82,
    #     use_rsr=True,   # ← rot+tra 2-step planner
    # ),
    # "firebird": RobotConfig(
    #     csv_file="firebird.csv",
    #     kp_linear=0.8, kp_angular=0.8, ky=1.0,
    #     v_max=0.26, w_max=1.9, use_rsr=False,
    # ),
    # "tb4_1": RobotConfig(
    #     csv_file="tb4_1.csv",
    #     kp_linear=1.5, kp_angular=2.5, ky=3.5,
    #     v_max=0.26, w_max=1.90, use_rsr=False,
    # ),
}

ROBOTS         = list(ROBOT_CONFIG.keys())
SYNC_DELAY     = 1.0
GOAL_TOLERANCE = 0.08   # m
V_FLOOR_FRAC   = 0.12   # fraction of v_max minimum forward speed (spline robots)

# ── RSR planner constants ─────────────────────────────────────────────────────
MIN_ROT_RAD     = 0.05   # rad — skip trivially small rotations
MIN_DIST_M      = 0.02   # m   — treat as pure rotation if distance < this
MIN_WP_DT       = 0.10   # s   — drop near-duplicate CSV rows
ROT_TIME_FACTOR = 1.35   # robot achieves ~74% of w_max → give 35% extra time

QOS_RELIABLE    = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,    depth=10)
QOS_BEST_EFFORT = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)

# ═════════════════════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def _norm(a: float) -> float:
    """Wrap angle to [-π, π]."""
    return math.atan2(math.sin(a), math.cos(a))

# ═════════════════════════════════════════════════════════════════════════════
#  CSV LOADER
# ═════════════════════════════════════════════════════════════════════════════
def load_waypoints(csv_path: str, drop_duplicates: bool = False) -> np.ndarray:
    """
    Load CSV (time, x, y, theta[, ...]).
    Returns (N, 4) sorted by time, t[0]=0.
    If drop_duplicates=True, rows with dt < MIN_WP_DT are removed (RSR planner).
    """
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    data = data[np.argsort(data[:, 0])]
    data[:, 0] -= data[0, 0]
    if drop_duplicates:
        keep = [0]
        for i in range(1, len(data)):
            if data[i, 0] - data[keep[-1], 0] >= MIN_WP_DT:
                keep.append(i)
        data = data[np.ix_(keep, [0, 1, 2, 3])]
    return data[:, :4]   # (N, 4): t, x, y, theta

# ═════════════════════════════════════════════════════════════════════════════
#  ── CUBIC SPLINE PLANNER  (burger / firebird / tb4_1) ──────────────────────
# ═════════════════════════════════════════════════════════════════════════════
def cubic_coeffs(t0, tf, p0, v0, pf, vf):
    T = tf - t0
    if T <= 1e-9:
        return np.array([p0, 0.0, 0.0, 0.0, t0])
    h    = pf - p0
    a2   = (3*h / T**2) - (2*v0 + vf) / T
    a3   = (-2*h / T**3) + (v0 + vf)  / T**2
    return np.array([p0, v0, a2, a3, t0])

def eval_cubic(t, c):
    a0, a1, a2, a3, t0 = c
    tau = max(0.0, t - t0)
    return (a0 + a1*tau + a2*tau**2 + a3*tau**3,
            a1 + 2*a2*tau + 3*a3*tau**2)

def build_spline(waypoints, pose_offset):
    """
    Cubic spline planner with central-difference knot velocities and
    linear theta interpolation (avoids ±π ringing).
    """
    dx, dy = pose_offset
    N      = len(waypoints)
    times  = waypoints[:, 0]
    xs     = waypoints[:, 1] + dx
    ys     = waypoints[:, 2] + dy
    thetas = np.unwrap(waypoints[:, 3])

    vx = np.zeros(N)
    vy = np.zeros(N)
    for i in range(1, N - 1):
        dt_p = times[i] - times[i-1]
        dt_n = times[i+1] - times[i]
        if dt_p < 1e-9 or dt_n < 1e-9:
            continue
        vx[i] = 0.5 * ((xs[i]-xs[i-1])/dt_p + (xs[i+1]-xs[i])/dt_n)
        vy[i] = 0.5 * ((ys[i]-ys[i-1])/dt_p + (ys[i+1]-ys[i])/dt_n)
    vx[0] = vy[0] = vx[-1] = vy[-1] = 0.0

    MIN_SEG = 0.1
    segments = []
    for i in range(N - 1):
        t0, tf = float(times[i]), float(times[i+1])
        if tf - t0 < MIN_SEG:
            continue
        # Linear theta — avoids cubic overshoot across ±π boundaries
        slope = _norm(thetas[i+1] - thetas[i]) / (tf - t0)
        ct    = np.array([thetas[i], slope, 0.0, 0.0, t0])
        segments.append({
            'type':  'spline',
            't0': t0, 'tf': tf,
            'cx': cubic_coeffs(t0, tf, xs[i],     vx[i], xs[i+1],     vx[i+1]),
            'cy': cubic_coeffs(t0, tf, ys[i],     vy[i], ys[i+1],     vy[i+1]),
            'ct': ct,
        })
    return segments, float(times[-1])

# ═════════════════════════════════════════════════════════════════════════════
#  ── ROT + TRA 2-STEP PLANNER  (waffle) ─────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
def _make_rot(t0, dt, x, y, th0, th1, wp_idx):
    """
    Rotate-in-place sub-segment.
    Clock is advanced by dt*ROT_TIME_FACTOR; w_ff is based on ideal dt
    so feedforward ≈ achievable angular rate.
    """
    dth       = _norm(th1 - th0)
    w_ff      = math.copysign(abs(dth) / max(dt, 1e-9), dth)
    actual_dt = dt * ROT_TIME_FACTOR
    return {
        'type':      'rot',
        'wp_idx':    wp_idx,
        'is_wp_end': False,
        't0': t0, 'tf': t0 + actual_dt,
        'x0': x,  'y0': y,  'th0': th0,
        'x1': x,  'y1': y,  'th1': th1,
        'v_ff': 0.0,
        'w_ff': w_ff,
    }

def build_rsr_trajectory(waypoints, pose_offset, v_max, w_max):
    """
    2-step planner: STEP1 rotate to face next WP, STEP2 drive straight.
    No STEP3 — feedback controller handles final heading correction.
    _norm() automatically picks the shorter CW/CCW rotation.
    """
    dx_off, dy_off = pose_offset
    poses = [(float(wp[1])+dx_off, float(wp[2])+dy_off, float(wp[3]))
             for wp in waypoints]

    segments = []
    t = 0.0

    for i in range(len(poses) - 1):
        x0, y0, _ = poses[i]
        x1, y1, th1 = poses[i+1]
        th0 = segments[-1]['th1'] if segments else poses[0][2]

        ddx     = x1 - x0
        ddy     = y1 - y0
        dist    = math.hypot(ddx, ddy)
        bearing = math.atan2(ddy, ddx)

        wp_segs = []

        if dist < MIN_DIST_M:
            dth = _norm(th1 - th0)
            if abs(dth) > MIN_ROT_RAD:
                dt = abs(dth) / w_max
                wp_segs.append(_make_rot(t, dt, x0, y0, th0, th1, i))
                t += dt * ROT_TIME_FACTOR
        else:
            # STEP 1 — rotate to face next waypoint (shorter direction)
            turn = _norm(bearing - th0)
            if abs(turn) > MIN_ROT_RAD:
                dt = abs(turn) / w_max
                wp_segs.append(_make_rot(t, dt, x0, y0, th0, bearing, i))
                t  += dt * ROT_TIME_FACTOR
                th0 = bearing

            # STEP 2 — drive straight; feedback corrects any lateral drift
            dt = dist / v_max
            wp_segs.append({
                'type':      'tra',
                'wp_idx':    i,
                'is_wp_end': True,
                't0': t, 'tf': t + dt,
                'x0': x0, 'y0': y0, 'th0': th0,
                'x1': x1, 'y1': y1, 'th1': th0,
                'v_ff': v_max,
                'w_ff': 0.0,
            })
            t += dt

        if wp_segs and not any(s['is_wp_end'] for s in wp_segs):
            wp_segs[-1]['is_wp_end'] = True
        segments.extend(wp_segs)

    return segments, t

def eval_rsr_seg(t, seg):
    """Evaluate desired pose + feedforward for a rot or tra segment."""
    t0  = seg['t0']
    tf  = seg['tf']
    T   = max(tf - t0, 1e-9)
    tau = max(0.0, min(t - t0, T))
    if seg['type'] == 'rot':
        return (seg['x0'], seg['y0'],
                _norm(seg['th0'] + seg['w_ff'] * tau),
                0.0, seg['w_ff'])
    else:
        frac = tau / T
        return (seg['x0'] + (seg['x1']-seg['x0'])*frac,
                seg['y0'] + (seg['y1']-seg['y0'])*frac,
                seg['th0'],
                seg['v_ff'], 0.0)

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

    def report_ready(self, robot_name, now_sec):
        with self._lock:
            self._ready.add(robot_name)
            if self._ready >= self._expected and self._start_time is None:
                self._start_time = now_sec + self._delay
        return self._start_time

# ═════════════════════════════════════════════════════════════════════════════
#  ROBOT NODE
# ═════════════════════════════════════════════════════════════════════════════
class PathFollower(Node):
    def __init__(self, robot_name, coordinator):
        super().__init__(f'{robot_name}_controller')
        self.robot = robot_name
        self.coord = coordinator
        self.cfg   = ROBOT_CONFIG[robot_name]

        self.current_pose    = None
        self.segments        = []
        self.traj_created    = False
        self.start_time      = None
        self.last_debug_time = 0.0
        self.logged_wps      = set()

        csv_path = os.path.join(PATH_DIR, self.cfg.csv_file)
        try:
            self.waypoints = load_waypoints(csv_path,
                                            drop_duplicates=self.cfg.use_rsr)
            self.get_logger().info(
                f"[CSV]  {self.robot}: {len(self.waypoints)} waypoints  "
                f"total_time={self.waypoints[-1,0]:.2f}s  "
                f"planner={'RSR' if self.cfg.use_rsr else 'SPLINE'}  "
                f"({csv_path})"
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
        self.get_logger().debug(
            f"[ODOM] {self.robot} x={pos.x:.3f} y={pos.y:.3f} yaw={yaw:.3f}")

    # ── Build trajectory ──────────────────────────────────────────────────────
    def create_trajectory(self, shared_start_time):
        if self.waypoints is None:
            self.get_logger().error(f"[TRAJ] {self.robot}: no waypoints")
            self.traj_created = True
            return
        x0, y0, _ = self.current_pose
        dx = x0 - float(self.waypoints[0, 1])
        dy = y0 - float(self.waypoints[0, 2])

        if self.cfg.use_rsr:
            self.segments, dur = build_rsr_trajectory(
                self.waypoints, (dx, dy), self.cfg.v_max, self.cfg.w_max)
            n_seg = len(self.segments)
            label = f"{len(self.waypoints)-1} waypoints → {n_seg} sub-segments"
        else:
            self.segments, dur = build_spline(self.waypoints, (dx, dy))
            n_seg = len(self.segments)
            label = f"{n_seg} cubic segments"

        self.start_time   = shared_start_time
        self.traj_created = True
        self.get_logger().info(
            f"[TRAJ] {self.robot}: {label}  total={dur:.2f}s  "
            f"offset=({dx:.3f},{dy:.3f})  start_at={shared_start_time:.3f}s"
        )

    # ── Active segment ────────────────────────────────────────────────────────
    def _active_segment(self, t):
        for seg in self.segments:
            if seg['t0'] <= t <= seg['tf']:
                return seg
        last = self.segments[-1]
        if t > last['tf'] and t - last['tf'] < 0.5:
            return last
        return None

    # ── Waypoint crossing logger ──────────────────────────────────────────────
    def _log_waypoint_crossings(self, t):
        x, y, yaw = self.current_pose

        if self.cfg.use_rsr:
            # RSR: log when is_wp_end segment's tf is passed
            for seg in self.segments:
                if not seg.get('is_wp_end', False):
                    continue
                wi = seg['wp_idx']
                if wi in self.logged_wps or t < seg['tf']:
                    continue
                self.logged_wps.add(wi)
                xd, yd = seg['x1'], seg['y1']
                td = _norm(seg['th1'])
                self.get_logger().info(
                    f"\n========== [{self.robot}] WAYPOINT {wi+1}/"
                    f"{len(self.waypoints)-1} ==========\n"
                    f"  Time error    : {t-seg['tf']:+.4f} s\n"
                    f"  Desired pose  : ({xd:.3f}, {yd:.3f}, {td:.3f})\n"
                    f"  Actual  pose  : ({x:.3f}, {y:.3f}, {yaw:.3f})\n"
                    f"  Position error: {math.hypot(xd-x, yd-y):.4f} m\n"
                    f"  Angular error : {_norm(td-yaw):.4f} rad\n"
                    f"==========================================="
                )
        else:
            # Spline: log at end of each spline segment
            for i, seg in enumerate(self.segments):
                if i in self.logged_wps or t < seg['tf']:
                    continue
                self.logged_wps.add(i)
                xd, _ = eval_cubic(seg['tf'], seg['cx'])
                yd, _ = eval_cubic(seg['tf'], seg['cy'])
                td, _ = eval_cubic(seg['tf'], seg['ct'])
                td    = _norm(td)
                self.get_logger().info(
                    f"\n========== [{self.robot}] WAYPOINT {i+1}/"
                    f"{len(self.segments)} ==========\n"
                    f"  Time error    : {t-seg['tf']:+.4f} s\n"
                    f"  Desired pose  : ({xd:.3f}, {yd:.3f}, {td:.3f})\n"
                    f"  Actual  pose  : ({x:.3f}, {y:.3f}, {yaw:.3f})\n"
                    f"  Position error: {math.hypot(xd-x, yd-y):.4f} m\n"
                    f"  Angular error : {_norm(td-yaw):.4f} rad\n"
                    f"==========================================="
                )

    # ── Control loop (50 Hz) ──────────────────────────────────────────────────
    def control_loop(self):
        if self.current_pose is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        if not self.traj_created:
            shared_start = self.coord.report_ready(self.robot, now)
            if shared_start is not None:
                self.create_trajectory(shared_start)
            return

        if not self.segments:
            return

        t = now - self.start_time

        if t < 0:
            if now - self.last_debug_time > 1.0:
                self.last_debug_time = now
                self.get_logger().info(f"[WAIT] {self.robot} → starts in {-t:.2f}s")
            return

        self._log_waypoint_crossings(t)

        # ── Phase 3 — SEEK after trajectory ends ─────────────────────────────
        last = self.segments[-1]
        if t > last['tf']:
            if self.cfg.use_rsr:
                xd_g, yd_g = last['x1'], last['y1']
            else:
                xd_g, _ = eval_cubic(last['tf'], last['cx'])
                yd_g, _ = eval_cubic(last['tf'], last['cy'])

            x, y, yaw = self.current_pose
            pos_err   = math.hypot(xd_g - x, yd_g - y)

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
                    f"des=({xd_g:.2f},{yd_g:.2f})  act=({x:.2f},{y:.2f})")
            return

        # ── Phase 4 — trajectory tracking ────────────────────────────────────
        seg = self._active_segment(t)
        if seg is None:
            return

        if self.cfg.use_rsr:
            self._control_rsr(t, seg, now)
        else:
            self._control_spline(t, seg, now)

    # ── RSR controller ────────────────────────────────────────────────────────
    def _control_rsr(self, t, seg, now):
        xd, yd, th_d, v_d, w_d = eval_rsr_seg(t, seg)
        x, y, yaw = self.current_pose
        dx = xd - x
        dy = yd - y

        ex      =  math.cos(yaw)*dx + math.sin(yaw)*dy
        ey      = -math.sin(yaw)*dx + math.cos(yaw)*dy
        e_theta =  _norm(th_d - yaw)

        pos_error     = math.hypot(dx, dy)
        bearing_to_d  = math.atan2(dy, dx)
        heading_error = _norm(bearing_to_d - yaw)
        v_blend       = min(1.0, abs(v_d) / 0.08)

        cfg = self.cfg
        lateral_correction = (
            cfg.ky          * v_d * ey
            + v_blend       * cfg.kp_angular * pos_error * math.sin(heading_error)
            + (1.0-v_blend) * cfg.kp_angular * math.sin(e_theta)
        )

        if seg['type'] == 'rot':
            v_cmd = 0.0   # strictly in-place — suppress kp_linear*ex
        else:
            v_cmd = v_d * math.cos(e_theta) + cfg.kp_linear * ex
            if seg['v_ff'] > 0:
                v_cmd = max(v_cmd, 0.0)   # never reverse on a forward segment
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

    # ── Spline controller ─────────────────────────────────────────────────────
    def _control_spline(self, t, seg, now):
        xd,  xd_dot  = eval_cubic(t, seg['cx'])
        yd,  yd_dot  = eval_cubic(t, seg['cy'])
        th_d, th_dot = eval_cubic(t, seg['ct'])
        th_d = _norm(th_d)

        x, y, yaw = self.current_pose
        dx = xd - x
        dy = yd - y

        ex      =  math.cos(yaw)*dx + math.sin(yaw)*dy
        ey      = -math.sin(yaw)*dx + math.cos(yaw)*dy
        e_theta =  _norm(th_d - yaw)

        v_d = xd_dot * math.cos(th_d) + yd_dot * math.sin(th_d)
        w_d = th_dot

        pos_error     = math.hypot(dx, dy)
        bearing_to_d  = math.atan2(dy, dx)
        heading_error = _norm(bearing_to_d - yaw)
        v_blend       = min(1.0, abs(v_d) / 0.08)

        cfg = self.cfg
        lateral_correction = (
            cfg.ky          * v_d * ey
            + v_blend       * cfg.kp_angular * pos_error * math.sin(heading_error)
            + (1.0-v_blend) * cfg.kp_angular * math.sin(e_theta)
        )

        v_cmd = v_d * math.cos(e_theta) + cfg.kp_linear * ex
        if pos_error > GOAL_TOLERANCE and math.cos(heading_error) > 0.5:
            v_cmd = max(v_cmd, cfg.v_max * V_FLOOR_FRAC)

        cmd = Twist()
        cmd.linear.x  = float(np.clip(v_cmd, -cfg.v_max, cfg.v_max))
        cmd.angular.z = float(np.clip(
            w_d + lateral_correction, -cfg.w_max, cfg.w_max))
        self.cmd_pub.publish(cmd)

        if now - self.last_debug_time > 0.5:
            self.last_debug_time = now
            self.get_logger().info(
                f"[CTRL] {self.robot} | t={t:.2f}s "
                f"seg=[{seg['t0']:.1f},{seg['tf']:.1f}] | "
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
