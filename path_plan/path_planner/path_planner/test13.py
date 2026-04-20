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

# New controller

# ═════════════════════════════════════════════════════════════════════════════
#  PATH DIRECTORY
# ═════════════════════════════════════════════════════════════════════════════
PATH_DIR = (
    "/home/suraj/murogai_nav/src/MuRoGAI-Nav/"
    "path_plan/path_planner/path_planner/trajectory_logs2"
)

# ═════════════════════════════════════════════════════════════════════════════
#  PER-ROBOT CONFIG
#  use_xy_only=True  → Kanayama + cubic x,y spline, theta ignored (waffle)
#  use_xy_only=False → full cubic spline with theta tracking (all others)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class RobotConfig:
    csv_file:    str   = ""
    kp_linear:   float = 1.5   # forward error gain (ex → v)
    kp_angular:  float = 2.5   # bearing error gain (sin(hdg_err) → w)
    ky:          float = 3.5   # Kanayama lateral gain (v_d · ey → w)
    v_max:       float = 0.22
    w_max:       float = 2.84
    use_xy_only: bool  = False

ROBOT_CONFIG: dict[str, RobotConfig] = {
    "burger1": RobotConfig(
        csv_file="burger1.csv",
        kp_linear=1.5, kp_angular=1.3, ky=4.2,
        v_max=0.22, w_max=2.84
    ),
    "burger2": RobotConfig(
        csv_file="burger2.csv",
        kp_linear=3.2, kp_angular=3.7, ky=4.2,
        v_max=0.22, w_max=2.84
    ),
    "burger3": RobotConfig(
        csv_file="burger3.csv",
        kp_linear=3.2, kp_angular=3.7, ky=4.2,
        v_max=0.22, w_max=2.84
    ),
    "waffle": RobotConfig(
        csv_file="waffle.csv",
        kp_linear=3.2, kp_angular=1.5, ky=2.5,
        v_max=0.26, w_max=1.82,
        use_xy_only=True,   # ← Kanayama x,y-only tracking
    ),
    "firebird": RobotConfig(
        csv_file="firebird.csv",
        kp_linear=1.2, kp_angular=1.2, ky=2.32,
        v_max=0.26, w_max=1.9, use_xy_only=False,
    ),
    # "tb4_1": RobotConfig(
    #     csv_file="tb4_1.csv",
    #     kp_linear=2.8, kp_angular=2.5, ky=3.5,
    #     v_max=0.26, w_max=1.90, use_xy_only=False
    # ),
}

ROBOTS         = list(ROBOT_CONFIG.keys())
SYNC_DELAY     = 1.0
GOAL_TOLERANCE = 0.08   # m
V_FLOOR_FRAC   = 0.12

QOS_RELIABLE    = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,    depth=10)
QOS_BEST_EFFORT = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)

# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def _norm(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))

# ═════════════════════════════════════════════════════════════════════════════
#  CSV LOADER
# ═════════════════════════════════════════════════════════════════════════════
def load_waypoints(csv_path: str) -> np.ndarray:
    """Load CSV (time, x, y, theta[,...]). Returns (N,4), t[0]=0."""
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    data = data[np.argsort(data[:, 0])]
    data[:, 0] -= data[0, 0]
    return data[:, :4]

# ═════════════════════════════════════════════════════════════════════════════
#  CUBIC SPLINE  (shared by both planners)
# ═════════════════════════════════════════════════════════════════════════════
def cubic_coeffs(t0, tf, p0, v0, pf, vf):
    T = tf - t0
    if T <= 1e-9:
        return np.array([p0, 0.0, 0.0, 0.0, t0])
    h  = pf - p0
    a2 = (3*h / T**2) - (2*v0 + vf) / T
    a3 = (-2*h / T**3) + (v0 + vf)  / T**2
    return np.array([p0, v0, a2, a3, t0])

def eval_cubic(t, c):
    a0, a1, a2, a3, t0 = c
    tau = max(0.0, t - t0)
    return (a0 + a1*tau + a2*tau**2 + a3*tau**3,
            a1 + 2*a2*tau + 3*a3*tau**2)

# ─────────────────────────────────────────────────────────────────────────────
#  WAFFLE PLANNER: x,y cubic spline only — theta column ignored
#  Central-difference knot velocities so the robot does NOT stop at every WP.
#  Near-duplicate rows (dt < 0.1 s) are dropped to avoid degenerate segments.
# ─────────────────────────────────────────────────────────────────────────────
def build_xy_spline(waypoints, pose_offset):
    """
    Builds cubic splines for x and y only.
    Theta from the CSV is completely ignored — the Kanayama controller
    derives desired heading from the instantaneous velocity direction.
    Returns: segments (list of dicts), duration (float)
    Each segment: {t0, tf, cx, cy, x1, y1}
    """
    dx_off, dy_off = pose_offset
    # Drop near-duplicate rows
    MIN_DT = 0.1
    keep = [0]
    for i in range(1, len(waypoints)):
        if waypoints[i, 0] - waypoints[keep[-1], 0] >= MIN_DT:
            keep.append(i)
    wps   = waypoints[keep]
    N     = len(wps)
    times = wps[:, 0]
    xs    = wps[:, 1] + dx_off
    ys    = wps[:, 2] + dy_off

    # Central-difference knot velocities
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

    segments = []
    for i in range(N - 1):
        t0, tf = float(times[i]), float(times[i+1])
        if tf - t0 < MIN_DT:
            continue
        segments.append({
            'type': 'xy',
            't0': t0, 'tf': tf,
            'cx': cubic_coeffs(t0, tf, xs[i], vx[i], xs[i+1], vx[i+1]),
            'cy': cubic_coeffs(t0, tf, ys[i], vy[i], ys[i+1], vy[i+1]),
            'x1': xs[i+1],   # waypoint target for logging
            'y1': ys[i+1],
        })
    return segments, float(times[-1])

# ─────────────────────────────────────────────────────────────────────────────
#  FULL SPLINE PLANNER: x, y, theta  (firebird / burger / tb4_1)
# ─────────────────────────────────────────────────────────────────────────────
def build_spline(waypoints, pose_offset):
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
    segments = []
    for i in range(N - 1):
        t0, tf = float(times[i]), float(times[i+1])
        if tf - t0 < 0.1:
            continue
        slope = _norm(thetas[i+1] - thetas[i]) / (tf - t0)
        segments.append({
            'type': 'spline',
            't0': t0, 'tf': tf,
            'cx': cubic_coeffs(t0, tf, xs[i], vx[i], xs[i+1], vx[i+1]),
            'cy': cubic_coeffs(t0, tf, ys[i], vy[i], ys[i+1], vy[i+1]),
            'ct': np.array([thetas[i], slope, 0.0, 0.0, t0]),
        })
    return segments, float(times[-1])

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
        self.logged_segs     = set()

        csv_path = os.path.join(PATH_DIR, self.cfg.csv_file)
        try:
            self.waypoints = load_waypoints(csv_path)
            self.get_logger().info(
                f"[CSV]  {self.robot}: {len(self.waypoints)} waypoints  "
                f"total_time={self.waypoints[-1,0]:.2f}s  "
                f"controller={'XY-Kanayama' if self.cfg.use_xy_only else 'SPLINE'}  "
                f"({csv_path})"
            )
        except Exception as e:
            self.get_logger().error(f"[CSV]  {self.robot}: FAILED — {e}")
            self.waypoints = None

        sub_qos = QOS_BEST_EFFORT if robot_name == "tb4_1" else QOS_RELIABLE
        self.get_logger().info(
            f"[INIT] {self.robot}  "
            f"sub_qos={'BE' if sub_qos is QOS_BEST_EFFORT else 'RE'}  pub_qos=RE  "
            f"kp_lin={self.cfg.kp_linear} kp_ang={self.cfg.kp_angular} "
            f"ky={self.cfg.ky}  v_max={self.cfg.v_max} w_max={self.cfg.w_max}"
        )
        self.create_subscription(
            Odometry, f'/{self.robot}/odom_world', self.odom_callback, sub_qos)
        self.cmd_pub = self.create_publisher(
            Twist, f'/{self.robot}/cmd_vel', QOS_RELIABLE)
        self.timer = self.create_timer(0.02, self.control_loop)

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        q   = msg.pose.pose.orientation
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y),
                         1.0 - 2.0*(q.y*q.y + q.z*q.z))
        self.current_pose = (pos.x, pos.y, yaw)

    def create_trajectory(self, shared_start_time):
        if self.waypoints is None:
            self.get_logger().error(f"[TRAJ] {self.robot}: no waypoints")
            self.traj_created = True
            return
        x0, y0, _ = self.current_pose
        dx = x0 - float(self.waypoints[0, 1])
        dy = y0 - float(self.waypoints[0, 2])

        if self.cfg.use_xy_only:
            self.segments, dur = build_xy_spline(self.waypoints, (dx, dy))
            label = f"{len(self.segments)} xy-segments"
        else:
            self.segments, dur = build_spline(self.waypoints, (dx, dy))
            label = f"{len(self.segments)} cubic segments"

        self.start_time   = shared_start_time
        self.traj_created = True
        self.get_logger().info(
            f"[TRAJ] {self.robot}: {label}  total={dur:.2f}s  "
            f"offset=({dx:.3f},{dy:.3f})  start_at={shared_start_time:.3f}s"
        )

    def _active_segment(self, t):
        for seg in self.segments:
            if seg['t0'] <= t <= seg['tf']:
                return seg
        last = self.segments[-1]
        if t > last['tf'] and t - last['tf'] < 0.5:
            return last
        return None

    def _log_waypoint_crossings(self, t):
        x, y, yaw = self.current_pose
        for i, seg in enumerate(self.segments):
            if i in self.logged_segs or t < seg['tf']:
                continue
            self.logged_segs.add(i)
            xd, _ = eval_cubic(seg['tf'], seg['cx'])
            yd, _ = eval_cubic(seg['tf'], seg['cy'])
            if self.cfg.use_xy_only:
                td_str = "N/A"
                ang_str = "N/A"
            else:
                td, _ = eval_cubic(seg['tf'], seg['ct'])
                td    = _norm(td)
                td_str  = f"{td:.3f}"
                ang_str = f"{_norm(td-yaw):.4f} rad"
            pos_err = math.hypot(xd - x, yd - y)
            self.get_logger().info(
                f"\n========== [{self.robot}] WAYPOINT {i+1}/{len(self.segments)} "
                f"==========\n"
                f"  Time error    : {t-seg['tf']:+.4f} s\n"
                f"  Desired pose  : ({xd:.3f}, {yd:.3f}, {td_str})\n"
                f"  Actual  pose  : ({x:.3f}, {y:.3f}, {yaw:.3f})\n"
                f"  Position error: {pos_err:.4f} m\n"
                f"  Angular error : {ang_str}\n"
                f"==========================================="
            )

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

        # Phase 3 — SEEK after trajectory ends
        last = self.segments[-1]
        if t > last['tf']:
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

        # Phase 4 — tracking
        seg = self._active_segment(t)
        if seg is None:
            return

        if self.cfg.use_xy_only:
            self._control_kanayama_xy(t, seg, now)
        else:
            self._control_spline(t, seg, now)

    # ── Kanayama x,y-only controller (waffle) ────────────────────────────────
    def _control_kanayama_xy(self, t, seg, now):
        """
        Pure x,y tracking using Kanayama controller.
        Theta from CSV is never used.

        Desired heading = direction of the instantaneous velocity vector:
            theta_d = atan2(yd_dot, xd_dot)

        This is always aligned with where the robot is actually going, so:
        - No large heading errors at waypoints
        - No theta spline ringing
        - No cubic theta overshoot
        - The robot simply follows the x,y curve efficiently

        Control law:
            v = v_d * cos(e_theta) + kp_linear * ex
            w = ky * v_d * ey + kp_angular * sin(e_theta)

        where e_theta = atan2(yd_dot, xd_dot) - yaw  (bearing of motion - actual heading)
        """
        xd, xd_dot = eval_cubic(t, seg['cx'])
        yd, yd_dot = eval_cubic(t, seg['cy'])

        # Desired heading = direction of velocity vector
        # When v_d ≈ 0 (start/end of segment), fall back to bearing to target
        v_d = math.hypot(xd_dot, yd_dot)
        if v_d > 0.01:
            theta_d = math.atan2(yd_dot, xd_dot)
        else:
            # Near zero velocity: point toward the waypoint end
            dx_goal = seg['x1'] - self.current_pose[0]
            dy_goal = seg['y1'] - self.current_pose[1]
            theta_d = math.atan2(dy_goal, dx_goal) if math.hypot(dx_goal, dy_goal) > 0.01 \
                      else self.current_pose[2]

        x, y, yaw = self.current_pose
        dx = xd - x
        dy = yd - y

        # Errors in robot body frame
        ex      =  math.cos(yaw)*dx + math.sin(yaw)*dy
        ey      = -math.sin(yaw)*dx + math.cos(yaw)*dy
        e_theta =  _norm(theta_d - yaw)

        pos_error = math.hypot(dx, dy)

        cfg = self.cfg

        # Kanayama control law — x,y only, no theta feedforward
        v_cmd = v_d * math.cos(e_theta) + cfg.kp_linear * ex
        w_cmd = cfg.ky * v_d * ey + cfg.kp_angular * math.sin(e_theta)

        # Velocity floor so robot doesn't stall near segment ends
        if pos_error > GOAL_TOLERANCE and math.cos(e_theta) > 0.5:
            v_cmd = max(v_cmd, cfg.v_max * V_FLOOR_FRAC)

        cmd = Twist()
        cmd.linear.x  = float(np.clip(v_cmd, -cfg.v_max, cfg.v_max))
        cmd.angular.z = float(np.clip(w_cmd, -cfg.w_max, cfg.w_max))
        self.cmd_pub.publish(cmd)

        if now - self.last_debug_time > 0.5:
            self.last_debug_time = now
            self.get_logger().info(
                f"[CTRL] {self.robot} | t={t:.2f}s "
                f"seg=[{seg['t0']:.1f},{seg['tf']:.1f}] | "
                f"des=({xd:.2f},{yd:.2f},thd={math.degrees(theta_d):.1f}°) "
                f"act=({x:.2f},{y:.2f},{math.degrees(yaw):.1f}°) | "
                f"v={cmd.linear.x:.2f} w={cmd.angular.z:.2f}"
            )

    # ── Full spline controller (firebird / burger / tb4_1) ───────────────────
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