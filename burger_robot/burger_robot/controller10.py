#!/usr/bin/env python3
import math
import threading
from dataclasses import dataclass, field
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from path_planner_interface.msg import RobotTrajectoryArray
from navigation_manager_interface.msg import StopRobotsRequest, RobotGoalStatus

# ═════════════════════════════════════════════════════════════════════════════
#  PER-ROBOT CONFIG
#  use_xy_only=True  → Kanayama + cubic x,y spline, theta ignored (waffle)
#  use_xy_only=False → full cubic spline with theta tracking (all others)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class RobotConfig:
    kp_linear:   float = 1.5
    kp_angular:  float = 2.5
    ky:          float = 3.5
    v_max:       float = 0.22
    w_max:       float = 2.84
    use_xy_only: bool  = False

# Add your robots here. These defaults are used if a robot_name is NOT found
# in this dict — so unknown robots will still work with sensible fallback gains.
ROBOT_CONFIG: dict[str, RobotConfig] = {
    "delivery_bot1": RobotConfig(
        kp_linear=1.5, kp_angular=1.3, ky=4.2,
        v_max=0.25, w_max=0.4,
    ),
    "delivery_bot2": RobotConfig(
        kp_linear=1.5, kp_angular=1.3, ky=4.2,
        v_max=0.25, w_max=0.4,
    ),
    "delivery_bot3": RobotConfig(
        kp_linear=1.5, kp_angular=1.5, ky=3.5,
        v_max=0.25, w_max=0.4,
        use_xy_only=True,
    ),
    "cleaning_bot": RobotConfig(
        kp_linear=1.5, kp_angular=1.5, ky=3.5,
        v_max=0.25, w_max=0.4,
        use_xy_only=True,
    ),
    # Add more robots here as needed
}

DEFAULT_CONFIG = RobotConfig()   # fallback for unknown robot names

GOAL_TOLERANCE = 0.08   # metres
V_FLOOR_FRAC   = 0.12   # fraction of v_max used as minimum speed near segment ends

QOS_RELIABLE    = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,    depth=10)
QOS_BEST_EFFORT = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def _norm(a: float) -> float:
    """Wrap angle to [-π, π]."""
    return math.atan2(math.sin(a), math.cos(a))


# ═════════════════════════════════════════════════════════════════════════════
#  CUBIC SPLINE
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


# ═════════════════════════════════════════════════════════════════════════════
#  SEGMENT BUILDERS  (called when a trajectory message arrives)
# ═════════════════════════════════════════════════════════════════════════════
def build_xy_spline(time_list, x_list, y_list):
    """
    Cubic x,y spline — theta column is completely ignored.
    Central-difference knot velocities so the robot does NOT stop at every WP.
    Near-duplicate rows (dt < 0.1 s) are dropped to avoid degenerate segments.
    Returns: (segments, duration)
    """
    MIN_DT = 0.1
    times = list(time_list)
    xs    = list(x_list)
    ys    = list(y_list)

    # Drop near-duplicate rows
    keep_idx = [0]
    for i in range(1, len(times)):
        if times[i] - times[keep_idx[-1]] >= MIN_DT:
            keep_idx.append(i)

    times = [times[i] for i in keep_idx]
    xs    = [xs[i]    for i in keep_idx]
    ys    = [ys[i]    for i in keep_idx]
    N     = len(times)

    # Central-difference knot velocities
    vx = [0.0] * N
    vy = [0.0] * N
    for i in range(1, N - 1):
        dt_p = times[i]   - times[i-1]
        dt_n = times[i+1] - times[i]
        if dt_p < 1e-9 or dt_n < 1e-9:
            continue
        vx[i] = 0.5 * ((xs[i]-xs[i-1])/dt_p + (xs[i+1]-xs[i])/dt_n)
        vy[i] = 0.5 * ((ys[i]-ys[i-1])/dt_p + (ys[i+1]-ys[i])/dt_n)

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
            'x1': xs[i+1],
            'y1': ys[i+1],
        })
    return segments, float(times[-1])


def build_spline(time_list, x_list, y_list, theta_list):
    """
    Cubic x, y, theta spline.
    Returns: (segments, duration)
    """
    MIN_DT = 0.1
    N      = len(time_list)
    times  = list(time_list)
    xs     = list(x_list)
    ys     = list(y_list)
    thetas = list(np.unwrap(theta_list))

    vx = [0.0] * N
    vy = [0.0] * N
    for i in range(1, N - 1):
        dt_p = times[i]   - times[i-1]
        dt_n = times[i+1] - times[i]
        if dt_p < 1e-9 or dt_n < 1e-9:
            continue
        vx[i] = 0.5 * ((xs[i]-xs[i-1])/dt_p + (xs[i+1]-xs[i])/dt_n)
        vy[i] = 0.5 * ((ys[i]-ys[i-1])/dt_p + (ys[i+1]-ys[i])/dt_n)

    segments = []
    for i in range(N - 1):
        t0, tf = float(times[i]), float(times[i+1])
        if tf - t0 < MIN_DT:
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
#  ROBOT NODE
# ═════════════════════════════════════════════════════════════════════════════
class PathFollower(Node):
    """
    One node per robot.
    - Subscribes to /path_planner/paths  (RobotTrajectoryArray)
    - Subscribes to /navigation/stop_robots  (StopRobotsRequest)
    - Publishes  to /controller/goal_status  (RobotGoalStatus)
    - Publishes  to /<robot>/cmd_vel
    """

    def __init__(self, robot_name: str):
        super().__init__(f'{robot_name}_controller')
        self.robot = robot_name
        self.cfg   = ROBOT_CONFIG.get(robot_name, DEFAULT_CONFIG)

        # ── State ──────────────────────────────────────────────────────────
        self.current_pose    = None
        self.segments        = []
        self.start_time      = None
        self.active          = False
        self.stop_requested  = False
        self.last_debug_time = 0.0
        self.logged_segs     = set()
        self._lock           = threading.Lock()

        # ── QoS ────────────────────────────────────────────────────────────
        # tb4 odometry is BEST_EFFORT; everything else RELIABLE
        sub_qos = (QOS_BEST_EFFORT if robot_name == "tb4_1"
                   else QOS_RELIABLE)

        self.get_logger().info(
            f"[INIT] {self.robot}  "
            f"sub_qos={'BE' if sub_qos is QOS_BEST_EFFORT else 'RE'}  "
            f"kp_lin={self.cfg.kp_linear} kp_ang={self.cfg.kp_angular} "
            f"ky={self.cfg.ky}  v_max={self.cfg.v_max} w_max={self.cfg.w_max}  "
            f"mode={'XY-Kanayama' if self.cfg.use_xy_only else 'SPLINE'}"
        )

        # ── Subscribers ────────────────────────────────────────────────────
        self.create_subscription(
            Odometry,
            f'/{self.robot}/odom',
            self.odom_callback,
            sub_qos,
        )
        self.create_subscription(
            RobotTrajectoryArray,
            '/path_planner/paths',
            self.trajectory_array_callback,
            10,
        )
        self.create_subscription(
            StopRobotsRequest,
            '/navigation/stop_robots',
            self.stop_callback,
            10,
        )

        # ── Publishers ─────────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(
            Twist,
            f'/{self.robot}/cmd_vel',
            QOS_RELIABLE,
        )
        self.goal_status_pub = self.create_publisher(
            RobotGoalStatus,
            '/controller/goal_status',
            10,
        )

        # ── Control timer (50 Hz) ──────────────────────────────────────────
        self.create_timer(0.02, self.control_loop)

    # ════════════════════════════════════════════════════════════════════════
    #  CALLBACKS
    # ════════════════════════════════════════════════════════════════════════
    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        q   = msg.pose.pose.orientation
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y),
                         1.0 - 2.0*(q.y*q.y + q.z*q.z))
        self.current_pose = (pos.x, pos.y, yaw)

    def trajectory_array_callback(self, msg: RobotTrajectoryArray):
        """Parse the incoming trajectory for THIS robot and build spline segments."""
        for robot_traj in msg.robot_trajectories:
            if robot_traj.robot_name != self.robot:
                continue

            robot_type = robot_traj.robot_type

            # ── Extract raw waypoint arrays ────────────────────────────────
            if robot_type == "diff-drive":
                traj = robot_traj.diff_drive_trajectories[0]
                time_list  = traj.time
                x_list     = traj.x
                y_list     = traj.y
                theta_list = traj.theta
            elif robot_type == "holonomic":
                traj = robot_traj.holo_trajectories[0]
                time_list  = traj.time
                x_list     = traj.x
                y_list     = traj.y
                theta_list = None
            else:
                self.get_logger().warn(
                    f"[TRAJ] {self.robot}: unknown robot_type '{robot_type}' — skipping"
                )
                continue

            # ── Build segments ─────────────────────────────────────────────
            if self.cfg.use_xy_only or theta_list is None:
                segments, duration = build_xy_spline(time_list, x_list, y_list)
                label = f"{len(segments)} xy-segments"
            else:
                segments, duration = build_spline(
                    time_list, x_list, y_list, theta_list)
                label = f"{len(segments)} cubic segments"

            if not segments:
                self.get_logger().error(
                    f"[TRAJ] {self.robot}: trajectory produced 0 segments — ignoring")
                continue

            # ── Atomic state update ────────────────────────────────────────
            with self._lock:
                self.segments       = segments
                self.logged_segs    = set()
                # t=0 on the spline maps to wall-clock now
                t_offset            = float(time_list[0])   # usually 0.0
                self.start_time     = (
                    self.get_clock().now().nanoseconds * 1e-9 - t_offset
                )
                self.active         = True
                self.stop_requested = False

            self.get_logger().info(
                f"[TRAJ] {self.robot}: {label}  total={duration:.2f}s  "
                f"type={robot_type}  "
                f"mode={'XY-Kanayama' if (self.cfg.use_xy_only or theta_list is None) else 'SPLINE'}"
            )
            break   # handled — stop scanning

    def stop_callback(self, msg: StopRobotsRequest):
        """Stop this robot if its name appears in the stop request."""
        if self.robot in msg.robot_names:
            with self._lock:
                self.stop_requested = True
                self.active         = False
            self._publish_zero()
            self.get_logger().info(f"[STOP] {self.robot}: stop requested")

    # ════════════════════════════════════════════════════════════════════════
    #  CONTROL LOOP  (50 Hz)
    # ════════════════════════════════════════════════════════════════════════
    def control_loop(self):
        # Guard: need pose and an active trajectory
        if self.current_pose is None:
            return
        if self.stop_requested:
            self._publish_zero()
            return
        if not self.active or not self.segments or self.start_time is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        t   = now - self.start_time

        # Wait for trajectory start
        if t < 0:
            if now - self.last_debug_time > 1.0:
                self.last_debug_time = now
                self.get_logger().info(
                    f"[WAIT] {self.robot} → starts in {-t:.2f}s")
            return

        # Log every waypoint crossing once
        self._log_waypoint_crossings(t, now)

        last = self.segments[-1]

        # ── Phase A: trajectory finished — seek to final goal ────────────
        if t > last['tf']:
            xd_g, _ = eval_cubic(last['tf'], last['cx'])
            yd_g, _ = eval_cubic(last['tf'], last['cy'])
            x, y, yaw = self.current_pose
            pos_err   = math.hypot(xd_g - x, yd_g - y)

            if pos_err < GOAL_TOLERANCE:
                self._publish_zero()
                self._publish_goal_reached()
                with self._lock:
                    self.active   = False
                    self.segments = []
                self.get_logger().info(
                    f"[DONE] {self.robot}  final_err={pos_err:.3f}m")
                return

            # Steer toward goal
            bearing_g     = math.atan2(yd_g - y, xd_g - x)
            heading_err_g = _norm(bearing_g - yaw)
            cfg = self.cfg
            cmd = Twist()
            cmd.linear.x  = float(np.clip(
                cfg.kp_linear * pos_err * math.cos(heading_err_g),
                0.0, cfg.v_max))
            cmd.angular.z = float(np.clip(
                cfg.kp_angular * math.sin(heading_err_g),
                -cfg.w_max, cfg.w_max))
            self.cmd_pub.publish(cmd)

            if now - self.last_debug_time > 0.5:
                self.last_debug_time = now
                self.get_logger().info(
                    f"[SEEK] {self.robot}  pos_err={pos_err:.3f}m  "
                    f"des=({xd_g:.2f},{yd_g:.2f})  "
                    f"act=({x:.2f},{y:.2f})")
            return

        # ── Phase B: active tracking ──────────────────────────────────────
        seg = self._active_segment(t)
        if seg is None:
            return

        if self.cfg.use_xy_only or seg.get('type') == 'xy':
            self._control_kanayama_xy(t, seg, now)
        else:
            self._control_spline(t, seg, now)

    # ════════════════════════════════════════════════════════════════════════
    #  CONTROL METHODS
    # ════════════════════════════════════════════════════════════════════════
    def _control_kanayama_xy(self, t, seg, now):
        """
        Pure x,y tracking — Kanayama controller.
        Desired heading = direction of instantaneous velocity vector.
        No theta feedforward.
        """
        xd, xd_dot = eval_cubic(t, seg['cx'])
        yd, yd_dot = eval_cubic(t, seg['cy'])

        v_d = math.hypot(xd_dot, yd_dot)
        if v_d > 0.01:
            theta_d = math.atan2(yd_dot, xd_dot)
        else:
            # Near-zero velocity: point toward waypoint end
            dx_goal = seg['x1'] - self.current_pose[0]
            dy_goal = seg['y1'] - self.current_pose[1]
            theta_d = (math.atan2(dy_goal, dx_goal)
                       if math.hypot(dx_goal, dy_goal) > 0.01
                       else self.current_pose[2])

        x, y, yaw = self.current_pose
        dx = xd - x
        dy = yd - y

        ex      =  math.cos(yaw)*dx + math.sin(yaw)*dy
        ey      = -math.sin(yaw)*dx + math.cos(yaw)*dy
        e_theta =  _norm(theta_d - yaw)
        pos_error = math.hypot(dx, dy)

        cfg   = self.cfg
        v_cmd = v_d * math.cos(e_theta) + cfg.kp_linear * ex
        w_cmd = cfg.ky * v_d * ey + cfg.kp_angular * math.sin(e_theta)

        # Velocity floor — avoid stalling near segment ends
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

    def _control_spline(self, t, seg, now):
        """Full x,y,theta spline tracking with Kanayama lateral correction."""
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

        # Blend between Kanayama (high speed) and bearing correction (low speed)
        v_blend = min(1.0, abs(v_d) / 0.08)

        cfg = self.cfg
        lateral_correction = (
            cfg.ky      * v_d * ey
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

    # ════════════════════════════════════════════════════════════════════════
    #  UTILITIES
    # ════════════════════════════════════════════════════════════════════════
    def _active_segment(self, t):
        for seg in self.segments:
            if seg['t0'] <= t <= seg['tf']:
                return seg
        last = self.segments[-1]
        if t > last['tf'] and t - last['tf'] < 0.5:
            return last
        return None

    def _log_waypoint_crossings(self, t, now):
        x, y, yaw = self.current_pose
        for i, seg in enumerate(self.segments):
            if i in self.logged_segs or t < seg['tf']:
                continue
            self.logged_segs.add(i)
            xd, _ = eval_cubic(seg['tf'], seg['cx'])
            yd, _ = eval_cubic(seg['tf'], seg['cy'])
            pos_err = math.hypot(xd - x, yd - y)

            if seg.get('type') == 'xy' or self.cfg.use_xy_only:
                td_str  = "N/A"
                ang_str = "N/A"
            else:
                td, _ = eval_cubic(seg['tf'], seg['ct'])
                td    = _norm(td)
                td_str  = f"{td:.3f}"
                ang_str = f"{_norm(td - yaw):.4f} rad"

            self.get_logger().info(
                f"\n========== [{self.robot}] WAYPOINT {i+1}/{len(self.segments)} "
                f"==========\n"
                f"  Time error    : {t - seg['tf']:+.4f} s\n"
                f"  Desired pose  : ({xd:.3f}, {yd:.3f}, {td_str})\n"
                f"  Actual  pose  : ({x:.3f}, {y:.3f}, {yaw:.3f})\n"
                f"  Position error: {pos_err:.4f} m\n"
                f"  Angular error : {ang_str}\n"
                f"==========================================="
            )

    def _publish_zero(self):
        self.cmd_pub.publish(Twist())

    def _publish_goal_reached(self):
        msg = RobotGoalStatus()
        msg.robot_name   = self.robot
        msg.goal_reached = True
        self.goal_status_pub.publish(msg)
        self.get_logger().info(f"[GOAL] {self.robot}: goal_reached published")


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
#  Reads ROBOTS list from ROBOT_CONFIG — no launch file required.
#  Simply run:  python3 controller10.py
# ═════════════════════════════════════════════════════════════════════════════
def main():
    rclpy.init()

    robots   = list(ROBOT_CONFIG.keys())
    nodes    = [PathFollower(r) for r in robots]
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