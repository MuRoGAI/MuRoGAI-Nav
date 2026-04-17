#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from path_planner_interface.msg import RobotTrajectoryArray
from navigation_manager_interface.msg import StopRobotsRequest, RobotGoalStatus
from rclpy.qos import QoSProfile, ReliabilityPolicy

# ============================================================
# Cubic Hermite Spline Utilities
# (kept as compute_quintic_coeffs / eval_quintic so nothing
#  else in the project needs renaming)
# ============================================================

def compute_quintic_coeffs(t0, tf, p0, v0, a0, pf, vf, af):
    """
    Cubic Hermite spline. Acceleration args are accepted but ignored.
    Returns [a0, a1, a2, a3, t0] evaluated in local time tau = t - t0.
    """
    T = tf - t0
    if T <= 1e-9:
        return np.array([p0, 0.0, 0.0, 0.0, t0])

    a0_c = p0
    a1_c = v0
    a2_c = (3 * (pf - p0) / T**2) - (2 * v0 + vf) / T
    a3_c = (-2 * (pf - p0) / T**3) + (v0 + vf) / T**2

    return np.array([a0_c, a1_c, a2_c, a3_c, t0])


def eval_quintic(t, c):
    """Evaluate cubic Hermite spline at global time t."""
    a0_c, a1_c, a2_c, a3_c, t0 = c
    tau = max(t - t0, 0.0)
    p = a0_c + a1_c * tau + a2_c * tau**2 + a3_c * tau**3
    v = a1_c + 2 * a2_c * tau + 3 * a3_c * tau**2
    return p, v


# ============================================================
# Controller Node
# ============================================================

class Controller(Node):
    def __init__(self):
        super().__init__('trajectory_controller')

        # ---------------- Parameters ----------------
        self.declare_parameter("kp_linear",   1.5)
        self.declare_parameter("kp_angular",  2.5)
        self.declare_parameter("kd_linear",   0.5)
        self.declare_parameter("kd_angular",  0.3)
        self.declare_parameter("ky",          3.5)
        self.declare_parameter("max_lin_x",   0.22)
        self.declare_parameter("max_lin_y",   0.0)
        self.declare_parameter("max_ang_z",   2.84)
        self.declare_parameter("max_lin_acc", 1.2)
        self.declare_parameter("max_ang_acc", 4.5)
        self.declare_parameter("robot_name",  "tb4_1")

        self.kp_linear   = self.get_parameter("kp_linear").value
        self.kp_angular  = self.get_parameter("kp_angular").value
        self.kd_linear   = self.get_parameter("kd_linear").value
        self.kd_angular  = self.get_parameter("kd_angular").value
        self.ky          = self.get_parameter("ky").value
        self.max_lin_x   = self.get_parameter("max_lin_x").value
        self.max_lin_y   = self.get_parameter("max_lin_y").value
        self.max_ang_z   = self.get_parameter("max_ang_z").value
        self.max_lin_acc = self.get_parameter("max_lin_acc").value
        self.max_ang_acc = self.get_parameter("max_ang_acc").value
        self.robot_name  = self.get_parameter("robot_name").value

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        # ---------------- Internal State ----------------
        self.current_pose    = None
        self.robot_type      = None
        self.segments        = []
        self.start_time      = None
        self.t_offset        = 0.0
        self.active          = False
        self.stop_requested  = False
        self._traj_pending   = False   # True between callback and first control tick
        self.prev_error_lin  = 0.0
        self.prev_error_ang  = 0.0
        self.prev_cmd        = Twist()
        self.prev_time       = self.get_clock().now().nanoseconds * 1e-9
        self.logged_segments = set()

        # ---------------- Subscribers / Publishers ----------------
        if self.robot_name == 'tb4_1':
            self.odom_sub = self.create_subscription(
                Odometry,
                f'/{self.robot_name}/odom',
                self.odom_callback,
                qos_profile
            )
            self.cmd_vel_pub = self.create_publisher(
                Twist,
                f'/{self.robot_name}/cmd_vel',
                qos_profile
            )
        else:
            self.odom_sub = self.create_subscription(
                Odometry,
                f'/{self.robot_name}/odom',
                self.odom_callback,
                10
            )
            self.cmd_vel_pub = self.create_publisher(
                Twist,
                f'/{self.robot_name}/cmd_vel',
                10
            )

        self.target_sub = self.create_subscription(
            RobotTrajectoryArray,
            '/path_planner/paths',
            self.trajectory_array_callback,
            10
        )
        self.stop_sub = self.create_subscription(
            StopRobotsRequest,
            "/navigation/stop_robots",
            self.stop_callback,
            10
        )

        self.goal_status_pub = self.create_publisher(
            RobotGoalStatus,
            "/controller/goal_status",
            10
        )

        self.timer = self.create_timer(0.02, self.control_loop)
        self.get_logger().info(f"[{self.robot_name}] Controller started.")

    # ============================================================
    # Callbacks
    # ============================================================

    def odom_callback(self, msg):
        x   = msg.pose.pose.position.x
        y   = msg.pose.pose.position.y
        q   = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        self.current_pose = (x, y, yaw)

    def trajectory_array_callback(self, msg):
        for robot_traj in msg.robot_trajectories:
            if robot_traj.robot_name != self.robot_name:
                continue

            self.robot_type = robot_traj.robot_type
            self.segments.clear()
            self.logged_segments.clear()

            # ── Extract waypoint lists ───────────────────────────────────
            if self.robot_type == "diff-drive":
                if not robot_traj.diff_drive_trajectories:
                    self.get_logger().warn(
                        f"[{self.robot_name}] diff-drive traj has no data, ignoring."
                    )
                    return
                traj = robot_traj.diff_drive_trajectories[0]
                self.build_segments(
                    list(traj.time), list(traj.x), list(traj.y),
                    theta_list=list(traj.theta)
                )

            elif self.robot_type == "holonomic":
                if not robot_traj.holo_trajectories:
                    self.get_logger().warn(
                        f"[{self.robot_name}] holonomic traj has no data, ignoring."
                    )
                    return
                traj = robot_traj.holo_trajectories[0]
                # NOTE: vx/vy from CSV may be nan — build_segments uses
                # finite differences internally, so we only need x, y, time.
                self.build_segments(
                    list(traj.time), list(traj.x), list(traj.y)
                )

            else:
                self.get_logger().warn(
                    f"[{self.robot_name}] Unknown robot_type '{self.robot_type}', ignoring."
                )
                return

            if not self.segments:
                self.get_logger().warn(
                    f"[{self.robot_name}] No segments built from trajectory, ignoring."
                )
                return

            # ── Anchor wall-clock start time at first control tick ───────
            self.t_offset      = traj.time[0]   # usually 0.0
            self.start_time    = None            # will be set on first control tick
            self._traj_pending = True
            self.active         = True
            self.stop_requested = False

            self.get_logger().info(
                f"[{self.robot_name}] Trajectory received: "
                f"type={self.robot_type}, "
                f"waypoints={len(traj.time)}, "
                f"segments={len(self.segments)}, "
                f"duration={traj.time[-1]:.2f}s"
            )
            break   # matched — no need to keep iterating

    def stop_callback(self, msg):
        if self.robot_name in msg.robot_names:
            self.stop_requested = True
            self.active = False
            self.publish_zero()

    # ============================================================
    # Trajectory Builder
    # Interior knot velocities via central finite differences so
    # the robot does NOT stop at every waypoint.
    # ============================================================

    def build_segments(self, time_list, x_list, y_list, theta_list=None):
        n = len(time_list)
        if n < 2:
            self.get_logger().warn(
                f"[{self.robot_name}] Need at least 2 waypoints, got {n}."
            )
            return

        vx = [0.0] * n
        vy = [0.0] * n
        vt = ([0.0] * n) if (theta_list is not None) else None

        for i in range(1, n - 1):
            dt_prev = time_list[i]     - time_list[i - 1]
            dt_next = time_list[i + 1] - time_list[i]
            if dt_prev < 1e-9 or dt_next < 1e-9:
                continue
            vx[i] = 0.5 * (
                (x_list[i]     - x_list[i - 1]) / dt_prev +
                (x_list[i + 1] - x_list[i])     / dt_next
            )
            vy[i] = 0.5 * (
                (y_list[i]     - y_list[i - 1]) / dt_prev +
                (y_list[i + 1] - y_list[i])     / dt_next
            )
            if theta_list is not None:
                dth_prev = self.normalize_angle(
                    theta_list[i]     - theta_list[i - 1]) / dt_prev
                dth_next = self.normalize_angle(
                    theta_list[i + 1] - theta_list[i])     / dt_next
                vt[i] = 0.5 * (dth_prev + dth_next)

        for i in range(n - 1):
            t0 = time_list[i]
            tf = time_list[i + 1]

            if abs(tf - t0) < 1e-9:
                self.get_logger().warn(
                    f"[{self.robot_name}] Skipping duplicate time at index {i} (t={t0})"
                )
                continue

            seg = {
                "t0": t0,
                "tf": tf,
                "cx": compute_quintic_coeffs(
                    t0, tf,
                    x_list[i],     vx[i],     0.0,
                    x_list[i + 1], vx[i + 1], 0.0
                ),
                "cy": compute_quintic_coeffs(
                    t0, tf,
                    y_list[i],     vy[i],     0.0,
                    y_list[i + 1], vy[i + 1], 0.0
                ),
            }
            if theta_list is not None:
                seg["ct"] = compute_quintic_coeffs(
                    t0, tf,
                    theta_list[i],     vt[i],     0.0,
                    theta_list[i + 1], vt[i + 1], 0.0
                )
            self.segments.append(seg)

    # ============================================================
    # Control Loop
    # ============================================================

    def control_loop(self):
        if not self.active:
            return
        if self.stop_requested:
            self.publish_zero()
            return
        if not self.segments:
            self.get_logger().warn(
                f"[{self.robot_name}] STUCK: active=True but segments list is empty.",
                throttle_duration_sec=2.0
            )
            return
        if self.current_pose is None:
            self.get_logger().warn(
                f"[{self.robot_name}] STUCK: waiting for odometry on "
                f"/{self.robot_name}/odom ...",
                throttle_duration_sec=2.0
            )
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        # Re-anchor start_time on the very first control tick after a new
        # trajectory arrives. Done here (not in callback) so odom-arrival
        # delay is absorbed and t=0 truly means "robot is ready to move".
        if self._traj_pending:
            self.start_time    = now - self.t_offset
            self._traj_pending = False
            self.get_logger().info(
                f"[{self.robot_name}] >>> CONTROL STARTED: "
                f"start_time={self.start_time:.3f}s  t_offset={self.t_offset:.3f}s  "
                f"pose=({self.current_pose[0]:.3f}, {self.current_pose[1]:.3f}, "
                f"{self.current_pose[2]:.3f})"
            )

        if self.start_time is None:
            self.get_logger().warn(
                f"[{self.robot_name}] STUCK: start_time is None (traj_pending={self._traj_pending}).",
                throttle_duration_sec=2.0
            )
            return

        t = now - self.start_time

        # ── Waypoint crossing log ────────────────────────────────────────
        for i, seg in enumerate(self.segments):
            if i not in self.logged_segments and t >= seg["tf"]:
                self.logged_segments.add(i)
                xd_end, _ = eval_quintic(seg["tf"], seg["cx"])
                yd_end, _ = eval_quintic(seg["tf"], seg["cy"])
                td_end = 0.0
                if "ct" in seg:
                    td_end, _ = eval_quintic(seg["tf"], seg["ct"])

                x, y, yaw = self.current_pose
                self.get_logger().info(
                    f"\n========== WAYPOINT {i} ==========\n"
                    f"Time           : desired={seg['tf']:.3f}  actual={t:.3f}  "
                    f"err={t - seg['tf']:.4f}s\n"
                    f"Desired Pose   : ({xd_end:.3f}, {yd_end:.3f}, {td_end:.3f})\n"
                    f"Actual Pose    : ({x:.3f}, {y:.3f}, {yaw:.3f})\n"
                    f"Position Error : {math.hypot(xd_end - x, yd_end - y):.4f} m\n"
                    f"Angular Error  : {self.normalize_angle(td_end - yaw):.4f} rad\n"
                    f"========================================="
                )

        # ── End of trajectory ────────────────────────────────────────────
        if t > self.segments[-1]["tf"]:
            last_seg    = self.segments[-1]
            final_theta = 0.0
            if "ct" in last_seg:
                final_theta, _ = eval_quintic(last_seg["tf"], last_seg["ct"])

            x, y, yaw = self.current_pose
            yaw_error = self.normalize_angle(final_theta - yaw)
            cmd = Twist()

            self.get_logger().info(
                f"[{self.robot_name}] Trajectory ended (t={t:.2f}s > tf={self.segments[-1]['tf']:.2f}s). "
                f"yaw_error={yaw_error:.4f} rad  final_theta={final_theta:.3f}",
                throttle_duration_sec=1.0
            )

            if abs(yaw_error) > 0.02:          # ~1° tolerance
                cmd.angular.z = float(np.clip(
                    self.kp_angular * yaw_error,
                    -self.max_ang_z, self.max_ang_z
                ))
                self.cmd_vel_pub.publish(cmd)
                return

            self.publish_zero()
            self.publish_goal_reached()
            self.active = False
            return

        # ── Find active segment ──────────────────────────────────────────
        seg = None
        for s in self.segments:
            if s["t0"] <= t <= s["tf"]:
                seg = s
                break
        if seg is None:
            self.get_logger().warn(
                f"[{self.robot_name}] STUCK: t={t:.3f}s falls in no segment "
                f"(first t0={self.segments[0]['t0']:.3f}, last tf={self.segments[-1]['tf']:.3f}).",
                throttle_duration_sec=2.0
            )
            return

        cmd = Twist()

        # ════════════════════════════════════════════════════════════════
        # DIFF-DRIVE  (Kanayama-style controller)
        # ════════════════════════════════════════════════════════════════
        if self.robot_type == "diff-drive":
            xd, xd_dot = eval_quintic(t, seg["cx"])
            yd, yd_dot = eval_quintic(t, seg["cy"])
            x, y, yaw  = self.current_pose

            # Reference heading: path tangent (or hold current if stopped)
            if math.hypot(xd_dot, yd_dot) > 1e-6:
                desired_theta     = math.atan2(yd_dot, xd_dot)
                desired_theta_dot = 0.0
            else:
                desired_theta     = yaw
                desired_theta_dot = 0.0

            # Near goal: enforce final waypoint theta
            last_seg = self.segments[-1]
            if seg is last_seg and "ct" in last_seg:
                xf, _ = eval_quintic(last_seg["tf"], last_seg["cx"])
                yf, _ = eval_quintic(last_seg["tf"], last_seg["cy"])
                if math.hypot(xf - x, yf - y) < 0.05:
                    final_theta, _    = eval_quintic(last_seg["tf"], last_seg["ct"])
                    desired_theta     = final_theta
                    desired_theta_dot = 0.0

            dx = xd - x
            dy = yd - y
            ex =  math.cos(yaw) * dx + math.sin(yaw) * dy   # forward error
            ey = -math.sin(yaw) * dx + math.cos(yaw) * dy   # lateral error
            e_theta = self.normalize_angle(desired_theta - yaw)

            v_d = xd_dot * math.cos(desired_theta) + yd_dot * math.sin(desired_theta)
            w_d = desired_theta_dot

            pos_error          = math.hypot(dx, dy)
            bearing_to_desired = math.atan2(dy, dx)
            heading_error      = self.normalize_angle(bearing_to_desired - yaw)

            lateral_correction = (
                self.ky          * v_d * ey
                + self.kp_angular * pos_error * math.sin(heading_error)
            )

            cmd.linear.x  = float(np.clip(
                v_d * math.cos(e_theta) + self.kp_linear * ex,
                -self.max_lin_x, self.max_lin_x
            ))
            cmd.angular.z = float(np.clip(
                w_d + lateral_correction,
                -self.max_ang_z, self.max_ang_z
            ))

            self.get_logger().info(
                f"[{self.robot_name}] DIFF t={t:.2f}s "
                f"seg=[{seg['t0']:.1f},{seg['tf']:.1f}] "
                f"pose=({x:.2f},{y:.2f},{yaw:.2f}) "
                f"desired=({xd:.2f},{yd:.2f},{desired_theta:.2f}) "
                f"ex={ex:.3f} ey={ey:.3f} eth={e_theta:.3f} "
                f"vd={v_d:.3f} cmd_v={cmd.linear.x:.3f} cmd_w={cmd.angular.z:.3f}",
                throttle_duration_sec=0.5
            )

        # ════════════════════════════════════════════════════════════════
        # HOLONOMIC  (independent x/y P-control, no angular needed)
        # ════════════════════════════════════════════════════════════════
        elif self.robot_type == "holonomic":
            xd, xd_dot = eval_quintic(t, seg["cx"])
            yd, yd_dot = eval_quintic(t, seg["cy"])
            x, y, yaw  = self.current_pose

            ex = xd - x
            ey = yd - y

            # Feedforward + proportional correction in world frame
            vx_cmd = xd_dot + self.kp_linear * ex
            vy_cmd = yd_dot + self.kp_linear * ey

            cmd.linear.x = float(np.clip(vx_cmd, -self.max_lin_x, self.max_lin_x))
            cmd.linear.y = float(np.clip(vy_cmd, -self.max_lin_y, self.max_lin_y)) \
                           if self.max_lin_y > 1e-6 else 0.0
            cmd.angular.z = 0.0    # holonomic robot has no heading requirement

            self.get_logger().info(
                f"[{self.robot_name}] HOLO t={t:.2f}s "
                f"seg=[{seg['t0']:.1f},{seg['tf']:.1f}] "
                f"pose=({x:.2f},{y:.2f}) "
                f"desired=({xd:.2f},{yd:.2f}) "
                f"err=({ex:.3f},{ey:.3f}) "
                f"ff=({xd_dot:.3f},{yd_dot:.3f}) "
                f"cmd=({cmd.linear.x:.3f},{cmd.linear.y:.3f})",
                throttle_duration_sec=0.5
            )

        else:
            self.get_logger().warn(
                f"[{self.robot_name}] STUCK: unknown robot_type='{self.robot_type}' in control loop.",
                throttle_duration_sec=2.0
            )
            return

        self.cmd_vel_pub.publish(cmd)

    # ============================================================
    # Acceleration Limiter (available but disabled by default)
    # ============================================================

    def apply_acceleration_limits(self, cmd):
        current_time = self.get_clock().now().nanoseconds * 1e-9
        dt = current_time - self.prev_time
        if dt <= 0.0:
            return cmd

        max_delta_v = self.max_lin_acc * dt
        max_delta_w = self.max_ang_acc * dt

        delta_v = np.clip(
            cmd.linear.x  - self.prev_cmd.linear.x,  -max_delta_v, max_delta_v)
        delta_w = np.clip(
            cmd.angular.z - self.prev_cmd.angular.z, -max_delta_w, max_delta_w)

        cmd.linear.x  = self.prev_cmd.linear.x  + delta_v
        cmd.angular.z = self.prev_cmd.angular.z + delta_w

        self.prev_cmd  = cmd
        self.prev_time = current_time
        return cmd

    # ============================================================
    # Utilities
    # ============================================================

    def publish_zero(self):
        self.cmd_vel_pub.publish(Twist())

    def publish_goal_reached(self):
        status_msg = RobotGoalStatus()
        status_msg.robot_name   = self.robot_name
        status_msg.goal_reached = True
        self.goal_status_pub.publish(status_msg)
        self.get_logger().info(f"[{self.robot_name}] Goal reached!")

    @staticmethod
    def normalize_angle(a):
        return math.atan2(math.sin(a), math.cos(a))


def main(args=None):
    rclpy.init(args=args)
    node = Controller()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()