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
# Quintic Polynomial Utilities
# ============================================================
# def compute_quintic_coeffs(t0, tf, p0, v0, a0, pf, vf, af):
#     A = np.array([
#         [1, t0,   t0**2,    t0**3,     t0**4,      t0**5],
#         [0,  1,  2*t0,    3*t0**2,   4*t0**3,    5*t0**4],
#         [0,  0,   2,      6*t0,     12*t0**2,   20*t0**3],
#         [1, tf,   tf**2,    tf**3,     tf**4,      tf**5],
#         [0,  1,  2*tf,    3*tf**2,   4*tf**3,    5*tf**4],
#         [0,  0,   2,      6*tf,     12*tf**2,   20*tf**3],
#     ], dtype=float)
#     b = np.array([p0, v0, a0, pf, vf, af], dtype=float)
#     return np.linalg.solve(A, b)
def compute_quintic_coeffs(t0, tf, p0, v0, a0, pf, vf, af):
    """
    Now computes CUBIC Hermite spline coefficients.
    Signature kept identical so rest of code remains unchanged.
    Acceleration inputs are ignored.
    """

    T = tf - t0
    if T <= 1e-9:
        return np.array([p0, 0.0, 0.0, 0.0])

    # Shift to local time tau = t - t0
    # Cubic polynomial:
    # p(tau) = a0 + a1*tau + a2*tau^2 + a3*tau^3

    a0_c = p0
    a1_c = v0
    a2_c = (3*(pf - p0)/(T**2)) - (2*v0 + vf)/T
    a3_c = (-2*(pf - p0)/(T**3)) + (v0 + vf)/(T**2)

    # We store t0 so evaluation works in global time
    return np.array([a0_c, a1_c, a2_c, a3_c, t0])
# def eval_quintic(t, c):
#     p = c[0] + c[1]*t + c[2]*t**2 + c[3]*t**3 + c[4]*t**4 + c[5]*t**5
#     v = c[1] + 2*c[2]*t + 3*c[3]*t**2 + 4*c[4]*t**3 + 5*c[5]*t**4
#     return p, v

def eval_quintic(t, c):
    """
    Evaluates cubic Hermite spline.
    Function name unchanged to avoid modifying controller.
    """

    a0_c, a1_c, a2_c, a3_c, t0 = c

    tau = t - t0
    if tau < 0:
        tau = 0.0

    p = a0_c + a1_c*tau + a2_c*tau**2 + a3_c*tau**3
    v = a1_c + 2*a2_c*tau + 3*a3_c*tau**2

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
        self.declare_parameter("ky",          3.5)   # lateral Kanayama gain (separate from kp_linear)
        self.declare_parameter("max_lin_x",   0.22)
        self.declare_parameter("max_lin_y",   0.0)
        self.declare_parameter("max_ang_z",   2.84)
        self.declare_parameter("max_lin_acc", 1.2)
        self.declare_parameter("max_ang_acc", 4.5)
        self.declare_parameter("robot_name",  "tb4_1")

        # Read parameters
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
        self.prev_error_lin  = 0.0
        self.prev_error_ang  = 0.0
        self.prev_cmd        = Twist()
        self.prev_time       = self.get_clock().now().nanoseconds * 1e-9
        self.logged_segments = set()

        # ---------------- Subscribers ----------------
        if self.robot_name == 'tb4_1':
            self.odom_sub = self.create_subscription(
                Odometry,
                f'/{self.robot_name}/odom_world',
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
                f'/{self.robot_name}/odom_world',
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

        # ---------------- Publishers ----------------
        self.goal_status_pub = self.create_publisher(
            RobotGoalStatus,
            "/controller/goal_status",
            10
        )

        self.timer = self.create_timer(0.02, self.control_loop)

    # ============================================================
    # Callbacks
    # ============================================================
    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        )
        self.current_pose = (x, y, yaw)

    def trajectory_array_callback(self, msg):
        for robot_traj in msg.robot_trajectories:
            if robot_traj.robot_name != self.robot_name:
                continue

            self.robot_type = robot_traj.robot_type
            self.segments.clear()
            self.logged_segments.clear()

            if self.robot_type == "diff-drive":
                traj = robot_traj.diff_drive_trajectories[0]
                self.build_segments(traj.time, traj.x, traj.y, traj.theta)
            elif self.robot_type == "holonomic":
                traj = robot_traj.holo_trajectories[0]
                self.build_segments(traj.time, traj.x, traj.y)

            # Anchor start time to the trajectory's own time base so that
            # t = (wall_now - start_time) maps correctly onto the spline domain.
            self.t_offset   = traj.time[0]   # usually 0.0; guards against non-zero bases
            self.start_time = (
                self.get_clock().now().nanoseconds * 1e-9 - self.t_offset
            )
            self.active         = True
            self.stop_requested = False
            self.get_logger().info(f"[{self.robot_name}] Trajectory received.")

    def stop_callback(self, msg):
        if self.robot_name in msg.robot_names:
            self.stop_requested = True
            self.active = False
            self.publish_zero()

    # ============================================================
    # Trajectory Builder
    # Interior knot velocities estimated via central finite differences
    # so the robot does NOT decelerate to zero at every waypoint.
    # ============================================================
    def build_segments(self, time_list, x_list, y_list, theta_list=None):
        n = len(time_list)

        # Estimate velocity at each knot via weighted central differences.
        # First and last knots keep v=0 (start from rest, stop at goal).
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

        # Build one quintic segment per consecutive knot pair
        for i in range(n - 1):
            t0 = time_list[i]
            tf = time_list[i + 1]

            if abs(tf - t0) < 1e-9:
                self.get_logger().warn(
                    f"Skipping duplicate time segment at index {i} (t={t0})"
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
        if not self.active or self.current_pose is None:
            return
        if self.stop_requested:
            self.publish_zero()
            return

        t = self.get_clock().now().nanoseconds * 1e-9 - self.start_time

        # -------- Waypoint Cross Detection --------
        for i, seg in enumerate(self.segments):
            if i not in self.logged_segments and t >= seg["tf"]:
                self.logged_segments.add(i)
                desired_time = seg["tf"]
                actual_time  = t
                time_error   = actual_time - desired_time

                xd_end, _ = eval_quintic(seg["tf"], seg["cx"])
                yd_end, _ = eval_quintic(seg["tf"], seg["cy"])
                td_end = 0.0
                if "ct" in seg:
                    td_end, _ = eval_quintic(seg["tf"], seg["ct"])

                x, y, yaw = self.current_pose
                pos_error = math.hypot(xd_end - x, yd_end - y)
                ang_error = self.normalize_angle(td_end - yaw)

                self.get_logger().info(
                    f"\n========== WAYPOINT {i} ==========\n"
                    f"Desired Time   : {desired_time:.3f}\n"
                    f"Actual Time    : {actual_time:.3f}\n"
                    f"Time Error     : {time_error:.4f} s\n"
                    f"Desired Pose   : ({xd_end:.3f}, {yd_end:.3f}, {td_end:.3f})\n"
                    f"Actual Pose    : ({x:.3f}, {y:.3f}, {yaw:.3f})\n"
                    f"Position Error : {pos_error:.4f} m\n"
                    f"Angular Error  : {ang_error:.4f} rad\n"
                    f"========================================="
                )

        # -------- Stop at End --------
        if t > self.segments[-1]["tf"]:

            last_seg = self.segments[-1]

            # Final desired yaw from waypoint
            final_theta = 0.0
            if "ct" in last_seg:
                final_theta, _ = eval_quintic(last_seg["tf"], last_seg["ct"])

            x, y, yaw = self.current_pose
            yaw_error = self.normalize_angle(final_theta - yaw)

            cmd = Twist()

            # Keep rotating until yaw matches
            if abs(yaw_error) > 0.02:   # ~1 degree tolerance
                cmd.linear.x = 0.0
                cmd.angular.z = float(np.clip(
                    self.kp_angular * yaw_error,
                    -self.max_ang_z,
                    self.max_ang_z
                ))
                self.cmd_vel_pub.publish(cmd)
                return

            # Once yaw matched, stop and finish
            self.publish_zero()
            self.publish_goal_reached()
            self.active = False
            return

        # -------- Find Active Segment --------
        seg = None
        for s in self.segments:
            if s["t0"] <= t <= s["tf"]:
                seg = s
                break
        if seg is None:
            return

        cmd = Twist()

        if self.robot_type == "diff-drive":

            # Desired position and velocities from spline
            xd, xd_dot = eval_quintic(t, seg["cx"])
            yd, yd_dot = eval_quintic(t, seg["cy"])

            x, y, yaw = self.current_pose

            # ------------------------------------------------------
            # 1️⃣ Use path tangent heading for all intermediate motion
            # ------------------------------------------------------
            if math.hypot(xd_dot, yd_dot) > 1e-6:
                desired_theta = math.atan2(yd_dot, xd_dot)
                desired_theta_dot = 0.0
            else:
                desired_theta = yaw
                desired_theta_dot = 0.0

            # ------------------------------------------------------
            # 2️⃣ If at final segment AND near final position,
            #     then enforce final waypoint theta
            # ------------------------------------------------------
            last_seg = self.segments[-1]
            if seg == last_seg:
                xf, _ = eval_quintic(last_seg["tf"], last_seg["cx"])
                yf, _ = eval_quintic(last_seg["tf"], last_seg["cy"])

                pos_error_to_goal = math.hypot(xf - x, yf - y)

                if pos_error_to_goal < 0.05:   # 5 cm tolerance
                    if "ct" in last_seg:
                        final_theta, _ = eval_quintic(last_seg["tf"], last_seg["ct"])
                        desired_theta = final_theta
                        desired_theta_dot = 0.0

            # Current pose
            x, y, yaw = self.current_pose

            # World-frame position error
            dx = xd - x
            dy = yd - y

            # Transform position error into robot body frame
            ex      =  math.cos(yaw) * dx + math.sin(yaw) * dy
            ey      = -math.sin(yaw) * dx + math.cos(yaw) * dy
            e_theta =  self.normalize_angle(desired_theta - yaw)

            # Feedforward reference velocities
            v_d = xd_dot * math.cos(desired_theta) + yd_dot * math.sin(desired_theta)
            w_d = desired_theta_dot

            # Gains
            kx     = self.kp_linear
            ky     = self.ky
            ktheta = self.kp_angular

            # ----------------------------------------------------------
            # Augmented angular correction (two complementary terms):
            #
            #   1. Kanayama lateral term:  ky * v_d * ey
            #      Works well when the robot is moving (v_d large) and
            #      is close to the path (ey small).
            #
            #   2. Bearing correction: ktheta * pos_error * sin(heading_error)
            #      Steers directly toward the desired point independent of
            #      v_d. Recovers large lateral deviations that accumulate
            #      during slow / knot-point phases where v_d ≈ 0.
            #      sin(heading_error) keeps it smooth and bounded (zero
            #      when already aimed at the target, saturates at ±1).
            # ----------------------------------------------------------
            pos_error          = math.hypot(dx, dy)
            bearing_to_desired = math.atan2(dy, dx)
            heading_error      = self.normalize_angle(bearing_to_desired - yaw)

            lateral_correction = (
                ky * v_d * ey
                + ktheta * pos_error * math.sin(heading_error)
            )

            cmd.linear.x  = v_d * math.cos(e_theta) + kx * ex
            cmd.angular.z = w_d + lateral_correction

            # Clip to hardware limits
            cmd.linear.x  = float(np.clip(cmd.linear.x,  -self.max_lin_x, self.max_lin_x))
            cmd.angular.z = float(np.clip(cmd.angular.z, -self.max_ang_z,  self.max_ang_z))

        elif self.robot_type == "holonomic":
            xd, _ = eval_quintic(t, seg["cx"])
            yd, _ = eval_quintic(t, seg["cy"])
            x, y, yaw = self.current_pose
            ex = xd - x
            ey = yd - y

            cmd.linear.x  = float(np.clip(self.kp_linear * ex, -self.max_lin_x, self.max_lin_x))
            # Guard against max_lin_y=0 zeroing out lateral motion on holonomic robots
            cmd.linear.y  = (
                float(np.clip(self.kp_linear * ey, -self.max_lin_y, self.max_lin_y))
                if self.max_lin_y > 1e-6 else 0.0
            )
            cmd.angular.z = float(np.clip(cmd.angular.z, -self.max_ang_z, self.max_ang_z))

        # cmd = self.apply_acceleration_limits(cmd)
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