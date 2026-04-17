#!/usr/bin/env python3

import math
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


# ── Config ────────────────────────────────────────────────────────────────────
ROBOTS = [
    "firebird",
    "burger3",
    "waffle",
    "burger1",
    "burger2",
    # "tb4_1",
]

SYNC_DELAY    = 1.0   # seconds after ALL robots have their first odom → start together
TRAJ_DURATION = 5.0   # seconds


# ── Quintic polynomial ────────────────────────────────────────────────────────

def compute_quintic_coeffs(t0, tf, p0, v0, a0, pf, vf, af):
    """
    True 5th-order polynomial satisfying:
      p(t0)=p0, p'(t0)=v0, p''(t0)=a0
      p(tf)=pf, p'(tf)=vf, p''(tf)=af
    """
    T = tf - t0
    if T <= 1e-9:
        return np.array([p0, 0.0, 0.0, 0.0, 0.0, 0.0, t0])

    h    = pf - p0
    a0_c = p0
    a1_c = v0
    a2_c = a0 / 2.0
    a3_c = ( 20*h - (8*vf + 12*v0)*T - (3*af - a0)*T**2) / (2 * T**3)
    a4_c = (-30*h + (14*vf + 16*v0)*T + (3*af - 2*a0)*T**2) / (2 * T**4)
    a5_c = ( 12*h -  6*(vf + v0)*T   - (af - a0)*T**2)      / (2 * T**5)

    return np.array([a0_c, a1_c, a2_c, a3_c, a4_c, a5_c, t0])


def eval_quintic(t, c):
    a0_c, a1_c, a2_c, a3_c, a4_c, a5_c, t0 = c
    tau = max(0.0, t - t0)
    p = (a0_c + a1_c*tau   + a2_c*tau**2 + a3_c*tau**3
              + a4_c*tau**4 + a5_c*tau**5)
    v = (a1_c + 2*a2_c*tau  + 3*a3_c*tau**2
              + 4*a4_c*tau**3 + 5*a5_c*tau**4)
    return p, v


# ── Coordinator ───────────────────────────────────────────────────────────────

class Coordinator:
    """
    Thread-safe synchronized start gate.

    Each robot calls report_ready() when its first odom arrives.
    Once ALL robots have reported, every call returns the shared
    absolute wall-clock start time.  Until then it returns None.

    Timeline:
        t=0          nodes start
        t~0.1        all robots receive first odom  →  start_time = now + SYNC_DELAY
        t=SYNC_DELAY all trajectories start simultaneously
    """

    def __init__(self, robot_names, sync_delay=SYNC_DELAY):
        self._expected   = set(robot_names)
        self._ready      = set()
        self._start_time = None
        self._lock       = threading.Lock()
        self._delay      = sync_delay

    def report_ready(self, robot_name: str, now_sec: float):
        """
        Returns shared start_time once every robot has called in, else None.
        Safe to call from multiple threads simultaneously.
        """
        with self._lock:
            self._ready.add(robot_name)
            if self._ready >= self._expected and self._start_time is None:
                self._start_time = now_sec + self._delay
            return self._start_time


# ── Robot node ────────────────────────────────────────────────────────────────

class OneMeterImmediate(Node):

    def __init__(self, robot_name: str, coordinator: Coordinator):
        # Unique node name per robot — no DDS conflicts
        super().__init__(f'{robot_name}_controller')

        self.robot       = robot_name
        self.coord       = coordinator

        self.current_pose    = None
        self.segment         = None
        self.traj_created    = False
        self.start_time      = None   # absolute wall-clock seconds (shared)
        self.last_debug_time = 0.0

        self.get_logger().info(f"[INIT] {self.robot}")

        self.create_subscription(
            Odometry,
            f'/{self.robot}/odom_world',
            self.odom_callback,
            10
        )
        self.cmd_pub = self.create_publisher(
            Twist,
            f'/{self.robot}/cmd_vel',
            10
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
        # DEBUG — not INFO — keeps log quiet at 30-100 Hz
        self.get_logger().debug(
            f"[ODOM] {self.robot} x={pos.x:.3f} y={pos.y:.3f} yaw={yaw:.3f}"
        )

    # ── Trajectory ───────────────────────────────────────────────────────────

    def create_trajectory(self, shared_start_time: float):
        x0, y0, yaw = self.current_pose

        xf = x0 + math.cos(yaw)
        yf = y0 + math.sin(yaw)

        t0 = 0.0
        tf = TRAJ_DURATION

        self.segment = {
            "t0": t0,
            "tf": tf,
            "cx": compute_quintic_coeffs(t0, tf, x0, 0.0, 0.0, xf, 0.0, 0.0),
            "cy": compute_quintic_coeffs(t0, tf, y0, 0.0, 0.0, yf, 0.0, 0.0),
        }

        self.start_time   = shared_start_time
        self.traj_created = True

        self.get_logger().info(
            f"[TRAJ] {self.robot}: ({x0:.2f},{y0:.2f}) → ({xf:.2f},{yf:.2f})  "
            f"start_at={shared_start_time:.3f}s"
        )

    # ── Control loop ─────────────────────────────────────────────────────────

    def control_loop(self):

        if self.current_pose is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        # ── Phase 1: wait until ALL robots have odom, then lock trajectory ───
        if not self.traj_created:
            shared_start = self.coord.report_ready(self.robot, now)
            if shared_start is not None:
                # All robots ready → compute trajectory for this robot
                self.create_trajectory(shared_start)
            return

        if self.segment is None:
            return

        t = now - self.start_time

        # ── Phase 2: countdown (holding position until shared start time) ────
        if t < 0:
            if now - self.last_debug_time > 1.0:
                self.last_debug_time = now
                self.get_logger().info(
                    f"[WAIT] {self.robot} → starts in {-t:.2f}s"
                )
            return

        # ── Phase 3: done ─────────────────────────────────────────────────────
        if t > self.segment["tf"]:
            self.cmd_pub.publish(Twist())
            self.get_logger().info(f"[DONE] {self.robot}")
            self.segment = None
            return

        # ── Desired state ─────────────────────────────────────────────────────
        xd,  xd_dot = eval_quintic(t, self.segment["cx"])
        yd,  yd_dot = eval_quintic(t, self.segment["cy"])

        x, y, yaw = self.current_pose

        # ── Tracking errors in robot frame ────────────────────────────────────
        dx = xd - x
        dy = yd - y
        desired_theta = math.atan2(yd_dot, xd_dot)

        ex      =  math.cos(yaw)*dx + math.sin(yaw)*dy
        ey      = -math.sin(yaw)*dx + math.cos(yaw)*dy
        e_theta =  math.atan2(
            math.sin(desired_theta - yaw),
            math.cos(desired_theta - yaw)
        )

        # ── Control law ───────────────────────────────────────────────────────
        v = xd_dot * math.cos(e_theta) + 1.5 * ex
        w = 2.5 * e_theta + 3.5 * v * ey

        cmd = Twist()
        cmd.linear.x  = float(np.clip(v, -0.22,  0.22))
        cmd.angular.z = float(np.clip(w, -2.84,  2.84))
        self.cmd_pub.publish(cmd)

        # ── Throttled debug log ───────────────────────────────────────────────
        if now - self.last_debug_time > 0.5:
            self.last_debug_time = now
            self.get_logger().info(
                f"[CTRL] {self.robot} | t={t:.2f}s | "
                f"des=({xd:.2f},{yd:.2f}) act=({x:.2f},{y:.2f}) | "
                f"v={cmd.linear.x:.2f} w={cmd.angular.z:.2f}"
            )


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    rclpy.init()

    coordinator = Coordinator(ROBOTS)

    # All robots in ONE process → shared Coordinator, no DDS startup race
    nodes = [OneMeterImmediate(r, coordinator) for r in ROBOTS]

    # One thread per robot → truly parallel callback execution
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