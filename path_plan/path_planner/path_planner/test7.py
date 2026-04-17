#!/usr/bin/env python3

import math
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
#  PER-ROBOT CONFIG  —  edit gains here
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class RobotConfig:
    # QoS
    best_effort: bool = False

    # Trajectory
    traj_duration: float = 5.0     # seconds
    distance: float      = 1.0     # metres to travel forward

    # Control gains
    k_ex:    float = 1.5   # forward error gain         (ex  → v)
    k_eth:   float = 2.5   # heading error gain          (e_θ → ω)
    k_ey:    float = 3.5   # lateral error gain          (ey  → ω)

    # Velocity limits
    v_max:   float = 0.22  # m/s
    w_max:   float = 2.84  # rad/s


ROBOT_CONFIG: dict[str, RobotConfig] = {
    # ── burger robots ────────────────────────────────────────────────────────
    "burger1": RobotConfig(
        k_ex=1.5, k_eth=2.5, k_ey=3.5,
        v_max=0.22, w_max=2.84,
    ),
    "burger2": RobotConfig(
        k_ex=1.5, k_eth=2.5, k_ey=3.5,
        v_max=0.22, w_max=2.84,
    ),
    "burger3": RobotConfig(
        k_ex=1.5, k_eth=2.5, k_ey=3.5,
        v_max=0.22, w_max=2.84,
    ),
    # ── waffle ───────────────────────────────────────────────────────────────
    "waffle": RobotConfig(
        k_ex=1.5, k_eth=2.5, k_ey=3.5,
        v_max=0.22, w_max=2.84,
    ),
    # ── firebird ─────────────────────────────────────────────────────────────
    "firebird": RobotConfig(
        k_ex=1.5, k_eth=2.5, k_ey=3.5,
        v_max=0.22, w_max=2.84,
    ),
    # ── tb4 (BEST_EFFORT QoS) ────────────────────────────────────────────────
    "tb4_1": RobotConfig(
        best_effort=True,
        k_ex=1.5, k_eth=2.5, k_ey=3.5,
        v_max=0.26, w_max=1.90,
    ),
}

# Derive ROBOTS list from the config dict — no need to maintain two lists
ROBOTS = list(ROBOT_CONFIG.keys())

# ─────────────────────────────────────────────────────────────────────────────

SYNC_DELAY = 1.0   # seconds after ALL robots have odom → all start together

QOS_RELIABLE    = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,    depth=10)
QOS_BEST_EFFORT = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)


# ── Quintic polynomial ────────────────────────────────────────────────────────

def compute_quintic_coeffs(t0, tf, p0, v0, a0, pf, vf, af):
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


# ── Robot node ────────────────────────────────────────────────────────────────

class OneMeterImmediate(Node):

    def __init__(self, robot_name: str, coordinator: Coordinator):
        super().__init__(f'{robot_name}_controller')

        self.robot = robot_name
        self.coord = coordinator
        self.cfg   = ROBOT_CONFIG[robot_name]   # ← typed config for this robot

        self.current_pose    = None
        self.segment         = None
        self.traj_created    = False
        self.start_time      = None
        self.last_debug_time = 0.0

        qos = QOS_BEST_EFFORT if self.cfg.best_effort else QOS_RELIABLE

        self.get_logger().info(
            f"[INIT] {self.robot}  qos={'BE' if self.cfg.best_effort else 'RE'}  "
            f"gains k_ex={self.cfg.k_ex} k_eth={self.cfg.k_eth} k_ey={self.cfg.k_ey}  "
            f"v_max={self.cfg.v_max} w_max={self.cfg.w_max}"
        )

        self.create_subscription(Odometry, f'/{self.robot}/odom', self.odom_callback, qos)
        self.cmd_pub = self.create_publisher(Twist, f'/{self.robot}/cmd_vel', qos)
        self.timer   = self.create_timer(0.02, self.control_loop)

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

    # ── Trajectory ───────────────────────────────────────────────────────────

    def create_trajectory(self, shared_start_time: float):
        x0, y0, yaw = self.current_pose
        d  = self.cfg.distance
        tf = self.cfg.traj_duration

        xf = x0 + math.cos(yaw) * d
        yf = y0 + math.sin(yaw) * d

        self.segment = {
            "tf": tf,
            "cx": compute_quintic_coeffs(0.0, tf, x0, 0.0, 0.0, xf, 0.0, 0.0),
            "cy": compute_quintic_coeffs(0.0, tf, y0, 0.0, 0.0, yf, 0.0, 0.0),
        }
        self.start_time   = shared_start_time
        self.traj_created = True

        self.get_logger().info(
            f"[TRAJ] {self.robot}: ({x0:.2f},{y0:.2f}) → ({xf:.2f},{yf:.2f})  "
            f"tf={tf}s  start_at={shared_start_time:.3f}s"
        )

    # ── Control loop ─────────────────────────────────────────────────────────

    def control_loop(self):
        if self.current_pose is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        # Phase 1 — wait for all robots to have odom
        if not self.traj_created:
            shared_start = self.coord.report_ready(self.robot, now)
            if shared_start is not None:
                self.create_trajectory(shared_start)
            return

        if self.segment is None:
            return

        t = now - self.start_time

        # Phase 2 — countdown
        if t < 0:
            if now - self.last_debug_time > 1.0:
                self.last_debug_time = now
                self.get_logger().info(f"[WAIT] {self.robot} → starts in {-t:.2f}s")
            return

        # Phase 3 — done
        if t > self.segment["tf"]:
            self.cmd_pub.publish(Twist())
            self.get_logger().info(f"[DONE] {self.robot}")
            self.segment = None
            return

        # ── Desired state ─────────────────────────────────────────────────────
        xd, xd_dot = eval_quintic(t, self.segment["cx"])
        yd, yd_dot = eval_quintic(t, self.segment["cy"])

        x, y, yaw = self.current_pose
        dx = xd - x
        dy = yd - y
        desired_theta = math.atan2(yd_dot, xd_dot)

        ex      =  math.cos(yaw)*dx + math.sin(yaw)*dy
        ey      = -math.sin(yaw)*dx + math.cos(yaw)*dy
        e_theta =  math.atan2(
            math.sin(desired_theta - yaw),
            math.cos(desired_theta - yaw)
        )

        # ── Control law — uses per-robot gains ────────────────────────────────
        cfg = self.cfg
        v = xd_dot * math.cos(e_theta) + cfg.k_ex  * ex
        w = cfg.k_eth * e_theta         + cfg.k_ey  * v * ey

        cmd = Twist()
        cmd.linear.x  = float(np.clip(v, -cfg.v_max, cfg.v_max))
        cmd.angular.z = float(np.clip(w, -cfg.w_max, cfg.w_max))
        self.cmd_pub.publish(cmd)

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
    nodes = [OneMeterImmediate(r, coordinator) for r in ROBOTS]

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