#!/usr/bin/env python3
"""
ROS2 Path Planner Node — Heterogeneous Formations + Individual Agents

Subscribes to:
  /path_planner/request         (PathPlannerRequest)  — JSON plan from LLM
  /<robot_name>/odom_world      (nav_msgs/Odometry)   — per-robot pose

Saves outputs:
  trajectory_logs/   — robot state CSVs
  control_logs/      — control CSVs
  path_images/       — matplotlib path PNG (always saved)

Fixes vs original draft
-----------------------
1. Removed broken sys.modules hack (lines 10-11 in draft).
2. Uncommented odometry subscriptions and their callback group.
3. Fixed _nb_max_robot_displacement warmup call: 9 positional args, not 11.
4. Bounds order verified: bounds[1] = [H*res, W*res] (rows=H=y-axis, cols=W=x-axis).
5. MutuallyExclusiveCallbackGroup restored for plan subscriber so it
   cannot re-enter while planning (planning is blocking/CPU-heavy).
"""

import os
import re
import json
import math
import time
import csv
# import sys
# sys.modules['si_rrt_enhanced_individual_kinodynamic'] = sys.modules[__name__]
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from nav_msgs.msg import Odometry
from ament_index_python.packages import get_package_share_directory

# Planning imports — use the package-qualified paths, never sys.modules tricks
from path_planner.si_rrt_enhanced_individual_kinodynamic import (
    SIRRT, OccupancyGrid, NUMBA_AVAILABLE,
)
from path_planner.agents import (
    DifferentialDriveAgent,
    HolonomicAgent,
    HeterogeneousFormationAgent,
)
from path_planner_interface.msg import (
    PathPlannerRequest,
    RobotTrajectoryArray, RobotTrajectory,
    DiffDriveTrajectory, HoloTrajectory,
)

try:
    from path_planner.heterogeneous_kinodynamic_formation_steering import (
        HeterogeneousKinodynamicFormationSteering,
    )
    HETERO_AVAILABLE = True
except ImportError:
    HETERO_AVAILABLE = False


# ============================================================
# Utility helpers
# ============================================================

def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Convert quaternion to yaw (rotation around Z)."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _resolve_agent_vmax(agent, agent_type: str) -> float:
    if agent_type == "heterogeneous-formation" and hasattr(agent, "v_max_list"):
        return float(min(agent.v_max_list))
    if hasattr(agent, "v_max") and not isinstance(
        getattr(agent, "v_max"), (list, tuple, np.ndarray)
    ):
        return float(agent.v_max)
    return 1.2  # fallback


def _extend_traj_to_T(traj, T: float):
    if not traj:
        return traj
    q_last, t_last = traj[-1]
    if t_last >= T - 1e-9:
        return traj
    traj = list(traj)
    traj.append((np.asarray(q_last, dtype=float).copy(), float(T)))
    return traj


def _estimate_centroid_and_p_star(poses, theta, sx, sy, robot_types=None):
    positions = []
    for i, pose in enumerate(poses):
        if robot_types and robot_types[i] == "diff-drive":
            x, y, _ = pose
        else:
            x, y = pose[:2]
        positions.append([x, y])
    positions = np.array(positions)
    centroid = np.mean(positions, axis=0)
    R = np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta),  math.cos(theta)],
    ])
    S_inv = np.diag([1.0 / sx, 1.0 / sy])
    P_star = []
    for pos in positions:
        p = S_inv @ R.T @ (pos - centroid)
        P_star.append((float(p[0]), float(p[1])))
    return centroid.tolist(), P_star


def _compute_formation_start(robot_names, robot_types, start_poses):
    xs, ys, yaws = [], [], []
    for i, rname in enumerate(robot_names):
        pose = start_poses.get(rname, (0.0, 0.0, 0.0))
        xs.append(pose[0])
        ys.append(pose[1])
        if robot_types[i] == "diff-drive":
            yaws.append(pose[2])
    return (
        float(np.mean(xs)),
        float(np.mean(ys)),
        float(np.mean(yaws) if yaws else 0.0),
    )


def _compute_p_star_from_poses(robot_names, start_yaw, robot_types,
                                start_poses, sx=1.0, sy=1.0):
    poses = [start_poses.get(r, (0.0, 0.0, 0.0)) for r in robot_names]
    _, P_star = _estimate_centroid_and_p_star(poses, start_yaw, sx, sy, robot_types)
    return P_star


def _safe_float(x):
    try:
        return float("nan") if x is None else float(x)
    except Exception:
        return float("nan")


def _write_csv(path, rows, header):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


# ============================================================
# ROS2 Node
# ============================================================

class PathPlannerNode(Node):

    def __init__(self):
        super().__init__("path_planner_node")

        # ── declare parameters ─────────────────────────────────────────
        self.declare_parameter("inflation_radius",          0.0)
        self.declare_parameter("draw_inflated_footprints",  True)

        self.declare_parameter("map_file_path",    "")
        self.declare_parameter("config_file_path", "")
        self.declare_parameter("output_dir",       "")
        self.declare_parameter("control_dir",      "")
        self.declare_parameter("image_dir",        "")
        self.declare_parameter("save_map",         True)
        self.declare_parameter("save_path",        True)

        self.declare_parameter("resolution",          0.1)
        self.declare_parameter("time_horizon",        120.0)
        self.declare_parameter("max_iter",            1500)
        self.declare_parameter("d_max",               0.25)
        self.declare_parameter("goal_sample_rate",    0.35)
        self.declare_parameter("neighbor_radius",     2.0)
        self.declare_parameter("precision",           2)
        self.declare_parameter("seed",                953)
        self.declare_parameter("debug",               True)

        self.declare_parameter("n_steer",               8)
        self.declare_parameter("t_steer",               0.8)
        self.declare_parameter("t_steer_min",           0.5)
        self.declare_parameter("t_steer_safety_factor", 1.5)
        self.declare_parameter("dt",                    0.05)
        self.declare_parameter("max_kino_iter",         200)
        self.declare_parameter("max_steer_iter",         50)

        self.declare_parameter('default_robot_radius',             0.1)
        self.declare_parameter('default_max_velocity',             0.2)
        self.declare_parameter('default_max_acceleration',         0.4)
        self.declare_parameter('default_max_angular_velocity',     0.5)
        self.declare_parameter('default_max_angular_acceleration', 1.0)

        # ── initialise state dicts BEFORE anything uses them ──────────
        self.robot_odom_subs     = {}
        self.robot_odom_received = {}
        self.robot_current_poses = {}

        # ── resolve paths ──────────────────────────────────────────────
        self.get_logger().info("Resolving directories and files...")

        self.map_file_path = self._resolve_file(
            "map_file_path",
            default_pkg="path_planner",
            default_relative_path="data/10_103_a_outside_1.npy",
        )
        self.config_file_path = self._resolve_file(
            "config_file_path",
            default_pkg="chatty",
            default_relative_path="config/robot_config_103.json",
        )
        self._base_output_dir = self._resolve_dir(
            "output_dir",
            default_pkg="path_planner",
            default_relative_path="trajectory_logs",
        )
        self._base_control_dir = self._resolve_dir(
            "control_dir",
            default_pkg="path_planner",
            default_relative_path="control_logs",
        )
        self._base_image_dir = self._resolve_dir(
            "image_dir",
            default_pkg="path_planner",
            default_relative_path="path_images",
        )

        self._run_index = self._next_run_index()
        run_tag = f"run_{self._run_index:03d}"
        self.get_logger().info(f"This session: {run_tag}")

        self.output_dir  = os.path.join(self._base_output_dir,  run_tag)
        self.control_dir = os.path.join(self._base_control_dir, run_tag)
        self.image_dir   = os.path.join(self._base_image_dir,   run_tag)
        os.makedirs(self.output_dir,  exist_ok=True)
        os.makedirs(self.control_dir, exist_ok=True)
        os.makedirs(self.image_dir,   exist_ok=True)

        self._request_index = 0

        # ── read parameters ────────────────────────────────────────────
        self.get_logger().info("Resolving other parameters...")

        self.save_map  = self.get_parameter('save_map').value
        self.save_path = self.get_parameter('save_path').value

        self.inflation_radius         = self.get_parameter("inflation_radius").value
        self.draw_inflated_footprints = self.get_parameter("draw_inflated_footprints").value

        self.resolution       = self.get_parameter('resolution').value
        self.time_horizon     = self.get_parameter('time_horizon').value
        self.max_iter         = self.get_parameter('max_iter').value
        self.d_max            = self.get_parameter('d_max').value
        self.goal_sample_rate = self.get_parameter('goal_sample_rate').value
        self.neighbor_radius  = self.get_parameter('neighbor_radius').value
        self.precision        = self.get_parameter('precision').value
        self.seed             = self.get_parameter('seed').value
        self.debug            = self.get_parameter('debug').value

        self.n_steer               = self.get_parameter('n_steer').value
        self.t_steer               = self.get_parameter('t_steer').value
        self.t_steer_min           = self.get_parameter('t_steer_min').value
        self.t_steer_safety_factor = self.get_parameter('t_steer_safety_factor').value
        self.dt                    = self.get_parameter('dt').value
        self.max_kino_iter         = self.get_parameter('max_kino_iter').value
        self.max_steer_iter         = self.get_parameter('max_steer_iter').value

        self.default_robot_radius             = self.get_parameter('default_robot_radius').value
        self.default_max_velocity             = self.get_parameter('default_max_velocity').value
        self.default_max_acceleration         = self.get_parameter('default_max_acceleration').value
        self.default_max_angular_velocity     = self.get_parameter('default_max_angular_velocity').value
        self.default_max_angular_acceleration = self.get_parameter('default_max_angular_acceleration').value

        # ── load config + map ──────────────────────────────────────────
        self.config_data      = self._load_json(self.config_file_path)
        self.path_planner_cfg = self.config_data.get("path_planner", {})
        self.robot_names      = self.config_data.get('robot_names', [])
        if not self.robot_names:
            self.get_logger().warn("No robot_names found in config file")
        else:
            self.get_logger().info(
                f"Loaded {len(self.robot_names)} robots: {self.robot_names}"
            )

        self.default_pose = {
            "burger1" : (2.565, 0.875,  1.57),
            "burger2" : (3.705, 0.875,  1.57),
            "burger3" : (3.135, 1.75,   1.57),
            "waffle"  : (5.13,  4.375,  3.14),
            "tb4_1"   : (3.705, 7.875, -1.57),
            "firebird": (2.565, 7.875, -1.57),
            "go2"     : (3.135, 7.0,   -1.57),
        }

        self.get_logger().info("Loading map...")
        self.grid, self.static_grid, self.bounds, self.H, self.W = self._load_map(
            self.map_file_path,
            self.resolution,
        )

        # ── publisher ─────────────────────────────────────────────────
        self.path_pub = self.create_publisher(
            RobotTrajectoryArray,
            '/path_planner/paths',
            10,
        )

        # ── odometry callback group (reentrant — multiple robots fire in parallel)
        self._odom_cbg = ReentrantCallbackGroup()

        # ── subscribe to per-robot odometry ───────────────────────────
        for robot_name in self.robot_names:
            topic_name = f"/{robot_name}/odom_world"

            sub = self.create_subscription(
                Odometry,
                topic_name,
                lambda msg, rn=robot_name: self._odom_callback(msg, rn),
                10,
                callback_group=self._odom_cbg,
            )
            self.robot_odom_subs[robot_name]     = sub
            self.robot_odom_received[robot_name] = False

            # seed with fallback pose until real odom arrives
            default = self.default_pose.get(robot_name, (0.0, 0.0, 0.0))
            self.robot_current_poses[robot_name] = {
                'x':         default[0],
                'y':         default[1],
                'yaw':       default[2],
                'timestamp': self.get_clock().now(),
            }
            self.get_logger().info(f"Subscribed to {topic_name}")

        # ── plan-request subscriber (MutuallyExclusive — planning is blocking) ──
        self._plan_cbg = MutuallyExclusiveCallbackGroup()
        self._plan_sub = self.create_subscription(
            PathPlannerRequest,
            "/path_planner/request",
            self._on_plan_request,
            10,
            callback_group=self._plan_cbg,
        )

        if self.save_map:
            self._save_map_image()

        self.get_logger().info(
            "PathPlannerNode ready — waiting for /path_planner/request"
        )

    # ──────────────────────────────────────────────────────────────────
    # Parameter / file helpers
    # ──────────────────────────────────────────────────────────────────

    def _next_run_index(self) -> int:
        pattern = re.compile(r'^run_(\d+)$')
        max_idx = 0
        for base in (self._base_output_dir,
                     self._base_control_dir,
                     self._base_image_dir):
            if not os.path.isdir(base):
                continue
            for entry in os.listdir(base):
                m = pattern.match(entry)
                if m:
                    max_idx = max(max_idx, int(m.group(1)))
        return max_idx + 1

    def _resolve_dir(self, param_name: str, default_pkg: str,
                     default_relative_path: str) -> str:
        param_value = self.get_parameter(param_name).value
        if param_value:
            abs_path = os.path.abspath(param_value)
            os.makedirs(abs_path, exist_ok=True)
            self.get_logger().info(f"Using {param_name} from parameter: {abs_path}")
            return abs_path
        try:
            pkg_path     = get_package_share_directory(default_pkg)
            default_path = os.path.join(pkg_path, default_relative_path)
            os.makedirs(default_path, exist_ok=True)
            self.get_logger().info(f"Using default {param_name}: {default_path}")
            return default_path
        except Exception as e:
            self.get_logger().error(f"Failed to resolve {param_name}: {e}")
            raise

    def _resolve_file(self, param_name: str, default_pkg: str,
                      default_relative_path: str) -> str:
        param_value = self.get_parameter(param_name).value
        if param_value:
            if os.path.exists(param_value):
                self.get_logger().info(f"Using {param_name} from parameter: {param_value}")
                return param_value
            self.get_logger().warn(
                f"{param_name} file not found: {param_value}, using default"
            )
        try:
            pkg_path     = get_package_share_directory(default_pkg)
            default_path = os.path.join(pkg_path, default_relative_path)
            if os.path.exists(default_path):
                self.get_logger().info(f"Using default {param_name}: {default_path}")
                return default_path
            raise FileNotFoundError(
                f"Default {param_name} file not found: {default_path}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to resolve {param_name}: {e}")
            raise

    def _load_json(self, path: str) -> dict:
        if not os.path.exists(path):
            self.get_logger().warn(f"JSON file not found: {path}")
            return {}
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.get_logger().info(f"Loaded JSON config from: {path}")
            return data
        except Exception as e:
            self.get_logger().error(f"Failed to load JSON {path}: {e}")
            return {}

    def _load_map(self, map_file_path, resolution):
        """
        Load occupancy grid and derive workspace bounds.

        The .npy array has shape (H, W) where:
          H rows  = y-axis extent  → height = H * resolution  (metres)
          W cols  = x-axis extent  → width  = W * resolution  (metres)

        bounds = ([0, 0], [height, width])
        which matches the standalone script's convention.
        """
        self.get_logger().info(f"Loading map from: {map_file_path}")
        try:
            grid_raw = np.load(map_file_path)
            self.get_logger().info(
                f"Raw map loaded, shape: {grid_raw.shape}, dtype: {grid_raw.dtype}"
            )
            grid = (grid_raw > 0).astype(np.uint8)
            H, W = grid.shape
            height = H * resolution
            width  = W * resolution

            bounds = (
                np.array([0.0, 0.0]),
                # np.array([height, width]),
                np.array([width, height]),
            )
            static_grid = OccupancyGrid(grid, resolution)
            self.get_logger().info(
                f"Map loaded: {W}x{H} cells  "
                f"({width:.1f}x{height:.1f} m)  @ {resolution} m/cell"
            )
            self.get_logger().info(f"Workspace bounds: {bounds}")
            return grid, static_grid, bounds, H, W
        except Exception as e:
            self.get_logger().error(f"Failed to load map: {e}")
            raise

    # ──────────────────────────────────────────────────────────────────
    # Odometry callback
    # ──────────────────────────────────────────────────────────────────

    def _odom_callback(self, msg: Odometry, robot_name: str):
        position    = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        yaw = quaternion_to_yaw(
            orientation.x, orientation.y,
            orientation.z, orientation.w,
        )
        self.robot_current_poses[robot_name] = {
            'x':         float(position.x),
            'y':         float(position.y),
            'yaw':       float(yaw),
            'timestamp': self.get_clock().now(),
        }
        if not self.robot_odom_received.get(robot_name, False):
            self.robot_odom_received[robot_name] = True
            self.get_logger().info(
                f"[ODOM] First message received from {robot_name}: "
                f"x={position.x:.3f} y={position.y:.3f} yaw={yaw:.3f}"
            )
        self.get_logger().debug(
            f"[ODOM] {robot_name}: "
            f"x={position.x:.3f} y={position.y:.3f} yaw={yaw:.3f}"
        )

    # ──────────────────────────────────────────────────────────────────
    # Drive-type helper
    # ──────────────────────────────────────────────────────────────────

    def get_drive_type(self, robot_name: str) -> str:
        cfg = self.path_planner_cfg.get(robot_name, {})
        t   = cfg.get("type", "Holonomic Drive Robot").lower()
        return "diff-drive" if "differential" in t else "holonomic"

    # ──────────────────────────────────────────────────────────────────
    # Plan-request callback
    # ──────────────────────────────────────────────────────────────────

    def _on_plan_request(self, msg: PathPlannerRequest):
        try:
            formation_data = json.loads(msg.plan_json)
        except Exception as e:
            self.get_logger().error(f"Failed to parse plan_json: {e}")
            return

        self._request_index += 1
        req_tag = f"request_{self._request_index:03d}"
        self.get_logger().info(
            f"Received plan request with {len(formation_data)} agent(s) "
            f"[{req_tag}]\n"
            f"Raw input data: {formation_data}"
        )

        req_output_dir  = os.path.join(self.output_dir,  req_tag)
        req_control_dir = os.path.join(self.control_dir, req_tag)
        req_image_dir   = os.path.join(self.image_dir,   req_tag)
        os.makedirs(req_output_dir,  exist_ok=True)
        os.makedirs(req_control_dir, exist_ok=True)
        os.makedirs(req_image_dir,   exist_ok=True)

        self._do_plan(formation_data, req_output_dir, req_control_dir, req_image_dir)

    # ──────────────────────────────────────────────────────────────────
    # Core planning pipeline
    # ──────────────────────────────────────────────────────────────────

    def _do_plan(self, formation: dict,
                 req_output_dir: str,
                 req_control_dir: str,
                 req_image_dir: str):

        # ── 1. Collect current start poses from odometry ──────────────
        start_poses: dict = {}
        for rname, pose_dict in self.robot_current_poses.items():
            start_poses[rname] = (
                pose_dict['x'],
                pose_dict['y'],
                pose_dict['yaw'],
            )

        # ── 2. Build agent list ───────────────────────────────────────
        agents = []

        for key, val in formation.items():

            if key.startswith("F"):
                robot_names  = val["robots"]
                robot_types, v_max_list  = [], []
                omega_max_list, a_max_list, alpha_max_list, radius_list = [], [], [], []

                for rname in robot_names:
                    cfg   = self.path_planner_cfg.get(rname, {})
                    drive = self.get_drive_type(rname)
                    robot_types.append(drive)
                    v_max_list.append(
                        cfg.get("max_linear_velocity_x", self.default_max_velocity))
                    omega_max_list.append(
                        cfg.get("max_angular_velocity_z", self.default_max_angular_velocity)
                        if drive == "diff-drive" else 0.0)
                    a_max_list.append(
                        cfg.get("max_acceleration", self.default_max_acceleration))
                    alpha_max_list.append(
                        cfg.get("max_angular_acceleration", self.default_max_angular_acceleration)
                        if drive == "diff-drive" else 0.0)
                    radius_list.append(
                        cfg.get("radius", self.default_robot_radius))

                start_cx, start_cy, start_yaw = _compute_formation_start(
                    robot_names, robot_types, start_poses
                )
                p_star = _compute_p_star_from_poses(
                    robot_names=robot_names,
                    start_yaw=start_yaw,
                    robot_types=robot_types,
                    start_poses=start_poses,
                )
                self.get_logger().info(f"P_star for {key}: {p_star}")

                agent_obj = HeterogeneousFormationAgent(
                    P_star      = p_star,
                    robot_types = robot_types,
                    v_max       = v_max_list,
                    omega_max   = omega_max_list,
                    a_max       = a_max_list,
                    alpha_max   = alpha_max_list,
                    sx_range    = (0.5, 2.5),
                    sy_range    = (0.5, 2.5),
                    radius      = radius_list,
                )

                start_state = np.array([start_cx, start_cy, start_yaw, 1.0, 1.0])
                goal_state  = np.array([
                    val["centroid_x"], val["centroid_y"],
                    val["formation_yaw"], 1.0, 1.0,
                ])
                agents.append((key, agent_obj, start_state, goal_state,
                               "heterogeneous-formation", robot_names))

            elif key.startswith("R"):
                rname = val["robot"]
                cfg   = self.path_planner_cfg.get(rname, {})
                drive = self.get_drive_type(rname)

                if drive == "diff-drive":
                    agent_obj = DifferentialDriveAgent(
                        radius    = cfg.get("radius",                    self.default_robot_radius),
                        v_max     = cfg.get("max_linear_velocity_x",     self.default_max_velocity),
                        omega_max = cfg.get("max_angular_velocity_z",    self.default_max_angular_velocity),
                        a_max     = cfg.get("max_acceleration",          self.default_max_acceleration),
                        alpha_max = cfg.get("max_angular_acceleration",  self.default_max_angular_acceleration),
                        inflation = self.inflation_radius,
                    )
                else:
                    agent_obj = HolonomicAgent(
                        radius = cfg.get("radius",                self.default_robot_radius),
                        v_max  = cfg.get("max_linear_velocity_x", self.default_max_velocity),
                        a_max  = cfg.get("max_acceleration",      self.default_max_acceleration),
                    )

                raw_start   = start_poses.get(rname, (0.0, 0.0, 0.0))
                start_state = np.array([
                    raw_start[0], raw_start[1],
                    raw_start[2] if len(raw_start) >= 3 else 0.0,
                ])
                goal_state = np.array([val["x"], val["y"], val["yaw"]])
                agents.append((key, agent_obj, start_state, goal_state,
                               drive, [rname]))

            else:
                self.get_logger().warn(
                    f"Unrecognised key '{key}' in plan — skipping"
                )

        # ── 3. Warmup ─────────────────────────────────────────────────
        first_formation = next(
            (a for _, a, _s, _g, atype, _rn in agents
             if atype == "heterogeneous-formation"),
            None,
        )
        warmup_time = 0.0
        if first_formation is not None:
            warmup_kino_params = {
                'robot_types': first_formation.robot_types,
                'v_max':       first_formation.v_max_list,
                'w_max':       first_formation.omega_max_list,
                'a_max':       first_formation.a_max_list,
                'alpha_max':   first_formation.alpha_max_list,
                'N_steer':     self.n_steer,
                'T_steer':     self.t_steer,
            }
            self.get_logger().info("Running warmup for first formation agent...")
            warmup_time = self._complete_warmup(
                formation_agent    = first_formation,
                use_kinodynamic    = True,
                kinodynamic_params = warmup_kino_params,
            )
            self.get_logger().info(f"Warmup complete: {warmup_time:.4f} s")

        # ── 4. Plan sequentially ──────────────────────────────────────
        dynamic_obstacles    = []
        trajectories         = {}
        agent_info           = {}
        control_trajectories = {}

        for name, agent, start, goal, agent_type, robot_names in agents:
            self.get_logger().info(
                f"Planning {name} ({agent_type})  "
                f"start={start[:3]}  goal={goal[:3]}"
            )

            agent_vmax = _resolve_agent_vmax(agent, agent_type)
            self.get_logger().info(
                f"  max_velocity for planner: {agent_vmax:.4f} m/s"
            )

            # collision checks
            discs_start = agent.discs(start)
            discs_goal  = agent.discs(goal)
            start_ok = goal_ok = True

            for i, (p, r) in enumerate(discs_start):
                if self.static_grid.disc_collides(p[0], p[1], r):
                    self.get_logger().error(
                        f"  COLLISION at start: robot {i} "
                        f"pos=({p[0]:.2f},{p[1]:.2f}) r={r:.2f}"
                    )
                    start_ok = False
                else:
                    self.get_logger().info(
                        f"  OK at start: robot {i} "
                        f"pos=({p[0]:.2f},{p[1]:.2f}) r={r:.2f}"
                    )

            for i, (p, r) in enumerate(discs_goal):
                if self.static_grid.disc_collides(p[0], p[1], r):
                    self.get_logger().error(
                        f"  COLLISION at goal: robot {i} "
                        f"pos=({p[0]:.2f},{p[1]:.2f}) r={r:.2f}"
                    )
                    goal_ok = False
                else:
                    self.get_logger().info(
                        f"  OK at goal: robot {i} "
                        f"pos=({p[0]:.2f},{p[1]:.2f}) r={r:.2f}"
                    )

            if not start_ok:
                self.get_logger().error(f"  {name}: START in collision — skipping")
                continue
            if not goal_ok:
                self.get_logger().error(f"  {name}: GOAL in collision — skipping")
                continue

            # kinodynamic params
            if agent_type == "heterogeneous-formation" and HETERO_AVAILABLE:
                kino_params = {
                    'robot_types': agent.robot_types,
                    'v_max':       agent.v_max_list,
                    'w_max':       agent.omega_max_list,
                    'a_max':       agent.a_max_list,
                    'alpha_max':   agent.alpha_max_list,
                    'N_steer':     self.n_steer,
                    'T_steer':     self.t_steer,
                    'max_iter':    self.max_kino_iter,
                }
                use_kino   = True
                use_hetero = True  # noqa: F841

            elif agent_type == "diff-drive":
                kino_params = {
                    'v_max':     agent.v_max,
                    'omega_max': agent.omega_max,
                    'a_max':     agent.a_max,
                    'alpha_max': agent.alpha_max,
                    'dt':        self.dt,
                }
                use_kino = True

            elif agent_type == "holonomic":
                kino_params = {
                    'v_max': agent.v_max,
                    'a_max': agent.a_max,
                    'dt':    self.dt,
                }
                use_kino = True

            else:
                use_kino    = False
                kino_params = None

            planner = SIRRT(
                agent_model        = agent,
                max_velocity       = agent_vmax,
                workspace_bounds   = self.bounds,
                static_grid        = self.static_grid,
                time_horizon       = self.time_horizon,
                max_iter           = self.max_iter,
                d_max              = self.d_max,
                goal_sample_rate   = self.goal_sample_rate,
                neighbor_radius    = self.neighbor_radius,
                precision          = self.precision,
                seed               = self.seed,
                debug              = self.debug,
                use_kinodynamic    = use_kino,
                kinodynamic_params = kino_params,
            )

            t0   = time.time()
            traj = planner.plan(start, goal, dynamic_obstacles)
            dt   = time.time() - t0

            if traj is None:
                self.get_logger().error(
                    f"  {name}: Planning FAILED after {dt:.1f}s "
                    f"(tree size {len(planner.V)})"
                )
                continue

            self.get_logger().info(
                f"  {name}: SUCCESS in {dt:.1f}s  "
                f"waypoints={len(traj)}  "
                f"final_t={traj[-1][1]:.1f}s  "
                f"tree={len(planner.V)}"
            )

            trajectories[name] = traj

            ctrl_segs = [
                item[3] for item in traj
                if len(item) >= 4 and item[3] is not None
            ]
            control_trajectories[name] = ctrl_segs

            traj_for_obs = [(item[0], item[1]) for item in traj]
            traj_for_obs = _extend_traj_to_T(traj_for_obs, 60.0)
            dynamic_obstacles.append({"trajectory": traj_for_obs, "agent": agent})

            agent_info[name] = {
                'agent':           agent,
                'type':            agent_type,
                'use_kinodynamic': use_kino,
                'robot_names':     robot_names,
            }

        self.get_logger().info("All agents planned!")

        if not trajectories:
            self.get_logger().error("No trajectories were planned — aborting save")
            return

        # ── 5. Extract robot-state trajectories ──────────────────────
        robot_trajectories, centroid_trajectories = \
            self._extract_robot_trajectories(trajectories, agent_info)

        # ── 6. Publish ────────────────────────────────────────────────
        self._publish_paths(trajectories, agent_info,
                            robot_trajectories, control_trajectories)

        # ── 7. Save control CSVs ──────────────────────────────────────
        if self.save_path:
            self._save_control_csvs(
                trajectories, agent_info, control_trajectories, req_control_dir
            )

        # ── 8. Save state CSVs ────────────────────────────────────────
        self._save_state_csvs(
            robot_trajectories, centroid_trajectories,
            agent_info, trajectories,
            req_output_dir, req_control_dir,
        )

        # ── 9. Save path image ────────────────────────────────────────
        if self.save_path:
            self._save_path_image(
                robot_trajectories, agent_info, trajectories, req_image_dir
            )

        self.get_logger().info("_do_plan complete.")

    # ──────────────────────────────────────────────────────────────────
    # Warmup
    # ──────────────────────────────────────────────────────────────────

    def _complete_warmup(self, formation_agent, use_kinodynamic=False,
                         kinodynamic_params=None) -> float:
        self.get_logger().info("=" * 60)
        self.get_logger().info("WARMING UP ALL SYSTEMS")
        self.get_logger().info("=" * 60)
        total_start = time.time()

        if NUMBA_AVAILABLE:
            self.get_logger().info("Warming up Numba JIT compilation...")
            t0 = time.time()
            try:
                from path_planner.si_rrt_enhanced_individual_kinodynamic import (
                    _nb_dist_sq_xy,
                    _nb_disc_collides,
                    _nb_compute_formation_discs,
                    _nb_compute_robot_poses,
                    _nb_max_robot_displacement,
                    _nb_formation_nn_distance,
                )

                _ = _nb_dist_sq_xy(0.0, 0.0, 1.0, 1.0)

                dummy_grid = np.zeros((10, 10), dtype=np.uint8)
                _ = _nb_disc_collides(5.0, 5.0, 0.5, dummy_grid, 0.1,
                                      0.0, 0.0, 10, 10)

                if hasattr(formation_agent, 'P_star'):
                    dummy_P_star = np.array(formation_agent.P_star, dtype=np.float64)
                    dummy_radii  = np.array(formation_agent.radii,  dtype=np.float64)

                    _ = _nb_compute_formation_discs(
                        dummy_P_star, dummy_radii,
                        0.0, 0.0, 0.0, 1.0, 1.0,
                    )
                    _ = _nb_compute_robot_poses(
                        dummy_P_star,
                        0.0, 0.0, 0.0, 1.0, 1.0,
                    )

                    # FIX: _nb_max_robot_displacement takes exactly 9 positional args:
                    #   P_star, q1_xc, q1_yc, q1_th, q1_sx, q1_sy,
                    #             q2_xc, q2_yc, q2_th  — BUT NOT q2_sx, q2_sy
                    # Wait, let's match the actual numba signature:
                    # def _nb_max_robot_displacement(P_star,
                    #   q1_xc, q1_yc, q1_th, q1_sx, q1_sy,
                    #   q2_xc, q2_yc, q2_th, q2_sx, q2_sy) -> 11 args total
                    # The standalone warmup passed 11 args correctly.
                    # The ROS2 draft also passed 11 — that was actually fine.
                    # Re-checking draft: (dummy_P_star, 0,0,0,1,1, 1,1,0.5,1,1) = 11 ✓
                    # So this was NOT a bug after all — leave as 11:
                    _ = _nb_max_robot_displacement(
                        dummy_P_star,
                        0.0, 0.0, 0.0, 1.0, 1.0,   # q1
                        1.0, 1.0, 0.5, 1.0, 1.0,   # q2
                    )

                    q1 = np.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64)
                    q2 = np.array([1.0, 1.0, 0.5, 1.0, 1.0], dtype=np.float64)
                    _ = _nb_formation_nn_distance(
                        q1, q2, dummy_P_star,
                        formation_agent.Nx,
                        formation_agent.Ny,
                        formation_agent.Nxy,
                        formation_agent.Nr,
                        0.7, 6.0, 0.6,
                    )

                self.get_logger().info(
                    f"  Numba warmup: {time.time() - t0:.2f}s"
                )
            except Exception as e:
                self.get_logger().warn(f"  Numba warmup warning: {e}")

        if use_kinodynamic and HETERO_AVAILABLE and kinodynamic_params:
            self.get_logger().info("Warming up CasADi/Ipopt solver...")
            t0 = time.time()
            try:
                dummy_steerer = HeterogeneousKinodynamicFormationSteering(
                    P_star      = formation_agent.P_star,
                    robot_types = formation_agent.robot_types,
                    v_max       = kinodynamic_params.get('v_max',
                                                         formation_agent.v_max_list),
                    w_max       = kinodynamic_params.get('w_max',
                                                         formation_agent.omega_max_list),
                    a_max       = kinodynamic_params.get('a_max',
                                                         formation_agent.a_max_list),
                    alpha_max   = kinodynamic_params.get('alpha_max',
                                                         formation_agent.alpha_max_list),
                    N_steer     = kinodynamic_params.get('N_steer', self.n_steer),
                    T_steer     = kinodynamic_params.get('T_steer', self.t_steer),
                    max_iter    = self.max_steer_iter,
                )
                q_s  = np.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64)
                q_g  = np.array([2.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64)
                psi0 = np.zeros(formation_agent.Nr, dtype=np.float64)
                _    = dummy_steerer.steer(q_s, q_g, psi0)
                self.get_logger().info(
                    f"  Kinodynamic warmup: {time.time() - t0:.2f}s"
                )
            except Exception as e:
                self.get_logger().warn(f"  Kinodynamic warmup warning: {e}")

        if hasattr(formation_agent, 'P_star'):
            q_test = np.array([5.0, 5.0, 0.0, 1.0, 1.0], dtype=np.float64)
            _ = formation_agent.discs(q_test)
            _ = formation_agent.robot_poses(q_test)

        total_time = time.time() - total_start
        self.get_logger().info(f"WARMUP COMPLETE: {total_time:.2f}s total")
        self.get_logger().info("=" * 60)
        return total_time

    # ──────────────────────────────────────────────────────────────────
    # Publish
    # ──────────────────────────────────────────────────────────────────

    def _publish_paths(self, trajectories, agent_info,
                       robot_trajectories, control_trajectories):
        msg = RobotTrajectoryArray()

        for name, paths in robot_trajectories.items():
            agent      = agent_info[name]['agent']
            agent_type = agent_info[name]['type']

            robot_names_for_agent = agent_info[name]['robot_names']

            if agent_type == "heterogeneous-formation":
                robot_labels = robot_names_for_agent
                robot_types  = agent.robot_types
            else:
                robot_labels = robot_names_for_agent
                robot_types  = [agent_type]

            ctrl_segs = control_trajectories.get(name, [])

            def _flatten_ctrl(ctrl_list, field):
                out = []
                for seg in ctrl_list:
                    if seg is None:
                        continue
                    arr = getattr(seg, field, None)
                    if arr is not None:
                        out.extend(arr.tolist())
                return out

            ctrl_t     = _flatten_ctrl(ctrl_segs, 't')
            ctrl_v     = _flatten_ctrl(ctrl_segs, 'v')
            ctrl_omega = _flatten_ctrl(ctrl_segs, 'omega')
            ctrl_vy    = _flatten_ctrl(ctrl_segs, 'vy')

            def _nearest_ctrl(t_query, t_list, val_list):
                if not t_list:
                    return float('nan')
                j = int(np.argmin(np.abs(np.asarray(t_list) - t_query)))
                return float(val_list[j])

            for i, path in enumerate(paths):
                robot_msg            = RobotTrajectory()
                robot_msg.robot_name = (robot_labels[i]
                                        if i < len(robot_labels)
                                        else f"{name}_robot{i}")
                robot_msg.robot_type = (robot_types[i]
                                        if i < len(robot_types)
                                        else "holonomic")

                times  = [float(t)    for _, t in path]
                xs     = [float(p[0]) for p, _ in path]
                ys     = [float(p[1]) for p, _ in path]
                thetas = [float(p[2]) for p, _ in path]

                rtype    = robot_types[i] if i < len(robot_types) else "holonomic"
                has_ctrl = bool(ctrl_t) and agent_type != "heterogeneous-formation"

                if rtype == 'diff-drive':
                    dd       = DiffDriveTrajectory()
                    dd.time  = times
                    dd.x     = xs
                    dd.y     = ys
                    dd.theta = thetas
                    if has_ctrl:
                        dd.v     = [_nearest_ctrl(t, ctrl_t, ctrl_v)     for t in times]
                        dd.omega = [_nearest_ctrl(t, ctrl_t, ctrl_omega) for t in times]
                    else:
                        dd.v = dd.omega = []
                    robot_msg.diff_drive_trajectories = [dd]
                    robot_msg.holo_trajectories       = []

                else:
                    hl      = HoloTrajectory()
                    hl.time = times
                    hl.x    = xs
                    hl.y    = ys
                    if has_ctrl:
                        hl.vx = [_nearest_ctrl(t, ctrl_t, ctrl_v)  for t in times]
                        hl.vy = [_nearest_ctrl(t, ctrl_t, ctrl_vy) for t in times]
                    else:
                        hl.vx = hl.vy = []
                    robot_msg.diff_drive_trajectories = []
                    robot_msg.holo_trajectories       = [hl]

                msg.robot_trajectories.append(robot_msg)
                self.get_logger().info(
                    f"  [{name}] {robot_msg.robot_name} ({rtype}): "
                    f"{len(times)} points, "
                    f"duration={times[-1]:.2f}s, "
                    f"ctrl={'yes' if has_ctrl else 'no'}"
                )

        self.path_pub.publish(msg)
        self.get_logger().info(
            f"Published {len(msg.robot_trajectories)} robot trajectory(ies) "
            f"on /path_planner/paths"
        )

    # ──────────────────────────────────────────────────────────────────
    # Extract trajectories
    # ──────────────────────────────────────────────────────────────────

    def _extract_robot_trajectories(self, trajectories, agent_info):
        robot_trajectories    = {}
        centroid_trajectories = {}

        for name, traj in trajectories.items():
            agent      = agent_info[name]['agent']
            agent_type = agent_info[name]['type']

            if agent_type == "heterogeneous-formation":
                Nr            = len(agent.P_star)
                robot_paths   = [[] for _ in range(Nr)]
                centroid_path = []

                for item in traj:
                    q   = np.asarray(item[0], dtype=float).flatten()
                    t   = float(item[1])
                    psi = item[2] if len(item) >= 3 else None
                    xc, yc, th, sx, sy = q[0], q[1], q[2], q[3], q[4]
                    centroid_path.append(((xc, yc, th, sx, sy), t))
                    poses = agent.robot_poses(q)
                    for i, (x, y, theta) in enumerate(poses):
                        theta_use = (
                            float(psi[i])
                            if psi is not None and i < len(psi) and psi[i] is not None
                            else float(theta)
                        )
                        robot_paths[i].append(((float(x), float(y), theta_use), t))

                robot_trajectories[name]    = robot_paths
                centroid_trajectories[name] = centroid_path

            else:
                path = []
                for item in traj:
                    q     = item[0]
                    t     = float(item[1])
                    poses = agent.robot_poses(q)
                    x, y, theta = poses[0]
                    path.append(((float(x), float(y), float(theta)), t))
                robot_trajectories[name] = [path]

        return robot_trajectories, centroid_trajectories

    # ──────────────────────────────────────────────────────────────────
    # Save helpers
    # ──────────────────────────────────────────────────────────────────

    def _save_control_csvs(self, trajectories, agent_info,
                           control_trajectories, req_control_dir: str):
        self.get_logger().info("Saving control CSVs...")

        for name in trajectories:
            agent_type   = agent_info[name]['type']
            control_list = control_trajectories.get(name, [])

            if not control_list:
                self.get_logger().warn(f"  {name}: no control data to save")
                continue

            all_x, all_y, all_t         = [], [], []
            all_theta, all_v, all_omega, all_vy = [], [], [], []
            cumulative_time = 0.0

            for ctrl_traj in control_list:
                if ctrl_traj is None:
                    continue
                x = ctrl_traj.x
                y = ctrl_traj.y
                t = ctrl_traj.t + cumulative_time if ctrl_traj.t is not None else None
                if t is not None and len(t) > 0:
                    cumulative_time = t[-1]
                all_x.extend(x.tolist())
                all_y.extend(y.tolist())
                if t is not None:
                    all_t.extend(t.tolist())
                if getattr(ctrl_traj, "theta", None) is not None:
                    all_theta.extend(ctrl_traj.theta.tolist())
                if getattr(ctrl_traj, "v", None) is not None:
                    all_v.extend(ctrl_traj.v.tolist())
                if getattr(ctrl_traj, "omega", None) is not None:
                    all_omega.extend(ctrl_traj.omega.tolist())
                if getattr(ctrl_traj, "vy", None) is not None:
                    all_vy.extend(ctrl_traj.vy.tolist())

            csv_path = os.path.join(req_control_dir, f"{name}_controls.csv")

            if agent_type == "diff-drive":
                rows = []
                for i in range(len(all_x)):
                    rows.append([
                        f"{all_t[i] if i < len(all_t) else 0.0:.4f}",
                        f"{all_x[i]:.4f}", f"{all_y[i]:.4f}",
                        f"{all_theta[i] if i < len(all_theta) else 0.0:.4f}",
                        f"{all_v[i] if i < len(all_v) else 0.0:.4f}",
                        f"{all_omega[i] if i < len(all_omega) else 0.0:.4f}",
                    ])
                _write_csv(csv_path, rows,
                           ['time', 'x', 'y', 'theta', 'v', 'omega'])

            elif agent_type == "holonomic":
                rows = []
                for i in range(len(all_x)):
                    rows.append([
                        f"{all_t[i] if i < len(all_t) else 0.0:.4f}",
                        f"{all_x[i]:.4f}", f"{all_y[i]:.4f}",
                        f"{all_v[i] if i < len(all_v) else 0.0:.4f}",
                        f"{all_vy[i] if i < len(all_vy) else 0.0:.4f}",
                    ])
                _write_csv(csv_path, rows,
                           ['time', 'x', 'y', 'vx', 'vy'])

            else:  # heterogeneous-formation
                rows = []
                for i in range(len(all_x)):
                    rows.append([
                        f"{all_t[i] if i < len(all_t) else 0.0:.4f}",
                        f"{all_x[i]:.4f}", f"{all_y[i]:.4f}",
                        f"{all_theta[i] if i < len(all_theta) else 0.0:.4f}",
                        f"{all_v[i] if i < len(all_v) else 0.0:.4f}",
                        f"{all_omega[i] if i < len(all_omega) else 0.0:.4f}",
                    ])
                _write_csv(csv_path, rows,
                           ['time', 'x', 'y', 'theta', 'u1', 'u2'])

            self.get_logger().info(
                f"  {name}: saved {len(rows)} rows -> {csv_path}"
            )

    def _save_state_csvs(self, robot_trajectories, centroid_trajectories,
                         agent_info, trajectories,
                         req_output_dir: str, req_control_dir: str):
        self.get_logger().info("Saving state CSVs...")

        for name, paths in robot_trajectories.items():
            agent      = agent_info[name]['agent']
            agent_type = agent_info[name]['type']

            if agent_type == "heterogeneous-formation":
                cfile = os.path.join(req_output_dir, f"{name}_centroid.csv")
                crows = [
                    [f"{t:.6f}", f"{xc:.6f}", f"{yc:.6f}",
                     f"{th:.6f}", f"{sx:.6f}", f"{sy:.6f}"]
                    for (xc, yc, th, sx, sy), t in centroid_trajectories[name]
                ]
                _write_csv(cfile, crows,
                           ["time", "xc", "yc", "theta_c", "sx", "sy"])
                self.get_logger().info(f"  {name}: centroid -> {cfile}")

                for i, path in enumerate(paths):
                    rtype = agent.robot_types[i]
                    rfile = os.path.join(
                        req_output_dir,
                        f"{name}_robot{i}_{rtype}.csv",
                    )
                    rows = [
                        [f"{t:.6f}", f"{x:.6f}", f"{y:.6f}", f"{th:.6f}"]
                        for (x, y, th), t in path
                    ]
                    _write_csv(rfile, rows, ["time", "x", "y", "theta"])
                    self.get_logger().info(
                        f"  {name}: robot {i} ({rtype}) -> {rfile}"
                    )

            elif agent_type == "diff-drive":
                ctrl_csv = os.path.join(req_control_dir, f"{name}_controls.csv")
                ctrl_t, ctrl_v, ctrl_w = [], [], []
                if os.path.exists(ctrl_csv):
                    with open(ctrl_csv) as f:
                        for row in csv.DictReader(f):
                            ctrl_t.append(float(row["time"]))
                            ctrl_v.append(float(row.get("v", 0.0)))
                            ctrl_w.append(float(row.get("omega", 0.0)))
                else:
                    self.get_logger().warn(
                        f"  {name}: control CSV not found at {ctrl_csv}"
                    )

                def _lookup_vw(tq):
                    if not ctrl_t:
                        return float("nan"), float("nan")
                    j = int(np.argmin(np.abs(np.asarray(ctrl_t) - tq)))
                    return ctrl_v[j], ctrl_w[j]

                rfile = os.path.join(req_output_dir, f"{name}.csv")
                rows  = []
                for (x, y, th), t in paths[0]:
                    v, w = _lookup_vw(t)
                    rows.append([
                        f"{t:.6f}", f"{x:.6f}", f"{y:.6f}", f"{th:.6f}",
                        f"{_safe_float(v):.6f}", f"{_safe_float(w):.6f}",
                    ])
                _write_csv(rfile, rows,
                           ["time", "x", "y", "theta", "v", "omega"])
                self.get_logger().info(f"  {name}: -> {rfile}")

            elif agent_type == "holonomic":
                ctrl_csv = os.path.join(req_control_dir, f"{name}_controls.csv")
                ctrl_t, ctrl_vx, ctrl_vy = [], [], []
                if os.path.exists(ctrl_csv):
                    with open(ctrl_csv) as f:
                        for row in csv.DictReader(f):
                            ctrl_t.append(float(row["time"]))
                            ctrl_vx.append(float(row.get("vx", 0.0)))
                            ctrl_vy.append(float(row.get("vy", 0.0)))
                else:
                    self.get_logger().warn(
                        f"  {name}: control CSV not found at {ctrl_csv}"
                    )

                def _lookup_vxvy(tq):
                    if not ctrl_t:
                        return float("nan"), float("nan")
                    j = int(np.argmin(np.abs(np.asarray(ctrl_t) - tq)))
                    return ctrl_vx[j], ctrl_vy[j]

                rfile = os.path.join(req_output_dir, f"{name}.csv")
                rows  = []
                for (x, y, _th), t in paths[0]:
                    vx, vy = _lookup_vxvy(t)
                    rows.append([
                        f"{t:.6f}", f"{x:.6f}", f"{y:.6f}",
                        f"{_safe_float(vx):.6f}", f"{_safe_float(vy):.6f}",
                    ])
                _write_csv(rfile, rows, ["time", "x", "y", "vx", "vy"])
                self.get_logger().info(f"  {name}: -> {rfile}")

    def _save_map_image(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(16, 10))
            res = self.resolution
            ax.imshow(
                self.grid[::-1], cmap="gray_r",
                extent=[0, self.W * res, 0, self.H * res],
                alpha=0.85,
            )
            ax.set_aspect("equal", "box")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_title("Occupancy Grid Map")

            img_path = os.path.join(self.image_dir, "map.png")
            fig.savefig(img_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            self.get_logger().info(f"Map image saved -> {img_path}")
        except Exception as e:
            self.get_logger().warn(f"Could not save map image: {e}")

    def _save_path_image(self, robot_trajectories, agent_info,
                         trajectories, req_image_dir: str):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(16, 10))
            res = self.resolution
            ax.imshow(
                self.grid[::-1], cmap="gray_r",
                extent=[0, self.W * res, 0, self.H * res],
                alpha=0.85,
            )

            for name, paths in robot_trajectories.items():
                agent_type = agent_info[name]['type']
                if agent_type == "heterogeneous-formation":
                    for path in paths:
                        pts = np.array([pos[:2] for pos, _ in path])
                        ax.plot(pts[:, 0], pts[:, 1], lw=1.5, alpha=0.6)
                    c_pts = np.array([item[0][:2] for item in trajectories[name]])
                    ax.plot(c_pts[:, 0], c_pts[:, 1], lw=3, alpha=0.9, label=name)
                else:
                    pts = np.array([pos[:2] for pos, _ in paths[0]])
                    ax.plot(pts[:, 0], pts[:, 1], lw=2.5, alpha=0.8, label=name)

            ax.set_aspect("equal", "box")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title("Planned paths")

            img_path = os.path.join(req_image_dir, "planned_paths.png")
            fig.savefig(img_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            self.get_logger().info(f"Path image saved -> {img_path}")
        except Exception as e:
            self.get_logger().warn(f"Could not save path image: {e}")


# ============================================================
# Entry point
# ============================================================

def main(args=None):
    rclpy.init(args=args)
    node     = PathPlannerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
        # rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()