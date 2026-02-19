#!/usr/bin/env python3
"""
ROS2 Path Planner Node - UPDATED VERSION
Compatible with agents that store their own velocity/acceleration limits

Supports:
- Individual robots (diff-drive, holonomic) with internal limits
- Heterogeneous formations with per-robot limits and radii
- Automatic extraction of kinodynamic parameters from agent objects
- Request cancellation and thread management
"""

import os
import json
import time
import math
import traceback
import threading
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
from typing import Optional, Dict, List

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from nav_msgs.msg import Odometry
from path_planner_interface.msg import (
    PathPlannerRequest,
    RobotTrajectory,
    RobotTrajectoryArray,
    DiffDriveTrajectory,
    HoloTrajectory
)

# Import your planning core
from path_planner.si_rrt_enhanced_individual_kinodynamic import (
    SIRRT,
    OccupancyGrid,
)
from path_planner.agents1 import (
    DifferentialDriveAgent,
    HolonomicAgent,
    HeterogeneousFormationAgent
)
try:
    from heterogeneous_kinodynamic_formation_steering import *
    HETERO_AVAILABLE = True
except ImportError:
    HETERO_AVAILABLE = False

def quaternion_to_yaw(qx, qy, qz, qw) -> float:
    """Convert quaternion to yaw angle (rotation around Z axis)."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw


def find_Pstar(positions, yaw):
    """
    Calculate P_star (formation shape in centroid frame) from robot positions
    
    Args:
        positions: np.array of shape (N, 2) with robot positions
        yaw: formation orientation angle
    
    Returns:
        list of [x, y] positions in centroid frame
    """
    p = []
    centroid = np.mean(positions, axis=0)
    Rinv = np.array([[np.cos(yaw), np.sin(yaw)],
                     [-np.sin(yaw), np.cos(yaw)]])
    
    h, _ = np.shape(positions)
    
    for i in range(h):
        delta = positions[i] - centroid
        pistar = Rinv @ np.transpose(delta)
        p.append(pistar.tolist())
    
    return p


# def find_sx_sy(P_star, radius):
#     """
#     Calculate formation scaling factors based on P_star and desired radius
    
#     Args:
#         P_star: list of [x, y] positions in formation frame
#         radius: desired formation radius
    
#     Returns:
#         tuple (sx, sy) scaling factors with safety margin
#     """
#     D = []
#     p = 10000
#     I = 10000
#     k = 0
#     for i in P_star:
#         d = np.sqrt(i[0]**2 + i[1]**2)
#         if d < p:
#             p = d
#             I = k
#         k = k + 1
    
#     X, Y = P_star[int(I)]
    
#     sy = radius / (np.sqrt(X**2 + Y**2))
#     sx = radius / (np.sqrt(X**2 + Y**2))
#     sx *=1.2 # with Factor of Safety
#     sy *=1.2
#     if sx >= 1.8:
#         sx = 1.2
#     if sy >= 1.8:
#         sy = 1.2
    
#     return sx, sy  


# ============================================================
# Formation collision check
# ============================================================

def formation_collides(P_star, radius, centroid, theta, sx, sy, occupancy):
    """
    P_star   : list of [x, y]
    radius   : list of floats
    centroid : [x, y]
    theta    : float (rad)
    sx, sy   : floats
    occupancy: OccupancyGrid
    """
    c = math.cos(theta)
    s = math.sin(theta)

    for i in range(len(P_star)):
        px, py = P_star[i]

        # R(theta) * S(sx, sy) * p + centroid
        x = centroid[0] + c * (sx * px) - s * (sy * py)
        y = centroid[1] + s * (sx * px) + c * (sy * py)

        if occupancy.disc_collides(x, y, radius[i]):
            return True   # collision

    return False  # collision-free


# ============================================================
# Find best (sx, sy) in given ranges
# ============================================================

def find_percentile_scaling(
    P_star,
    radius,
    centroid,
    theta,
    grid,
    resolution,
    origin=(0.0, 0.0),
    sx_range=(0.5, 2.5),
    sy_range=(0.5, 2.5),
    step=0.02,
    percentile=20
):
    occupancy = OccupancyGrid(grid, resolution, origin)

    feasible = []   # (sx, sy, area)

    sx_vals = np.arange(sx_range[0], sx_range[1] + 1e-9, step)
    sy_vals = np.arange(sy_range[0], sy_range[1] + 1e-9, step)

    for sx in sx_vals:
        for sy in sy_vals:
            if not formation_collides(
                P_star, radius, centroid, theta, sx, sy, occupancy
            ):
                feasible.append((sx, sy, sx * sy))

    if not feasible:
        print("[WARN] No collision-free scaling found.")
        return None, None, None

    areas = np.array([a for _, _, a in feasible])
    target_area = np.percentile(areas, percentile)

    idx = np.argmin(np.abs(areas - target_area))
    best_sx, best_sy, best_area = feasible[idx]

    return best_sx, best_sy, target_area

# def find_best_scaling(
#     P_star,
#     radius,
#     centroid,
#     theta,
#     grid,
#     resolution,
#     origin=(0.0, 0.0),
#     sx_range=(0.8, 2.5),
#     sy_range=(0.8, 2.5),
#     step=0.02
# ):
#     """
#     Returns the (sx, sy) with maximum area sx*sy
#     that is collision-free.
#     """

#     occupancy = OccupancyGrid(grid, resolution, origin)

#     best_sx = sx_range[0]
#     best_sy = sy_range[0]
#     best_area = best_sx * best_sy

#     sx_vals = np.arange(sx_range[0], sx_range[1] + 1e-9, step)
#     sy_vals = np.arange(sy_range[0], sy_range[1] + 1e-9, step)

#     for sx in sx_vals:
#         for sy in sy_vals:
#             if not formation_collides(
#                 P_star, radius, centroid, theta, sx, sy, occupancy
#             ):
#                 area = sx * sy
#                 if area > best_area:
#                     best_area = area
#                     best_sx = sx
#                     best_sy = sy
#                 # print("found")
#             # else:
#                 # print("oh no")
#     return best_sx, best_sy


def find_robot_pose(theta, centroid, P_star, best_sx, best_sy):
    print("\nRobot positions:")
    try:
        c = math.cos(theta)
        s = math.sin(theta)
        for i, (px, py) in enumerate(P_star):
            x = centroid[0] + c * (best_sx * px) - s * (best_sy * py)
            y = centroid[1] + s * (best_sx * px) + c * (best_sy * py)
            print(f"  Robot {i}: ({x:.2f}, {y:.2f})")
    except Exception as e:
        print(f" Robot Pose can not be found: {e}")



class CancellableEvent:
    """Thread-safe event that can be checked for cancellation"""
    def __init__(self):
        self._cancelled = False
        self._lock = threading.Lock()
    
    def cancel(self):
        with self._lock:
            self._cancelled = True
    
    def is_cancelled(self):
        with self._lock:
            return self._cancelled
    
    def reset(self):
        with self._lock:
            self._cancelled = False


class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner_node')
        
        self.get_logger().info("="*60)
        self.get_logger().info("Path Planner Node starting (UPDATED with internal limits)...")
        self.get_logger().info("="*60)
        
        # ============================================================
        # Thread management
        # ============================================================
        self._planning_lock = threading.Lock()
        self._planning_thread = None
        self._cancel_event = CancellableEvent()
        self._shutdown = False
        
        # ============================================================
        # Planning state control
        # ============================================================
        self._planning_active = False
        self._current_request_id = 0
        self._pending_request = None
        self._pending_request_lock = threading.Lock()

        # ============================================================
        # Declare Parameters (with defaults)
        # ============================================================
        self.declare_parameter('map_file_path', '')
        self.declare_parameter('config_file_path', '')
        self.declare_parameter('resolution', 0.1)
        self.declare_parameter('max_velocity', 0.6)
        self.declare_parameter('time_horizon', 100.0)
        self.declare_parameter('max_iter', 2000)
        self.declare_parameter('d_max', 0.1)
        self.declare_parameter('goal_sample_rate', 0.22)
        self.declare_parameter('neighbor_radius', 1.5)
        self.declare_parameter('precision', 4)
        self.declare_parameter('n_steer', 8)
        self.declare_parameter('t_steer', 0.8)
        self.declare_parameter('t_steer_min', 0.5)
        self.declare_parameter('t_steer_safety_factor', 1.5)
        self.declare_parameter('dt', 0.05)
        self.declare_parameter('seed', 9)
        self.declare_parameter('debug', False)
        self.declare_parameter('show_initial_map', True)
        self.declare_parameter('save_paths', True)
        self.declare_parameter('save_base_dir', '')
        self.declare_parameter('save_image', True)
        self.declare_parameter('save_video', True)
        
        # Default robot parameters (fallbacks only)
        self.declare_parameter('default_robot_radius', 0.2)
        self.declare_parameter('default_max_velocity', 0.2)
        self.declare_parameter('default_max_acceleration', 0.4)
        self.declare_parameter('default_max_angular_velocity', 0.5)
        self.declare_parameter('default_max_angular_acceleration', 1.0)
        self.declare_parameter('odom_timeout', 2.0)
        
        # Get parameters
        self.resolution = self.get_parameter('resolution').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.time_horizon = self.get_parameter('time_horizon').value
        self.max_iter = self.get_parameter('max_iter').value
        self.d_max = self.get_parameter('d_max').value
        self.goal_sample_rate = self.get_parameter('goal_sample_rate').value
        self.neighbor_radius = self.get_parameter('neighbor_radius').value
        self.precision = self.get_parameter('precision').value
        self.n_steer = self.get_parameter('n_steer').value
        self.t_steer = self.get_parameter('t_steer').value
        self.t_steer_min = self.get_parameter('t_steer_min').value
        self.t_steer_safety_factor = self.get_parameter('t_steer_safety_factor').value
        self.dt = self.get_parameter('dt').value
        self.seed = self.get_parameter('seed').value
        self.debug = self.get_parameter('debug').value
        self.show_initial_map = self.get_parameter('show_initial_map').value
        self.save_paths = self.get_parameter('save_paths').value
        self.save_image = self.get_parameter('save_image').value
        self.save_video = self.get_parameter('save_video').value
        
        # Default fallbacks
        self.default_robot_radius = self.get_parameter('default_robot_radius').value
        self.default_max_velocity = self.get_parameter('default_max_velocity').value
        self.default_max_acceleration = self.get_parameter('default_max_acceleration').value
        self.default_max_angular_velocity = self.get_parameter('default_max_angular_velocity').value
        self.default_max_angular_acceleration = self.get_parameter('default_max_angular_acceleration').value
        self.odom_timeout = self.get_parameter('odom_timeout').value

        self.get_logger().info(f"Show initial map: {self.show_initial_map}")
        self.get_logger().info(f"Max iterations: {self.max_iter}")
        self.get_logger().info(f"Time horizon: {self.time_horizon}s")
        self.get_logger().info(f"N_steer: {self.n_steer}, T_steer: {self.t_steer}s")

        # ============================================================
        # Resolve Directories and Files
        # ============================================================
        self.get_logger().info("="*60)
        self.get_logger().info("Resolving directories and files...")
        self.get_logger().info("="*60)
        
        self.save_base_dir = self._resolve_dir(
            param_name='save_base_dir',
            default_pkg='path_planner',
            default_relative_path='saved_paths'
        )

        self.map_file_path = self._resolve_file(
            param_name='map_file_path',
            default_pkg='path_planner',
            default_relative_path='data/restaurant.npy'
        )
        
        self.config_file_path = self._resolve_file(
            param_name='config_file_path',
            default_pkg='chatty',
            default_relative_path='config/robot_config_restaurant.json'
        )
        
        # ============================================================
        # Load Robot Config
        # ============================================================
        self.get_logger().info("="*60)
        self.get_logger().info("Loading robot configuration...")
        self.get_logger().info("="*60)
        
        self.config_data = self._load_json(self.config_file_path)
        
        # Load robot names from config
        self.robot_names = self.config_data.get('robot_names', [])
        if not self.robot_names:
            self.get_logger().warn("No robot_names found in config file")
        else:
            self.get_logger().info(f"Loaded {len(self.robot_names)} robots: {self.robot_names}")
        
        # ============================================================
        # Robot Offsets (default positions if odom not available)
        # ============================================================
        self.offsets = {
            "robot1": (4.0, 6.0, 0.0, 0.0),
            "robot2": (4.0, 4.0, 0.0, 0.0),
            "robot3": (5.0, 5.0, 0.0, 0.0),
            "robot4": (16.0, 8.0, 0.0, 0.0),
            "robot5": (16.0, 10.0, 0.0, 0.0),
            "robot6": (15.0, 9.0, 0.0, 0.0),
        }
        
        # ============================================================
        # Load Map
        # ============================================================
        self.get_logger().info("="*60)
        self.get_logger().info("Loading map...")
        self.get_logger().info("="*60)
        
        self.grid, self.static_grid, self.bounds, self.H, self.W = self._load_map(
            self.map_file_path, 
            self.resolution
        )
        
        # ============================================================
        # Robot pose tracking from odometry
        # ============================================================
        self.robot_current_poses = {}
        self.robot_odom_subs = {}
        self.robot_odom_received = {}
        
        # ============================================================
        # Subscribe to odometry for all robots in config
        # ============================================================
        self.get_logger().info("="*60)
        self.get_logger().info("Setting up odometry subscribers...")
        self.get_logger().info("="*60)
        
        for idx, robot_name in enumerate(self.robot_names):
            topic_name = f"/{robot_name}/odom_world"
            
            sub = self.create_subscription(
                Odometry,
                topic_name,
                lambda msg, rn=robot_name: self._odom_callback(msg, rn),
                10
            )
            self.robot_odom_subs[robot_name] = sub
            self.robot_odom_received[robot_name] = False
            self.get_logger().info(f"Subscribed to {topic_name}")
            
            # Initialize using offsets as fallback
            offset_x, offset_y, offset_yaw, timestamp = self.offsets.get(robot_name, (0.0, 0.0, 0.0, 0.0))
            self.robot_current_poses[robot_name] = {
                'x': offset_x,
                'y': offset_y,
                'yaw': offset_yaw,
                'timestamp': self.get_clock().now()
            }
            
            self.get_logger().debug(
                f"Initialized {robot_name} default pose: "
                f"x={self.robot_current_poses[robot_name]['x']:.2f}, "
                f"y={self.robot_current_poses[robot_name]['y']:.2f}, "
                f"yaw={self.robot_current_poses[robot_name]['yaw']:.2f}"
            )

        # ============================================================
        # Show Initial Map if enabled
        # ============================================================
        if self.show_initial_map:
            self.get_logger().info("="*60)
            self.get_logger().info("Displaying initial map...")
            self.get_logger().info("="*60)
            self._display_initial_map()
        
        # ============================================================
        # Publisher & Subscriber
        # ============================================================
        self.get_logger().info("="*60)
        self.get_logger().info("Setting up ROS2 publishers and subscribers...")
        self.get_logger().info("="*60)
        
        self.path_pub = self.create_publisher(
            RobotTrajectoryArray,
            '/path_planner/paths',
            10
        )
        self.get_logger().info("Publisher created: /path_planner/paths")
        
        self.path_plan_sub = self.create_subscription(
            PathPlannerRequest,
            '/path_planner/request',
            self.path_plan_callback,
            10
        )
        self.get_logger().info("Subscriber created: /path_planner/request")
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'completed': 0,
            'failed': 0,
            'cancelled': 0
        }
        
        self.get_logger().info('='*60)
        self.get_logger().info('Path Planner Node initialized and ready')
        self.get_logger().info('='*60)
    
    def _load_map(self, map_file_path, resolution):
        """Load the occupancy grid map from file"""
        self.get_logger().info(f'Loading map from: {map_file_path}')
        try:
            grid_raw = np.load(map_file_path)
            self.get_logger().info(f'Raw map loaded, shape: {grid_raw.shape}, dtype: {grid_raw.dtype}')
            
            grid = (grid_raw > 0).astype(np.uint8)
            self.get_logger().info(f'Binary map created, occupied cells: {np.sum(grid)}')
            
            H, W = grid.shape
            
            bounds = (
                np.array([0.0, 0.0]), 
                np.array([W * resolution, H * resolution])
            )
            static_grid = OccupancyGrid(grid, resolution)
            
            self.get_logger().info(f'Map loaded: {W}x{H} @ {resolution}m/cell')
            self.get_logger().info(f'Workspace bounds: {bounds}')
            
            return grid, static_grid, bounds, H, W
            
        except Exception as e:
            self.get_logger().error("="*60)
            self.get_logger().error(f'Failed to load map: {e}')
            self.get_logger().error("="*60)
            raise

    def _resolve_dir(self, param_name: str, default_pkg: str, default_relative_path: str) -> str:
        """Resolve directory path from parameter or use default from package."""
        param_value = self.get_parameter(param_name).value

        if param_value:
            abs_path = os.path.abspath(param_value)
            os.makedirs(abs_path, exist_ok=True)
            self.get_logger().info(f"Using {param_name} from parameter: {abs_path}")
            return abs_path

        try:
            pkg_path = get_package_share_directory(default_pkg)
            default_path = os.path.join(pkg_path, default_relative_path)
            os.makedirs(default_path, exist_ok=True)
            self.get_logger().info(f"Using default {param_name}: {default_path}")
            return default_path

        except Exception as e:
            self.get_logger().error("="*60)
            self.get_logger().error(f"Failed to resolve {param_name}: {e}")
            self.get_logger().error("="*60)
            raise

    def _resolve_file(self, param_name: str, default_pkg: str, default_relative_path: str) -> str:
        """Resolve file path from parameter or use default from package."""
        param_value = self.get_parameter(param_name).value
        
        if param_value:
            if os.path.exists(param_value):
                self.get_logger().info(f"Using {param_name} from parameter: {param_value}")
                return param_value
            else:
                self.get_logger().warn(f"{param_name} file not found: {param_value}, using default")
        
        try:
            pkg_path = get_package_share_directory(default_pkg)
            default_path = os.path.join(pkg_path, default_relative_path)
            
            if os.path.exists(default_path):
                self.get_logger().info(f"Using default {param_name}: {default_path}")
                return default_path
            else:
                self.get_logger().error(f"Default {param_name} file not found: {default_path}")
                raise FileNotFoundError(f"Default {param_name} file not found: {default_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to resolve {param_name}: {e}")
            raise
    
    def _load_json(self, path: str) -> dict:
        """Load JSON file"""
        if not os.path.exists(path):
            self.get_logger().warn(f"JSON file not found: {path}")
            return {}
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self.get_logger().info(f"Loaded JSON config from: {path}")
            return data
        except Exception as e:
            self.get_logger().error(f"Failed to load JSON {path}: {e}")
            return {}
    
    def _odom_callback(self, msg: Odometry, robot_name: str):
        """Store current robot poses from odometry"""
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        
        yaw = quaternion_to_yaw(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        )
        
        self.robot_current_poses[robot_name] = {
            'x': float(position.x),
            'y': float(position.y),
            'yaw': float(yaw),
            'timestamp': self.get_clock().now()
        }
        
        if not self.robot_odom_received.get(robot_name, False):
            self.robot_odom_received[robot_name] = True
            self.get_logger().info(f"[ODOM] First message received from {robot_name}")
        
        self.get_logger().debug(
            f"[ODOM] {robot_name}: x={position.x:.3f}, y={position.y:.3f}, yaw={yaw:.3f}"
        )

    def _display_initial_map(self):
        """Display the map on startup with robot positions and save to file"""
        try:
            plt.ioff()
            fig, ax = plt.subplots(figsize=(10, 8))
            
            ax.imshow(
                self.grid[::-1],
                cmap="gray_r",
                extent=[0, self.W * self.resolution, 0, self.H * self.resolution],
                alpha=0.9
            )
            
            ax.set_xlim(0, self.W * self.resolution)
            ax.set_ylim(0, self.H * self.resolution)
            ax.set_aspect("equal")
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_title("Path Planner - Initial Map with Robot Positions")
            
            colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
            
            for idx, robot_name in enumerate(self.robot_names):
                if robot_name in self.robot_current_poses:
                    pose = self.robot_current_poses[robot_name]
                    x = pose['x']
                    y = pose['y']
                    yaw = pose['yaw']
                    
                    color = colors[idx % len(colors)]
                    radius = self.default_robot_radius
                    
                    circle = Circle((x, y), radius, color=color, alpha=0.5)
                    ax.add_patch(circle)
                    
                    circle_outline = Circle((x, y), radius, fill=False, 
                                          edgecolor=color, linewidth=2)
                    ax.add_patch(circle_outline)
                    
                    arrow_length = radius * 1.5
                    dx = arrow_length * np.cos(yaw)
                    dy = arrow_length * np.sin(yaw)
                    ax.arrow(x, y, dx, dy, 
                            head_width=radius*0.5, head_length=radius*0.5,
                            fc=color, ec=color, linewidth=2, alpha=0.8)
                    
                    ax.text(x, y - radius - 0.3, robot_name, 
                           fontsize=9, ha='center', va='top', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='white', edgecolor=color, alpha=0.8))
                    
                    coord_text = f"({x:.2f}, {y:.2f})"
                    ax.text(x, y + radius + 0.3, coord_text, 
                           fontsize=7, ha='center', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.2', 
                                   facecolor='lightyellow', alpha=0.7))
            
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=colors[i % len(colors)], alpha=0.5, 
                      edgecolor=colors[i % len(colors)], label=robot_name)
                for i, robot_name in enumerate(self.robot_names)
            ]
            ax.legend(handles=legend_elements, loc='upper right', 
                     fontsize=8, title='Robots')
            
            base_dir = Path(self.save_base_dir)
            base_dir.mkdir(exist_ok=True)
            
            existing = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
            if existing:
                ids = [int(d.name.split("_")[1]) for d in existing if d.name.split("_")[1].isdigit()]
                run_id = max(ids) + 1 if ids else 1
            else:
                run_id = 1
            
            run_dir = base_dir / f"run_{run_id:04d}"
            run_dir.mkdir()

            initial_map_path = run_dir / "initial_map.png"
            fig.savefig(initial_map_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            self.get_logger().info(f'Initial map saved to: {initial_map_path}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to display initial map: {e}')
            traceback.print_exc()
    
    def path_plan_callback(self, msg: PathPlannerRequest):
        """Handle incoming path planning requests"""
        
        self.stats['total_requests'] += 1
        request_id = self.stats['total_requests']
        
        self.get_logger().info("="*60)
        self.get_logger().info(f"Received planning request #{request_id}")
        self.get_logger().info("="*60)
        
        if not hasattr(msg, 'plan_json') or not msg.plan_json:
            self.get_logger().error("Missing or empty 'plan_json' field")
            self.stats['failed'] += 1
            return
        
        try:
            plan_json = json.loads(msg.plan_json)
            self.get_logger().info(f"Plan JSON parsed successfully")
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Invalid JSON in plan request: {e}")
            self.stats['failed'] += 1
            return
        
        if not isinstance(plan_json, dict) or not plan_json:
            self.get_logger().error("plan_json must be a non-empty dictionary")
            self.stats['failed'] += 1
            return
        
        with self._planning_lock:
            if self._planning_thread is not None and self._planning_thread.is_alive():
                self.get_logger().warn("="*60)
                self.get_logger().warn(f"Cancelling previous request to start #{request_id}")
                self.get_logger().warn("="*60)
                
                self._cancel_event.cancel()
                self._planning_thread.join(timeout=2.0)
                
                if self._planning_thread.is_alive():
                    self.get_logger().warn("Previous planning thread did not stop gracefully")
                
                self.stats['cancelled'] += 1
            
            self._cancel_event.reset()
            self._current_request_id = request_id
            
            self._planning_thread = threading.Thread(
                target=self._planning_thread_worker,
                args=(request_id, plan_json),
                daemon=True
            )
            self._planning_thread.start()
            
        self.get_logger().info(f"Started planning thread for request #{request_id}")
    
    def _planning_thread_worker(self, request_id: int, plan_json: dict):
        """Worker thread for path planning"""
        
        try:
            self.get_logger().info(f"[Thread-{request_id}] Planning started")
            
            if self._cancel_event.is_cancelled():
                self.get_logger().warn(f"[Thread-{request_id}] Cancelled before planning")
                return
            
            success = self._plan_paths(plan_json, request_id)
            
            if self._cancel_event.is_cancelled():
                self.get_logger().warn(f"[Thread-{request_id}] ✗ Cancelled during planning")
                return
            
            if success:
                self.stats['completed'] += 1
                self.get_logger().info(f"[Thread-{request_id}] ✓ Completed successfully")
            else:
                self.stats['failed'] += 1
                self.get_logger().error(f"[Thread-{request_id}] ✗ Failed")
            
            self.get_logger().info(
                f"Stats: {self.stats['completed']} completed, "
                f"{self.stats['failed']} failed, "
                f"{self.stats['cancelled']} cancelled"
            )
            
        except Exception as e:
            self.get_logger().error(f"[Thread-{request_id}] Exception: {e}")
            traceback.print_exc()
            self.stats['failed'] += 1

    def extend_traj_to_T(self, traj, T):
        """
        Extend a trajectory so that the robot stays at its final configuration
        until the time horizon T (stationary obstacle).
        """
        if not traj:
            return traj

        q_last, t_last = traj[-1]

        if t_last >= T - 1e-9:
            return traj

        traj = list(traj)
        traj.append((np.asarray(q_last, dtype=float).copy(), float(T)))
        return traj


    def _plan_paths(self, plan_json: dict, request_id: int) -> bool:
        """Plan paths for all robots/formations in the request"""
        
        try:
            self.get_logger().info("="*60)
            self.get_logger().info(f"[Thread-{request_id}] Parsing plan JSON...")
            self.get_logger().info("="*60)
            
            agents_list = self._parse_plan_json(plan_json)
            
            if not agents_list:
                self.get_logger().warn(f"[Thread-{request_id}] No valid agents extracted from plan")
                return False
            
            if self._cancel_event.is_cancelled():
                return False
            
            self.get_logger().info(f"[Thread-{request_id}] Planning for {len(agents_list)} agents/formations")
            
            self.get_logger().info("="*60)
            self.get_logger().info(f"[Thread-{request_id}] Starting sequential path planning...")
            self.get_logger().info("="*60)
            
            dynamic_obstacles = []
            all_results = []
            
            for agent_entry in agents_list:
                if self._cancel_event.is_cancelled():
                    self.get_logger().warn(f"[Thread-{request_id}] Cancelled during agent planning")
                    return False
                
                name = agent_entry['name']
                agent = agent_entry['agent']
                start = agent_entry['start']
                goal = agent_entry['goal']
                agent_type = agent_entry['type']
                robot_names = agent_entry['robot_names']
                robot_types = agent_entry['robot_types']
                
                self.get_logger().info(f"\n[Thread-{request_id}] >>> Planning for {name} ({agent_type})...")
                self.get_logger().info(f"  Start: {start}")
                self.get_logger().info(f"  Goal: {goal}")
                
                traj = self._plan_single_agent(
                    request_id=request_id,
                    name=name,
                    agent=agent,
                    start=start,
                    goal=goal,
                    agent_type=agent_type,
                    dynamic_obstacles=dynamic_obstacles
                )
                
                if self._cancel_event.is_cancelled():
                    self.get_logger().warn(f"[Thread-{request_id}] Cancelled after planning {name}")
                    return False
                
                if traj is None:
                    self.get_logger().error(f"[Thread-{request_id}] Planning failed for {name}, skipping...")
                    continue
                
                all_results.append({
                    'name': name,
                    'agent': agent,
                    'traj': traj,
                    'robot_names': robot_names,
                    'robot_types': robot_types,
                    'color': agent_entry.get('color', 'gray')
                })
                
                traj_for_obstacle = [(item[0], item[1]) for item in traj]
                dynamic_obstacles.append({
                    'trajectory': traj_for_obstacle,
                    'agent': agent
                })

                # traj_for_obstacle = [(item[0], item[1]) for item in traj]

                # # IMPORTANT: keep robot stationary until time horizon
                # traj_for_obstacle = self.extend_traj_to_T(traj_for_obstacle, 60.0)

                # dynamic_obstacles.append({
                #     "trajectory": traj_for_obstacle,
                #     "agent": agent
                # })


                self.get_logger().info(f"  ✓ Planned {len(traj)} waypoints, arrival: {traj[-1][1]:.2f}s")
            
            if self._cancel_event.is_cancelled():
                self.get_logger().warn(f"[Thread-{request_id}] Cancelled before publishing")
                return False
            
            if all_results:
                self.get_logger().info("="*60)
                self.get_logger().info(f"[Thread-{request_id}] Publishing paths for {len(all_results)} agents")
                self.get_logger().info("="*60)
                
                self._publish_paths(all_results)
                
                if self.save_paths:
                    self._save_trajectories(all_results)
                
                if self.save_image:
                    self._save_path_image(all_results)
                
                if self.save_video:
                    self._save_path_video(all_results)
                
                return True
            else:
                self.get_logger().warn(f"[Thread-{request_id}] No paths were successfully planned")
                return False
        
        except Exception as e:
            self.get_logger().error(f"[Thread-{request_id}] Planning error: {e}")
            traceback.print_exc()
            return False
    
    def _parse_plan_json(self, plan_json: dict) -> list:
        """
        Parse plan_json into agent configurations with INTERNAL LIMITS
        
        Returns list of dicts with keys:
        - name, agent, start, goal, type
        - robot_names, robot_types
        - color
        """
        agents_list = []
        
        for group_name, group_data in plan_json.items():
            self.get_logger().debug(f"Processing group: {group_name}")
            
            if group_name.startswith('F') and isinstance(group_data, dict):
                formation_config = self._parse_formation_group(group_name, group_data)
                if formation_config:
                    agents_list.append(formation_config)
            
            elif group_name.startswith('R') and isinstance(group_data, list):
                for robot_entry in group_data:
                    robot_config = self._parse_individual_robot(group_name, robot_entry)
                    if robot_config:
                        agents_list.append(robot_config)
        
        if agents_list:
            self.get_logger().info("="*60)
            self.get_logger().info("AGENTS LIST (with internal limits):")
            self.get_logger().info("="*60)
            self.get_logger().info("agents = [")
            
            for agent_config in agents_list:
                name = agent_config['name']
                agent = agent_config['agent']
                start = agent_config['start']
                goal = agent_config['goal']
                agent_type = agent_config['type']
                color = agent_config.get('color', 'gray')
                
                self.get_logger().info(f'    ("{name}", {agent.__class__.__name__},')
                self.get_logger().info(f'     {start},  # start')
                self.get_logger().info(f'     {goal},  # goal')
                self.get_logger().info(f'     "{agent_type}"),  # color={color}')
                
                # Print agent's internal limits
                if agent_type == "heterogeneous-formation":
                    self.get_logger().info(f'     # Per-robot limits:')
                    self.get_logger().info(f'     #   radii: {agent.radii}')
                    self.get_logger().info(f'     #   v_max: {agent.v_max_list}')
                    self.get_logger().info(f'     #   omega_max: {agent.omega_max_list}')
                    self.get_logger().info(f'     #   a_max: {agent.a_max_list}')
                    self.get_logger().info(f'     #   alpha_max: {agent.alpha_max_list}')
                elif hasattr(agent, 'v_max'):
                    limit_str = f'     # Limits: v_max={agent.v_max:.3f}'
                    if hasattr(agent, 'omega_max'):
                        limit_str += f', omega_max={agent.omega_max:.3f}'
                    limit_str += f', a_max={agent.a_max:.3f}'
                    self.get_logger().info(limit_str)
                
                self.get_logger().info("")
            
            self.get_logger().info("]")
            self.get_logger().info("="*60)
        
        return agents_list
    
    def _parse_formation_group(self, group_name: str, group_data: dict) -> Optional[dict]:
        """Parse formation group with per-robot limits stored in agent"""
        
        try:
            centroid_x = float(group_data.get('centroid_x', 0.0))
            centroid_y = float(group_data.get('centroid_y', 0.0))
            formation_yaw = float(group_data.get('formation_yaw', 0.0))
            desired_radius = float(group_data.get('desired_radius', 0.5))
            
            robots = group_data.get('robots', [])
            if not robots:
                self.get_logger().warn(f"{group_name}: No robots in formation")
                return None
            
            robot_names = []
            robot_types = []
            robot_positions = []
            robot_radii = []
            robot_colors = []
            v_max_list = []
            w_max_list = []
            a_max_list = []
            alpha_max_list = []
            
            for robot_name in robots:
                robot_cfg = self.config_data.get('path_planner', {}).get(robot_name, {})
                pose = self._get_robot_pose(robot_name)
                
                robot_names.append(robot_name)
                robot_positions.append([pose['x'], pose['y']])
                robot_radii.append(robot_cfg.get('radius', self.default_robot_radius))
                robot_colors.append(robot_cfg.get('colour', 'gray'))
                
                rtype = robot_cfg.get('type', 'Differential Drive Robot').lower()
                if 'holonomic' in rtype or 'omnidirectional' in rtype:
                    robot_types.append('holonomic')
                    v_max_list.append(robot_cfg.get('max_linear_velocity_x', self.default_max_velocity))
                    w_max_list.append(0.0)
                    a_max_list.append(robot_cfg.get('max_acceleration', self.default_max_acceleration))
                    alpha_max_list.append(0.0)
                else:
                    robot_types.append('diff-drive')
                    v_max_list.append(robot_cfg.get('max_linear_velocity_x', self.default_max_velocity))
                    w_max_list.append(robot_cfg.get('max_angular_velocity_z', self.default_max_angular_velocity))
                    a_max_list.append(robot_cfg.get('max_acceleration', self.default_max_acceleration))
                    alpha_max_list.append(robot_cfg.get('max_angular_acceleration', self.default_max_angular_acceleration))
            
            if not robot_names:
                return None
            
            start_pose_yaw = 0.0
            positions_array = np.array(robot_positions)
            P_star = find_Pstar(positions_array, start_pose_yaw)
            formation_centroid = np.mean(positions_array, axis=0)
            
            # Create agent with INTERNAL limits
            agent = HeterogeneousFormationAgent(
                P_star=P_star,
                robot_types=robot_types,
                radius=robot_radii,         # per-robot radii
                v_max=v_max_list,           # per-robot v_max
                omega_max=w_max_list,       # per-robot omega_max
                a_max=a_max_list,           # per-robot a_max
                alpha_max=alpha_max_list,   # per-robot alpha_max
            )

            start = np.array([
                formation_centroid[0],
                formation_centroid[1],
                start_pose_yaw,
                1.0,
                1.0
            ])
            
            # sx, sy = find_sx_sy(P_star, desired_radius)

            sx, sy, _= find_percentile_scaling(
                P_star=P_star,
                radius=robot_radii,
                centroid=[centroid_x, centroid_y],
                theta=formation_yaw,
                grid=self.grid,
                resolution=self.resolution,
                # origin=(0.0, 0.0),
                # sx_range=(0.5, 2.5),
                # sy_range=(0.5, 2.5),
                # step=0.02,
                # percentile=20
            )
            
            goal = np.array([
                centroid_x,
                centroid_y,
                formation_yaw,
                sx,
                sy
            ])

            
            find_robot_pose(theta=formation_yaw, centroid=[centroid_x, centroid_y], P_star=P_star, best_sx=sx, best_sy=sy)


            self.get_logger().info(f"  Formation {group_name}: {len(robot_names)} robots")
            self.get_logger().info(f"    P_star: {P_star}")
            self.get_logger().info(f"    Types: {robot_types}")
            self.get_logger().info(f"    Radii: {robot_radii}")
            self.get_logger().info(f"    v_max: {v_max_list}")
            self.get_logger().info(f"    omega_max: {w_max_list}")
            self.get_logger().info(f"    formation_yaw: {formation_yaw}")
            self.get_logger().info(f"    centroid : {centroid_x, centroid_y}")
            self.get_logger().info(f"    sx: {sx}")
            self.get_logger().info(f"    sy: {sy}")
            
            for i, robot_name in enumerate(robot_names):
                self.get_logger().info(
                    f"    - {robot_name}: type={robot_types[i]}, color={robot_colors[i]}, "
                    f"radius={robot_radii[i]:.3f}, v_max={v_max_list[i]:.3f}"
                )
            
            return {
                'name': group_name,
                'agent': agent,
                'start': start,
                'goal': goal,
                'type': 'heterogeneous-formation',
                'robot_names': robot_names,
                'robot_types': robot_types,
                'color': robot_colors[0] if robot_colors else 'gray'
            }
            
        except Exception as e:
            self.get_logger().error(f"Failed to parse formation {group_name}: {e}")
            traceback.print_exc()
            return None
    
    def _parse_individual_robot(self, group_name: str, robot_data: dict) -> Optional[dict]:
        """Parse individual robot with INTERNAL limits"""
        
        try:
            robot_name = robot_data.get('robot')
            if not robot_name:
                self.get_logger().warn(f"{group_name}: Missing robot name")
                return None
            
            robot_cfg = self.config_data.get('path_planner', {}).get(robot_name, {})
            
            start_pose = self._get_robot_pose(robot_name)
            
            goal_x = robot_data.get('x')
            goal_y = robot_data.get('y')
            if goal_x is None or goal_y is None:
                self.get_logger().error(f"{robot_name}: Missing goal x or y")
                return None
            
            goal_yaw = robot_data.get('yaw', 0.0)
            
            rtype = robot_cfg.get('type', 'Differential Drive Robot').lower()
            radius = robot_cfg.get('radius', self.default_robot_radius)
            color = robot_cfg.get('colour', 'gray')
            
            if 'holonomic' in rtype or 'omnidirectional' in rtype:
                agent_type = 'holonomic'
                
                # Create agent with INTERNAL limits
                agent = HolonomicAgent(
                    radius=radius,
                    v_max=robot_cfg.get('max_linear_velocity_x', self.default_max_velocity),
                    a_max=robot_cfg.get('max_acceleration', self.default_max_acceleration)
                )
                
                start = np.array([start_pose['x'], start_pose['y']])
                goal = np.array([goal_x, goal_y])
                
            else:
                agent_type = 'diff-drive'
                
                # Create agent with INTERNAL limits
                agent = DifferentialDriveAgent(
                    radius=radius,
                    v_max=robot_cfg.get('max_linear_velocity_x', self.default_max_velocity),
                    omega_max=robot_cfg.get('max_angular_velocity_z', self.default_max_angular_velocity),
                    a_max=robot_cfg.get('max_acceleration', self.default_max_acceleration),
                    alpha_max=robot_cfg.get('max_angular_acceleration', self.default_max_angular_acceleration)
                )
                
                start = np.array([start_pose['x'], start_pose['y'], start_pose['yaw']])
                goal = np.array([goal_x, goal_y, goal_yaw])
            
            self.get_logger().info(f"  Robot {robot_name} ({agent_type})")
            self.get_logger().info(f"    - color={color}, radius={radius:.3f}")
            self.get_logger().info(f"    - v_max={agent.v_max:.3f}, a_max={agent.a_max:.3f}")
            if hasattr(agent, 'omega_max'):
                self.get_logger().info(f"    - omega_max={agent.omega_max:.3f}")
            
            return {
                'name': robot_name,
                'agent': agent,
                'start': start,
                'goal': goal,
                'type': agent_type,
                'robot_names': [robot_name],
                'robot_types': [agent_type],
                'color': color
            }
            
        except Exception as e:
            self.get_logger().error(f"Failed to parse robot: {e}")
            traceback.print_exc()
            return None
    
    def _get_robot_pose(self, robot_name: str) -> dict:
        """Get robot pose from odometry with timeout"""
        
        if robot_name in self.robot_current_poses:
            pose_data = self.robot_current_poses[robot_name]
            return pose_data
        
        self.get_logger().warn(f"No recent odom for {robot_name}, using default (0, 0, 0)")
        return {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
    
    def _plan_single_agent(self, request_id: int, name: str, agent, start: np.ndarray, goal: np.ndarray,
                          agent_type: str, dynamic_obstacles: list):
        """
        Plan path for single agent using SI-RRT with kinodynamic constraints
        EXTRACTS limits directly from agent object
        """
        
        try:
            # ============================================================
            # Build kinodynamic parameters from AGENT'S INTERNAL LIMITS
            # ============================================================
            if agent_type == "heterogeneous-formation" and HETERO_AVAILABLE:
                kino_params = {
                    'robot_types': agent.robot_types,
                    'v_max': agent.v_max_list,
                    'w_max': agent.omega_max_list,
                    'a_max': agent.a_max_list,
                    'alpha_max': agent.alpha_max_list,
                    'N_steer': self.n_steer,
                    'T_steer': self.t_steer,
                    'max_iter': self.max_iter,
                }
                use_kino = True
                use_hetero = True
                
                self.get_logger().info(f"  Formation kinodynamic params:")
                self.get_logger().info(f"    v_max: {kino_params['v_max']}")
                self.get_logger().info(f"    w_max: {kino_params['w_max']}")
                self.get_logger().info(f"    a_max: {kino_params['a_max']}")
                self.get_logger().info(f"    alpha_max: {kino_params['alpha_max']}")
                
            elif agent_type == "diff-drive":
                kino_params = {
                    'v_max': agent.v_max,
                    'omega_max': agent.omega_max,
                    'a_max': agent.a_max,
                    'alpha_max': agent.alpha_max,
                    'dt': self.dt,
                }
                use_kino = True
                use_hetero = False
                
                self.get_logger().info(f"  DD kinodynamic params: v_max={agent.v_max:.3f}, omega_max={agent.omega_max:.3f}")
                
            elif agent_type == "holonomic":
                kino_params = {
                    'v_max': agent.v_max,
                    'a_max': agent.a_max,
                    'dt': self.dt,
                }
                use_kino = True
                use_hetero = False
                
                self.get_logger().info(f"  Holonomic kinodynamic params: v_max={agent.v_max:.3f}, a_max={agent.a_max:.3f}")
            else:
                use_kino = False
                use_hetero = False
                kino_params = None
            
            planner = SIRRT(
                agent_model=agent,
                max_velocity=self.max_velocity,
                workspace_bounds=self.bounds,
                static_grid=self.static_grid,
                time_horizon=self.time_horizon,
                max_iter=self.max_iter,
                d_max=self.d_max,
                goal_sample_rate=self.goal_sample_rate,
                neighbor_radius=self.neighbor_radius,
                precision=self.precision,
                seed=self.seed,
                debug=self.debug,
                use_kinodynamic=use_kino,
                kinodynamic_params=kino_params,
            )
            
            self.get_logger().info(f"[Thread-{request_id}] Starting planner for {name}...")

            if use_hetero and HETERO_AVAILABLE:
                print(f"  Using heterogeneous formation steering")
                HeterogeneousKinodynamicFormationSteering(
                    P_star=agent.P_star,
                    robot_types=agent.robot_types,
                    v_max=agent.v_max_list,
                    w_max=agent.omega_max_list,
                    a_max=agent.a_max_list,
                    alpha_max=agent.alpha_max_list,
                    N_steer=8,
                    T_steer=0.8,
                )

            traj = planner.plan(start, goal, dynamic_obstacles)
            
            if self._cancel_event.is_cancelled():
                self.get_logger().warn(f"[Thread-{request_id}] Cancelled after planning {name}")
                return None
            
            if traj is None:
                self.get_logger().error(f"[Thread-{request_id}] Planner returned None for {name}")
            
            return traj
            
        except Exception as e:
            self.get_logger().error(f"[Thread-{request_id}] Planning exception for {name}: {e}")
            traceback.print_exc()
            return None
    
    def _publish_paths(self, all_results: list):
        """Publish computed paths using new message types"""
        
        msg = RobotTrajectoryArray()
        
        for result in all_results:
            name = result['name']
            agent = result['agent']
            traj = result['traj']
            robot_names = result['robot_names']
            robot_types = result['robot_types']


            # ADD THIS DEBUG BLOCK
            self.get_logger().info(f"[DEBUG] Trajectory structure for {name}:")
            self.get_logger().info(f"  - Number of waypoints: {len(traj)}")
            if len(traj) > 0:
                self.get_logger().info(f"  - First waypoint length: {len(traj[0])}")
                self.get_logger().info(f"  - First waypoint: {traj[0]}")
                if len(traj[0]) >= 4:
                    self.get_logger().info(f"  - Control object type: {type(traj[0][3])}")
                    if traj[0][3] is not None:
                        self.get_logger().info(f"  - Control attributes: {dir(traj[0][3])}")
            

            control_trajs = [item[3] if len(item) >= 4 else None for item in traj]
            
            flattened = self._flatten_trajectory_data(
                traj, control_trajs, agent, robot_types
            )
            
            for i, robot_name in enumerate(robot_names):
                robot_msg = RobotTrajectory()
                robot_msg.robot_name = robot_name
                robot_msg.robot_type = robot_types[i]
                
                data = flattened[i]
                
                if robot_types[i] == 'diff-drive':
                    dd_traj = DiffDriveTrajectory()
                    dd_traj.time = data['t'].tolist()
                    dd_traj.x = data['x'].tolist()
                    dd_traj.y = data['y'].tolist()
                    dd_traj.theta = data['theta'].tolist()
                    dd_traj.v = data['v'].tolist()
                    dd_traj.omega = data['omega'].tolist()
                    
                    robot_msg.diff_drive_trajectories = [dd_traj]
                    robot_msg.holo_trajectories = []
                    
                elif robot_types[i] == 'holonomic':
                    holo_traj = HoloTrajectory()
                    holo_traj.time = data['t'].tolist()
                    holo_traj.x = data['x'].tolist()
                    holo_traj.y = data['y'].tolist()
                    holo_traj.vx = data['vx'].tolist()
                    holo_traj.vy = data['vy'].tolist()
                    
                    robot_msg.diff_drive_trajectories = []
                    robot_msg.holo_trajectories = [holo_traj]
                
                msg.robot_trajectories.append(robot_msg)
                
                # self.get_logger().info(
                #     f"  - {robot_name}: {len(data['t'])} points, duration={data['t'][-1]:.2f}s"
                # )
        
        self.path_pub.publish(msg)
        self.get_logger().info("Paths published successfully")
    
    def _flatten_trajectory_data(self, traj: list, control_trajs: list, agent, robot_types: list) -> dict:
        """Flatten trajectory data into per-robot arrays with fallback for missing control data"""
        
        if hasattr(agent, 'P_star'):
            Nr = len(agent.P_star)
        else:
            Nr = 1
            robot_types = [robot_types[0]]
        
        flattened = {}
        
        for i in range(Nr):
            flat_t, flat_x, flat_y, flat_theta = [], [], [], []
            flat_v, flat_omega, flat_vx, flat_vy = [], [], [], []
            
            cumulative_time = 0.0
            
            # Check if we have valid control data
            has_control_data = any(
                ctrl is not None and (
                    (hasattr(ctrl, 't_traj') and hasattr(ctrl, 'q_traj')) or
                    (hasattr(ctrl, 't') and ctrl.t is not None)
                )
                for ctrl in control_trajs
            )
            
            if not has_control_data:
                # Fallback: Extract from waypoints directly
                self.get_logger().warn(f"No control data available, extracting from waypoints for robot {i}")
                
                for waypoint in traj:
                    q_state = waypoint[0]  # State vector
                    t = waypoint[1]         # Time
                    
                    # For formations, decompose state to get individual robot positions
                    if hasattr(agent, 'P_star'):
                        xc, yc, th, sx, sy = q_state
                        R = np.array([[np.cos(th), -np.sin(th)],
                                    [np.sin(th), np.cos(th)]])
                        D = np.diag([sx, sy])
                        p_star = agent.P_star[i]
                        pos = np.array([xc, yc]) + R @ D @ p_star
                        
                        flat_x.append(float(pos[0]))
                        flat_y.append(float(pos[1]))
                        
                        # For diff-drive robots in formation, use formation orientation
                        # (This is approximate - ideally should come from control)
                        if robot_types[i] == 'diff-drive':
                            flat_theta.append(float(th))
                        else:
                            flat_theta.append(0.0)
                    else:
                        # Individual robot
                        flat_x.append(float(q_state[0]))
                        flat_y.append(float(q_state[1]))
                        
                        if robot_types[i] == 'diff-drive' and len(q_state) > 2:
                            flat_theta.append(float(q_state[2]))
                        else:
                            flat_theta.append(0.0)
                    
                    flat_t.append(float(t))
                    
                    # Estimate velocities from position differences (basic numerical derivative)
                    if len(flat_x) > 1:
                        dt = flat_t[-1] - flat_t[-2]
                        if dt > 0:
                            dx = flat_x[-1] - flat_x[-2]
                            dy = flat_y[-1] - flat_y[-2]
                            
                            if robot_types[i] == 'diff-drive':
                                # Linear velocity
                                v = np.sqrt(dx**2 + dy**2) / dt
                                flat_v.append(float(v))
                                
                                # Angular velocity
                                if len(flat_theta) > 1:
                                    dtheta = flat_theta[-1] - flat_theta[-2]
                                    # Handle angle wrapping
                                    if dtheta > np.pi:
                                        dtheta -= 2 * np.pi
                                    elif dtheta < -np.pi:
                                        dtheta += 2 * np.pi
                                    omega = dtheta / dt
                                    flat_omega.append(float(omega))
                                else:
                                    flat_omega.append(0.0)
                            elif robot_types[i] == 'holonomic':
                                vx = dx / dt
                                vy = dy / dt
                                flat_vx.append(float(vx))
                                flat_vy.append(float(vy))
                        else:
                            flat_v.append(0.0)
                            flat_omega.append(0.0)
                            flat_vx.append(0.0)
                            flat_vy.append(0.0)
                    else:
                        # First waypoint - zero velocity
                        flat_v.append(0.0)
                        flat_omega.append(0.0)
                        flat_vx.append(0.0)
                        flat_vy.append(0.0)
            
            else:
                # Original control-based extraction
                for ctrl in control_trajs:
                    if ctrl is None:
                        continue
                    
                    # Formation control with detailed trajectories
                    if hasattr(ctrl, 't_traj') and hasattr(ctrl, 'q_traj'):
                        seg_t = ctrl.t_traj
                        q_traj = ctrl.q_traj
                        
                        for k in range(len(seg_t)):
                            xc, yc, th, sx, sy = q_traj[:, k]
                            R = np.array([[np.cos(th), -np.sin(th)],
                                        [np.sin(th), np.cos(th)]])
                            D = np.diag([sx, sy])
                            p_star = agent.P_star[i]
                            pos = np.array([xc, yc]) + R @ D @ p_star
                            
                            flat_x.append(float(pos[0]))
                            flat_y.append(float(pos[1]))
                            
                            if robot_types[i] == 'diff-drive' and hasattr(ctrl, 'psi_traj') and i in ctrl.psi_traj:
                                flat_theta.append(float(ctrl.psi_traj[i][k]))
                            else:
                                flat_theta.append(float(th))
                        
                        if robot_types[i] == 'diff-drive' and hasattr(ctrl, 'v_traj') and i in ctrl.v_traj:
                            flat_v.extend([float(v) for v in ctrl.v_traj[i]])
                            flat_omega.extend([float(w) for w in ctrl.omega_traj[i]])
                        elif robot_types[i] == 'holonomic' and hasattr(ctrl, 'vx_traj') and i in ctrl.vx_traj:
                            flat_vx.extend([float(vx) for vx in ctrl.vx_traj[i]])
                            flat_vy.extend([float(vy) for vy in ctrl.vy_traj[i]])
                        
                        flat_t.extend([float(t) + cumulative_time for t in seg_t])
                        cumulative_time += float(seg_t[-1])
                    
                    # Simple control with basic trajectories
                    elif hasattr(ctrl, 't') and ctrl.t is not None:
                        seg_t = ctrl.t
                        flat_x.extend([float(x) for x in ctrl.x])
                        flat_y.extend([float(y) for y in ctrl.y])
                        
                        if ctrl.theta is not None:
                            flat_theta.extend([float(th) for th in ctrl.theta])
                        else:
                            flat_theta.extend([0.0] * len(seg_t))
                        
                        if robot_types[i] == 'diff-drive':
                            flat_v.extend([float(v) for v in ctrl.v])
                            flat_omega.extend([float(w) for w in ctrl.omega])
                        elif robot_types[i] == 'holonomic':
                            flat_vx.extend([float(vx) for vx in ctrl.v])
                            flat_vy.extend([float(vy) for vy in ctrl.vy])
                        
                        flat_t.extend([float(t) + cumulative_time for t in seg_t])
                        cumulative_time += float(seg_t[-1])
            
            # Ensure all arrays have the same length
            n = len(flat_t)
            
            if len(flat_x) < n:
                self.get_logger().warn(f"Padding flat_x from {len(flat_x)} to {n}")
                flat_x.extend([flat_x[-1] if flat_x else 0.0] * (n - len(flat_x)))
            
            if len(flat_y) < n:
                self.get_logger().warn(f"Padding flat_y from {len(flat_y)} to {n}")
                flat_y.extend([flat_y[-1] if flat_y else 0.0] * (n - len(flat_y)))
            
            if len(flat_theta) < n:
                flat_theta.extend([flat_theta[-1] if flat_theta else 0.0] * (n - len(flat_theta)))
            
            if len(flat_v) < n:
                flat_v.extend([0.0] * (n - len(flat_v)))
            
            if len(flat_omega) < n:
                flat_omega.extend([0.0] * (n - len(flat_omega)))
            
            if len(flat_vx) < n:
                flat_vx.extend([0.0] * (n - len(flat_vx)))
            
            if len(flat_vy) < n:
                flat_vy.extend([0.0] * (n - len(flat_vy)))
            
            # Store flattened data
            flattened[i] = {
                't': np.array(flat_t),
                'x': np.array(flat_x),
                'y': np.array(flat_y),
                'theta': np.array(flat_theta),
                'v': np.array(flat_v),
                'omega': np.array(flat_omega),
                'vx': np.array(flat_vx),
                'vy': np.array(flat_vy),
            }
            
            # Log extracted data for debugging
            if len(flat_t) > 0:
                self.get_logger().info(
                    f"Robot {i} ({robot_types[i]}): {len(flat_t)} points, "
                    f"duration={flat_t[-1]:.2f}s, "
                    f"pos=({flat_x[0]:.2f},{flat_y[0]:.2f})->({flat_x[-1]:.2f},{flat_y[-1]:.2f})"
                )
            else:
                self.get_logger().warn(f"Robot {i}: No trajectory data extracted!")
        
        return flattened
    
    def _save_trajectories(self, all_results: list):
        """Save trajectories to disk"""
        
        base_dir = Path(self.save_base_dir)
        existing = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        if existing:
            existing.sort(key=lambda x: x.name)
            run_dir = existing[-1]
        else:
            run_dir = base_dir / "run_0001"
            run_dir.mkdir(parents=True, exist_ok=True)
        
        self.get_logger().info(f"Saving trajectories to: {run_dir}")
        
        for result in all_results:
            name = result['name']
            agent = result['agent']
            traj = result['traj']
            robot_names = result['robot_names']
            robot_types = result['robot_types']
            
            control_trajs = [item[3] if len(item) >= 4 else None for item in traj]
            flattened = self._flatten_trajectory_data(traj, control_trajs, agent, robot_types)
            
            for i, robot_name in enumerate(robot_names):
                data = flattened[i]
                
                arr = np.zeros((len(data['t']), 3))
                arr[:, 0] = data['x']
                arr[:, 1] = data['y']
                arr[:, 2] = data['t']
                
                np.save(run_dir / f"{name}_{robot_name}.npy", arr)
        
        self.get_logger().info(f"Saved trajectories to {run_dir}")

    def _save_path_video(self, all_results: list):
        """Save path visualization as animated video"""
        
        try:
            self.get_logger().info("Creating path animation video...")
            
            base_dir = Path(self.save_base_dir)
            existing = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
            if existing:
                existing.sort(key=lambda x: x.name)
                run_dir = existing[-1]
            else:
                run_dir = base_dir / "run_0001"
                run_dir.mkdir(parents=True, exist_ok=True)
            
            video_path = run_dir / "planned_paths.mp4"
            
            color_map = {
                'blue': 'tab:blue',
                'orange': 'tab:orange', 
                'green': 'tab:green',
                'red': 'tab:red',
                'purple': 'tab:purple',
                'brown': 'tab:brown',
                'yellow': 'gold',
                'pink': 'pink',
                'cyan': 'cyan',
                'magenta': 'magenta'
            }
            default_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
            
            all_traj_data = []
            max_time = 0.0
            
            for idx, result in enumerate(all_results):
                name = result['name']
                agent = result['agent']
                traj = result['traj']
                robot_names = result['robot_names']
                robot_types = result['robot_types']
                agent_color = result.get('color', 'gray')
                color = color_map.get(agent_color, default_colors[idx % len(default_colors)])
                
                control_trajs = [item[3] if len(item) >= 4 else None for item in traj]
                flattened = self._flatten_trajectory_data(traj, control_trajs, agent, robot_types)
                
                for robot_idx in range(len(robot_names)):
                    robot_data = flattened[robot_idx]
                    if len(robot_data['t']) == 0:
                        self.get_logger().warn(f"Empty trajectory for robot {robot_idx} in video, skipping...")
                        continue
                    if len(robot_data['t']) > 0:
                        max_time = max(max_time, robot_data['t'][-1])
                
                all_traj_data.append({
                    'name': name,
                    'agent': agent,
                    'color': color,
                    'robot_names': robot_names,
                    'robot_types': robot_types,
                    'flattened': flattened
                })
            
            if max_time == 0:
                self.get_logger().error("No valid trajectory data found")
                return
            
            fps = 10
            duration = max(max_time, 3.0)
            num_frames = min(int(duration * fps), 300)
            
            self.get_logger().info(f"Video config: {num_frames} frames at {fps} fps, duration={num_frames/fps:.1f}s")
            
            plt.ioff()
            fig, ax = plt.subplots(figsize=(10, 8), dpi=80)
            
            ax.imshow(
                self.grid[::-1],
                cmap="gray_r",
                extent=[0, self.W * self.resolution, 0, self.H * self.resolution],
                alpha=0.9
            )
            
            ax.set_xlim(0, self.W * self.resolution)
            ax.set_ylim(0, self.H * self.resolution)
            ax.set_aspect("equal")
            ax.set_xlabel("X [m]", fontsize=10)
            ax.set_ylabel("Y [m]", fontsize=10)
            ax.set_title("Path Planner - Robot Trajectories", fontsize=12)
            
            # Pre-draw start/goal markers
            for traj_data in all_traj_data:
                num_robots = len(traj_data['robot_names'])
                for robot_idx in range(num_robots):
                    flattened = traj_data['flattened'][robot_idx]
                    robot_type = traj_data['robot_types'][robot_idx]
                    
                    if len(flattened['x']) > 0:
                        ax.plot(flattened['x'][0], flattened['y'][0], 'o', 
                            color=traj_data['color'], ms=6, 
                            markeredgecolor='white', markeredgewidth=1.5)
                        
                        ax.plot(flattened['x'][-1], flattened['y'][-1], '*', 
                            color=traj_data['color'], ms=12, 
                            markeredgecolor='white', markeredgewidth=1.5)
                        
                        if robot_type == 'diff-drive' and len(flattened['theta']) > 0:
                            theta_start = flattened['theta'][0]
                            arrow_len = 0.3
                            dx = arrow_len * np.cos(theta_start)
                            dy = arrow_len * np.sin(theta_start)
                            ax.arrow(flattened['x'][0], flattened['y'][0], dx, dy,
                                head_width=0.15, head_length=0.15,
                                fc=traj_data['color'], ec='white', 
                                linewidth=1, alpha=0.7, zorder=3)
            
            robot_elements = []
            
            for traj_data in all_traj_data:
                num_robots = len(traj_data['robot_names'])
                agent = traj_data['agent']
                
                for robot_idx in range(num_robots):
                    robot_name = traj_data['robot_names'][robot_idx]
                    robot_type = traj_data['robot_types'][robot_idx]
                    flattened = traj_data['flattened'][robot_idx]
                    
                    # Get radius - HANDLE PER-ROBOT RADII
                    if hasattr(agent, 'radii') and isinstance(agent.radii, list):
                        radius = agent.radii[robot_idx]
                    elif hasattr(agent, 'radius'):
                        radius = agent.radius
                    else:
                        radius = 0.2
                    
                    line, = ax.plot([], [], lw=1.5, color=traj_data['color'], alpha=0.6)
                    
                    circle = Circle((0, 0), radius, color=traj_data['color'], alpha=0.8, zorder=5)
                    ax.add_patch(circle)
                    
                    text = ax.text(0, 0, robot_name, fontsize=7, ha='center', va='center',
                                color='white', fontweight='bold', zorder=6)
                    
                    if robot_type == 'diff-drive':
                        arrow = ax.arrow(0, 0, 0, 0, head_width=0.15, head_length=0.15,
                                    fc=traj_data['color'], ec='white', 
                                    linewidth=1.5, alpha=0.9, zorder=6)
                    else:
                        arrow = None
                    
                    robot_elements.append({
                        'line': line,
                        'circle': circle,
                        'text': text,
                        'arrow': arrow,
                        'robot_name': robot_name,
                        'robot_type': robot_type,
                        'x_data': np.array(flattened['x']),
                        'y_data': np.array(flattened['y']),
                        'theta_data': np.array(flattened['theta']) if robot_type == 'diff-drive' else None,
                        't_data': np.array(flattened['t']),
                        'color': traj_data['color'],
                        'radius': radius
                    })
            
            time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                            verticalalignment='top', fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            
            def animate(frame):
                current_time = (frame / (num_frames - 1)) * max_time if num_frames > 1 else 0
                
                for elem in robot_elements:
                    x_data = elem['x_data']
                    y_data = elem['y_data']
                    theta_data = elem['theta_data']
                    t_data = elem['t_data']
                    
                    if len(t_data) == 0:
                        elem['circle'].set_visible(False)
                        elem['text'].set_visible(False)
                        if elem['arrow'] is not None:
                            elem['arrow'].set_visible(False)
                        continue
                    
                    if current_time < t_data[0]:
                        elem['circle'].set_visible(False)
                        elem['text'].set_visible(False)
                        elem['line'].set_data([], [])
                        if elem['arrow'] is not None:
                            elem['arrow'].set_visible(False)
                        continue
                    
                    mask = t_data <= current_time
                    
                    if np.any(mask):
                        elem['line'].set_data(x_data[mask], y_data[mask])
                        
                        if current_time >= t_data[-1]:
                            current_x = x_data[-1]
                            current_y = y_data[-1]
                            current_theta = theta_data[-1] if theta_data is not None else 0.0
                        else:
                            idx = np.searchsorted(t_data, current_time)
                            if idx > 0 and idx < len(t_data):
                                t0, t1 = t_data[idx-1], t_data[idx]
                                x0, x1 = x_data[idx-1], x_data[idx]
                                y0, y1 = y_data[idx-1], y_data[idx]
                                
                                if t1 > t0:
                                    alpha = (current_time - t0) / (t1 - t0)
                                    current_x = x0 + alpha * (x1 - x0)
                                    current_y = y0 + alpha * (y1 - y0)
                                    
                                    if theta_data is not None:
                                        theta0, theta1 = theta_data[idx-1], theta_data[idx]
                                        theta_diff = theta1 - theta0
                                        if theta_diff > np.pi:
                                            theta_diff -= 2 * np.pi
                                        elif theta_diff < -np.pi:
                                            theta_diff += 2 * np.pi
                                        current_theta = theta0 + alpha * theta_diff
                                    else:
                                        current_theta = 0.0
                                else:
                                    current_x = x0
                                    current_y = y0
                                    current_theta = theta_data[idx-1] if theta_data is not None else 0.0
                            else:
                                current_x = x_data[0]
                                current_y = y_data[0]
                                current_theta = theta_data[0] if theta_data is not None else 0.0
                        
                        elem['circle'].center = (current_x, current_y)
                        elem['circle'].set_visible(True)
                        elem['text'].set_position((current_x, current_y))
                        elem['text'].set_visible(True)
                        
                        if elem['robot_type'] == 'diff-drive' and elem['arrow'] is not None:
                            elem['arrow'].remove()
                            
                            arrow_len = elem['radius'] * 1.5
                            dx = arrow_len * np.cos(current_theta)
                            dy = arrow_len * np.sin(current_theta)
                            
                            new_arrow = ax.arrow(current_x, current_y, dx, dy,
                                            head_width=0.15, head_length=0.15,
                                            fc=elem['color'], ec='white',
                                            linewidth=1.5, alpha=0.9, zorder=6)
                            elem['arrow'] = new_arrow
                    else:
                        elem['circle'].set_visible(False)
                        elem['text'].set_visible(False)
                        if elem['arrow'] is not None:
                            elem['arrow'].set_visible(False)
                
                time_text.set_text(f'{current_time:.1f}s / {max_time:.1f}s')
                
                return ([elem['line'] for elem in robot_elements] + 
                        [elem['circle'] for elem in robot_elements] + 
                        [elem['text'] for elem in robot_elements] + 
                        [elem['arrow'] for elem in robot_elements if elem['arrow'] is not None] +
                        [time_text])
            
            anim = FuncAnimation(fig, animate, frames=num_frames, 
                            interval=1000/fps, blit=False, repeat=True)
            
            self.get_logger().info(f"Saving video to: {video_path}")
            
            try:
                writer = FFMpegWriter(fps=fps, bitrate=1500)
                anim.save(str(video_path), writer=writer, dpi=80, 
                        savefig_kwargs={'facecolor': 'white'})
                self.get_logger().info(f"✓ Video saved: {video_path}")
            except Exception as e:
                self.get_logger().warn(f"FFmpeg failed: {e}")
                try:
                    gif_path = run_dir / "planned_paths.gif"
                    self.get_logger().info(f"Trying GIF format...")
                    anim.save(str(gif_path), writer='pillow', fps=fps)
                    self.get_logger().info(f"✓ GIF saved: {gif_path}")
                except Exception as e2:
                    self.get_logger().error(f"Failed to save animation: {e2}")
                    traceback.print_exc()
            
            plt.close(fig)
            self.get_logger().info("Video generation complete")
            
        except Exception as e:
            self.get_logger().error(f'Failed to create path video: {e}')
            traceback.print_exc()

    def _save_path_image(self, all_results: list):
        """Save path visualization image"""
        
        try:
            plt.ioff()
            fig, ax = plt.subplots(figsize=(12, 10))
            
            ax.imshow(
                self.grid[::-1],
                cmap="gray_r",
                extent=[0, self.W * self.resolution, 0, self.H * self.resolution],
                alpha=0.9
            )
            
            default_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
            color_map = {
                'blue': 'tab:blue', 'orange': 'tab:orange', 'green': 'tab:green',
                'red': 'tab:red', 'purple': 'tab:purple', 'brown': 'tab:brown',
                'yellow': 'gold', 'pink': 'pink', 'cyan': 'cyan', 'magenta': 'magenta'
            }
            
            for idx, result in enumerate(all_results):
                name = result['name']
                agent = result['agent']
                traj = result['traj']
                robot_names = result['robot_names']
                robot_types = result['robot_types']
                
                agent_color = result.get('color', 'gray')
                color = color_map.get(agent_color, default_colors[idx % len(default_colors)])
                
                control_trajs = [item[3] if len(item) >= 4 else None for item in traj]
                flattened = self._flatten_trajectory_data(traj, control_trajs, agent, robot_types)
                
                num_robots = len(robot_names)
                
                for robot_idx in range(num_robots):
                    robot_name = robot_names[robot_idx]
                    data = flattened[robot_idx]
                    
                    x_vals = data['x']
                    y_vals = data['y']

                    if len(x_vals) == 0 or len(y_vals) == 0:
                        self.get_logger().warn(f"Empty trajectory for {robot_name} in image, skipping...")
                        continue
                    
                    ax.plot(x_vals, y_vals, lw=2, color=color, alpha=0.7,
                           label=f"{name} - {robot_name}")
                    
                    ax.plot(x_vals[0], y_vals[0], 'o', color=color, ms=6,
                           markeredgecolor='black', markeredgewidth=1)
                    
                    ax.plot(x_vals[-1], y_vals[-1], 'x', color=color, ms=8,
                           markeredgecolor='black', markeredgewidth=2)
                    
                    # Get radius - HANDLE PER-ROBOT RADII
                    if hasattr(agent, 'radii') and isinstance(agent.radii, list):
                        radius = agent.radii[robot_idx]
                    elif hasattr(agent, 'radius'):
                        radius = agent.radius
                    else:
                        radius = 0.2
                    
                    start_circle = Circle((x_vals[0], y_vals[0]), radius, 
                                         color=color, alpha=0.25)
                    ax.add_patch(start_circle)
                    start_circle_outline = Circle((x_vals[0], y_vals[0]), radius, 
                                                  fill=False, lw=2, color=color, alpha=0.8)
                    ax.add_patch(start_circle_outline)
                    
                    goal_circle = Circle((x_vals[-1], y_vals[-1]), radius, 
                                        fill=False, lw=2, linestyle="--", color=color)
                    ax.add_patch(goal_circle)
                    
                    ax.text(x_vals[0], y_vals[0], robot_name, 
                           fontsize=8, ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                    ax.text(x_vals[-1], y_vals[-1], robot_name, 
                           fontsize=8, ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
            ax.set_xlim(0, self.W * self.resolution)
            ax.set_ylim(0, self.H * self.resolution)
            ax.set_aspect("equal")
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.legend(loc="upper right", fontsize=8, ncol=2)
            ax.set_title(f"Path Planner - Planned Paths (All Robots)")
            
            base_dir = Path(self.save_base_dir)
            existing = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
            if existing:
                existing.sort(key=lambda x: x.name)
                run_dir = existing[-1]
            else:
                run_dir = base_dir / "run_0001"
                run_dir.mkdir(parents=True, exist_ok=True)
            
            image_path = run_dir / "planned_paths.png"
            fig.savefig(image_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            self.get_logger().info(f"Saved path image to: {image_path}")
            
        except Exception as e:
            self.get_logger().error(f'Failed to save path image: {e}')
            traceback.print_exc()

    def destroy_node(self):
        """Clean up resources when node is destroyed"""
        self.get_logger().info("Shutting down Path Planner Node...")
        
        self._shutdown = True
        self._cancel_event.cancel()
        
        with self._planning_lock:
            if self._planning_thread is not None and self._planning_thread.is_alive():
                self.get_logger().info("Waiting for planning thread to finish...")
                self._planning_thread.join(timeout=3.0)
                
                if self._planning_thread.is_alive():
                    self.get_logger().warn("Planning thread did not stop gracefully")
        
        plt.close('all')
        
        self.get_logger().info("Path Planner Node shutdown complete")
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    node = PathPlannerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()