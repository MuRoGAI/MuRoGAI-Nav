#!/usr/bin/env python3
"""
ROS2 Path Planner Node - HETEROGENEOUS FORMATIONS UPDATE
Supports mixed robot types (diff-drive + holonomic) in formations
Publishes trajectories using DiffDriveTrajectory and HoloTrajectory messages
"""

import os
import json
import time
import math
import traceback
import csv
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from pathlib import Path
from typing import Optional, Dict, List

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from nav_msgs.msg import Odometry
from path_planner_interface.msg import (
    PathPlannerRequest,
    DiffDriveTrajectory,
    HoloTrajectory,
    RobotTrajectoryArray
)

from path_planner.si_rrt_enhanced_individual_kinodynamic import (
    SIRRT,
    OccupancyGrid,
)

from path_planner.agents import (
    DifferentialDriveAgent,
    HolonomicAgent,
    HeterogeneousFormationAgent
)

# Try to import heterogeneous formation steering
try:
    from path_planner.heterogeneous_kinodynamic_formation_steering import *
    HETERO_AVAILABLE = True
except ImportError:
    HETERO_AVAILABLE = False
    print("Warning: Heterogeneous formation steering not available")


def quaternion_to_yaw(qx, qy, qz, qw) -> float:
    """Convert quaternion to yaw angle (rotation around Z axis)."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw


def find_Pstar(positions, yaw):
    """
    Convert world positions to formation-frame P_star coordinates.
    
    Args:
        positions: List of [x, y] positions in world frame
        yaw: Formation heading angle
    
    Returns:
        List of [px, py] in formation frame
    """
    p = []
    positions = np.array(positions)
    centroid = np.mean(positions, axis=0)
    
    # Inverse rotation matrix
    Rinv = np.array([
        [np.cos(yaw), np.sin(yaw)],
        [-np.sin(yaw), np.cos(yaw)]
    ])
    
    for i in range(len(positions)):
        delta = positions[i] - centroid
        pistar = Rinv @ delta
        p.append(pistar.tolist())
    
    return p


class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner_node')
        
        self.get_logger().info("="*60)
        self.get_logger().info("Path Planner Node (Heterogeneous) starting...")
        self.get_logger().info("="*60)
        
        # ============================================================
        # Planning state control
        # ============================================================
        self._planning_active = False
        self._cancel_current_planning = False
        self._current_request_id = 0
        
        # ============================================================
        # Declare Parameters
        # ============================================================
        self.declare_parameter('map_file_path', '')
        self.declare_parameter('config_file_path', '')
        self.declare_parameter('resolution', 0.1)
        self.declare_parameter('max_velocity', 0.6)
        self.declare_parameter('time_horizon', 100.0)
        self.declare_parameter('max_iter', 2000)
        self.declare_parameter('d_max', 0.5)
        self.declare_parameter('goal_sample_rate', 0.22)
        self.declare_parameter('neighbor_radius', 1.5)
        self.declare_parameter('precision', 2)
        self.declare_parameter('seed', 671)
        self.declare_parameter('debug', False)
        self.declare_parameter('show_initial_map', True)
        self.declare_parameter('show_path_visualization', True)
        self.declare_parameter('save_paths', True)
        self.declare_parameter('save_base_dir', '')
        self.declare_parameter('default_robot_radius', 0.2)
        self.declare_parameter('odom_timeout', 2.0)  # seconds to wait for odom
        
        # Get parameters
        self.resolution = self.get_parameter('resolution').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.time_horizon = self.get_parameter('time_horizon').value
        self.max_iter = self.get_parameter('max_iter').value
        self.d_max = self.get_parameter('d_max').value
        self.goal_sample_rate = self.get_parameter('goal_sample_rate').value
        self.neighbor_radius = self.get_parameter('neighbor_radius').value
        self.precision = self.get_parameter('precision').value
        self.seed = self.get_parameter('seed').value
        self.debug = self.get_parameter('debug').value
        self.show_initial_map = self.get_parameter('show_initial_map').value
        self.show_path_visualization = self.get_parameter('show_path_visualization').value
        self.save_paths = self.get_parameter('save_paths').value
        self.default_robot_radius = self.get_parameter('default_robot_radius').value
        self.odom_timeout = self.get_parameter('odom_timeout').value

        self.get_logger().info(f"Show initial map: {self.show_initial_map}")
        self.get_logger().info(f"Show path visualization: {self.show_path_visualization}")
        self.get_logger().info(f"Max iterations: {self.max_iter}")
        self.get_logger().info(f"Time horizon: {self.time_horizon}s")
        self.get_logger().info(f"Odom timeout: {self.odom_timeout}s")

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
            default_relative_path='data/restaurant_map.npy'
        )
        
        self.config_file_path = self._resolve_file(
            param_name='config_file_path',
            default_pkg='chatty',
            default_relative_path='config/robot_config_room_202.json'
        )
        
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
        # Load Robot Config
        # ============================================================
        self.get_logger().info("="*60)
        self.get_logger().info("Loading robot configuration...")
        self.get_logger().info("="*60)
        
        self.config_data = self._load_json(self.config_file_path)
        self.robot_names = self.config_data.get('robot_names', [])
        
        if not self.robot_names:
            self.get_logger().warn("="*60)
            self.get_logger().warn("No robot_names found in config file")
            self.get_logger().warn("="*60)
        
        self.get_logger().info(f"Loaded {len(self.robot_names)} robots: {self.robot_names}")
        
        # ============================================================
        # Odometry subscribers for current poses
        # ============================================================
        self.get_logger().info("="*60)
        self.get_logger().info("Setting up odometry subscribers...")
        self.get_logger().info("="*60)
        
        self.robot_current_poses = {}
        self.robot_odom_subs = {}
        self.robot_odom_received = {}  # Track if odom was received
        
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
            
            # CRITICAL: Default pose as fallback (in case odom not available)
            # This ensures robots have valid initial positions even without odom
            base_x = 0.5
            gap_x = 0.6

            self.robot_current_poses[robot_name] = {
                'x': base_x + idx * gap_x, 
                'y': 1.0,
                'yaw': 0.0
            }

            self.get_logger().debug(
                f"Initialized {robot_name} default pose: "
                f"x={self.robot_current_poses[robot_name]['x']:.2f}, "
                f"y={self.robot_current_poses[robot_name]['y']:.2f}, "
                f"yaw={self.robot_current_poses[robot_name]['yaw']:.2f}"
            )
        
        # Wait briefly for odom messages to arrive
        self.get_logger().info(f"Waiting up to {self.odom_timeout}s for odometry messages...")
        start_time = self.get_clock().now()
        
        while (self.get_clock().now() - start_time).nanoseconds / 1e9 < self.odom_timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            
            # Check if all robots have received odom
            all_received = all(self.robot_odom_received.values())
            if all_received:
                self.get_logger().info("All robots' odometry received!")
                break
        
        # Report odom status
        for robot_name in self.robot_names:
            if self.robot_odom_received[robot_name]:
                pose = self.robot_current_poses[robot_name]
                self.get_logger().info(
                    f"  ✓ {robot_name}: odom available at "
                    f"({pose['x']:.3f}, {pose['y']:.3f}, {pose['yaw']:.3f})"
                )
            else:
                pose = self.robot_current_poses[robot_name]
                self.get_logger().warn(
                    f"  ⚠ {robot_name}: no odom, using default "
                    f"({pose['x']:.3f}, {pose['y']:.3f}, {pose['yaw']:.3f})"
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
            '/path_planner/trajectories',
            10
        )
        self.get_logger().info("Publisher created: /path_planner/trajectories")
        
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
            
            self.get_logger().debug(f'Raw map value range: [{grid_raw.min()}, {grid_raw.max()}]')
            
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
                self.get_logger().error("="*60)
                self.get_logger().error(f"{param_name} file not found: {param_value}")
                self.get_logger().error("="*60)
                raise FileNotFoundError(f"{param_name} file not found: {param_value}")
        
        try:
            pkg_path = get_package_share_directory(default_pkg)
            default_path = os.path.join(pkg_path, default_relative_path)
            
            if os.path.exists(default_path):
                self.get_logger().info(f"Using default {param_name}: {default_path}")
                return default_path
            else:
                self.get_logger().error("="*60)
                self.get_logger().error(f"Default {param_name} file not found: {default_path}")
                self.get_logger().error("="*60)
                raise FileNotFoundError(f"Default {param_name} file not found: {default_path}")
        except Exception as e:
            self.get_logger().error("="*60)
            self.get_logger().error(f"Failed to resolve {param_name}: {e}")
            self.get_logger().error("="*60)
            raise
    
    def _load_json(self, path: str) -> dict:
        """Load JSON file"""
        if not os.path.exists(path):
            self.get_logger().error("="*60)
            self.get_logger().error(f"JSON file not found: {path}")
            self.get_logger().error("="*60)
            return {}
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self.get_logger().info(f"Loaded JSON config from: {path}")
            return data
        except Exception as e:
            self.get_logger().error("="*60)
            self.get_logger().error(f"Failed to load JSON {path}: {e}")
            self.get_logger().error("="*60)
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
            'yaw': float(yaw)
        }
        
        # Mark as received
        if not self.robot_odom_received[robot_name]:
            self.robot_odom_received[robot_name] = True
            self.get_logger().info(
                f"[ODOM] First message from {robot_name}: "
                f"x={position.x:.3f}, y={position.y:.3f}, yaw={yaw:.3f}"
            )
        else:
            self.get_logger().debug(
                f"[ODOM] {robot_name}: x={position.x:.3f}, y={position.y:.3f}, yaw={yaw:.3f}"
            )

    def _display_initial_map(self):
        """Display the map on startup (non-blocking)"""
        try:
            plt.ion()
            fig, ax = plt.subplots(figsize=(8, 8))
            
            ax.imshow(
                self.grid[::-1],
                cmap="gray_r",
                extent=[0, self.W * self.resolution, 0, self.H * self.resolution],
                alpha=0.9
            )
            
            # Draw robot initial positions
            default_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
            for idx, robot_name in enumerate(self.robot_names):
                pose = self.robot_current_poses[robot_name]
                radius = self.default_robot_radius
                color = default_colors[idx % len(default_colors)]
                
                # Different style for odom vs default
                if self.robot_odom_received.get(robot_name, False):
                    linestyle = '-'
                    alpha = 1.0
                else:
                    linestyle = '--'
                    alpha = 0.5
                
                circle = Circle(
                    (pose['x'], pose['y']), 
                    radius, 
                    fill=False, 
                    lw=2, 
                    linestyle=linestyle,
                    alpha=alpha,
                    color=color,
                    label=robot_name
                )
                ax.add_patch(circle)
                ax.plot(pose['x'], pose['y'], 'o', color=color, ms=6, alpha=alpha)
            
            ax.set_xlim(0, self.W * self.resolution)
            ax.set_ylim(0, self.H * self.resolution)
            ax.set_aspect("equal")
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_title("Path Planner - Initial Map with Robot Positions\n(Solid=Odom, Dashed=Default)")
            ax.legend(loc="upper right")
            
            plt.draw()
            plt.pause(0.1)
            
            self.get_logger().info('Initial map displayed successfully')
            
        except Exception as e:
            self.get_logger().error("="*60)
            self.get_logger().error(f'Failed to display initial map: {e}')
            self.get_logger().error("="*60)
            traceback.print_exc()
    
    def path_plan_callback(self, msg: PathPlannerRequest):
        """Handle incoming path planning requests"""
        
        self.stats['total_requests'] += 1
        self._current_request_id = self.stats['total_requests']
        
        # Check if planning is already active
        if self._planning_active:
            self.get_logger().warn("="*60)
            self.get_logger().warn(f"New request #{self._current_request_id} received while planning in progress")
            self.get_logger().warn("Cancelling current planning...")
            self.get_logger().warn("="*60)
            self._cancel_current_planning = True
            time.sleep(0.1)
        
        self.get_logger().info("="*60)
        self.get_logger().info(f"Processing planning request #{self._current_request_id}")
        self.get_logger().info("="*60)
        
        # Validate and parse JSON
        if not hasattr(msg, 'plan_json') or not msg.plan_json:
            self.get_logger().error("="*60)
            self.get_logger().error("ERROR: Missing or empty 'plan_json' field")
            self.get_logger().error("="*60)
            self.stats['failed'] += 1
            return
        
        try:
            plan_json = json.loads(msg.plan_json)
            self.get_logger().info(f"Plan JSON parsed successfully")
            self.get_logger().debug("="*60)
            self.get_logger().debug(f"Plan content: {json.dumps(plan_json, indent=2)}")
            self.get_logger().debug("="*60)
        except json.JSONDecodeError as e:
            self.get_logger().error("="*60)
            self.get_logger().error(f"ERROR: Invalid JSON in plan request: {e}")
            self.get_logger().error("="*60)
            self.stats['failed'] += 1
            return
        
        if not isinstance(plan_json, dict) or not plan_json:
            self.get_logger().error("="*60)
            self.get_logger().error("ERROR: plan_json must be a non-empty dictionary")
            self.get_logger().error("="*60)
            self.stats['failed'] += 1
            return
        
        # Process request synchronously
        self._planning_active = True
        self._cancel_current_planning = False
        
        success = self._plan_paths(plan_json)
        
        self._planning_active = False
        
        if self._cancel_current_planning:
            self.stats['cancelled'] += 1
            self.get_logger().warn("="*60)
            self.get_logger().warn(f"✗ Request #{self._current_request_id} was cancelled")
            self.get_logger().warn("="*60)
        elif success:
            self.stats['completed'] += 1
            self.get_logger().info("="*60)
            self.get_logger().info(f"✓ Request #{self._current_request_id} completed")
            self.get_logger().info("="*60)
        else:
            self.stats['failed'] += 1
            self.get_logger().error("="*60)
            self.get_logger().error(f"✗ Request #{self._current_request_id} failed")
            self.get_logger().error("="*60)
        
        self.get_logger().info(
            f"Stats: {self.stats['completed']} completed, "
            f"{self.stats['failed']} failed, "
            f"{self.stats['cancelled']} cancelled"
        )
    
    def _plan_paths(self, plan_json: dict) -> bool:
        """Plan paths for all robots/formations in the request"""
        
        try:
            # Extract agent configurations from plan_json
            self.get_logger().info("="*60)
            self.get_logger().info("Extracting agent configurations...")
            self.get_logger().info("="*60)
            
            agents_config = self._extract_agents_from_plan(plan_json)
            
            if not agents_config:
                self.get_logger().warn("="*60)
                self.get_logger().warn("No agents extracted from plan")
                self.get_logger().warn("="*60)
                return False
            
            self.get_logger().info(f"Planning for {len(agents_config)} agents")
            for name, config in agents_config.items():
                self.get_logger().info(f"  - {name}: {config['type']}")
            
            # Sequential planning with dynamic obstacles
            self.get_logger().info("="*60)
            self.get_logger().info("Starting sequential path planning...")
            self.get_logger().info("="*60)
            
            dynamic_obstacles = []
            agent_paths = {}
            agent_models = {}
            agent_info = {}
            control_trajectories = {}
            
            for name, config in agents_config.items():
                
                # Check for cancellation
                if self._cancel_current_planning:
                    self.get_logger().warn(f"Planning cancelled before {name}")
                    return False
                
                self.get_logger().info(f"\n>>> Planning path for {name} ({config['type']})...")
                
                # Create agent model
                agent = config['agent']
                agent_models[name] = agent
                
                # Get start and goal
                start = config['start']
                goal = config['goal']
                
                self.get_logger().info(f"  Start: {start}")
                self.get_logger().info(f"  Goal: {goal}")
                
                # Log dynamic obstacles
                if dynamic_obstacles:
                    self.get_logger().info(f"  Dynamic obstacles: {len(dynamic_obstacles)}")
                else:
                    self.get_logger().info(f"  Dynamic obstacles: 0 (first agent)")
                
                # Plan path
                traj = self._plan_single_agent(
                    name=name,
                    agent=agent,
                    start=start,
                    goal=goal,
                    dynamic_obstacles=dynamic_obstacles,
                    config=config
                )
                
                # Check for cancellation
                if self._cancel_current_planning:
                    self.get_logger().warn(f"Planning cancelled after {name}")
                    return False
                
                if traj is None:
                    self.get_logger().error("="*60)
                    self.get_logger().error(f"Planning failed for {name}")
                    self.get_logger().error("="*60)
                    self.get_logger().warn(f"Skipping {name}, continuing with other agents...")
                    continue
                
                # Store trajectory
                agent_paths[name] = traj
                agent_info[name] = config
                
                # Extract control trajectories
                control_traj_list = []
                for item in traj:
                    if len(item) >= 4 and item[3] is not None:
                        control_traj_list.append(item[3])
                control_trajectories[name] = control_traj_list
                
                # Add to dynamic obstacles for next agent
                traj_for_obstacle = [(item[0], item[1]) for item in traj]
                dynamic_obstacles.append({
                    'trajectory': traj_for_obstacle,
                    'agent': agent
                })
                
                self.get_logger().info(
                    f"  ✓ {name}: planned {len(traj)} waypoints, "
                    f"arrival time: {traj[-1][1]:.2f}s"
                )
            
            # Check for cancellation before publishing
            if self._cancel_current_planning:
                self.get_logger().warn("Planning cancelled before publishing")
                return False
            
            # Publish paths
            if agent_paths:
                self.get_logger().info("="*60)
                self.get_logger().info(f"Publishing trajectories for {len(agent_paths)} agents")
                self.get_logger().info("="*60)
                
                self._publish_trajectories(agent_paths, agent_models, agent_info, control_trajectories)
                
                # Save if enabled
                if self.save_paths:
                    self.get_logger().info("="*60)
                    self.get_logger().info("Saving trajectories to disk...")
                    self.get_logger().info("="*60)
                    self._save_trajectories(agent_paths, agent_models, agent_info, control_trajectories)
                
                # Show visualization
                if self.show_path_visualization:
                    self.get_logger().info("="*60)
                    self.get_logger().info("Displaying final paths...")
                    self.get_logger().info("="*60)
                    self._visualize_all_paths(agent_paths, agent_models, agent_info)
                
                return True
            else:
                self.get_logger().warn("="*60)
                self.get_logger().warn("No paths were successfully planned")
                self.get_logger().warn("="*60)
                return False
        
        except Exception as e:
            self.get_logger().error("="*60)
            self.get_logger().error(f"Planning error: {e}")
            self.get_logger().error("="*60)
            traceback.print_exc()
            return False
    
    def _extract_agents_from_plan(self, plan_json: dict) -> dict:
        """Extract agent configurations from plan JSON
        
        Returns:
            dict: {agent_name: {
                'type': 'formation' or 'individual',
                'agent': AgentModel,
                'start': np.array,
                'goal': np.array,
                'vel_limits': dict,
                ...
            }}
        """
        agents = {}
        
        for group_name, group_data in plan_json.items():
            
            self.get_logger().debug(f"Processing group: {group_name}")
            
            # Formation group (F*)
            if group_name.startswith('F') and isinstance(group_data, dict):
                self.get_logger().info(f"  - Processing formation {group_name}")
                
                try:
                    formation_config = self._parse_formation_group(group_name, group_data)
                    if formation_config:
                        agents[group_name] = formation_config
                except Exception as e:
                    self.get_logger().error(f"Error parsing formation {group_name}: {e}")
                    traceback.print_exc()
                    continue
            
            # Individual robot group (R*)
            elif group_name.startswith('R') and isinstance(group_data, list):
                self.get_logger().debug(f"  - Processing individual robots in {group_name}")
                
                for robot_entry in group_data:
                    try:
                        robot_config = self._parse_individual_robot(robot_entry)
                        if robot_config:
                            robot_name = robot_entry.get('robot', f"{group_name}_robot")
                            agents[robot_name] = robot_config
                    except Exception as e:
                        self.get_logger().error(f"Error parsing robot in {group_name}: {e}")
                        traceback.print_exc()
                        continue
        
        return agents
    
    def _parse_formation_group(self, group_name: str, group_data: dict) -> Optional[dict]:
        """Parse formation group configuration"""
        
        if 'robots' not in group_data or not group_data['robots']:
            self.get_logger().error(f"Formation {group_name} has no robots")
            return None
        
        robots = group_data['robots']
        
        # Extract formation centroid and heading from goal
        centroid_x = group_data.get('centroid_x')
        centroid_y = group_data.get('centroid_y')
        formation_yaw = group_data.get('formation_yaw', 0.0)
        desired_radius = group_data.get('desired_radius', 0.4)
        
        if centroid_x is None or centroid_y is None:
            self.get_logger().error(f"Formation {group_name} missing centroid position")
            return None
        
        # Extract robot types and positions
        robot_types = []
        robot_positions = []
        robot_names = []
        robot_radii = []
        vel_limits_list = []
        
        for robot_entry in robots:
            robot_name = robot_entry.get('robot')
            if not robot_name:
                self.get_logger().warn(f"Robot in {group_name} missing name, skipping")
                continue
            
            robot_names.append(robot_name)
            
            # Get robot type
            robot_type_str = robot_entry.get('type', 'Differential Drive Robot')
            if 'holonomic' in robot_type_str.lower() or 'omnidirectional' in robot_type_str.lower():
                robot_type = 'holonomic'
            else:
                robot_type = 'diff-drive'
            
            robot_types.append(robot_type)
            
            # CRITICAL: Get robot CURRENT position from ODOM (not from plan JSON)
            # The plan JSON positions are GOAL positions, not start positions!
            # We need the actual current position to compute the formation start centroid
            if robot_name in self.robot_current_poses:
                pose = self.robot_current_poses[robot_name]
                x = pose['x']
                y = pose['y']
                self.get_logger().info(f"    Robot {robot_name} current position (from odom): ({x:.3f}, {y:.3f})")
            else:
                self.get_logger().error(f"Cannot get current position for {robot_name} - no odom available")
                return None
            
            robot_positions.append([x, y])
            
            # Get robot radius
            radius = robot_entry.get('radius', self.default_robot_radius)
            robot_radii.append(radius)
            
            # Get velocity limits
            vel_limits = {}
            
            if robot_type == 'diff-drive':
                vel_limits['v_max'] = robot_entry.get('max_linear_velocity_x', 0.3)
                vel_limits['v_min'] = -vel_limits['v_max']
                vel_limits['omega_max'] = robot_entry.get('max_angular_velocity_z', 2.0)
            else:  # holonomic
                vel_limits['v_max'] = max(
                    robot_entry.get('max_linear_velocity_x', 0.3),
                    robot_entry.get('max_linear_velocity_y', 0.3)
                )
            
            vel_limits['a_max'] = robot_entry.get('max_acceleration', 1.5)
            vel_limits_list.append(vel_limits)
        
        if not robot_types:
            self.get_logger().error(f"No valid robots in formation {group_name}")
            return None
        
        # Compute P_star from current positions
        P_star = find_Pstar(robot_positions, formation_yaw)
        
        self.get_logger().info(f"  Formation {group_name}:")
        self.get_logger().info(f"    Robot types: {robot_types}")
        self.get_logger().info(f"    P_star: {P_star}")
        self.get_logger().info(f"    Desired radius: {desired_radius}")
        
        # Create heterogeneous formation agent
        if not HETERO_AVAILABLE:
            self.get_logger().error("Heterogeneous formation steering not available!")
            return None
        
        agent = HeterogeneousFormationAgent(
            P_star=P_star,
            robot_types=robot_types,
            radius=desired_radius
        )
        
        # Compute start configuration from CURRENT robot positions
        # For formation: [xc, yc, theta, sx, sy]
        centroid_start = np.mean(robot_positions, axis=0)
        
        self.get_logger().info(f"    Computed start centroid from robot positions: ({centroid_start[0]:.3f}, {centroid_start[1]:.3f})")
        self.get_logger().info(f"    Goal centroid from plan: ({centroid_x:.3f}, {centroid_y:.3f})")
        
        start = np.array([
            centroid_start[0],
            centroid_start[1],
            formation_yaw,
            1.0,  # sx
            1.0   # sy
        ])
        
        # Goal configuration
        goal = np.array([
            centroid_x,
            centroid_y,
            formation_yaw,
            1.0,  # sx
            1.0   # sy
        ])
        
        # Aggregate velocity limits for formation
        # Use per-robot lists
        v_max_list = [vl.get('v_max', 0.3) for vl in vel_limits_list]
        w_max_list = [vl.get('omega_max', 2.0) if robot_types[i] == 'diff-drive' else 0.0 
                      for i, vl in enumerate(vel_limits_list)]
        
        formation_vel_limits = {
            'v_max': v_max_list,
            'w_max': w_max_list,
            'a_max': max([vl.get('a_max', 1.5) for vl in vel_limits_list])
        }
        
        return {
            'type': 'heterogeneous-formation',
            'agent': agent,
            'start': start,
            'goal': goal,
            'vel_limits': formation_vel_limits,
            'robot_types': robot_types,
            'robot_names': robot_names,
            'P_star': P_star
        }
    
    def _parse_individual_robot(self, robot_entry: dict) -> Optional[dict]:
        """Parse individual robot configuration"""
        
        robot_name = robot_entry.get('robot')
        if not robot_name:
            self.get_logger().error("Robot entry missing 'robot' field")
            return None
        
        # Get robot type
        robot_type_str = robot_entry.get('type', 'Differential Drive Robot')
        if 'holonomic' in robot_type_str.lower() or 'omnidirectional' in robot_type_str.lower():
            robot_type = 'holonomic'
            agent = HolonomicAgent(radius=robot_entry.get('radius', self.default_robot_radius))
        else:
            robot_type = 'diff-drive'
            agent = DifferentialDriveAgent(radius=robot_entry.get('radius', self.default_robot_radius))
        
        # Get start position from odometry
        if robot_name in self.robot_current_poses:
            pose = self.robot_current_poses[robot_name]
            if robot_type == 'diff-drive':
                start = np.array([pose['x'], pose['y'], pose['yaw']])
            else:
                start = np.array([pose['x'], pose['y']])
            self.get_logger().info(f"    Using odom for {robot_name} start: {start}")
        else:
            self.get_logger().warn(f"No odometry for {robot_name}, using default")
            if robot_type == 'diff-drive':
                start = np.array([0.0, 0.0, 0.0])
            else:
                start = np.array([0.0, 0.0])
        
        # Get goal position
        x = robot_entry.get('x')
        y = robot_entry.get('y')
        yaw = robot_entry.get('yaw', 0.0)
        
        if x is None or y is None:
            self.get_logger().error(f"Robot {robot_name} missing goal position")
            return None
        
        if robot_type == 'diff-drive':
            goal = np.array([x, y, yaw])
        else:
            goal = np.array([x, y])
        
        # Get velocity limits
        vel_limits = {}
        
        if robot_type == 'diff-drive':
            vel_limits['v_max'] = robot_entry.get('max_linear_velocity_x', 0.3)
            vel_limits['v_min'] = -vel_limits['v_max']
            vel_limits['omega_max'] = robot_entry.get('max_angular_velocity_z', 2.0)
        else:
            vel_limits['v_max'] = max(
                robot_entry.get('max_linear_velocity_x', 0.3),
                robot_entry.get('max_linear_velocity_y', 0.3)
            )
        
        vel_limits['a_max'] = robot_entry.get('max_acceleration', 1.5)
        
        self.get_logger().info(f"  Robot {robot_name} ({robot_type}):")
        self.get_logger().info(f"    Velocity limits: {vel_limits}")
        
        return {
            'type': robot_type,
            'agent': agent,
            'start': start,
            'goal': goal,
            'vel_limits': vel_limits,
            'robot_name': robot_name
        }
    
    def _plan_single_agent(self, name: str, agent, start: np.ndarray,
                          goal: np.ndarray, dynamic_obstacles: list,
                          config: dict):
        """Plan path for a single agent using kinodynamic SI-RRT"""
        
        agent_type = config['type']
        vel_limits = config['vel_limits']
        
        # Build kinodynamic parameters
        if agent_type == "heterogeneous-formation" and HETERO_AVAILABLE:
            kino_params = {
                'robot_types': config['robot_types'],
                'v_max': vel_limits.get('v_max', [0.8]*len(config['P_star'])),
                'w_max': vel_limits.get('w_max', [2.0]*len(config['P_star'])),
                'N_steer': 10,
                'T_steer': 1.0,
                'T_steer_min': 0.5,
                'T_steer_safety_factor': 1.5,
                'max_iter': 200,
            }
            use_kino = True
            self.get_logger().info(f"    Per-robot v_max: {kino_params['v_max']}")
            self.get_logger().info(f"    Per-robot w_max: {kino_params['w_max']}")
            
        elif agent_type == "diff-drive":
            kino_params = {
                'v_max': vel_limits.get('v_max', 0.8),
                'v_min': vel_limits.get('v_min', -0.3),
                'omega_max': vel_limits.get('omega_max', 2.0),
                'a_max': vel_limits.get('a_max', 1.5),
                'dt': 0.05,
            }
            use_kino = True
            self.get_logger().info(f"    v_max: {kino_params['v_max']}, omega_max: {kino_params['omega_max']}")
            
        elif agent_type == "holonomic":
            kino_params = {
                'v_max': vel_limits.get('v_max', 0.4),
                'a_max': vel_limits.get('a_max', 2.0),
                'dt': 0.05,
            }
            use_kino = True
            self.get_logger().info(f"    v_max: {kino_params['v_max']}")
            
        else:
            use_kino = False
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
        
        try:
            traj = planner.plan(start, goal, dynamic_obstacles)
            
            if traj is None:
                self.get_logger().error(f"  Planner returned None for {name}")
                return None
            
            self.get_logger().debug(f"  Planning succeeded: {len(traj)} waypoints")
            
            return traj
            
        except Exception as e:
            self.get_logger().error("="*60)
            self.get_logger().error(f"  Planning exception for {name}: {e}")
            self.get_logger().error("="*60)
            traceback.print_exc()
            return None
    
    def _publish_trajectories(self, agent_paths: dict, agent_models: dict, 
                             agent_info: dict, control_trajectories: dict):
        """Publish trajectories using DiffDriveTrajectory and HoloTrajectory messages"""
        
        msg = RobotTrajectoryArray()
        
        for agent_name, traj in agent_paths.items():
            agent = agent_models[agent_name]
            agent_type = agent_info[agent_name]['type']
            control_list = control_trajectories.get(agent_name, [])
            
            if agent_type == "heterogeneous-formation":
                # Formation - extract individual robot trajectories with controls
                Nr = len(agent.P_star)
                robot_types = agent_info[agent_name]['robot_types']
                robot_names = agent_info[agent_name]['robot_names']
                
                for robot_idx in range(Nr):
                    robot_name = robot_names[robot_idx]
                    robot_type = robot_types[robot_idx]
                    
                    # Extract data from control trajectories
                    all_t, all_x, all_y, all_theta = [], [], [], []
                    all_v, all_omega, all_vx, all_vy = [], [], [], []
                    
                    cumulative_time = 0.0
                    
                    for ctrl_traj in control_list:
                        if ctrl_traj is None or not hasattr(ctrl_traj, 'q_traj'):
                            continue
                        
                        # Extract formation state and robot-specific data
                        q_traj = ctrl_traj.q_traj  # (5, N)
                        t_traj = ctrl_traj.t_traj + cumulative_time
                        
                        cumulative_time = t_traj[-1]
                        
                        # Compute robot positions from formation state
                        for k in range(len(t_traj)):
                            xc, yc, th, sx, sy = q_traj[:, k]
                            
                            # Rotation matrix
                            R = np.array([[np.cos(th), -np.sin(th)],
                                        [np.sin(th), np.cos(th)]])
                            D = np.diag([sx, sy])
                            p_star_i = agent.P_star[robot_idx]
                            
                            # Robot position
                            p_i = np.array([xc, yc]) + R @ D @ p_star_i
                            all_x.append(p_i[0])
                            all_y.append(p_i[1])
                            all_t.append(t_traj[k])
                            
                            # Robot heading (psi for DD, formation heading for holonomic)
                            if robot_type == 'diff-drive' and robot_idx in ctrl_traj.psi_traj:
                                psi_val = ctrl_traj.psi_traj[robot_idx][k]
                                all_theta.append(psi_val)
                            else:
                                all_theta.append(th)
                        
                        # Extract controls (N points)
                        if robot_type == 'diff-drive' and robot_idx in ctrl_traj.v_traj:
                            all_v.extend(ctrl_traj.v_traj[robot_idx].tolist())
                            all_omega.extend(ctrl_traj.omega_traj[robot_idx].tolist())
                        elif robot_type == 'holonomic' and robot_idx in ctrl_traj.vx_traj:
                            all_vx.extend(ctrl_traj.vx_traj[robot_idx].tolist())
                            all_vy.extend(ctrl_traj.vy_traj[robot_idx].tolist())
                    
                    # Create appropriate message
                    if robot_type == 'diff-drive':
                        traj_msg = DiffDriveTrajectory()
                        traj_msg.time = all_t
                        traj_msg.x = all_x
                        traj_msg.y = all_y
                        traj_msg.theta = all_theta
                        traj_msg.v = all_v if all_v else [0.0] * len(all_t)
                        traj_msg.omega = all_omega if all_omega else [0.0] * len(all_t)
                        
                        msg.robot_names.append(robot_name)
                        msg.robot_types.append('diff-drive')
                        msg.diff_drive_trajectories.append(traj_msg)
                        
                        self.get_logger().info(
                            f"  - {robot_name} (DD): {len(all_t)} points, duration={all_t[-1]:.2f}s"
                        )
                    
                    else:  # holonomic
                        traj_msg = HoloTrajectory()
                        traj_msg.time = all_t
                        traj_msg.x = all_x
                        traj_msg.y = all_y
                        traj_msg.vx = all_vx if all_vx else [0.0] * len(all_t)
                        traj_msg.vy = all_vy if all_vy else [0.0] * len(all_t)
                        
                        msg.robot_names.append(robot_name)
                        msg.robot_types.append('holonomic')
                        msg.holo_trajectories.append(traj_msg)
                        
                        self.get_logger().info(
                            f"  - {robot_name} (Holo): {len(all_t)} points, duration={all_t[-1]:.2f}s"
                        )
            
            else:
                # Individual robot
                robot_name = agent_info[agent_name].get('robot_name', agent_name)
                
                # Extract data from control trajectories
                all_t, all_x, all_y, all_theta = [], [], [], []
                all_v, all_omega, all_vy = [], [], []
                
                cumulative_time = 0.0
                
                for ctrl_traj in control_list:
                    if ctrl_traj is None:
                        continue
                    
                    x = ctrl_traj.x
                    y = ctrl_traj.y
                    t = ctrl_traj.t + cumulative_time if ctrl_traj.t is not None else None
                    
                    if t is not None:
                        cumulative_time = t[-1]
                    
                    all_x.extend(x.tolist())
                    all_y.extend(y.tolist())
                    if t is not None:
                        all_t.extend(t.tolist())
                    
                    if ctrl_traj.theta is not None:
                        all_theta.extend(ctrl_traj.theta.tolist())
                    if ctrl_traj.v is not None:
                        all_v.extend(ctrl_traj.v.tolist())
                    if ctrl_traj.omega is not None:
                        all_omega.extend(ctrl_traj.omega.tolist())
                    if ctrl_traj.vy is not None:
                        all_vy.extend(ctrl_traj.vy.tolist())
                
                # Create appropriate message
                if agent_type == 'diff-drive':
                    traj_msg = DiffDriveTrajectory()
                    traj_msg.time = all_t
                    traj_msg.x = all_x
                    traj_msg.y = all_y
                    traj_msg.theta = all_theta if all_theta else [0.0] * len(all_t)
                    traj_msg.v = all_v if all_v else [0.0] * len(all_t)
                    traj_msg.omega = all_omega if all_omega else [0.0] * len(all_t)
                    
                    msg.robot_names.append(robot_name)
                    msg.robot_types.append('diff-drive')
                    msg.diff_drive_trajectories.append(traj_msg)
                    
                    self.get_logger().info(
                        f"  - {robot_name} (DD): {len(all_t)} points, duration={all_t[-1]:.2f}s"
                    )
                
                else:  # holonomic
                    traj_msg = HoloTrajectory()
                    traj_msg.time = all_t
                    traj_msg.x = all_x
                    traj_msg.y = all_y
                    traj_msg.vx = all_v if all_v else [0.0] * len(all_t)
                    traj_msg.vy = all_vy if all_vy else [0.0] * len(all_t)
                    
                    msg.robot_names.append(robot_name)
                    msg.robot_types.append('holonomic')
                    msg.holo_trajectories.append(traj_msg)
                    
                    self.get_logger().info(
                        f"  - {robot_name} (Holo): {len(all_t)} points, duration={all_t[-1]:.2f}s"
                    )
        
        self.path_pub.publish(msg)
        self.get_logger().info("Trajectories published successfully")
    
    def _save_trajectories(self, agent_paths: dict, agent_models: dict, 
                          agent_info: dict, control_trajectories: dict):
        """Save trajectories to disk with auto-versioning (matches upgraded code format)"""
        
        base_dir = Path(self.save_base_dir)
        base_dir.mkdir(exist_ok=True)
        
        # Find next run index
        existing = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        if existing:
            ids = [int(d.name.split("_")[1]) for d in existing if d.name.split("_")[1].isdigit()]
            run_id = max(ids) + 1 if ids else 1
        else:
            run_id = 1
        
        run_dir = base_dir / f"run_{run_id:04d}"
        run_dir.mkdir()
        
        self.get_logger().info(f"Saving trajectories to: {run_dir}")
        
        # Also create CSV directory
        csv_dir = run_dir / "trajectory_data"
        csv_dir.mkdir()
        
        saved = {}
        
        for agent_name, traj in agent_paths.items():
            agent = agent_models[agent_name]
            agent_type = agent_info[agent_name]['type']
            control_list = control_trajectories.get(agent_name, [])
            
            if agent_type == "heterogeneous-formation":
                # Formation - save per-robot data
                Nr = len(agent.P_star)
                robot_types = agent_info[agent_name]['robot_types']
                robot_names = agent_info[agent_name]['robot_names']
                robot_trajs = {}
                
                for robot_idx in range(Nr):
                    robot_name = robot_names[robot_idx]
                    robot_type = robot_types[robot_idx]
                    
                    # Extract comprehensive data
                    all_t, all_x, all_y, all_theta = [], [], [], []
                    all_v, all_omega, all_vx, all_vy = [], [], [], []
                    
                    cumulative_time = 0.0
                    
                    for ctrl_traj in control_list:
                        if ctrl_traj is None or not hasattr(ctrl_traj, 'q_traj'):
                            continue
                        
                        q_traj = ctrl_traj.q_traj
                        t_traj = ctrl_traj.t_traj + cumulative_time
                        
                        cumulative_time = t_traj[-1]
                        
                        for k in range(len(t_traj)):
                            xc, yc, th, sx, sy = q_traj[:, k]
                            R = np.array([[np.cos(th), -np.sin(th)],
                                        [np.sin(th), np.cos(th)]])
                            D = np.diag([sx, sy])
                            p_star_i = agent.P_star[robot_idx]
                            p_i = np.array([xc, yc]) + R @ D @ p_star_i
                            
                            all_x.append(p_i[0])
                            all_y.append(p_i[1])
                            all_t.append(t_traj[k])
                            
                            if robot_type == 'diff-drive' and robot_idx in ctrl_traj.psi_traj:
                                all_theta.append(ctrl_traj.psi_traj[robot_idx][k])
                            else:
                                all_theta.append(th)
                        
                        if robot_type == 'diff-drive' and robot_idx in ctrl_traj.v_traj:
                            all_v.extend(ctrl_traj.v_traj[robot_idx].tolist())
                            all_omega.extend(ctrl_traj.omega_traj[robot_idx].tolist())
                        elif robot_type == 'holonomic' and robot_idx in ctrl_traj.vx_traj:
                            all_vx.extend(ctrl_traj.vx_traj[robot_idx].tolist())
                            all_vy.extend(ctrl_traj.vy_traj[robot_idx].tolist())
                    
                    # Save CSV
                    csv_file = csv_dir / f"{robot_name}_{robot_type}.csv"
                    with open(csv_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        
                        if robot_type == 'diff-drive':
                            writer.writerow(['time', 'x', 'y', 'theta', 'v_body', 'omega', 'vx_world', 'vy_world'])
                            
                            for i in range(len(all_t)):
                                v_val = all_v[i] if i < len(all_v) else 0.0
                                omega_val = all_omega[i] if i < len(all_omega) else 0.0
                                theta_val = all_theta[i] if i < len(all_theta) else 0.0
                                vx_world = v_val * np.cos(theta_val)
                                vy_world = v_val * np.sin(theta_val)
                                
                                writer.writerow([
                                    f"{all_t[i]:.6f}",
                                    f"{all_x[i]:.6f}",
                                    f"{all_y[i]:.6f}",
                                    f"{theta_val:.6f}",
                                    f"{v_val:.6f}",
                                    f"{omega_val:.6f}",
                                    f"{vx_world:.6f}",
                                    f"{vy_world:.6f}"
                                ])
                        
                        else:  # holonomic
                            writer.writerow(['time', 'x', 'y', 'vx', 'vy', 'speed'])
                            
                            for i in range(len(all_t)):
                                vx_val = all_vx[i] if i < len(all_vx) else 0.0
                                vy_val = all_vy[i] if i < len(all_vy) else 0.0
                                speed = np.hypot(vx_val, vy_val)
                                
                                writer.writerow([
                                    f"{all_t[i]:.6f}",
                                    f"{all_x[i]:.6f}",
                                    f"{all_y[i]:.6f}",
                                    f"{vx_val:.6f}",
                                    f"{vy_val:.6f}",
                                    f"{speed:.6f}"
                                ])
                    
                    # Save numpy array
                    arr = np.zeros((len(all_t), 3))
                    arr[:, 0] = all_x
                    arr[:, 1] = all_y
                    arr[:, 2] = all_t
                    
                    robot_trajs[f"robot_{robot_idx}"] = arr
                    np.save(run_dir / f"{agent_name}_robot_{robot_idx}.npy", arr)
                
                saved[agent_name] = {
                    "times": np.array(all_t),
                    "robots": robot_trajs
                }
            
            else:
                # Individual robot
                robot_name = agent_info[agent_name].get('robot_name', agent_name)
                
                # Extract data
                all_t, all_x, all_y, all_theta = [], [], [], []
                all_v, all_omega, all_vy = [], [], []
                
                cumulative_time = 0.0
                
                for ctrl_traj in control_list:
                    if ctrl_traj is None:
                        continue
                    
                    t = ctrl_traj.t + cumulative_time if ctrl_traj.t is not None else None
                    if t is not None:
                        cumulative_time = t[-1]
                        all_t.extend(t.tolist())
                    
                    all_x.extend(ctrl_traj.x.tolist())
                    all_y.extend(ctrl_traj.y.tolist())
                    
                    if ctrl_traj.theta is not None:
                        all_theta.extend(ctrl_traj.theta.tolist())
                    if ctrl_traj.v is not None:
                        all_v.extend(ctrl_traj.v.tolist())
                    if ctrl_traj.omega is not None:
                        all_omega.extend(ctrl_traj.omega.tolist())
                    if ctrl_traj.vy is not None:
                        all_vy.extend(ctrl_traj.vy.tolist())
                
                # Save CSV
                csv_file = csv_dir / f"{robot_name}_{agent_type}.csv"
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    if agent_type == 'diff-drive':
                        writer.writerow(['time', 'x', 'y', 'theta', 'v_body', 'omega', 'vx_world', 'vy_world'])
                        
                        for i in range(len(all_t)):
                            v_val = all_v[i] if i < len(all_v) else 0.0
                            omega_val = all_omega[i] if i < len(all_omega) else 0.0
                            theta_val = all_theta[i] if i < len(all_theta) else 0.0
                            vx_world = v_val * np.cos(theta_val)
                            vy_world = v_val * np.sin(theta_val)
                            
                            writer.writerow([
                                f"{all_t[i]:.6f}",
                                f"{all_x[i]:.6f}",
                                f"{all_y[i]:.6f}",
                                f"{theta_val:.6f}",
                                f"{v_val:.6f}",
                                f"{omega_val:.6f}",
                                f"{vx_world:.6f}",
                                f"{vy_world:.6f}"
                            ])
                    
                    else:  # holonomic
                        writer.writerow(['time', 'x', 'y', 'vx', 'vy', 'speed'])
                        
                        for i in range(len(all_t)):
                            vx_val = all_v[i] if i < len(all_v) else 0.0
                            vy_val = all_vy[i] if i < len(all_vy) else 0.0
                            speed = np.hypot(vx_val, vy_val)
                            
                            writer.writerow([
                                f"{all_t[i]:.6f}",
                                f"{all_x[i]:.6f}",
                                f"{all_y[i]:.6f}",
                                f"{vx_val:.6f}",
                                f"{vy_val:.6f}",
                                f"{speed:.6f}"
                            ])
                
                # Save numpy array
                arr = np.zeros((len(all_t), 3))
                arr[:, 0] = all_x
                arr[:, 1] = all_y
                arr[:, 2] = all_t
                
                np.save(run_dir / f"{robot_name}.npy", arr)
                
                saved[robot_name] = {
                    "times": np.array(all_t),
                    "positions": arr
                }
        
        # Save summary
        summary_path = run_dir / "all_agents_paths.npy"
        np.save(summary_path, saved, allow_pickle=True)
        
        self.get_logger().info(f"Saved {len(agent_paths)} agent trajectories")
        self.get_logger().info(f"CSV files saved to: {csv_dir}")
    
    def _visualize_all_paths(self, agent_paths: dict, agent_models: dict, agent_info: dict):
        """Visualize all planned paths"""
        
        self.get_logger().info(f'Visualizing {len(agent_paths)} agent paths')
        
        try:
            plt.close('all')
            plt.ion()
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Draw map
            ax.imshow(
                self.grid[::-1],
                cmap="gray_r",
                extent=[0, self.W * self.resolution, 0, self.H * self.resolution],
                alpha=0.9
            )
            
            # Color mapping
            colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
            color_idx = 0
            
            # Plot trajectories
            for agent_name, traj in agent_paths.items():
                agent = agent_models[agent_name]
                agent_type = agent_info[agent_name]['type']
                color = colors[color_idx % len(colors)]
                color_idx += 1
                
                if agent_type == "heterogeneous-formation":
                    # Plot formation centroid path
                    pts = []
                    for item in traj:
                        if len(item) == 2:
                            q = item[0]
                        elif len(item) >= 3:
                            q = item[0]
                        else:
                            continue
                        pts.append(np.asarray(q)[:2])
                    
                    pts = np.array(pts)
                    ax.plot(pts[:, 0], pts[:, 1], '--', lw=1.5, color=color, alpha=0.5, label=f"{agent_name} (centroid)")
                    
                    # Plot individual robot paths and discs
                    Nr = len(agent.P_star)
                    for robot_idx in range(Nr):
                        robot_pts = []
                        for item in traj:
                            if len(item) == 2:
                                q = item[0]
                            elif len(item) >= 3:
                                q = item[0]
                            else:
                                continue
                            
                            xc, yc, th, sx, sy = q[0], q[1], q[2], q[3], q[4]
                            R = np.array([[np.cos(th), -np.sin(th)],
                                        [np.sin(th), np.cos(th)]])
                            D = np.diag([sx, sy])
                            p_i = np.array([xc, yc]) + R @ D @ agent.P_star[robot_idx]
                            robot_pts.append(p_i)
                        
                        robot_pts = np.array(robot_pts)
                        ax.plot(robot_pts[:, 0], robot_pts[:, 1], '-', lw=2, color=color)
                        
                        # Start and goal discs
                        ax.add_patch(Circle(robot_pts[0], agent.radius, color=color, alpha=0.25))
                        ax.add_patch(Circle(robot_pts[0], agent.radius, fill=False, lw=2, color=color, alpha=0.8))
                        ax.add_patch(Circle(robot_pts[-1], agent.radius, fill=False, lw=2, linestyle="--", color=color))
                        
                        # Markers
                        ax.plot(robot_pts[0, 0], robot_pts[0, 1], 'o', color=color, ms=6)
                        ax.plot(robot_pts[-1, 0], robot_pts[-1, 1], 'x', color=color, ms=8, mew=2)
                
                else:
                    # Individual robot
                    pts = []
                    for item in traj:
                        if len(item) == 2:
                            q = item[0]
                        elif len(item) >= 3:
                            q = item[0]
                        else:
                            continue
                        pts.append(np.asarray(q)[:2])
                    
                    pts = np.array(pts)
                    ax.plot(pts[:, 0], pts[:, 1], lw=2, color=color, label=agent_name)
                    
                    # Start and goal discs
                    start_item = traj[0]
                    goal_item = traj[-1]
                    
                    start_q = start_item[0] if len(start_item) >= 2 else start_item
                    goal_q = goal_item[0] if len(goal_item) >= 2 else goal_item
                    
                    for p, r in agent.discs(start_q):
                        ax.add_patch(Circle(p, r, color=color, alpha=0.25))
                        ax.add_patch(Circle(p, r, fill=False, lw=2, color=color, alpha=0.8))
                    
                    for p, r in agent.discs(goal_q):
                        ax.add_patch(Circle(p, r, fill=False, lw=2, linestyle="--", color=color))
                    
                    # Markers
                    ax.plot(pts[0, 0], pts[0, 1], 'o', color=color, ms=6)
                    ax.plot(pts[-1, 0], pts[-1, 1], 'x', color=color, ms=8, mew=2)
            
            ax.set_xlim(0, self.W * self.resolution)
            ax.set_ylim(0, self.H * self.resolution)
            ax.set_aspect("equal")
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.legend(loc="upper right")
            ax.set_title(f"Path Planner - {len(agent_paths)} Agents (Kinodynamic SI-RRT + CPP)")
            
            plt.draw()
            plt.pause(0.1)
            
            self.get_logger().info('Paths displayed. Close window to continue.')
            
        except Exception as e:
            self.get_logger().error("="*60)
            self.get_logger().error(f'Failed to visualize paths: {e}')
            self.get_logger().error("="*60)
            traceback.print_exc()


def main(args=None):
    rclpy.init(args=args)
    
    node = PathPlannerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("="*60)
        node.get_logger().info("Shutting down Path Planner Node...")
        node.get_logger().info("="*60)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()