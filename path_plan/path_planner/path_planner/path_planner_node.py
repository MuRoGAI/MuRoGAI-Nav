#!/usr/bin/env python3
"""
ROS2 Path Planner Node - FIXED VERSION
Fixes applied:
1. Goal dimension corrected to 2D [x, y]
2. Disc extraction added in trajectory saving
3. Disc visualization added at start/goal
4. Legend cleaned up
5. All logging preserved
"""

import os
import json
import time
import math
import traceback
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
from typing import Optional, Dict, List

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from nav_msgs.msg import Odometry
from path_planner_interface.msg import RobotPath, RobotPathArray, PathPlannerRequest

from path_planner.path_planning_core import (
    SIRRT,
    OccupancyGrid,
    IndividualAgent,
    FlexibleFormationAgent,
)


def quaternion_to_yaw(qx, qy, qz, qw) -> float:
    """Convert quaternion to yaw angle (rotation around Z axis)."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw


class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner_node')
        
        self.get_logger().info("="*60)
        self.get_logger().info("Path Planner Node starting...")
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
        self.declare_parameter('max_velocity', 0.15)
        self.declare_parameter('time_horizon', 600.0)
        self.declare_parameter('max_iter', 1500)
        self.declare_parameter('d_max', 1.2)
        self.declare_parameter('goal_sample_rate', 0.22)
        self.declare_parameter('neighbor_radius', 1.5)
        self.declare_parameter('precision', 2)
        self.declare_parameter('seed', 48)
        self.declare_parameter('debug', False)
        self.declare_parameter('show_initial_map', False)
        self.declare_parameter('show_path_visualization', False)
        self.declare_parameter('save_paths', True)
        self.declare_parameter('save_base_dir', '')
        self.declare_parameter('default_robot_radius', 0.15)
        
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

        self.get_logger().info(f"Show initial map: {self.show_initial_map}")
        self.get_logger().info(f"Show path visualization: {self.show_path_visualization}")
        self.get_logger().info(f"Max iterations: {self.max_iter}")
        self.get_logger().info(f"Time horizon: {self.time_horizon}s")

        # ============================================================
        # Resolve Directory
        # ============================================================
        self.get_logger().info("="*60)
        self.get_logger().info("Resolving directories and files...")
        self.get_logger().info("="*60)
        
        self.save_base_dir = self._resolve_dir(
            param_name='save_base_dir',
            default_pkg='path_planner',
            default_relative_path='saved_paths'
        )

        # ============================================================
        # Resolve Files
        # ============================================================
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
        # Robot radius mapping from config
        # ============================================================
        self.robot_radius = {}
        # robot_data = self.config_data.get('robots', {})
        # for robot_name in self.robot_names:
        #     if robot_name in robot_data:
        #         self.robot_radius[robot_name] = robot_data[robot_name].get(
        #             'radius', 
        #             self.default_robot_radius
        #         )
        #     else:
        #         self.robot_radius[robot_name] = self.default_robot_radius
        # self.get_logger().info(f"Robot radius mapping: {self.robot_radius}")
        for idx, robot_name in enumerate(self.robot_names):
            self.robot_radius[robot_name] = self.default_robot_radius
        
        # ============================================================
        # Odometry subscribers for current poses
        # ============================================================
        self.get_logger().info("="*60)
        self.get_logger().info("Setting up odometry subscribers...")
        self.get_logger().info("="*60)
        
        self.robot_current_poses = {}
        self.robot_odom_subs = {}
        
        for idx, robot_name in enumerate(self.robot_names):
            topic_name = f"/{robot_name}/odom_world"

            sub = self.create_subscription(
                Odometry,
                topic_name,
                lambda msg, rn=robot_name: self._odom_callback(msg, rn),
                10
            )
            self.robot_odom_subs[robot_name] = sub
            self.get_logger().info(f"Subscribed to {topic_name}")
            
            # Default pose with X spacing
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
    
        # Color mapping for robots
        self.robot_colors = {}
        default_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
        for idx, robot_name in enumerate(self.robot_names):
            self.robot_colors[robot_name] = default_colors[idx % len(default_colors)]
        
        self.get_logger().info(f"Robot colors: {self.robot_colors}")
        
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
            RobotPathArray,
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
        
        self.get_logger().debug(
                f"[ODOM] {robot_name}: x={position.x:.3f}, y={position.y:.3f}, yaw={yaw:.3f}"
            )

    def _display_initial_map(self):
        """Display the map on startup (non-blocking)"""
        try:
            plt.ion()  # Interactive mode
            fig, ax = plt.subplots(figsize=(8, 8))
            
            self.get_logger().debug(f'Creating initial map visualization:')
            self.get_logger().debug(f'  - Grid shape: {self.grid.shape}')
            self.get_logger().debug(f'  - Extent: [0, {self.W * self.resolution}, 0, {self.H * self.resolution}]')
            
            # Display map flipped vertically to match coordinate system
            ax.imshow(
                self.grid[::-1],
                cmap="gray_r",
                extent=[0, self.W * self.resolution, 0, self.H * self.resolution],
                alpha=0.9
            )
            
            # Draw robot initial positions
            for robot_name in self.robot_names:
                pose = self.robot_current_poses[robot_name]
                radius = self.robot_radius.get(robot_name, self.default_robot_radius)
                color = self.robot_colors[robot_name]
                
                self.get_logger().debug(
                    f'  - Drawing {robot_name} at ({pose["x"]:.3f}, {pose["y"]:.3f}), '
                    f'radius={radius}, color={color}'
                )
                
                circle = Circle(
                    (pose['x'], pose['y']), 
                    radius, 
                    fill=False, 
                    lw=2, 
                    color=color,
                    label=robot_name
                )
                ax.add_patch(circle)
                ax.plot(pose['x'], pose['y'], 'o', color=color, ms=6)
            
            ax.set_xlim(0, self.W * self.resolution)
            ax.set_ylim(0, self.H * self.resolution)
            ax.set_aspect("equal")
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_title("Path Planner - Initial Map with Robot Positions")
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
        """Handle incoming path planning requests (synchronous processing)"""
        
        self.stats['total_requests'] += 1
        self._current_request_id = self.stats['total_requests']
        
        # Check if planning is already active
        if self._planning_active:
            self.get_logger().warn("="*60)
            self.get_logger().warn(f"New request #{self._current_request_id} received while planning in progress")
            self.get_logger().warn("Cancelling current planning...")
            self.get_logger().warn("="*60)
            self._cancel_current_planning = True
            # Wait briefly for current planning to cancel
            time.sleep(0.1)
        
        self.get_logger().info("="*60)
        self.get_logger().info(f"Processing planning request #{self._current_request_id}")
        self.get_logger().info("="*60)
        
        # Validate message fields
        if not hasattr(msg, 'plan_json') or not msg.plan_json:
            self.get_logger().error("="*60)
            self.get_logger().error("ERROR: Missing or empty 'plan_json' field in PathPlannerRequest message")
            self.get_logger().error("="*60)
            self.stats['failed'] += 1
            return
        
        # Parse JSON
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
        
        # Validate plan_json structure
        if not isinstance(plan_json, dict):
            self.get_logger().error("="*60)
            self.get_logger().error(f"ERROR: plan_json must be a dictionary, got {type(plan_json)}")
            self.get_logger().error("="*60)
            self.stats['failed'] += 1
            return
        
        if not plan_json:
            self.get_logger().error("="*60)
            self.get_logger().error("ERROR: plan_json is empty - no robot groups found")
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
        """Plan paths for all robots in the request
        
        Returns:
            bool: True if planning succeeded, False otherwise
        """
        
        try:
            # Extract robot goals from the plan_json
            self.get_logger().info("="*60)
            self.get_logger().info("Extracting robot goals from plan...")
            self.get_logger().info("="*60)
            
            robot_goals = self._extract_robot_goals(plan_json)
            
            if not robot_goals:
                self.get_logger().warn("="*60)
                self.get_logger().warn("No robot goals extracted from plan")
                self.get_logger().warn("="*60)
                return False
            
            self.get_logger().info(f"Planning for {len(robot_goals)} robots")
            for robot_name, goal_info in robot_goals.items():
                self.get_logger().info(f"  - {robot_name}: goal={goal_info['goal']}")
            
            # Sequential planning with dynamic obstacles
            self.get_logger().info("="*60)
            self.get_logger().info("Starting sequential path planning...")
            self.get_logger().info("="*60)
            
            dynamic_obstacles = []
            robot_paths = {}
            robot_agents = {}
            
            for robot_name, goal_info in robot_goals.items():
                
                # Check for cancellation
                if self._cancel_current_planning:
                    self.get_logger().warn(f"Planning cancelled before {robot_name}")
                    return False
                
                self.get_logger().info(f"\n>>> Planning path for {robot_name}...")
                
                # Get start configuration from odometry (2D)
                start = self._get_start_pose(robot_name)
                if start is None:
                    self.get_logger().error(f"Cannot get start pose for {robot_name}")
                    continue
                
                self.get_logger().info(f"  Start: [{start[0]:.3f}, {start[1]:.3f}]")
                
                # Get goal (2D)
                goal = goal_info['goal']
                self.get_logger().info(f"  Goal:  [{goal[0]:.3f}, {goal[1]:.3f}]")
                
                # Get agent model with custom radius if provided
                agent = self._get_agent_model(robot_name, goal_info.get('radius'))
                robot_agents[robot_name] = agent
                self.get_logger().info(f"  Agent radius: {agent.radius}")
                
                # Get custom velocity if provided
                max_vel = goal_info.get('max_velocity', self.max_velocity)
                self.get_logger().info(f"  Max velocity: {max_vel}")
                
                # Log dynamic obstacles info
                if dynamic_obstacles:
                    self.get_logger().info(f"  Dynamic obstacles: {len(dynamic_obstacles)}")
                    for idx, obs in enumerate(dynamic_obstacles):
                        obs_traj = obs['trajectory']
                        self.get_logger().debug(
                            f"    Obstacle {idx}: {len(obs_traj)} waypoints, "
                            f"duration={obs_traj[-1][1]:.2f}s"
                        )
                else:
                    self.get_logger().info(f"  Dynamic obstacles: 0 (first robot)")
                
                # Plan path
                traj = self._plan_single_robot(
                    robot_name=robot_name,
                    agent=agent,
                    start=start,
                    goal=goal,
                    dynamic_obstacles=dynamic_obstacles,
                    max_velocity=max_vel
                )
                
                # Check for cancellation after planning
                if self._cancel_current_planning:
                    self.get_logger().warn(f"Planning cancelled after {robot_name}")
                    return False
                
                if traj is None:
                    self.get_logger().error("="*60)
                    self.get_logger().error(f"Planning failed for {robot_name}")
                    self.get_logger().error("Possible reasons:")
                    self.get_logger().error("  - No valid path exists (obstacles blocking)")
                    self.get_logger().error("  - Dynamic obstacle conflicts (timing issues)")
                    self.get_logger().error(f"  - Max iterations reached ({self.max_iter})")
                    self.get_logger().error(f"  - Time horizon too short ({self.time_horizon}s)")
                    self.get_logger().error("="*60)
                    self.get_logger().warn(f"Skipping {robot_name}, continuing with other robots...")
                    # Don't fail entire planning, just skip this robot
                    continue
                
                # Store trajectory
                robot_paths[robot_name] = traj
                
                # Add to dynamic obstacles for next robot
                dynamic_obstacles.append({
                    'trajectory': traj,
                    'agent': agent
                })
                
                self.get_logger().info(
                    f"  ✓ {robot_name}: planned {len(traj)} waypoints, "
                    f"arrival time: {traj[-1][1]:.2f}s"
                )
            
            # Check for cancellation before publishing
            if self._cancel_current_planning:
                self.get_logger().warn("Planning cancelled before publishing")
                return False
            
            # Publish paths
            if robot_paths:
                self.get_logger().info("="*60)
                self.get_logger().info(f"Publishing paths for {len(robot_paths)} robots")
                self.get_logger().info("="*60)
                
                self._publish_paths(robot_paths, robot_agents)
                
                # Save if enabled
                if self.save_paths:
                    self.get_logger().info("="*60)
                    self.get_logger().info("Saving trajectories to disk...")
                    self.get_logger().info("="*60)
                    self._save_trajectories(robot_paths, robot_agents)
                
                # Show final visualization with all robot paths
                if self.show_path_visualization:
                    self.get_logger().info("="*60)
                    self.get_logger().info("Displaying final paths for all robots...")
                    self.get_logger().info("="*60)
                    self._visualize_all_paths(robot_paths, robot_agents)
                
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
    
    def _extract_robot_goals(self, plan_json: dict) -> dict:
        """Extract individual robot goals from plan_json
        
        Returns:
            dict: {robot_name: {'goal': [x, y], 'radius': r, 'max_velocity': v, ...}}
        """
        robot_goals = {}
        missing_fields_errors = []
        
        for group_name, group_data in plan_json.items():
            
            self.get_logger().debug(f"  Processing group: {group_name}")
            
            # Formation group (F*) - IGNORED FOR NOW
            if group_name.startswith('F') and isinstance(group_data, dict):
                self.get_logger().warn(f"  - Ignoring formation group {group_name} (not implemented yet)")
                continue
            
            # Individual robot groups (R*)
            elif group_name.startswith('R') and isinstance(group_data, list):
                self.get_logger().debug(f"  - Processing {len(group_data)} robots in {group_name}")
                    
                for robot_entry in group_data:
                    # Validate robot entry has 'robot' field
                    robot_name = robot_entry.get('robot')
                    if not robot_name:
                        error_msg = f"Robot entry in {group_name} missing 'robot' field: {robot_entry}"
                        missing_fields_errors.append(error_msg)
                        self.get_logger().error(f"ERROR: {error_msg}")
                        continue
                    
                    # Validate required position fields
                    if 'x' not in robot_entry:
                        error_msg = f"Robot '{robot_name}' missing required field 'x'"
                        missing_fields_errors.append(error_msg)
                        self.get_logger().error(f"ERROR: {error_msg}")
                        continue
                    
                    if 'y' not in robot_entry:
                        error_msg = f"Robot '{robot_name}' missing required field 'y'"
                        missing_fields_errors.append(error_msg)
                        self.get_logger().error(f"ERROR: {error_msg}")
                        continue
                    
                    # FIX #1: Use 2D goal array [x, y] instead of 3D [x, y, yaw]
                    try:
                        goal = np.array([
                            float(robot_entry.get('x')),
                            float(robot_entry.get('y'))
                        ])
                    except (ValueError, TypeError) as e:
                        error_msg = f"Robot '{robot_name}' has invalid coordinate values: {e}"
                        missing_fields_errors.append(error_msg)
                        self.get_logger().error(f"ERROR: {error_msg}")
                        continue
                    
                    robot_goals[robot_name] = {
                        'goal': goal,
                        'radius': robot_entry.get('radius', None),
                        'max_velocity': robot_entry.get('max_velocity', None),
                        'colour': robot_entry.get('colour', None),
                        'type': robot_entry.get('type', None)
                    }
                    
                    self.get_logger().info(
                        f"    - {robot_name}: goal=[{goal[0]:.3f}, {goal[1]:.3f}]"
                    )
        
        # Report summary of errors if any
        if missing_fields_errors:
            self.get_logger().error("="*60)
            self.get_logger().error(f"Found {len(missing_fields_errors)} validation errors in plan_json")
            self.get_logger().error("="*60)


        self.get_logger().debug(f"Robot Goals: {robot_goals}")

        return robot_goals
    
    def _get_start_pose(self, robot_name: str) -> Optional[np.ndarray]:
        """Get start pose for robot from odometry - return 2D [x, y]"""
        
        if robot_name in self.robot_current_poses:
            pose = self.robot_current_poses[robot_name]
            # Return 2D like original code
            start = np.array([pose['x'], pose['y']])
            
            self.get_logger().debug(
                    f"Got start pose from odometry for {robot_name}: [{start[0]:.3f}, {start[1]:.3f}]"
                )
            
            return start
        else:
            self.get_logger().warn(
                f"No odometry data for {robot_name}, using default (0, 0)"
            )
            return np.array([0.0, 0.0])
    
    def _get_agent_model(self, robot_name: str, custom_radius: Optional[float] = None):
        """Get individual agent model for robot"""
        if custom_radius is not None:
            radius = custom_radius
            self.get_logger().debug(f"Using custom radius {radius} for {robot_name}")
        else:
            radius = self.robot_radius.get(robot_name, self.default_robot_radius)
            self.get_logger().debug(f"Using default radius {radius} for {robot_name}")
        
        return IndividualAgent(radius=radius)
    
    def _plan_single_robot(self, robot_name: str, agent, start: np.ndarray,
                          goal: np.ndarray, dynamic_obstacles: list,
                          max_velocity: float = None):
        """Plan path for a single robot using SI-RRT
        
        Uses 2D configurations [x, y] like the original code.
        """
        
        if max_velocity is None:
            max_velocity = self.max_velocity
        
        self.get_logger().debug(f"  Creating SI-RRT planner for {robot_name}...")
        self.get_logger().debug(f"  - Start: {start}")
        self.get_logger().debug(f"  - Goal: {goal}")
        self.get_logger().debug(f"  - Max velocity: {max_velocity}")
        self.get_logger().debug(f"  - Dynamic obstacles: {len(dynamic_obstacles)}")
        self.get_logger().debug(f"  - Max iterations: {self.max_iter}")
        self.get_logger().debug(f"  - Time horizon: {self.time_horizon}s")
        
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
            debug=self.debug
        )
        
        try:
            self.get_logger().debug(f"  Running planner.plan()...")
                
            traj = planner.plan(start, goal, dynamic_obstacles)
            
            if traj is None:
                self.get_logger().error(f"  Planner returned None for {robot_name}")
                return None
            
            self.get_logger().debug(f"  Planning succeeded: {len(traj)} waypoints")
            self.get_logger().debug(f"  First waypoint: {traj[0]}")
            self.get_logger().debug(f"  Last waypoint: {traj[-1]}")
            
            return traj
            
        except Exception as e:
            self.get_logger().error("="*60)
            self.get_logger().error(f"  Planning exception for {robot_name}: {e}")
            self.get_logger().error("="*60)
            traceback.print_exc()
            return None
    
    def _publish_paths(self, robot_paths: dict, robot_agents: dict):
        """Publish computed paths as RobotPathArray message"""
        
        msg = RobotPathArray()
        
        for robot_name, traj in robot_paths.items():
            robot_path = RobotPath()
            robot_path.robot_name = robot_name
            
            times = []
            x_positions = []
            y_positions = []
            
            for q, t in traj:
                q = np.asarray(q)
                times.append(float(t))
                x_positions.append(float(q[0]))
                y_positions.append(float(q[1]))
            
            robot_path.times = times
            robot_path.x_positions = x_positions
            robot_path.y_positions = y_positions
            
            msg.paths.append(robot_path)
            
            self.get_logger().info(
                f"  - {robot_name}: {len(times)} points, duration={times[-1]:.2f}s"
            )
        
        self.path_pub.publish(msg)
        self.get_logger().info("Paths published successfully")
    
    def _save_trajectories(self, robot_paths: dict, robot_agents: dict):
        """Save trajectories to disk with auto-versioning - matches original code format
        
          FIX #2: Properly extract disc positions from agent.discs()
        """
        
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
        
        saved = {}
        
        for robot_name, traj in robot_paths.items():
            agent = robot_agents[robot_name]
            times = np.array([t for _, t in traj])
            
            # ✅ Extract all disc positions (matches original code)
            all_robot_paths = []
            for q, _ in traj:
                discs = agent.discs(q)
                all_robot_paths.append([p for p, _ in discs])
            
            all_robot_paths = np.array(all_robot_paths)  # [T, N, 2]
            
            robot_trajs = {}
            
            # Save each disc/robot position separately
            for r in range(all_robot_paths.shape[1]):
                arr = np.zeros((len(times), 3))
                arr[:, 0] = all_robot_paths[:, r, 0]
                arr[:, 1] = all_robot_paths[:, r, 1]
                arr[:, 2] = times
                
                robot_trajs[f"robot_{r}"] = arr
                np.save(run_dir / f"{robot_name}_robot_{r}.npy", arr)
            
            saved[robot_name] = {
                "times": times,
                "robots": robot_trajs
            }
            
            self.get_logger().debug(f"  - Saved {robot_name} with {len(robot_trajs)} discs")
        
        summary_path = run_dir / "all_agents_paths.npy"
        np.save(summary_path, saved, allow_pickle=True)
        self.get_logger().info(f"Saved {len(robot_paths)} robot trajectories")
    
    def _visualize_all_paths(self, robot_paths: dict, robot_agents: dict):
        """Visualize all planned paths - matches original code visualization
        
          FIX #3: Add disc rendering at start and goal positions
        """
        
        self.get_logger().info(f'Visualizing {len(robot_paths)} robot paths')
        
        try:
            # Close any existing plots first
            plt.close('all')
            
            plt.ion()
            fig, ax = plt.subplots(figsize=(10, 8))
            
            self.get_logger().debug(f'Map dimensions: {self.W}x{self.H}, resolution: {self.resolution}m/cell')
            self.get_logger().debug(f'Map extent: [0, {self.W * self.resolution}, 0, {self.H * self.resolution}]')
            
            # Draw map flipped vertically
            ax.imshow(
                self.grid[::-1],
                cmap="gray_r",
                extent=[0, self.W * self.resolution, 0, self.H * self.resolution],
                alpha=0.9
            )
            
            # Plot all trajectories
            for robot_name, traj in robot_paths.items():
                self.get_logger().debug(f'Plotting trajectory for {robot_name}')
                
                agent = robot_agents[robot_name]
                color = self.robot_colors.get(robot_name, 'gray')
                
                # Extract positions (x, y) from trajectory
                pts = np.array([np.asarray(q)[:2] for q, _ in traj])
                
                # Plot the trajectory line
                ax.plot(pts[:, 0], pts[:, 1], lw=2, color=color, label=robot_name)
                
                #  FIX #3: Draw start discs (solid-ish)
                start_q = traj[0][0]
                for p, r in agent.discs(start_q):
                    ax.add_patch(Circle(p, r, color=color, alpha=0.25))
                    ax.add_patch(Circle(p, r, fill=False, lw=2, color=color, alpha=0.8))
                
                #  FIX #3: Draw goal discs (outline only)
                goal_q = traj[-1][0]
                for p, r in agent.discs(goal_q):
                    ax.add_patch(Circle(p, r, fill=False, lw=2, linestyle="--", color=color))
                
                #  FIX #4: Mark centroid start and goal (no extra labels)
                ax.plot(pts[0, 0], pts[0, 1], 'o', color=color, ms=6)
                ax.plot(pts[-1, 0], pts[-1, 1], 'x', color=color, ms=8, mew=2)
                
                self.get_logger().debug(
                    f'  - Start: ({pts[0, 0]:.3f}, {pts[0, 1]:.3f})'
                )
                self.get_logger().debug(
                    f'  - Goal: ({pts[-1, 0]:.3f}, {pts[-1, 1]:.3f})'
                )
                self.get_logger().debug(
                    f'  - Waypoints: {len(pts)}'
                )
            
            ax.set_xlim(0, self.W * self.resolution)
            ax.set_ylim(0, self.H * self.resolution)
            ax.set_aspect("equal")
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.legend(loc="upper right")
            ax.set_title(f"Path Planner - {len(robot_paths)} Robot Paths (SI-RRT + CPP)")
            
            plt.draw()
            plt.pause(0.1)
            
            self.get_logger().info('Robot paths displayed. Close window to continue.')
            
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