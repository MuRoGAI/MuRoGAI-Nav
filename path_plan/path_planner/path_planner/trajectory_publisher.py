#!/usr/bin/env python3
"""
Flexible trajectory publisher that handles CSV files with missing columns
"""

import rclpy
from rclpy.node import Node
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


from path_planner_interface.msg import (
    PathPlannerRequest,
    RobotTrajectory,
    RobotTrajectoryArray,
    DiffDriveTrajectory,
    HoloTrajectory
)

class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner_node')
        
        # Publisher
        self.path_pub = self.create_publisher(
            RobotTrajectoryArray,
            '/path_planner/paths',
            10
        )
        
        # Trajectory configuration
        self.TRAJECTORY_FILES = {
            'robot1': 'HeteroForm2_robot0_diff-drive.csv',
            'robot2': 'HeteroForm2_robot1_holonomic.csv',
            'robot3': 'HeteroForm2_robot2_diff-drive.csv',
            'robot4': 'HeteroForm_robot0_diff-drive.csv',
            'robot5': 'HeteroForm_robot1_holonomic.csv',
            'robot6': 'HeteroForm_robot2_diff-drive.csv',
        }
        
        # Base directory
        self.trajectory_dir = Path('/home/multi-robot/comuros/src/CoMuRoS/CoMuRoS/path_plan/path_planner/path_planner/trajectory_logs1')
        
        self.get_logger().info(f'Path Planner initialized. Trajectory dir: {self.trajectory_dir}')
    
    def load_trajectories_from_files(self):
        """
        Load all trajectory files and create all_results structure
        Handles missing columns gracefully by filling with zeros/defaults
        
        Returns:
            list: all_results in the format expected by _publish_paths
        """
        all_results = []
        
        self.get_logger().info("Loading trajectory files...")
        
        for robot_name, filename in self.TRAJECTORY_FILES.items():
            file_path = self.trajectory_dir / filename
            
            # Determine robot type from filename
            if 'diff-drive' in filename:
                robot_type = 'diff-drive'
            elif 'holonomic' in filename:
                robot_type = 'holonomic'
            else:
                self.get_logger().warn(f"Unknown type for {filename}, using diff-drive")
                robot_type = 'diff-drive'
            
            # Load CSV file
            try:
                df = pd.read_csv(file_path)
                self.get_logger().info(
                    f"  ✓ {robot_name}: {len(df)} waypoints from {filename}"
                )
                self.get_logger().info(f"    Available columns: {list(df.columns)}")
                
                # Add missing columns with default values
                df = self._add_missing_columns(df, robot_type)
                
                # Convert DataFrame to trajectory format
                traj = []
                
                if robot_type == 'diff-drive':
                    for _, row in df.iterrows():
                        # [x, y, theta, control_obj]
                        traj.append([row['x'], row['y'], row['theta'], None])
                
                else:  # holonomic
                    for _, row in df.iterrows():
                        # [x, y, vx, control_obj]
                        traj.append([row['x'], row['y'], row['vx'], None])
                
                # Create result dictionary
                result = {
                    'name': robot_name,
                    'agent': None,  # Not needed for file loading
                    'traj': traj,
                    'robot_names': [robot_name],
                    'robot_types': [robot_type],
                    'dataframe': df  # Store for plotting
                }
                
                all_results.append(result)
                
            except FileNotFoundError:
                self.get_logger().error(f"  ✗ File not found: {file_path}")
            except Exception as e:
                self.get_logger().error(f"  ✗ Error loading {filename}: {str(e)}")
                import traceback
                self.get_logger().error(traceback.format_exc())
        
        self.get_logger().info(f"Loaded {len(all_results)} trajectories successfully")
        return all_results
    
    def _add_missing_columns(self, df, robot_type):
        """
        Add any missing columns with appropriate default values
        
        Args:
            df: DataFrame from CSV
            robot_type: 'diff-drive' or 'holonomic'
        
        Returns:
            DataFrame with all required columns
        """
        n_points = len(df)
        
        # Common columns
        if 't' not in df.columns:
            # Create time array if missing
            if 'time' in df.columns:
                df['t'] = df['time']
            else:
                df['t'] = np.arange(n_points) * 0.1  # Assume 10Hz
                self.get_logger().warn(f"  ! 't' column missing, created with dt=0.1s")
        
        if 'x' not in df.columns:
            df['x'] = np.zeros(n_points)
            self.get_logger().warn(f"  ! 'x' column missing, filled with zeros")
        
        if 'y' not in df.columns:
            df['y'] = np.zeros(n_points)
            self.get_logger().warn(f"  ! 'y' column missing, filled with zeros")
        
        if robot_type == 'diff-drive':
            # Diff-drive specific columns
            if 'theta' not in df.columns:
                # Try to compute from velocities or positions
                if 'v' in df.columns and 'omega' in df.columns:
                    # Integrate angular velocity
                    dt = df['t'].diff().fillna(0.1)
                    df['theta'] = (df['omega'] * dt).cumsum()
                    self.get_logger().warn(f"  ! 'theta' computed from omega")
                else:
                    # Compute from position changes
                    dx = df['x'].diff().fillna(0)
                    dy = df['y'].diff().fillna(0)
                    df['theta'] = np.arctan2(dy, dx)
                    df['theta'].iloc[0] = 0  # First point
                    self.get_logger().warn(f"  ! 'theta' computed from position")
            
            if 'v' not in df.columns:
                # Compute linear velocity from position
                dx = df['x'].diff().fillna(0)
                dy = df['y'].diff().fillna(0)
                dt = df['t'].diff().fillna(0.1)
                dt[dt == 0] = 0.1  # Avoid division by zero
                df['v'] = np.sqrt(dx**2 + dy**2) / dt
                df['v'].iloc[0] = 0  # First point
                self.get_logger().warn(f"  ! 'v' computed from position")
            
            if 'omega' not in df.columns:
                # Compute angular velocity from theta
                if 'theta' in df.columns:
                    dtheta = df['theta'].diff().fillna(0)
                    dt = df['t'].diff().fillna(0.1)
                    dt[dt == 0] = 0.1
                    df['omega'] = dtheta / dt
                    df['omega'].iloc[0] = 0
                else:
                    df['omega'] = np.zeros(n_points)
                self.get_logger().warn(f"  ! 'omega' computed/filled with zeros")
        
        else:  # holonomic
            if 'vx' not in df.columns:
                # Compute from position
                dx = df['x'].diff().fillna(0)
                dt = df['t'].diff().fillna(0.1)
                dt[dt == 0] = 0.1
                df['vx'] = dx / dt
                df['vx'].iloc[0] = 0
                self.get_logger().warn(f"  ! 'vx' computed from position")
            
            if 'vy' not in df.columns:
                # Compute from position
                dy = df['y'].diff().fillna(0)
                dt = df['t'].diff().fillna(0.1)
                dt[dt == 0] = 0.1
                df['vy'] = dy / dt
                df['vy'].iloc[0] = 0
                self.get_logger().warn(f"  ! 'vy' computed from position")
        
        return df
    
    def _flatten_trajectory_data(self, traj, control_trajs, agent, robot_types):
        """
        Flatten trajectory data for each robot
        """
        flattened_data = []
        
        for i, robot_type in enumerate(robot_types):
            # Get the corresponding dataframe if available
            # Since we're only dealing with one robot per result, use index 0
            data = {
                'x': np.array([pt[0] for pt in traj]),
                'y': np.array([pt[1] for pt in traj]),
            }
            
            if robot_type == 'diff-drive':
                data['theta'] = np.array([pt[2] for pt in traj])
                # These will be filled from the dataframe in _publish_paths
                data['t'] = np.zeros(len(traj))
                data['v'] = np.zeros(len(traj))
                data['omega'] = np.zeros(len(traj))
            else:  # holonomic
                data['vx'] = np.array([pt[2] for pt in traj])
                data['t'] = np.zeros(len(traj))
                data['vy'] = np.zeros(len(traj))
            
            flattened_data.append(data)
        
        return flattened_data
    
    def _publish_paths(self, all_results: list):
        """
        Publish computed paths using RobotTrajectoryArray message
        """
        msg = RobotTrajectoryArray()
        
        for result in all_results:
            name = result['name']
            traj = result['traj']
            robot_names = result['robot_names']
            robot_types = result['robot_types']
            df = result.get('dataframe')
            
            # Debug logging
            self.get_logger().info(f"[DEBUG] Processing trajectory for {name}:")
            self.get_logger().info(f"  - Number of waypoints: {len(traj)}")
            
            # Create message for each robot
            for i, robot_name in enumerate(robot_names):
                robot_msg = RobotTrajectory()
                robot_msg.robot_name = robot_name
                robot_msg.robot_type = robot_types[i]
                
                if robot_types[i] == 'diff-drive':
                    dd_traj = DiffDriveTrajectory()
                    # Use data from dataframe if available
                    if df is not None:
                        dd_traj.time = df['t'].tolist()
                        dd_traj.x = df['x'].tolist()
                        dd_traj.y = df['y'].tolist()
                        dd_traj.theta = df['theta'].tolist()
                        dd_traj.v = df['v'].tolist()
                        dd_traj.omega = df['omega'].tolist()
                    else:
                        # Fallback to traj data
                        dd_traj.time = list(range(len(traj)))
                        dd_traj.x = [pt[0] for pt in traj]
                        dd_traj.y = [pt[1] for pt in traj]
                        dd_traj.theta = [pt[2] for pt in traj]
                        dd_traj.v = [0.0] * len(traj)
                        dd_traj.omega = [0.0] * len(traj)
                    
                    robot_msg.diff_drive_trajectories = [dd_traj]
                    robot_msg.holo_trajectories = []
                    
                    self.get_logger().info(
                        f"  - {robot_name}: {len(dd_traj.time)} points, "
                        f"duration={dd_traj.time[-1]:.2f}s"
                    )
                
                elif robot_types[i] == 'holonomic':
                    holo_traj = HoloTrajectory()
                    # Use data from dataframe if available
                    if df is not None:
                        holo_traj.time = df['t'].tolist()
                        holo_traj.x = df['x'].tolist()
                        holo_traj.y = df['y'].tolist()
                        holo_traj.vx = df['vx'].tolist()
                        holo_traj.vy = df['vy'].tolist()
                    else:
                        # Fallback to traj data
                        holo_traj.time = list(range(len(traj)))
                        holo_traj.x = [pt[0] for pt in traj]
                        holo_traj.y = [pt[1] for pt in traj]
                        holo_traj.vx = [pt[2] for pt in traj]
                        holo_traj.vy = [0.0] * len(traj)
                    
                    robot_msg.diff_drive_trajectories = []
                    robot_msg.holo_trajectories = [holo_traj]
                    
                    self.get_logger().info(
                        f"  - {robot_name}: {len(holo_traj.time)} points, "
                        f"duration={holo_traj.time[-1]:.2f}s"
                    )
                
                msg.robot_trajectories.append(robot_msg)
        
        # Publish the message
        self.path_pub.publish(msg)
        self.get_logger().info(
            f"✓ Paths published successfully for {len(msg.robot_trajectories)} robots"
        )
    
    def plot_trajectories(self, all_results: list):
        """
        Plot all trajectories after publishing
        """
        if not all_results:
            self.get_logger().warn("No trajectories to plot")
            return
        
        self.get_logger().info("Generating trajectory plot...")
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Color palette
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
        
        for idx, result in enumerate(all_results):
            robot_name = result['robot_names'][0]
            robot_type = result['robot_types'][0]
            
            # Use DataFrame if available
            if 'dataframe' in result:
                df = result['dataframe']
                x_data = df['x'].values
                y_data = df['y'].values
                has_df = True
            else:
                traj = result['traj']
                x_data = np.array([pt[0] for pt in traj])
                y_data = np.array([pt[1] for pt in traj])
                has_df = False
            
            # Plot main trajectory
            ax.plot(x_data, y_data,
                   color=colors[idx],
                   linewidth=2.5,
                   label=f"{robot_name} ({robot_type})",
                   alpha=0.85,
                   zorder=2)
            
            # Start marker
            ax.plot(x_data[0], y_data[0],
                   marker='o',
                   color=colors[idx],
                   markersize=14,
                   markeredgecolor='black',
                   markeredgewidth=2.5,
                   zorder=10,
                   label='_nolegend_')
            
            # Goal marker
            ax.plot(x_data[-1], y_data[-1],
                   marker='s',
                   color=colors[idx],
                   markersize=14,
                   markeredgecolor='black',
                   markeredgewidth=2.5,
                   zorder=10,
                   label='_nolegend_')
            
            # Add orientation arrows for diff-drive robots
            if robot_type == 'diff-drive' and has_df and 'theta' in df.columns:
                n_arrows = min(8, len(df))  # Don't exceed number of points
                if n_arrows > 1:
                    arrow_indices = np.linspace(0, len(df)-1, n_arrows, dtype=int)
                    
                    for i in arrow_indices:
                        x, y, theta = df.iloc[i][['x', 'y', 'theta']]
                        arrow_length = 0.5
                        dx = arrow_length * np.cos(theta)
                        dy = arrow_length * np.sin(theta)
                        
                        ax.arrow(x, y, dx, dy,
                                head_width=0.25,
                                head_length=0.18,
                                fc=colors[idx],
                                ec='black',
                                linewidth=1,
                                alpha=0.7,
                                zorder=5)
            
            # Add velocity arrows for holonomic robots
            elif robot_type == 'holonomic' and has_df and 'vx' in df.columns:
                n_arrows = min(8, len(df))
                if n_arrows > 1:
                    arrow_indices = np.linspace(0, len(df)-1, n_arrows, dtype=int)
                    
                    for i in arrow_indices:
                        row = df.iloc[i]
                        x, y, vx, vy = row['x'], row['y'], row['vx'], row['vy']
                        
                        v_mag = np.sqrt(vx**2 + vy**2)
                        if v_mag > 0.01:
                            scale_factor = 0.4 / v_mag
                            
                            ax.arrow(x, y, vx*scale_factor, vy*scale_factor,
                                    head_width=0.2,
                                    head_length=0.15,
                                    fc=colors[idx],
                                    ec='black',
                                    linewidth=1,
                                    alpha=0.7,
                                    zorder=5)
        
        # Formatting
        ax.set_xlabel('X Position (m)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y Position (m)', fontsize=14, fontweight='bold')
        ax.set_title('Multi-Robot Trajectory Paths', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
        ax.set_aspect('equal', adjustable='box')
        
        # Enhanced legend
        from matplotlib.lines import Line2D
        
        # Get existing handles and labels
        handles, labels = ax.get_legend_handles_labels()
        
        # Add marker descriptions
        marker_handles = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='gray', markersize=12,
                   label='Start Position',
                   markeredgecolor='black', markeredgewidth=2),
            Line2D([0], [0], marker='s', color='w',
                   markerfacecolor='gray', markersize=12,
                   label='Goal Position',
                   markeredgecolor='black', markeredgewidth=2)
        ]
        
        ax.legend(handles=handles + marker_handles,
                 loc='best',
                 fontsize=10,
                 framealpha=0.95,
                 edgecolor='black',
                 fancybox=True,
                 shadow=True)
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path('/tmp/multi_robot_trajectories.png')
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        self.get_logger().info(f"✓ Trajectory plot saved to: {output_path}")
        
        # Display plot
        plt.show()
    
    def run(self):
        """
        Main execution: Load trajectories, publish, and plot
        """
        # Step 1: Load trajectories from files
        all_results = self.load_trajectories_from_files()
        
        if not all_results:
            self.get_logger().error("No trajectories loaded! Check file paths.")
            return
        
        # Step 2: Publish trajectories
        self._publish_paths(all_results)
        
        # Step 3: Plot trajectories
        self.plot_trajectories(all_results)


def main(args=None):
    rclpy.init(args=args)
    
    node = PathPlannerNode()
    
    try:
        # Run the main workflow
        node.run()
        
        # Keep node alive briefly to ensure message delivery
        rclpy.spin_once(node, timeout_sec=2.0)
        
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user")
    except Exception as e:
        node.get_logger().error(f"Error: {str(e)}")
        import traceback
        node.get_logger().error(traceback.format_exc())
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()