#!/usr/bin/env python3
"""
Plot trajectories from CSV files for multiple robots
Reads CSV files from package config directory and visualizes trajectories
"""

import os
import csv
import matplotlib.pyplot as plt
from ament_index_python.packages import get_package_share_directory


def load_trajectory_from_csv(csv_path):
    """
    Load trajectory data from a CSV file
    
    Args:
        csv_path: Full path to CSV file
    
    Returns:
        Dictionary with 'time', 'x', 'y', and optionally 'theta' lists
        Returns None if file cannot be loaded
    """
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}")
        return None
    
    try:
        with open(csv_path, 'r') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            fieldnames = csv_reader.fieldnames
            
            # Check for required columns
            if 'time' not in fieldnames or 'x' not in fieldnames or 'y' not in fieldnames:
                print(f"ERROR: CSV must contain 'time', 'x', 'y' columns. Found: {fieldnames}")
                return None
            
            # Initialize data storage
            data = {
                'time': [],
                'x': [],
                'y': [],
                'theta': [] if 'theta' in fieldnames else None
            }
            
            # Read all rows
            for row in csv_reader:
                try:
                    data['time'].append(float(row['time']))
                    data['x'].append(float(row['x']))
                    data['y'].append(float(row['y']))
                    
                    if 'theta' in fieldnames:
                        data['theta'].append(float(row['theta']))
                        
                except (ValueError, KeyError) as e:
                    print(f"Warning: Skipping invalid row: {e}")
                    continue
            
            if len(data['time']) == 0:
                print(f"ERROR: No valid data found in {csv_path}")
                return None
            
            print(f"Loaded {len(data['time'])} waypoints from {os.path.basename(csv_path)}")
            return data
            
    except Exception as e:
        print(f"ERROR loading {csv_path}: {e}")
        return None


def plot_trajectories(package_name, file_names):
    """
    Plot multiple trajectories from CSV files in a single XY plot
    
    Args:
        package_name: Name of the ROS2 package containing config folder
        file_names: List of CSV filenames (without .csv extension)
    """
    # Get package share directory
    try:
        package_share = get_package_share_directory(package_name)
    except Exception as e:
        print(f"ERROR: Could not find package '{package_name}': {e}")
        return
    
    config_dir = os.path.join(package_share, "trajectory_logs")
    print(f"Looking for CSV files in: {config_dir}")
    
    # Load all trajectories
    trajectories = {}
    for file_name in file_names:
        csv_file = file_name if file_name.endswith('.csv') else file_name + '.csv'
        csv_path = os.path.join(config_dir, csv_file)
        
        data = load_trajectory_from_csv(csv_path)
        if data is not None:
            trajectories[file_name] = data
    
    if len(trajectories) == 0:
        print("ERROR: No valid trajectories loaded!")
        return
    
    # Create single figure for XY plot
    plt.figure(figsize=(12, 10))
    
    # Define colors for different robots
    colors = plt.cm.tab10(range(len(trajectories)))
    
    # Plot each trajectory
    for idx, (name, data) in enumerate(trajectories.items()):
        color = colors[idx]
        
        # Plot trajectory path
        plt.plot(data['x'], data['y'], 
                marker='o', markersize=2, 
                label=name, 
                linewidth=2, 
                color=color,
                alpha=0.7)
        
        # Mark start point (larger green circle)
        plt.plot(data['x'][0], data['y'][0], 
                'o', markersize=12, 
                color=color, 
                markeredgecolor='green', 
                markeredgewidth=3,
                label=f'{name} start')
        
        # Mark end point (larger red square)
        plt.plot(data['x'][-1], data['y'][-1], 
                's', markersize=12, 
                color=color, 
                markeredgecolor='red', 
                markeredgewidth=3,
                label=f'{name} end')
    
    plt.xlabel('X Position (m)', fontsize=14, fontweight='bold')
    plt.ylabel('Y Position (m)', fontsize=14, fontweight='bold')
    plt.title('Robot Trajectories (XY Plane)', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.axis('equal')
    
    # Add minor gridlines for better readability
    plt.grid(True, which='minor', alpha=0.2, linestyle=':')
    plt.minorticks_on()
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function to plot trajectories"""
    
    # Configuration
    package_name = "burger_robot"  # Change to your package name
    
    # List of CSV files to plot (without .csv extension)
    file_names = [
        "HeteroForm2_robot0_diff-drive",
        "HeteroForm2_robot1_holonomic",
        "HeteroForm2_robot2_diff-drive",
        "HeteroForm_robot0_diff-drive",
        "HeteroForm_robot1_holonomic",
        "HeteroForm_robot2_diff-drive",
    ]
    
    # Alternative: Use the same file for all
    # file_names = ["input"]
    
    print(f"Package: {package_name}")
    print(f"Files to plot: {file_names}")
    print("-" * 50)
    
    # Plot trajectories
    plot_trajectories(package_name, file_names)


if __name__ == '__main__':
    main()