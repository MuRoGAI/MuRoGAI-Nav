#!/usr/bin/env python3
"""
code: 2

Complete Test: Heterogeneous Formations + Individual Agents

Tests ALL robot types and configurations:
1. Individual diff-drive agents
2. Individual holonomic agents
3. Homogeneous formations (all DD)
4. HETEROGENEOUS formations (mixed DD + holonomic)

Every robot (individual or inside a formation) carries its own:
  - radius
  - v_max, omega_max   (velocity limits)
  - a_max, alpha_max   (acceleration limits)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.widgets import Slider, Button
import csv
import os

from si_rrt_enhanced_individual_kinodynamic import (
    SIRRT, OccupancyGrid
)

# Try to import heterogeneous formation steering
try:
    from heterogeneous_kinodynamic_formation_steering import *
    HETERO_AVAILABLE = True
except ImportError:
    HETERO_AVAILABLE = False
    print("Warning: Heterogeneous formation steering not available")


from agents1 import DifferentialDriveAgent, HeterogeneousFormationAgent, HolonomicAgent


# ============================================================
# Workspace + Map
# ============================================================

bounds = (np.array([0.0, 0.0]), np.array([20.0, 20.0]))
res = 0.1
H = W = int(20 / res)

grid = np.zeros((H, W), dtype=np.uint8)

def draw_circle(center, radius):
    cx, cy = center
    r2 = radius ** 2
    for i in range(H):
        for j in range(W):
            x = (j + 0.5) * res
            y = (i + 0.5) * res
            if (x - cx)**2 + (y - cy)**2 <= r2:
                grid[i, j] = 1

draw_circle((8.0, 10.0), 1.2)
draw_circle((12.0, 10.0), 1.2)

static_grid = OccupancyGrid(grid, res)

# ============================================================
# Agents - ALL TYPES  (each robot has its own limits)
# ============================================================

# --- Individual agents ---
dd_robot1 = DifferentialDriveAgent(
    radius=0.40,
    v_max=0.9, omega_max=2.0,
    a_max=1.8, alpha_max=3.5,
)
dd_robot2 = DifferentialDriveAgent(
    radius=0.35,
    v_max=0.8, omega_max=2.5,
    a_max=2.0, alpha_max=4.0,
)
holo_robot1 = HolonomicAgent(
    radius=0.40,
    v_max=1.0, a_max=2.0,
)
holo_robot2 = HolonomicAgent(
    radius=0.30,
    v_max=1.2, a_max=2.5,
)

# --- Heterogeneous formations (per-robot limits + per-robot radius) ---
hetero_form = HeterogeneousFormationAgent(
    P_star=[
        [-0.6, 0.0],  # Robot 0: Diff-drive
        [0.6, 0.0],   # Robot 1: Holonomic
        [0.0, 0.6]    # Robot 2: Diff-drive
    ],
    robot_types=['diff-drive', 'holonomic', 'diff-drive'],
    radius=[0.35, 0.30, 0.35],       # per-robot radii
    v_max=[0.8, 1.0, 0.8],           # per-robot velocity
    omega_max=[2.0, 0.0, 2.0],       # per-robot angular vel (0 for holonomic)
    a_max=[1.5, 2.0, 1.5],           # per-robot linear accel
    alpha_max=[3.0, 0.0, 3.0],       # per-robot angular accel (0 for holonomic)
)

hetero_form2 = HeterogeneousFormationAgent(
    P_star=[
        [-0.6, 0.0],  # Robot 0: Diff-drive
        [0.6, 0.0],   # Robot 1: Holonomic
        [0.0, 0.6]    # Robot 2: Diff-drive
    ],
    robot_types=['diff-drive', 'holonomic', 'diff-drive'],
    radius=[0.40, 0.35, 0.40],
    v_max=[0.7, 0.9, 0.7],
    omega_max=[1.8, 0.0, 1.8],
    a_max=[1.2, 1.8, 1.2],
    alpha_max=[2.5, 0.0, 2.5],
)

hetero_form3 = HeterogeneousFormationAgent(
    P_star=[
        [-0.6, 0.0],  # Robot 0: Diff-drive
        [0.6, 0.0],   # Robot 1: Holonomic
        [0.0, 0.6]    # Robot 2: Diff-drive
    ],
    robot_types=['diff-drive', 'holonomic', 'diff-drive'],
    radius=[0.30, 0.25, 0.30],
    v_max=[1.0, 1.2, 1.0],
    omega_max=[2.5, 0.0, 2.5],
    a_max=[2.0, 2.5, 2.0],
    alpha_max=[4.0, 0.0, 4.0],
)

agents = [

    ("HeteroForm", hetero_form,
     np.array([2.0, 10.0, 0.0, 1.0, 1.0]),
     np.array([18.0, 10.0, 0.0, 1.0, 1.0]),
     "heterogeneous-formation"),

    ("DD1", dd_robot1,
     np.array([3.0, 3.0, 0.0]),
     np.array([17.0, 3.0, 0.0]),
     "diff-drive"),

    ("Holo1", holo_robot1,
     np.array([10.0, 2.0]),
     np.array([10.0, 18.0]),
     "holonomic"),

    
    ("Holo2", holo_robot2,
     np.array([10.0, 18.0]),
     np.array([10.0, 2.0]),
     "holonomic"),

    ("HeteroForm2", hetero_form2,
     np.array([18.0, 10.0, 0.0, 1.0, 1.0]),
     np.array([2.0, 10.0, 0.0, 1.0, 1.0]),
     "heterogeneous-formation"),

    ("DD2", dd_robot2,
     np.array([17.0, 3.0, 0.0]),
     np.array([3.0, 3.0, 0.0]),
     "diff-drive"),

     ("HeteroForm3", hetero_form3,
     np.array([2.5, 15, 0.0, 1.0, 1.0]),
     np.array([17.5, 15.0, 0.0, 1.0, 1.0]),
     "heterogeneous-formation"),


]

colors = {
    "DD1": "tab:blue",
    "DD2": "tab:cyan",
    "Holo1": "tab:green",
    "HeteroForm": "tab:purple",
    "HeteroForm2": "tab:orange",
    "HeteroForm3": "tab:brown",
    "Holo2": "tab:red",
}

# ============================================================
# Planning
# ============================================================

print("\n" + "="*70)
print("HETEROGENEOUS KINODYNAMIC RRT*")
print("="*70)

dynamic_obstacles = []
trajectories = {}
agent_info = {}
control_trajectories = {}

for name, agent, start, goal, agent_type in agents:
    print(f"\nPlanning {name} ({agent_type})...")
    
    # ---------------------------------------------------------------
    # Build kino_params automatically from agent's stored limits
    # ---------------------------------------------------------------
    if agent_type == "heterogeneous-formation" and HETERO_AVAILABLE:
        # Pull per-robot limits directly from the agent object
        kino_params = {
            'robot_types': agent.robot_types,
            'v_max': agent.v_max_list,
            'w_max': agent.omega_max_list,
            'a_max': agent.a_max_list,
            'alpha_max': agent.alpha_max_list,
            'N_steer': 8,
            'T_steer': 0.8,
            'max_iter': 200,
        }
        use_kino = True
        use_hetero = True
    elif agent_type == "diff-drive":
        # Pull limits from the individual DD agent
        kino_params = {
            'v_max': agent.v_max,
            'omega_max': agent.omega_max,
            'a_max': agent.a_max,
            'alpha_max': agent.alpha_max,
            'dt': 0.05,
        }
        use_kino = True
        use_hetero = False
    elif agent_type == "holonomic":
        # Pull limits from the individual holonomic agent
        kino_params = {
            'v_max': agent.v_max,
            'a_max': agent.a_max,
            'dt': 0.05,
        }
        use_kino = True
        use_hetero = False
    else:
        use_kino = False
        use_hetero = False
        kino_params = None

    if agent_type == "heterogeneous-formation":
        max_vel = min(agent.v_max_list)
    elif hasattr(agent, "v_max"):
        max_vel = agent.v_max
    print("--"*30, "|-|"*10, "--"*30)
    print("-"*30, f"Max Velocity {max_vel}")
    print("--"*30, "|-|"*10, "--"*30)
    planner = SIRRT(
        agent_model=agent,
        max_velocity=max_vel,
        workspace_bounds=bounds,
        static_grid=static_grid,
        time_horizon=80.0,
        max_iter=2000,
        d_max=0.5,
        goal_sample_rate=0.22,
        neighbor_radius=1.5,
        precision=2,
        seed=211,
        debug=True,
        use_kinodynamic=use_kino,
        kinodynamic_params=kino_params,
    )
    
    # IMPORTANT: For heterogeneous formations, we need to replace the steering
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
    if traj is None:
        raise RuntimeError(f"Planning failed for {name}")

    trajectories[name] = traj
    
    # Extract control trajectories
    control_traj_list = []
    for item in traj:
        if len(item) >= 4 and item[3] is not None:
            control_traj_list.append(item[3])
    control_trajectories[name] = control_traj_list
    
    # For cooperative planning
    traj_for_obstacle = [(item[0], item[1]) for item in traj]
    dynamic_obstacles.append({"trajectory": traj_for_obstacle, "agent": agent})
    
    agent_info[name] = {
        'agent': agent,
        'type': agent_type,
        'use_kinodynamic': use_kino
    }

    print(f"  ✓ Complete: {traj[-1][1]:.1f}s, {len(traj)} waypoints")
    if control_traj_list:
        print(f"  ✓ Control trajectory segments: {len(control_traj_list)}")

print("\n✓ All agents planned!")

# ============================================================
# Save Control Trajectories
# ============================================================

print("\n" + "="*70)
print("SAVING CONTROL TRAJECTORIES")
print("="*70)

output_dir = "control_logs"
os.makedirs(output_dir, exist_ok=True)

for name in trajectories.keys():
    agent_type = agent_info[name]['type']
    control_list = control_trajectories[name]
    
    if not control_list:
        print(f"\n{name}: No control trajectory data")
        continue
    
    print(f"\n{name} ({agent_type}):")
    
    # Combine all segments
    all_x, all_y, all_t = [], [], []
    all_theta, all_v, all_omega, all_vy = [], [], [], []
    
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
    
    # Save to CSV
    csv_file = os.path.join(output_dir, f"{name}_controls.csv")
    
    with open(csv_file, 'w', newline='') as f:
        if agent_type == 'diff-drive':
            writer = csv.writer(f)
            writer.writerow(['time', 'x', 'y', 'theta', 'v', 'omega'])
            
            n_states = len(all_x)
            n_controls = len(all_v)
            
            for i in range(n_states):
                t_val = all_t[i] if i < len(all_t) else 0.0
                x_val = all_x[i]
                y_val = all_y[i]
                theta_val = all_theta[i] if i < len(all_theta) else 0.0
                v_val = all_v[i] if i < n_controls else 0.0
                omega_val = all_omega[i] if i < n_controls else 0.0
                
                writer.writerow([f"{t_val:.4f}", f"{x_val:.4f}", f"{y_val:.4f}", 
                               f"{theta_val:.4f}", f"{v_val:.4f}", f"{omega_val:.4f}"])
            
            print(f"  Saved {n_states} states to {csv_file}")
        
        elif agent_type == 'holonomic':
            writer = csv.writer(f)
            writer.writerow(['time', 'x', 'y', 'vx', 'vy'])
            
            n_states = len(all_x)
            n_controls = len(all_v)
            
            for i in range(n_states):
                t_val = all_t[i] if i < len(all_t) else 0.0
                x_val = all_x[i]
                y_val = all_y[i]
                vx_val = all_v[i] if i < n_controls else 0.0
                vy_val = all_vy[i] if i < len(all_vy) else 0.0
                
                writer.writerow([f"{t_val:.4f}", f"{x_val:.4f}", f"{y_val:.4f}", 
                               f"{vx_val:.4f}", f"{vy_val:.4f}"])
            
            print(f"  Saved {n_states} states to {csv_file}")

print("\n✓ Control logs saved!")

# ============================================================
# Extract Robot Paths
# ============================================================

def _wrap_angle(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

robot_trajectories = {}

for name, traj in trajectories.items():
    agent = agent_info[name]['agent']
    agent_type = agent_info[name]['type']
    
    if agent_type == "heterogeneous-formation":
        # Extract individual robot positions
        Nr = len(agent.P_star)
        robot_paths = [[] for _ in range(Nr)]
        
        for item in traj:
            q = item[0]
            t = item[1]
            psi = item[2] if len(item) >= 3 else None
            
            poses = agent.robot_poses(q)
            for i, (x, y, theta) in enumerate(poses):
                if psi is not None and i < len(psi):
                    theta_use = float(psi[i])
                else:
                    theta_use = float(theta)
                robot_paths[i].append(((x, y, theta_use), t))
        
        robot_trajectories[name] = robot_paths
    else:
        # Individual agent
        path = []
        for item in traj:
            q = item[0]
            t = item[1]
            poses = agent.robot_poses(q)
            x, y, theta = poses[0]
            path.append(((x, y, theta), t))
        robot_trajectories[name] = [path]

# ============================================================
# Visualization
# ============================================================

print("\n" + "="*70)
print("Creating visualization...")
print("="*70)

fig, ax = plt.subplots(figsize=(16, 10))
plt.subplots_adjust(bottom=0.15, right=0.80)

ax.imshow(grid[::-1], cmap="gray_r", extent=[0, 20, 0, 20], alpha=0.85)

# Plot paths
# ============================
# Build time vector
# ============================

all_times = sorted(set(
    t for paths in robot_trajectories.values()
    for rp in paths
    for (_, t) in rp
))
Tmax = all_times[-1]

# ============================
# Create robot artists (per-robot radius)
# ============================

robot_artists = []

def draw_robot(x, y, theta, radius, is_dd, color):
    circ = Circle((x, y), radius, fill=False, lw=2, color=color)
    ax.add_patch(circ)

    arrow = None
    if is_dd:
        L = radius * 1.8
        arrow = FancyArrowPatch(
            (x, y),
            (x + L*np.cos(theta), y + L*np.sin(theta)),
            arrowstyle='->',
            lw=2,
            color=color
        )
        ax.add_patch(arrow)

    return circ, arrow


for name, robot_paths in robot_trajectories.items():
    agent = agent_info[name]['agent']
    agent_type = agent_info[name]['type']
    col = colors[name]

    if agent_type == "heterogeneous-formation":
        for i in range(len(robot_paths)):
            r = agent.radii[i]                                    # ← per-robot radius
            is_dd = agent.robot_types[i] == 'diff-drive'
            circ, arr = draw_robot(0, 0, 0, r, is_dd, col)
            robot_artists.append((name, i, circ, arr))

    else:
        r = agent.radius
        is_dd = agent_type == "diff-drive"
        circ, arr = draw_robot(0, 0, 0, r, is_dd, col)
        robot_artists.append((name, 0, circ, arr))


# ============================
# Slider
# ============================

ax_time = plt.axes([0.2, 0.05, 0.6, 0.03])
time_slider = Slider(ax_time, "Time", 0.0, Tmax, valinit=0.0)

# ============================
# Pose lookup helper
# ============================

def interp_pose(path, t):
    for i in range(len(path)-1):
        (x1,y1,th1), t1 = path[i]
        (x2,y2,th2), t2 = path[i+1]
        if t1 <= t <= t2:
            a = (t - t1)/(t2 - t1 + 1e-9)
            x = x1 + a*(x2 - x1)
            y = y1 + a*(y2 - y1)
            th = th1 + a*((th2 - th1 + np.pi)%(2*np.pi)-np.pi)
            return x,y,th
    return path[-1][0]


# ============================
# Update function
# ============================

def update(val):
    t = time_slider.val

    for name, idx, circ, arr in robot_artists:
        path = robot_trajectories[name][idx]
        x,y,th = interp_pose(path, t)

        circ.center = (x, y)

        if arr is not None:
            L = circ.radius * 1.8
            arr.set_positions(
                (x, y),
                (x + L*np.cos(th), y + L*np.sin(th))
            )

    fig.canvas.draw_idle()

time_slider.on_changed(update)

update(0.0)

# ============================
# Draw trajectories (static)
# ============================

for name, robot_paths in robot_trajectories.items():
    agent = agent_info[name]['agent']
    agent_type = agent_info[name]['type']
    col = colors[name]

    if agent_type == "heterogeneous-formation":
        for i, path in enumerate(robot_paths):
            pts = np.array([pos[:2] for pos, _ in path])
            rtype = agent.robot_types[i]

            style = '-' if rtype == 'diff-drive' else ':'
            ax.plot(
                pts[:,0], pts[:,1],
                style,
                lw=1.8,
                color=col,
                alpha=0.6
            )

        # centroid path
        centroid_pts = np.array([item[0][:2] for item in trajectories[name]])
        ax.plot(
            centroid_pts[:,0], centroid_pts[:,1],
            lw=3,
            color=col,
            alpha=0.9
        )

    else:
        path = robot_paths[0]
        pts = np.array([pos[:2] for pos, _ in path])

        style = '-' if agent_type == "diff-drive" else ':'
        ax.plot(
            pts[:,0], pts[:,1],
            style,
            lw=2.5,
            color=col,
            alpha=0.8
        )

plt.show()