#!/usr/bin/env python3
"""
Complete Test: Heterogeneous Formations + Individual Agents

Tests ALL robot types and configurations:
1. Individual diff-drive agents
2. Individual holonomic agents
3. Homogeneous formations (all DD)
4. HETEROGENEOUS formations (mixed DD + holonomic)

Each robot type gets proper kinodynamic constraints!
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

# ===========================================================
# Agent Classes
# ===========================================================

class DifferentialDriveAgent:
    """Differential-drive robot"""
    def __init__(self, radius: float):
        self.radius = float(radius)
        self.is_holonomic = False
    
    def sample_q(self, bounds_xy):
        lo, hi = bounds_xy
        q = np.zeros(3, dtype=float)
        q[:2] = np.random.uniform(lo, hi)
        q[2] = np.random.uniform(-np.pi, np.pi)
        return q
    
    def interpolate_q(self, q1, q2, a):
        q = q1.copy()
        q[:2] = q1[:2] + a * (q2[:2] - q1[:2])
        th1, th2 = float(q1[2]), float(q2[2])
        dth = (th2 - th1 + np.pi) % (2*np.pi) - np.pi
        q[2] = th1 + a * dth
        return q
    
    def discs(self, q):
        p = np.asarray(q, dtype=float)[:2]
        return [(p, self.radius)]
    
    def centroid_xy(self, q):
        return np.asarray(q, dtype=float)[:2].copy()
    
    def robot_poses(self, q):
        q = np.asarray(q, dtype=float)
        return [(float(q[0]), float(q[1]), float(q[2]))]
    
    def max_robot_displacement(self, q1, q2):
        return np.hypot(float(q2[0] - q1[0]), float(q2[1] - q1[1]))
    
    def dist_for_nn(self, q1, q2):
        dx = float(q2[0] - q1[0])
        dy = float(q2[1] - q1[1])
        dth = abs((q2[2] - q1[2] + np.pi) % (2*np.pi) - np.pi)
        return np.hypot(dx, dy) + 0.3 * dth


class HolonomicAgent:
    """Holonomic robot"""
    def __init__(self, radius: float):
        self.radius = float(radius)
        self.is_holonomic = True
    
    def sample_q(self, bounds_xy):
        lo, hi = bounds_xy
        return np.random.uniform(lo, hi)
    
    def interpolate_q(self, q1, q2, a):
        return q1 + a * (q2 - q1)
    
    def discs(self, q):
        p = np.asarray(q, dtype=float)[:2]
        return [(p, self.radius)]
    
    def centroid_xy(self, q):
        return np.asarray(q, dtype=float)[:2].copy()
    
    def robot_poses(self, q):
        q = np.asarray(q, dtype=float)
        return [(float(q[0]), float(q[1]), 0.0)]
    
    def max_robot_displacement(self, q1, q2):
        return np.hypot(float(q2[0] - q1[0]), float(q2[1] - q1[1]))
    
    def dist_for_nn(self, q1, q2):
        return np.hypot(float(q2[0] - q1[0]), float(q2[1] - q1[1]))


class HeterogeneousFormationAgent:
    """Formation with MIXED robot types"""
    def __init__(
        self,
        P_star: list,
        robot_types: list,  # ['diff-drive', 'holonomic', ...]
        radius: float = 0.35,
        sx_range: tuple = (0.7, 3.0),
        sy_range: tuple = (0.7, 3.0),
    ):
        self.P_star = np.array(P_star, dtype=float)
        self.robot_types = robot_types
        self.Nr = len(P_star)
        self.radius = float(radius)
        self.sx_range = sx_range
        self.sy_range = sy_range
        
        # For disc decomposition cache
        self._disc_cache = {}
        
        print(f"Heterogeneous formation: {robot_types}")
    
    def sample_q(self, bounds_xy):
        lo, hi = bounds_xy
        q = np.zeros(5, dtype=float)
        q[:2] = np.random.uniform(lo, hi)
        q[2] = np.random.uniform(-np.pi, np.pi)
        q[3] = np.random.uniform(*self.sx_range)
        q[4] = np.random.uniform(*self.sy_range)
        return q
    
    def interpolate_q(self, q1, q2, a):
        q = q1.copy()
        q[:2] = q1[:2] + a * (q2[:2] - q1[:2])
        dth = (q2[2] - q1[2] + np.pi) % (2*np.pi) - np.pi
        q[2] = q1[2] + a * dth
        q[3:5] = q1[3:5] + a * (q2[3:5] - q1[3:5])
        return q
    
    def discs(self, q):
        """Decompose formation into discs for collision checking"""
        q = np.asarray(q, dtype=float).flatten()
        
        # Use cache
        qkey = tuple(np.round(q, decimals=2))
        if qkey in self._disc_cache:
            return self._disc_cache[qkey]
        
        xc, yc, th, sx, sy = q[0], q[1], q[2], q[3], q[4]
        
        R = np.array([[np.cos(th), -np.sin(th)],
                      [np.sin(th), np.cos(th)]])
        D = np.diag([sx, sy])
        
        discs = []
        for p_star in self.P_star:
            p = np.array([xc, yc]) + R @ D @ p_star
            discs.append((p, self.radius))
        
        self._disc_cache[qkey] = discs
        return discs
    
    def centroid_xy(self, q):
        return np.asarray(q, dtype=float)[:2].copy()
    
    def robot_poses(self, q):
        """Return (x, y, heading) for each robot"""
        q = np.asarray(q, dtype=float).flatten()
        xc, yc, th, sx, sy = q[0], q[1], q[2], q[3], q[4]
        
        R = np.array([[np.cos(th), -np.sin(th)],
                      [np.sin(th), np.cos(th)]])
        D = np.diag([sx, sy])
        
        poses = []
        for i, p_star in enumerate(self.P_star):
            p = np.array([xc, yc]) + R @ D @ p_star
            # For DD robots, use formation heading; for holonomic, 0.0
            theta = th if self.robot_types[i] == 'diff-drive' else 0.0
            poses.append((float(p[0]), float(p[1]), float(theta)))
        
        return poses
    
    def max_robot_displacement(self, q1, q2):
        poses1 = self.robot_poses(q1)
        poses2 = self.robot_poses(q2)
        max_d = 0.0
        for (x1, y1, _), (x2, y2, _) in zip(poses1, poses2):
            d = np.hypot(x2 - x1, y2 - y1)
            max_d = max(max_d, d)
        return max_d
    
    def dist_for_nn(self, q1, q2):
        """Distance metric for nearest neighbor (formations)"""
        q1, q2 = np.asarray(q1, dtype=float), np.asarray(q2, dtype=float)
        dx, dy = q2[0] - q1[0], q2[1] - q1[1]
        dth = abs((q2[2] - q1[2] + np.pi) % (2*np.pi) - np.pi)
        dsx = abs(q2[3] - q1[3])
        dsy = abs(q2[4] - q1[4])
        
        # Weighted distance
        return np.hypot(dx, dy) + 0.3 * dth + 0.2 * (dsx + dsy)
    
    def clear_cache(self):
        self._disc_cache.clear()


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
# Agents - Define geometry here, velocity limits in agents list
# ============================================================

dd_robot = DifferentialDriveAgent(radius=0.40)
holo_robot = HolonomicAgent(radius=0.40)

hetero_form = HeterogeneousFormationAgent(
    P_star=[
        [-0.6, 0.0],  # Robot 0: Diff-drive
        [0.6, 0.0],   # Robot 1: Holonomic
        [0.0, 0.6]    # Robot 2: Diff-drive
    ],
    robot_types=['diff-drive', 'holonomic', 'diff-drive'],
    radius=0.35,
)

# ============================================================
# Agent Configurations: (name, agent, start, goal, type, vel_limits)
# All velocity limits defined here alongside start/goal
# ============================================================
agents = [
    ("HeteroForm", hetero_form,
     np.array([2.0, 10.0, 0.0, 1.0, 1.0]),   # start: [xc, yc, θ, sx, sy]
     np.array([18.0, 10.0, 0.0, 1.0, 1.0]),  # goal
     "heterogeneous-formation",
     {  # Per-robot velocity limits
         'v_max': [0.22, 0.22, 0.22],   # Robot 0: 0.6, Robot 1: 0.8, Robot 2: 0.7
         'w_max': [1.2, 0.0, 1.2],   # omega limits (0 for holonomic)
         'a_max': 0.2,
     }),

    ("DD", dd_robot,
     np.array([3.0, 3.0, 0.0]),    # start: [x, y, θ]
     np.array([17.0, 3.0, 0.0]),   # goal
     "diff-drive",
     {  # DD velocity limits
         'v_max': 0.22,
         'v_min': -0.22,
         'omega_max': 1.2,
         'a_max': 0.2,
     }),

    ("Holo", holo_robot,
     np.array([10.0, 2.0]),        # start: [x, y]
     np.array([10.0, 18.0]),       # goal
     "holonomic",
     {  # Holonomic velocity limits
         'v_max': 0.15,
         'a_max': 1.0,
     }),
]

colors = {
    "DD": "tab:blue",
    "Holo": "tab:red",
    "HeteroForm": "tab:purple",
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

for name, agent, start, goal, agent_type, vel_limits in agents:
    print(f"\nPlanning {name} ({agent_type})...")
    
    # Build kinodynamic parameters from velocity limits defined in agents list
    if agent_type == "heterogeneous-formation" and HETERO_AVAILABLE:
        kino_params = {
            'robot_types': agent.robot_types,
            'v_max': vel_limits.get('v_max', [0.8]*len(agent.P_star)),
            'w_max': vel_limits.get('w_max', [2.0]*len(agent.P_star)),
            'N_steer': 10,
            'T_steer': 1.0,
            'T_steer_min': 0.5,
            'T_steer_safety_factor': 1.5,
            'max_iter': 200,
        }
        use_kino = True
        use_hetero = True
        print(f"    Per-robot v_max: {kino_params['v_max']}")
        print(f"    Per-robot w_max: {kino_params['w_max']}")
        
    elif agent_type == "diff-drive":
        kino_params = {
            'v_max': vel_limits.get('v_max', 0.8),
            'v_min': vel_limits.get('v_min', -0.3),
            'omega_max': vel_limits.get('omega_max', 2.0),
            'a_max': vel_limits.get('a_max', 1.5),
            'dt': 0.05,
        }
        use_kino = True
        use_hetero = False
        print(f"    v_max: {kino_params['v_max']}, omega_max: {kino_params['omega_max']}")
        
    elif agent_type == "holonomic":
        kino_params = {
            'v_max': vel_limits.get('v_max', 0.4),
            'a_max': vel_limits.get('a_max', 2.0),
            'dt': 0.05,
        }
        use_kino = True
        use_hetero = False
        print(f"    v_max: {kino_params['v_max']}")
        
    else:
        use_kino = False
        use_hetero = False
        kino_params = None

    planner = SIRRT(
        agent_model=agent,
        max_velocity=0.6,
        workspace_bounds=bounds,
        static_grid=static_grid,
        time_horizon=80.0,
        max_iter=2000,
        d_max=0.5,
        goal_sample_rate=0.22,
        neighbor_radius=1.5,
        precision=2,
        seed=20,
        debug=True,
        use_kinodynamic=use_kino,
        kinodynamic_params=kino_params,
    )
    
    # IMPORTANT: For heterogeneous formations, the steering is handled inside SIRRT
    # The kinodynamic_params are passed to the planner which creates the steerer
    if use_hetero and HETERO_AVAILABLE:
        print(f"  Using heterogeneous formation steering (created inside SIRRT)")

    traj = planner.plan(start, goal, dynamic_obstacles)
    if traj is None:
        raise RuntimeError(f"Planning failed for {name}")

    trajectories[name] = traj
    
    # Extract control trajectories (handle both individual and formation types)
    control_traj_list = []
    for item in traj:
        if len(item) >= 4 and item[3] is not None:
            ctrl = item[3]
            # Check if it's a FormationControlTrajectory or individual ControlTrajectory
            if hasattr(ctrl, 'q_traj'):
                # It's a FormationControlTrajectory - extract per-robot controls
                control_traj_list.append(ctrl)
            elif hasattr(ctrl, 'x'):
                # It's an individual ControlTrajectory
                control_traj_list.append(ctrl)
            else:
                # Unknown format, skip
                pass
    control_trajectories[name] = control_traj_list
    
    # For cooperative planning
    traj_for_obstacle = [(item[0], item[1]) for item in traj]
    dynamic_obstacles.append({"trajectory": traj_for_obstacle, "agent": agent})
    
    agent_info[name] = {
        'agent': agent,
        'type': agent_type,
        'use_kinodynamic': use_kino,
        'kino_params': kino_params,
        'vel_limits': vel_limits,  # Store velocity limits for plotting
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
    
    if agent_type == "heterogeneous-formation":
        # Handle FormationControlTrajectory - extract per-robot controls
        agent = agent_info[name]['agent']
        Nr = len(agent.P_star)
        robot_types = agent.robot_types
        
        # Per-robot control data
        for robot_idx in range(Nr):
            robot_type = robot_types[robot_idx]
            all_x, all_y, all_t = [], [], []
            all_theta, all_psi, all_v, all_omega, all_vx, all_vy = [], [], [], [], [], []
            
            cumulative_time = 0.0
            
            for form_ctrl_traj in control_list:
                if form_ctrl_traj is None or not hasattr(form_ctrl_traj, 'q_traj'):
                    continue
                
                # Extract formation state and robot-specific data
                q_traj = form_ctrl_traj.q_traj  # (5, N)
                t_traj = form_ctrl_traj.t_traj + cumulative_time
                
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
                    if robot_type == 'diff-drive' and robot_idx in form_ctrl_traj.psi_traj:
                        psi_val = form_ctrl_traj.psi_traj[robot_idx][k]
                        all_psi.append(psi_val)
                    else:
                        all_psi.append(th)  # Use formation heading
                
                # Extract controls (N-1 points)
                if robot_type == 'diff-drive' and robot_idx in form_ctrl_traj.v_traj:
                    all_v.extend(form_ctrl_traj.v_traj[robot_idx].tolist())
                    all_omega.extend(form_ctrl_traj.omega_traj[robot_idx].tolist())
                elif robot_type == 'holonomic' and robot_idx in form_ctrl_traj.vx_traj:
                    all_vx.extend(form_ctrl_traj.vx_traj[robot_idx].tolist())
                    all_vy.extend(form_ctrl_traj.vy_traj[robot_idx].tolist())
            
            # Save per-robot CSV
            csv_file = os.path.join(output_dir, f"{name}_robot{robot_idx}_controls.csv")
            
            with open(csv_file, 'w', newline='') as f:
                if robot_type == 'diff-drive':
                    writer = csv.writer(f)
                    writer.writerow(['time', 'x', 'y', 'psi', 'v', 'omega'])
                    
                    n_states = len(all_x)
                    n_controls = len(all_v)
                    
                    for i in range(n_states):
                        t_val = all_t[i] if i < len(all_t) else 0.0
                        x_val = all_x[i]
                        y_val = all_y[i]
                        psi_val = all_psi[i] if i < len(all_psi) else 0.0
                        v_val = all_v[i] if i < n_controls else 0.0
                        omega_val = all_omega[i] if i < n_controls else 0.0
                        
                        writer.writerow([f"{t_val:.4f}", f"{x_val:.4f}", f"{y_val:.4f}", 
                                       f"{psi_val:.4f}", f"{v_val:.4f}", f"{omega_val:.4f}"])
                    
                    print(f"  Robot {robot_idx} (DD): Saved {n_states} states to {csv_file}")
                
                elif robot_type == 'holonomic':
                    writer = csv.writer(f)
                    writer.writerow(['time', 'x', 'y', 'vx', 'vy'])
                    
                    n_states = len(all_x)
                    n_controls = len(all_vx)
                    
                    for i in range(n_states):
                        t_val = all_t[i] if i < len(all_t) else 0.0
                        x_val = all_x[i]
                        y_val = all_y[i]
                        vx_val = all_vx[i] if i < n_controls else 0.0
                        vy_val = all_vy[i] if i < len(all_vy) else 0.0
                        
                        writer.writerow([f"{t_val:.4f}", f"{x_val:.4f}", f"{y_val:.4f}", 
                                       f"{vx_val:.4f}", f"{vy_val:.4f}"])
                    
                    print(f"  Robot {robot_idx} (Holonomic): Saved {n_states} states to {csv_file}")
    
    else:
        # Individual agent - use existing logic
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
    control_list = control_trajectories[name]
    
    if agent_type == "heterogeneous-formation":
        # Extract individual robot positions with TRUE psi from optimization
        Nr = len(agent.P_star)
        robot_paths = [[] for _ in range(Nr)]
        
        ctrl_idx = 0
        for item_idx, item in enumerate(traj):
            q = item[0]
            t = item[1]
            ctrl_traj = item[3] if len(item) >= 4 else None
            
            # Get robot poses from formation config
            xc, yc, th, sx, sy = q[0], q[1], q[2], q[3], q[4]
            R = np.array([[np.cos(th), -np.sin(th)],
                          [np.sin(th), np.cos(th)]])
            D = np.diag([sx, sy])
            
            for i in range(Nr):
                p_star_i = agent.P_star[i]
                p_i = np.array([xc, yc]) + R @ D @ p_star_i
                x, y = p_i[0], p_i[1]
                
                # Use TRUE psi from FormationControlTrajectory if available
                if ctrl_traj is not None and hasattr(ctrl_traj, 'psi_traj') and i in ctrl_traj.psi_traj:
                    # Use final psi from this trajectory segment
                    theta_use = float(ctrl_traj.psi_traj[i][-1])
                else:
                    # Fallback to formation heading
                    theta_use = float(th)
                
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
# Visualization Preparation (Robust Flattening)
# ============================================================

print("\n" + "="*70)
print("Preparing Visualization Data...")
print("="*70)

# 1. FLATTEN DATA into a unified structure for robust lookup
# Structure: flattened_data[name][robot_idx] = {t, x, y, theta, v, omega...}
flattened_data = {}

for name in trajectories.keys():
    flattened_data[name] = {}
    agent_type = agent_info[name]['type']
    control_list = control_trajectories[name]
    
    # Identify how many robots we are dealing with
    if agent_type == "heterogeneous-formation":
        Nr = len(agent_info[name]['agent'].P_star)
        robot_types = agent_info[name]['agent'].robot_types
    else:
        Nr = 1
        robot_types = [agent_type]

    for i in range(Nr):
        # Arrays to hold flattened history for this specific robot
        flat_t, flat_x, flat_y, flat_theta = [], [], [], []
        flat_v, flat_omega, flat_vx, flat_vy = [], [], [], []
        
        cumulative_time = 0.0
        
        # Iterate through the trajectory segments (RRT* edges)
        for segment_idx, ctrl in enumerate(control_list):
            if ctrl is None: continue

            # --- CASE A: Heterogeneous Formation ---
            if agent_type == "heterogeneous-formation":
                if not hasattr(ctrl, 't_traj'): continue
                seg_t = ctrl.t_traj 
                
                # Re-calculate world pose from formation state to ensure perfect alignment
                q_traj = ctrl.q_traj
                for k in range(len(seg_t)):
                    xc, yc, th, sx, sy = q_traj[:, k]
                    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
                    D = np.diag([sx, sy])
                    p_star = agent_info[name]['agent'].P_star[i]
                    pos = np.array([xc, yc]) + R @ D @ p_star
                    
                    flat_x.append(pos[0])
                    flat_y.append(pos[1])
                    
                    # For heading: Use psi for DD, formation theta for Holo
                    # This ensures the arrow points exactly where the optimizer planned
                    if robot_types[i] == 'diff-drive' and i in ctrl.psi_traj:
                        flat_theta.append(float(ctrl.psi_traj[i][k]))
                    else:
                        flat_theta.append(th)

                # Extract Controls (N points - same as states now)
                if robot_types[i] == 'diff-drive':
                    vs = ctrl.v_traj[i]
                    ws = ctrl.omega_traj[i]
                    flat_v.extend(vs)
                    flat_omega.extend(ws)
                elif robot_types[i] == 'holonomic':
                    vxs = ctrl.vx_traj[i]
                    vys = ctrl.vy_traj[i]
                    flat_vx.extend(vxs)
                    flat_vy.extend(vys)
            
            # --- CASE B: Individual Agent ---
            else: 
                if ctrl.t is None: continue
                seg_t = ctrl.t
                flat_x.extend(ctrl.x)
                flat_y.extend(ctrl.y)
                
                # Handle Heading
                if ctrl.theta is not None: 
                    flat_theta.extend(ctrl.theta)
                else:
                    flat_theta.extend([0.0]*len(seg_t)) # Default for holonomic
                
                # Handle Controls
                if agent_type == 'diff-drive':
                    flat_v.extend(ctrl.v)
                    flat_omega.extend(ctrl.omega)
                elif agent_type == 'holonomic':
                    # Fix: Holonomic individual steering stores 'v' as vx and 'vy' as vy
                    flat_vx.extend(ctrl.v) 
                    flat_vy.extend(ctrl.vy)

            # Add cumulative time
            flat_t.extend(seg_t + cumulative_time)
            cumulative_time += seg_t[-1]

        # ============================================================
        # TWO-PHASE GOAL: Append zero-velocity dwell at final position
        # This ensures robots come to complete rest at the goal
        # ============================================================
        DWELL_TIME = 0.5  # seconds to dwell at goal with zero velocity
        DWELL_POINTS = 5  # number of points in dwell phase
        
        if len(flat_t) > 0 and len(flat_x) > 0:
            final_t = flat_t[-1]
            final_x = flat_x[-1]
            final_y = flat_y[-1]
            final_theta = flat_theta[-1] if flat_theta else 0.0
            
            dwell_dt = DWELL_TIME / DWELL_POINTS
            
            for dp in range(1, DWELL_POINTS + 1):
                flat_t.append(final_t + dp * dwell_dt)
                flat_x.append(final_x)
                flat_y.append(final_y)
                flat_theta.append(final_theta)
                
                # Zero velocities for dwell phase
                if robot_types[i] == 'diff-drive':
                    flat_v.append(0.0)
                    flat_omega.append(0.0)
                elif robot_types[i] == 'holonomic':
                    flat_vx.append(0.0)
                    flat_vy.append(0.0)
            
            print(f"    Robot {i} ({robot_types[i]}): Added {DWELL_POINTS} dwell points at goal (v=0)")

        # Convert to numpy arrays for fast indexing
        flattened_data[name][i] = {
            't': np.array(flat_t),
            'x': np.array(flat_x),
            'y': np.array(flat_y),
            'theta': np.array(flat_theta) if flat_theta else None,
            'v': np.array(flat_v) if flat_v else None,
            'omega': np.array(flat_omega) if flat_omega else None,
            'vx': np.array(flat_vx) if flat_vx else None,
            'vy': np.array(flat_vy) if flat_vy else None,
            'type': robot_types[i]
        }

# ============================================================
# SAVE COMPREHENSIVE TRAJECTORY DATA TO CSV
# ============================================================

print("\n" + "="*70)
print("Saving Comprehensive Trajectory Data...")
print("="*70)

trajectory_output_dir = "trajectory_data"
os.makedirs(trajectory_output_dir, exist_ok=True)

for name in trajectories.keys():
    agent_type = agent_info[name]['type']
    data_map = flattened_data[name]
    
    print(f"\n{name} ({agent_type}):")
    
    for robot_idx in data_map.keys():
        data = data_map[robot_idx]
        rtype = data['type']
        
        # Prepare filename
        if agent_type == "heterogeneous-formation":
            csv_filename = f"{name}_robot{robot_idx}_{rtype}.csv"
        else:
            csv_filename = f"{name}_{rtype}.csv"
        
        csv_path = os.path.join(trajectory_output_dir, csv_filename)
        
        # Get data arrays
        t_arr = data['t']
        x_arr = data['x']
        y_arr = data['y']
        theta_arr = data['theta'] if data['theta'] is not None else np.zeros_like(t_arr)
        
        n_points = len(t_arr)
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            if rtype == 'diff-drive':
                v_arr = data['v'] if data['v'] is not None else np.zeros(n_points)
                omega_arr = data['omega'] if data['omega'] is not None else np.zeros(n_points)
                
                # Compute world-frame velocities
                vx_world = v_arr * np.cos(theta_arr[:len(v_arr)]) if len(theta_arr) >= len(v_arr) else v_arr * np.cos(theta_arr)
                vy_world = v_arr * np.sin(theta_arr[:len(v_arr)]) if len(theta_arr) >= len(v_arr) else v_arr * np.sin(theta_arr)
                
                writer.writerow(['time', 'x', 'y', 'theta', 'v_body', 'omega', 'vx_world', 'vy_world'])
                
                for i in range(n_points):
                    v_val = v_arr[i] if i < len(v_arr) else 0.0
                    omega_val = omega_arr[i] if i < len(omega_arr) else 0.0
                    vx_val = vx_world[i] if i < len(vx_world) else 0.0
                    vy_val = vy_world[i] if i < len(vy_world) else 0.0
                    
                    writer.writerow([
                        f"{t_arr[i]:.6f}",
                        f"{x_arr[i]:.6f}",
                        f"{y_arr[i]:.6f}",
                        f"{theta_arr[i]:.6f}",
                        f"{v_val:.6f}",
                        f"{omega_val:.6f}",
                        f"{vx_val:.6f}",
                        f"{vy_val:.6f}"
                    ])
                
                # Print summary
                print(f"  Robot {robot_idx} (DD): {n_points} points, t=[{t_arr[0]:.2f}, {t_arr[-1]:.2f}]s")
                print(f"    v: [{np.min(v_arr):.3f}, {np.max(v_arr):.3f}], v_end={v_arr[-1]:.4f}")
                print(f"    ω: [{np.min(omega_arr):.3f}, {np.max(omega_arr):.3f}], ω_end={omega_arr[-1]:.4f}")
                print(f"    Saved to: {csv_path}")
                
            elif rtype == 'holonomic':
                vx_arr = data['vx'] if data['vx'] is not None else np.zeros(n_points)
                vy_arr = data['vy'] if data['vy'] is not None else np.zeros(n_points)
                
                writer.writerow(['time', 'x', 'y', 'vx', 'vy', 'speed'])
                
                for i in range(n_points):
                    vx_val = vx_arr[i] if i < len(vx_arr) else 0.0
                    vy_val = vy_arr[i] if i < len(vy_arr) else 0.0
                    speed = np.hypot(vx_val, vy_val)
                    
                    writer.writerow([
                        f"{t_arr[i]:.6f}",
                        f"{x_arr[i]:.6f}",
                        f"{y_arr[i]:.6f}",
                        f"{vx_val:.6f}",
                        f"{vy_val:.6f}",
                        f"{speed:.6f}"
                    ])
                
                # Print summary
                print(f"  Robot {robot_idx} (Holo): {n_points} points, t=[{t_arr[0]:.2f}, {t_arr[-1]:.2f}]s")
                print(f"    vx: [{np.min(vx_arr):.3f}, {np.max(vx_arr):.3f}], vx_end={vx_arr[-1]:.4f}")
                print(f"    vy: [{np.min(vy_arr):.3f}, {np.max(vy_arr):.3f}], vy_end={vy_arr[-1]:.4f}")
                print(f"    Saved to: {csv_path}")

# Save summary file
summary_path = os.path.join(trajectory_output_dir, "summary.txt")
with open(summary_path, 'w') as f:
    f.write("="*70 + "\n")
    f.write("TRAJECTORY DATA SUMMARY\n")
    f.write("="*70 + "\n\n")
    
    for name in trajectories.keys():
        agent_type = agent_info[name]['type']
        agent = agent_info[name]['agent']
        data_map = flattened_data[name]
        vel_limits = agent_info[name].get('vel_limits', {})
        
        f.write(f"\n{name} ({agent_type}):\n")
        f.write("-"*40 + "\n")
        
        # Write velocity limits
        f.write("  Velocity Limits:\n")
        for key, val in vel_limits.items():
            if isinstance(val, list):
                f.write(f"    {key} (per-robot): {val}\n")
            else:
                f.write(f"    {key}: {val}\n")
        f.write("\n")
        
        for robot_idx in data_map.keys():
            data = data_map[robot_idx]
            rtype = data['type']
            t_arr = data['t']
            
            f.write(f"  Robot {robot_idx} ({rtype}):\n")
            f.write(f"    Duration: {t_arr[-1]:.2f}s\n")
            f.write(f"    Points: {len(t_arr)}\n")
            f.write(f"    Start: ({data['x'][0]:.2f}, {data['y'][0]:.2f})\n")
            f.write(f"    End: ({data['x'][-1]:.2f}, {data['y'][-1]:.2f})\n")
            
            if rtype == 'diff-drive' and data['v'] is not None:
                f.write(f"    v range: [{np.min(data['v']):.4f}, {np.max(data['v']):.4f}]\n")
                f.write(f"    ω range: [{np.min(data['omega']):.4f}, {np.max(data['omega']):.4f}]\n")
                f.write(f"    Final v: {data['v'][-1]:.4f}\n")
                f.write(f"    Final ω: {data['omega'][-1]:.4f}\n")
            elif rtype == 'holonomic' and data['vx'] is not None:
                f.write(f"    vx range: [{np.min(data['vx']):.4f}, {np.max(data['vx']):.4f}]\n")
                f.write(f"    vy range: [{np.min(data['vy']):.4f}, {np.max(data['vy']):.4f}]\n")
                f.write(f"    Final vx: {data['vx'][-1]:.4f}\n")
                f.write(f"    Final vy: {data['vy'][-1]:.4f}\n")
            f.write("\n")

print(f"\n✓ Summary saved to: {summary_path}")
print(f"✓ All trajectory data saved to: {trajectory_output_dir}/")

# ============================================================
# Visualization Logic (Robust Update)
# ============================================================

print("Creating plot...")
fig, ax = plt.subplots(figsize=(16, 10))
plt.subplots_adjust(bottom=0.15, right=0.80)
ax.imshow(grid[::-1], cmap="gray_r", extent=[0, 20, 0, 20], alpha=0.85)

# 1. Static Paths (Background - still drawn as lines for context)
# for name, robot_paths in robot_trajectories.items():
#     col = colors[name]
#     for i, path in enumerate(robot_paths):
#         pts = np.array([p[0][:2] for p in path])
#         ax.plot(pts[:,0], pts[:,1], lw=1, color=col, alpha=0.5)

# 1. Static Paths - USE DENSE CONTROL TRAJECTORY DATA
for name in trajectories.keys():
    col = colors[name]
    data_map = flattened_data[name]
    
    for i in data_map.keys():
        data = data_map[i]
        if len(data['x']) > 0:
            ax.plot(data['x'], data['y'], lw=1, color=col, alpha=0.5)

# 2. Robot Artists
robot_artists = {}
for name in trajectories.keys():
    data_map = flattened_data[name]
    robot_artists[name] = []
    col = colors[name]
    
    for i in data_map.keys():
        rtype = data_map[i]['type']
        is_dd = (rtype == 'diff-drive')
        
        radius = agent_info[name]['agent'].radius
        # Circle
        circ = Circle((0,0), radius, fill=False, lw=2, color=col)
        ax.add_patch(circ)
        # Arrow (only visible for diff-drive)
        arrow = FancyArrowPatch((0,0), (0,0), arrowstyle='->', mutation_scale=15, color=col, visible=is_dd)
        ax.add_patch(arrow)
        
        robot_artists[name].append({'circ': circ, 'arrow': arrow, 'idx': i})

# 3. Slider Setup
all_max_times = [d[0]['t'][-1] for n, d in flattened_data.items() if 0 in d and len(d[0]['t']) > 0]
Tmax = max(all_max_times) if all_max_times else 10.0
ax_time = plt.axes([0.2, 0.05, 0.6, 0.03])
time_slider = Slider(ax_time, "Time", 0.0, Tmax, valinit=0.0)

# 4. Text Box
control_text = ax.text(1.02, 0.98, '', transform=ax.transAxes, verticalalignment='top',
                       fontfamily='monospace', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

def update(val):
    t_query = time_slider.val
    
    status_lines = [f"Time: {t_query:.2f}s", "="*25, ""]
    
    for name in trajectories.keys():
        status_lines.append(f"[{name}]")
        r_data_map = flattened_data[name]
        
        for i, artists in enumerate(robot_artists[name]):
            # Retrieve robust data
            data = r_data_map[i]
            times = data['t']
            
            if len(times) == 0: continue
            
            # --- ROBUST LOOKUP ---
            # 1. Clamp time to valid range (solves "disappearing data")
            t_safe = max(times[0], min(t_query, times[-1]))
            
            # 2. Binary search for nearest index (Zero-Order Hold)
            # This avoids linear interpolation artifacts with angles
            idx = np.searchsorted(times, t_safe)
            if idx >= len(times): idx = len(times) - 1
            
            # 3. Extract EXACT data point
            # No interpolation means the text and the drawing are identical.
            x = data['x'][idx]
            y = data['y'][idx]
            theta = data['theta'][idx] if data['theta'] is not None else 0.0
            
            # Update Visuals
            artists['circ'].center = (x, y)
            
            # Update Arrow (Matplotlib handles angle wrapping automatically here)
            if artists['arrow'].get_visible():
                L = agent_info[name]['agent'].radius * 1.5
                artists['arrow'].set_positions((x, y), (x + L*np.cos(theta), y + L*np.sin(theta)))
            
            # Update Text
            r_type = data['type']
            if r_type == 'diff-drive':
                v = data['v'][idx]
                w = data['omega'][idx]
                status_lines.append(f" R{i}(DD): ψ={theta:6.2f} v={v:5.2f} ω={w:5.2f}")
            elif r_type == 'holonomic':
                vx = data['vx'][idx]
                vy = data['vy'][idx]
                status_lines.append(f" R{i}(HL): vx={vx:5.2f} vy={vy:5.2f}")
        
        status_lines.append("")

    control_text.set_text("\n".join(status_lines))
    fig.canvas.draw_idle()

time_slider.on_changed(update)
update(0.0)

# ============================================================
# SAVE MAIN FIGURE
# ============================================================
main_fig_path = os.path.join(trajectory_output_dir, "main_trajectory_plot.png")
fig.savefig(main_fig_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Main figure saved to: {main_fig_path}")

# ============================================================
# CREATE AND SAVE VELOCITY PROFILE PLOTS
# ============================================================
print("\nCreating velocity profile plots...")

for name in trajectories.keys():
    agent_type = agent_info[name]['type']
    data_map = flattened_data[name]
    
    n_robots = len(data_map)
    
    if agent_type == "heterogeneous-formation":
        # Create subplot for each robot
        fig_vel, axes = plt.subplots(n_robots, 2, figsize=(14, 4*n_robots))
        if n_robots == 1:
            axes = axes.reshape(1, -1)
        
        fig_vel.suptitle(f'{name} - Velocity Profiles', fontsize=14)
        
        # Get velocity limits from vel_limits stored in agent_info
        vel_limits = agent_info[name].get('vel_limits', {})
        v_max_list = vel_limits.get('v_max', [1.0]*n_robots)
        w_max_list = vel_limits.get('w_max', [2.0]*n_robots)
        if not isinstance(v_max_list, list):
            v_max_list = [v_max_list] * n_robots
        if not isinstance(w_max_list, list):
            w_max_list = [w_max_list] * n_robots
        
        for robot_idx in data_map.keys():
            data = data_map[robot_idx]
            rtype = data['type']
            t = data['t']
            
            # Get limits for this robot
            v_lim = v_max_list[robot_idx] if robot_idx < len(v_max_list) else 1.0
            w_lim = w_max_list[robot_idx] if robot_idx < len(w_max_list) else 2.0
            
            if rtype == 'diff-drive':
                v = data['v'] if data['v'] is not None else np.zeros_like(t)
                omega = data['omega'] if data['omega'] is not None else np.zeros_like(t)
                
                # Pad if needed
                if len(v) < len(t):
                    v = np.concatenate([v, np.zeros(len(t) - len(v))])
                if len(omega) < len(t):
                    omega = np.concatenate([omega, np.zeros(len(t) - len(omega))])
                
                axes[robot_idx, 0].plot(t, v[:len(t)], 'b-', lw=1.5, label='v (body)')
                axes[robot_idx, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
                axes[robot_idx, 0].axhline(y=v_lim, color='r', linestyle=':', alpha=0.5, label=f'v_max={v_lim}')
                axes[robot_idx, 0].axhline(y=-v_lim, color='r', linestyle=':', alpha=0.5)
                axes[robot_idx, 0].set_ylabel('v [m/s]')
                axes[robot_idx, 0].set_title(f'Robot {robot_idx} (DD) - Forward Velocity')
                axes[robot_idx, 0].legend()
                axes[robot_idx, 0].grid(True, alpha=0.3)
                
                axes[robot_idx, 1].plot(t, omega[:len(t)], 'r-', lw=1.5, label='ω')
                axes[robot_idx, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
                axes[robot_idx, 1].axhline(y=w_lim, color='g', linestyle=':', alpha=0.5, label=f'ω_max={w_lim}')
                axes[robot_idx, 1].axhline(y=-w_lim, color='g', linestyle=':', alpha=0.5)
                axes[robot_idx, 1].set_ylabel('ω [rad/s]')
                axes[robot_idx, 1].set_title(f'Robot {robot_idx} (DD) - Angular Velocity')
                axes[robot_idx, 1].legend()
                axes[robot_idx, 1].grid(True, alpha=0.3)
                
            elif rtype == 'holonomic':
                vx = data['vx'] if data['vx'] is not None else np.zeros_like(t)
                vy = data['vy'] if data['vy'] is not None else np.zeros_like(t)
                
                # Pad if needed
                if len(vx) < len(t):
                    vx = np.concatenate([vx, np.zeros(len(t) - len(vx))])
                if len(vy) < len(t):
                    vy = np.concatenate([vy, np.zeros(len(t) - len(vy))])
                
                axes[robot_idx, 0].plot(t, vx[:len(t)], 'b-', lw=1.5, label='vx')
                axes[robot_idx, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
                axes[robot_idx, 0].axhline(y=v_lim, color='r', linestyle=':', alpha=0.5, label=f'v_max={v_lim}')
                axes[robot_idx, 0].axhline(y=-v_lim, color='r', linestyle=':', alpha=0.5)
                axes[robot_idx, 0].set_ylabel('vx [m/s]')
                axes[robot_idx, 0].set_title(f'Robot {robot_idx} (Holo) - X Velocity')
                axes[robot_idx, 0].legend()
                axes[robot_idx, 0].grid(True, alpha=0.3)
                
                axes[robot_idx, 1].plot(t, vy[:len(t)], 'r-', lw=1.5, label='vy')
                axes[robot_idx, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
                axes[robot_idx, 1].axhline(y=v_lim, color='g', linestyle=':', alpha=0.5, label=f'v_max={v_lim}')
                axes[robot_idx, 1].axhline(y=-v_lim, color='g', linestyle=':', alpha=0.5)
                axes[robot_idx, 1].set_ylabel('vy [m/s]')
                axes[robot_idx, 1].set_title(f'Robot {robot_idx} (Holo) - Y Velocity')
                axes[robot_idx, 1].legend()
                axes[robot_idx, 1].grid(True, alpha=0.3)
            
            axes[robot_idx, 0].set_xlabel('Time [s]')
            axes[robot_idx, 1].set_xlabel('Time [s]')
        
        plt.tight_layout()
        vel_fig_path = os.path.join(trajectory_output_dir, f"{name}_velocity_profiles.png")
        fig_vel.savefig(vel_fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig_vel)
        print(f"  ✓ {name} velocity profiles saved to: {vel_fig_path}")
    
    else:
        # Individual agent - single subplot
        data = data_map[0]
        rtype = data['type']
        t = data['t']
        vel_limits = agent_info[name].get('vel_limits', {})
        
        fig_vel, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig_vel.suptitle(f'{name} ({rtype}) - Velocity Profiles', fontsize=14)
        
        if rtype == 'diff-drive':
            v = data['v'] if data['v'] is not None else np.zeros_like(t)
            omega = data['omega'] if data['omega'] is not None else np.zeros_like(t)
            
            # Get limits from vel_limits
            v_lim = vel_limits.get('v_max', 0.22)
            w_lim = vel_limits.get('omega_max', 2.84)
            
            axes[0].plot(t, v[:len(t)], 'b-', lw=1.5, label='v (body)')
            axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            axes[0].axhline(y=v_lim, color='r', linestyle=':', alpha=0.5, label=f'v_max={v_lim}')
            axes[0].axhline(y=-v_lim, color='r', linestyle=':', alpha=0.5)
            axes[0].set_xlabel('Time [s]')
            axes[0].set_ylabel('v [m/s]')
            axes[0].set_title('Forward Velocity')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(t, omega[:len(t)], 'r-', lw=1.5, label='ω')
            axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            axes[1].axhline(y=w_lim, color='g', linestyle=':', alpha=0.5, label=f'ω_max={w_lim}')
            axes[1].axhline(y=-w_lim, color='g', linestyle=':', alpha=0.5)
            axes[1].set_xlabel('Time [s]')
            axes[1].set_ylabel('ω [rad/s]')
            axes[1].set_title('Angular Velocity')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
        elif rtype == 'holonomic':
            vx = data['vx'] if data['vx'] is not None else np.zeros_like(t)
            vy = data['vy'] if data['vy'] is not None else np.zeros_like(t)
            
            # Get limit from vel_limits
            v_lim = vel_limits.get('v_max', 0.4)
            
            axes[0].plot(t, vx[:len(t)], 'b-', lw=1.5, label='vx')
            axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            axes[0].axhline(y=v_lim, color='r', linestyle=':', alpha=0.5, label=f'v_max={v_lim}')
            axes[0].axhline(y=-v_lim, color='r', linestyle=':', alpha=0.5)
            axes[0].set_xlabel('Time [s]')
            axes[0].set_ylabel('vx [m/s]')
            axes[0].set_title('X Velocity')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(t, vy[:len(t)], 'r-', lw=1.5, label='vy')
            axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            axes[1].axhline(y=v_lim, color='g', linestyle=':', alpha=0.5, label=f'v_max={v_lim}')
            axes[1].axhline(y=-v_lim, color='g', linestyle=':', alpha=0.5)
            axes[1].set_xlabel('Time [s]')
            axes[1].set_ylabel('vy [m/s]')
            axes[1].set_title('Y Velocity')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        vel_fig_path = os.path.join(trajectory_output_dir, f"{name}_velocity_profiles.png")
        fig_vel.savefig(vel_fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig_vel)
        print(f"  ✓ {name} velocity profiles saved to: {vel_fig_path}")

print(f"\n" + "="*70)
print("ALL DATA SAVED SUCCESSFULLY")
print("="*70)
print(f"Output directory: {trajectory_output_dir}/")
print("Files:")
print("  - CSV trajectory files for each robot")
print("  - summary.txt with trajectory statistics")
print("  - main_trajectory_plot.png")
print("  - *_velocity_profiles.png for each agent")
print("="*70)

plt.show()