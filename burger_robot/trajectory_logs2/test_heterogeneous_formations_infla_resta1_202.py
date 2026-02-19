#!/usr/bin/env python3
"""
Complete Test: Heterogeneous Formations + Individual Agents
WITH:
  1) Inflation radius applied to *all* robots (individual + inside formations)
     -> affects collision checking via agent.discs()
     -> also affects visualization circles (optional toggle)
  2) Save trajectories of *all robots* (including each robot in a formation)
     to CSV with relevant variables.
  3) WARMUP: Pre-compile Numba and CasADi solvers before timing

Author: (you)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.widgets import Slider, Button
import csv
import os
import math
import time

from si_rrt_enhanced_individual_kinodynamic import (
    SIRRT, OccupancyGrid, NUMBA_AVAILABLE
)

# Try to import heterogeneous formation steering
try:
    from heterogeneous_kinodynamic_formation_steering import *
    HETERO_AVAILABLE = True
except ImportError:
    HETERO_AVAILABLE = False
    print("Warning: Heterogeneous formation steering not available")

# ===========================================================
# SYSTEM CHECK
# ===========================================================

print("\n" + "="*70)
print("SYSTEM CHECK")
print("="*70)

print(f"Numba available: {NUMBA_AVAILABLE}")
if NUMBA_AVAILABLE:
    import numba
    print(f"  Version: {numba.__version__}")
else:
    print("  ⚠️  Install with: pip install numba")

print(f"Heterogeneous steering available: {HETERO_AVAILABLE}")

try:
    from scipy.spatial import cKDTree
    KDTREE_AVAILABLE = True
    print(f"scipy KD-tree available: True")
except ImportError:
    KDTREE_AVAILABLE = False
    print(f"scipy KD-tree available: False")

try:
    import casadi
    print(f"CasADi available: v{casadi.__version__}")
except:
    print(f"CasADi available: False")

print("="*70 + "\n")

# ===========================================================
# GLOBAL SETTINGS
# ===========================================================

INFLATION_RADIUS = 0.0  # <-- change as you like (meters)
DRAW_INFLATED_FOOTPRINTS = True  # visualization circles show inflated radii if True

OUTPUT_DIR = "trajectory_logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONTROL_DIR = "control_logs"
os.makedirs(CONTROL_DIR, exist_ok=True)

# ===========================================================
# Agent Classes (inflation applied inside discs())
# ===========================================================

class DifferentialDriveAgent:
    """
    Individual differential-drive robot.

    Stores per-robot limits:
        radius, v_max, omega_max, a_max, alpha_max
    """
    def __init__(
        self,
        radius: float = 0.40,
        v_max: float = 1.0,
        omega_max: float = 2.0,
        a_max: float = 2.0,
        alpha_max: float = 4.0,
        inflation: float = 0.0,
    ):
        self.radius = float(radius)
        self.inflation = float(inflation)
        self.is_holonomic = False
        self.v_max = float(v_max)
        self.omega_max = float(omega_max)
        self.a_max = float(a_max)
        self.alpha_max = float(alpha_max)

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
        # inflation applied here (collision checking uses inflated radius)
        p = np.asarray(q, dtype=float)[:2]
        return [(p, self.radius + self.inflation)]

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
    """
    Individual holonomic (omnidirectional) robot.

    Stores per-robot limits:
        radius, v_max, a_max
    """
    def __init__(
        self,
        radius: float = 0.40,
        v_max: float = 1.0,
        a_max: float = 2.0,
        inflation: float = 0.0,
    ):
        self.radius = float(radius)
        self.inflation = float(inflation)
        self.is_holonomic = True
        self.v_max = float(v_max)
        self.a_max = float(a_max)

    def sample_q(self, bounds_xy):
        lo, hi = bounds_xy
        return np.random.uniform(lo, hi)

    def interpolate_q(self, q1, q2, a):
        return q1 + a * (q2 - q1)

    def discs(self, q):
        # inflation applied here
        p = np.asarray(q, dtype=float)[:2]
        return [(p, self.radius + self.inflation)]

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
    """
    Formation with MIXED robot types.

    Every robot has its own:
        radius, v_max, omega_max, a_max, alpha_max

    Inflation applied to EACH robot disc radius.
    """
    def __init__(
        self,
        P_star: list,
        robot_types: list,          # ['diff-drive', 'holonomic', ...]
        radius=0.25,                # float OR list[float]
        v_max=1.0,                  # float OR list[float]
        omega_max=2.0,              # float OR list[float]
        a_max=2.0,                  # float OR list[float]
        alpha_max=4.0,              # float OR list[float]
        sx_range: tuple = (1.1, 3.0),
        sy_range: tuple = (1.1, 3.0),
        inflation: float = 0.0,
    ):
        self.P_star = np.array(P_star, dtype=float)
        self.robot_types = robot_types
        self.Nr = len(P_star)
        self.sx_range = sx_range
        self.sy_range = sy_range
        self.inflation = float(inflation)

        if len(robot_types) != self.Nr:
            raise ValueError(f"robot_types length ({len(robot_types)}) != Nr ({self.Nr})")

        def _broadcast(val, name):
            if isinstance(val, (int, float)):
                return [float(val)] * self.Nr
            lst = [float(v) for v in val]
            if len(lst) != self.Nr:
                raise ValueError(f"{name} length ({len(lst)}) != Nr ({self.Nr})")
            return lst

        self.radii = _broadcast(radius, 'radius')       # per-robot radii
        self.radius = self.radii                         # alias (list)
        self.v_max_list = _broadcast(v_max, 'v_max')
        self.omega_max_list = _broadcast(omega_max, 'omega_max')
        self.a_max_list = _broadcast(a_max, 'a_max')
        self.alpha_max_list = _broadcast(alpha_max, 'alpha_max')

        self._disc_cache = {}
        
        # Precompute FF-RRT* metric constants
        from si_rrt_enhanced_individual_kinodynamic import precompute_constants
        self.Nx, self.Ny, self.Nxy = precompute_constants(self.P_star)

        print(f"Heterogeneous formation: {robot_types}")
        print(f"  radii      = {self.radii}")
        print(f"  v_max      = {self.v_max_list}")
        print(f"  omega_max  = {self.omega_max_list}")
        print(f"  a_max      = {self.a_max_list}")
        print(f"  alpha_max  = {self.alpha_max_list}")
        print(f"  inflation  = {self.inflation}")

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
        """Decompose formation into discs for collision checking (per-robot radius + inflation)."""
        q = np.asarray(q, dtype=float).flatten()

        qkey = tuple(np.round(q, decimals=2))
        if qkey in self._disc_cache:
            return self._disc_cache[qkey]

        xc, yc, th, sx, sy = q[0], q[1], q[2], q[3], q[4]

        R = np.array([[np.cos(th), -np.sin(th)],
                      [np.sin(th),  np.cos(th)]])
        D = np.diag([sx, sy])

        discs = []
        for i, p_star in enumerate(self.P_star):
            p = np.array([xc, yc]) + R @ D @ p_star
            discs.append((p, self.radii[i] + self.inflation))  # inflation HERE

        self._disc_cache[qkey] = discs
        return discs

    def centroid_xy(self, q):
        return np.asarray(q, dtype=float)[:2].copy()

    def robot_poses(self, q):
        """Return (x, y, heading) for each robot"""
        q = np.asarray(q, dtype=float).flatten()
        xc, yc, th, sx, sy = q[0], q[1], q[2], q[3], q[4]

        R = np.array([[np.cos(th), -np.sin(th)],
                      [np.sin(th),  np.cos(th)]])
        D = np.diag([sx, sy])

        poses = []
        for i, p_star in enumerate(self.P_star):
            p = np.array([xc, yc]) + R @ D @ p_star
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
        q1, q2 = np.asarray(q1, dtype=float), np.asarray(q2, dtype=float)
        dx, dy = q2[0] - q1[0], q2[1] - q1[1]
        dth = abs((q2[2] - q1[2] + np.pi) % (2*np.pi) - np.pi)
        dsx = abs(q2[3] - q1[3])
        dsy = abs(q2[4] - q1[4])
        return np.hypot(dx, dy) + 0.3 * dth + 0.2 * (dsx + dsy)

    def clear_cache(self):
        self._disc_cache.clear()


# ============================================================
# Workspace + Map
# ============================================================

# # ============================================================
grid = np.load('202_v4.npy')
res = 0.1

height, width = grid.shape
height, width = height*res, width*res

print(f"Grid width {width} ")
print(f"Grid height {height} ")

bounds = (np.array([0.0, 0.0]), np.array([height, width]))
W = int(width / res)
H = int(height / res)

static_grid = OccupancyGrid(grid, res)
# def draw_circle(center, radius):
#     cx, cy = center
#     r2 = radius ** 2
#     for i in range(H):
#         for j in range(W):
#             x = (j + 0.5) * res
#             y = (i + 0.5) * res
#             if (x - cx)**2 + (y - cy)**2 <= r2:
#                 grid[i, j] = 1

# draw_circle((8.0, 10.0), 1.2)
# draw_circle((12.0, 10.0), 1.2)

# static_grid = OccupancyGrid(grid, res)

# ============================================================
# Agents (inflation passed to ALL)
# ============================================================

dd_robot1 = DifferentialDriveAgent(
    radius=0.40,
    v_max=0.9, omega_max=2.0,
    a_max=1.8, alpha_max=3.5,
    inflation=INFLATION_RADIUS,
)
dd_robot2 = DifferentialDriveAgent(
    radius=0.25,
    v_max=0.8, omega_max=2.5,
    a_max=2.0, alpha_max=4.0,
    inflation=INFLATION_RADIUS,
)
# holo_robot1 = HolonomicAgent(
#     radius=0.40,
#     v_max=1.0, a_max=2.0,
#     inflation=INFLATION_RADIUS,
# )
# holo_robot2 = HolonomicAgent(
#     radius=0.30,
#     v_max=1.2, a_max=2.5,
#     inflation=INFLATION_RADIUS,
# )

hetero_form = HeterogeneousFormationAgent(
    P_star=[[-0.57, -0.19], [0.57, -0.19], [0.0, 0.38]],
    robot_types=['diff-drive', 'diff-drive', 'diff-drive'],
    radius=[0.25, 0.25, 0.25],       # per-robot radii
    v_max=[0.8, 0.8, 0.8],           # ⬅️ INCREASED from 0.2
    omega_max=[1.5, 1.5, 1.5],       # ⬅️ INCREASED from 0.4
    a_max=[1.5, 1.5, 1.5],           # ⬅️ INCREASED from 0.4
    alpha_max=[4.0, 4.0, 4.0],
    sx_range=(0.95, 1.3),            # ⬅️ LIMITED scaling range
    sy_range=(0.95, 1.3),
)
# ============================================================
# FIXED Agents with REALISTIC parameters
# ============================================================

# hetero_form = HeterogeneousFormationAgent(
#     P_star=[[-0.66, 0.0], [0.33, -1.0], [0.33, 1.0]],
#     robot_types=['diff-drive', 'holonomic', 'holonomic'],
#     radius=[0.25, 0.25, 0.25],
    
#     # ⬇️ CRITICAL FIXES
#     v_max=[0.12, 0.25, 0.25],           # 4x FASTER (was 0.2, 0.25, 0.2)
#     omega_max=[0.4, 0.0, 0.0],       # 3.75x FASTER (was 0.4, 0.0, 0.4)
#     a_max=[0.05, 0.4, 0.4],           # 30x FASTER! (was 0.05, 0.4, 0.05)
#     alpha_max=[0.3, 0.0, 0.0],       # Slightly increased
    
#     sx_range=(1.0, 4.0),            # Limited scaling
#     sy_range=(1.0, 4.0),
#     inflation=INFLATION_RADIUS,
# )

# hetero_form2 = HeterogeneousFormationAgent(
#     P_star=[[-0.33333333333333304, 1.0], [-0.33333333333333304, -1.0], [0.666666666666667, 0.0]],
#     robot_types=['diff-drive', 'diff-drive', 'holonomic'],
#     radius=[0.25, 0.25, 0.25],
    
#     # ⬇️ CRITICAL FIXES
#     v_max=[0.12, 0.12, 0.25],           # 4x FASTER
#     omega_max=[1.5, 0.0, 1.5],       # 3.75x FASTER
#     a_max=[0.05, 0.05, 0.4],           # 30x FASTER!
#     alpha_max=[0.3, 0.3, 0.0],
    
#     sx_range=(1.0, 4.0),
#     sy_range=(1.0, 4.0),
#     inflation=INFLATION_RADIUS,
# )

# hetero_form3 = HeterogeneousFormationAgent(
#     P_star=[[-0.8, 0.0], [0.8, 0.0], [0.0, 0.8]],
#     robot_types=['diff-drive', 'holonomic', 'diff-drive'],
#     radius=[0.30, 0.25, 0.30],
#     v_max=[1.0, 1.2, 1.0],           # Already good
#     omega_max=[2.5, 0.0, 2.5],       # Already good
#     a_max=[2.0, 2.5, 2.0],           # Already good
#     alpha_max=[4.0, 0.0, 4.0],       # Already good
#     sx_range=(0.95, 1.3),            # Limited
#     sy_range=(0.95, 1.3),
#     inflation=INFLATION_RADIUS,
# )

agents = [

             ("DD2", dd_robot2, # waffle
     np.array([2.85, 7.41, -1.57]),
     np.array([1.14, 1.71, -1.57]),
     "diff-drive"),

    ("HeteroForm", hetero_form,
     np.array([2.28, 1.33, 1.57, 1.0, 1.0]),
     np.array([1.71, 6.84, 1.57, 1.0, 1.0]),
     "heterogeneous-formation"),

    ("DD1", dd_robot1, # TB4
     np.array([0.57, 4.56, 0.0]),
     np.array([3.42, 5.13, 0.0]),
     "diff-drive"),




]


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

# agents = [

#     ("HeteroForm", hetero_form,
#      np.array([2.0, 10.0, 0.0, 1.0, 1.0]),
#      np.array([18.0, 10.0, 0.0, 1.0, 1.0]),
#      "heterogeneous-formation"),

#     ("DD1", dd_robot1,
#      np.array([3.0, 3.0, 0.0]),
#      np.array([17.0, 3.0, 0.0]),
#      "diff-drive"),

#     ("Holo1", holo_robot1,
#      np.array([10.0, 2.0]),
#      np.array([10.0, 18.0]),
#      "holonomic"),

    
#     ("Holo2", holo_robot2,
#      np.array([10.0, 18.0]),
#      np.array([10.0, 2.0]),
#      "holonomic"),

#     ("HeteroForm2", hetero_form2,
#      np.array([18.0, 10.0, 0.0, 1.0, 1.0]),
#      np.array([2.0, 10.0, 0.0, 1.0, 1.0]),
#      "heterogeneous-formation"),

#     ("DD2", dd_robot2,
#      np.array([17.0, 3.0, 0.0]),
#      np.array([3.0, 3.0, 0.0]),
#      "diff-drive"),

#      ("HeteroForm3", hetero_form3,
#      np.array([2.5, 15, 0.0, 1.0, 1.0]),
#      np.array([17.5, 15.0, 0.0, 1.0, 1.0]),
#      "heterogeneous-formation"),


# ]

colors = {
    "DD1": "tab:blue",
    "DD2": "tab:cyan",
    "Holo1": "tab:green",
    "HeteroForm": "tab:purple",
    "HeteroForm2": "tab:orange",
    "HeteroForm3": "tab:brown",
    "Holo2": "tab:red",
}


def extend_traj_to_T(traj, T):
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


# ===========================================================
# WARMUP FUNCTION (ADD HERE - before planning)
# ===========================================================

def _complete_warmup(formation_agent, use_kinodynamic=False, kinodynamic_params=None):
    """
    Complete warmup: Numba + Kinodynamic solver + all caches
    """
    print("\n" + "="*70)
    print("🔥 WARMING UP ALL SYSTEMS")
    print("="*70)
    
    total_start = time.time()
    
    # 1. Numba warmup
    if NUMBA_AVAILABLE:
        print("1️⃣  Warming up Numba JIT compilation...")
        t0 = time.time()
        
        try:
            # Import numba functions from your SIRRT file
            from si_rrt_enhanced_individual_kinodynamic import (
                _nb_dist_sq_xy,
                _nb_disc_collides,
                _nb_compute_formation_discs,
                _nb_compute_robot_poses,
                _nb_max_robot_displacement,
                _nb_formation_nn_distance,
            )
            
            # Warmup distance
            _ = _nb_dist_sq_xy(0.0, 0.0, 1.0, 1.0)
            
            # Warmup collision check
            dummy_grid = np.zeros((10, 10), dtype=np.uint8)
            _ = _nb_disc_collides(5.0, 5.0, 0.5, dummy_grid, 0.1, 0.0, 0.0, 10, 10)
            
            # Warmup formation functions with ACTUAL agent parameters
            if hasattr(formation_agent, 'P_star'):
                dummy_P_star = np.array(formation_agent.P_star, dtype=np.float64)
                dummy_radii = np.array(formation_agent.radii, dtype=np.float64)
                
                _ = _nb_compute_formation_discs(
                    dummy_P_star, dummy_radii, 0.0, 0.0, 0.0, 1.0, 1.0
                )
                _ = _nb_compute_robot_poses(
                    dummy_P_star, 0.0, 0.0, 0.0, 1.0, 1.0
                )
                _ = _nb_max_robot_displacement(
                    dummy_P_star,
                    0.0, 0.0, 0.0, 1.0, 1.0,
                    1.0, 1.0, 0.5, 1.0, 1.0
                )
                
                # Warmup distance metric
                q1 = np.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64)
                q2 = np.array([1.0, 1.0, 0.5, 1.0, 1.0], dtype=np.float64)
                
                # Get metric constants from agent
                Nx = formation_agent.Nx
                Ny = formation_agent.Ny
                Nxy = formation_agent.Nxy
                Nr = formation_agent.Nr
                
                _ = _nb_formation_nn_distance(
                    q1, q2, dummy_P_star, Nx, Ny, Nxy, Nr,
                    0.7, 6.0, 0.6
                )
            
            print(f"   ✓ Numba warmup: {time.time()-t0:.2f}s")
            
        except ImportError as e:
            print(f"   ⚠️  Could not import numba functions: {e}")
        except Exception as e:
            print(f"   ⚠️  Warmup error: {e}")
    else:
        print("1️⃣  Numba not available - skipping")
    
    # 2. Kinodynamic solver warmup (THE BIG ONE!)
    if use_kinodynamic and HETERO_AVAILABLE and kinodynamic_params:
        print("2️⃣  Warming up CasADi/Ipopt solver...")
        t0 = time.time()
        
        try:
            from heterogeneous_kinodynamic_formation_steering import (
                HeterogeneousKinodynamicFormationSteering
            )
            
            # Create a dummy steerer with same parameters
            dummy_steerer = HeterogeneousKinodynamicFormationSteering(
                P_star=formation_agent.P_star,
                robot_types=formation_agent.robot_types,
                v_max=kinodynamic_params.get('v_max', formation_agent.v_max_list),
                w_max=kinodynamic_params.get('w_max', formation_agent.omega_max_list),
                a_max=kinodynamic_params.get('a_max', formation_agent.a_max_list),
                alpha_max=kinodynamic_params.get('alpha_max', formation_agent.alpha_max_list),
                N_steer=kinodynamic_params.get('N_steer', 8),
                T_steer=kinodynamic_params.get('T_steer', 0.8),
                max_iter=50,  # Quick warmup solve
            )
            
            # Perform a dummy steer (this triggers Ipopt initialization)
            q_start = np.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64)
            q_goal = np.array([2.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64)
            psi_start = np.zeros(formation_agent.Nr, dtype=np.float64)
            
            # This triggers the expensive first-time compilation
            _ = dummy_steerer.steer(q_start, q_goal, psi_start)
            
            print(f"   ✓ Kinodynamic warmup: {time.time()-t0:.2f}s")
        except Exception as e:
            print(f"   ⚠️  Kinodynamic warmup failed: {e}")
    else:
        print("2️⃣  Kinodynamic solver not needed - skipping")
    
    # 3. Formation cache warmup
    if hasattr(formation_agent, 'P_star'):
        print("3️⃣  Warming up formation caches...")
        t0 = time.time()
        q_test = np.array([5.0, 5.0, 0.0, 1.0, 1.0], dtype=np.float64)
        _ = formation_agent.discs(q_test)
        _ = formation_agent.robot_poses(q_test)
        _ = formation_agent.max_robot_displacement(q_test, q_test + 0.1)
        _ = formation_agent.dist_for_nn(q_test, q_test + 0.1)
        print(f"   ✓ Cache warmup: {time.time()-t0:.2f}s")
    
    total_time = time.time() - total_start
    print(f"\n✅ WARMUP COMPLETE: {total_time:.2f}s total")
    print("="*70 + "\n")
    
    return total_time


# ===========================================================
# CALL WARMUP (before planning loop)
# ===========================================================

# Prepare warmup parameters (use first formation agent)
warmup_kino_params = {
    'robot_types': hetero_form.robot_types,
    'v_max': hetero_form.v_max_list,
    'w_max': hetero_form.omega_max_list,
    'a_max': hetero_form.a_max_list,
    'alpha_max': hetero_form.alpha_max_list,
    'N_steer': 8,
    'T_steer': 0.8,
}

# Call warmup ONCE
warmup_time = _complete_warmup(
    formation_agent=hetero_form,
    use_kinodynamic=True,
    kinodynamic_params=warmup_kino_params
)

# ============================================================
# Planning
# ============================================================

print("="*70)
print("HETEROGENEOUS KINODYNAMIC RRT* (WITH INFLATION)")
print("="*70)
print(f"Inflation radius: {INFLATION_RADIUS}")
print(f"Warmup completed in: {warmup_time:.2f}s\n")

dynamic_obstacles = []
trajectories = {}
agent_info = {}
control_trajectories = {}

for name, agent, start, goal, agent_type in agents:
    print(f"\nPlanning {name} ({agent_type})...")
    
    # Check start/goal validity
    if agent_type == "heterogeneous-formation":
        print(f"  Formation centroid start: {start[:3]}")
        print(f"  Formation centroid goal: {goal[:3]}")
    else:
        print(f"  Start: {start[:3]}")
        print(f"  Goal: {goal[:3]}")
    
    # Verify no collisions FOR EACH ROBOT
    # For formations: agent.discs() decomposes into individual robot positions
    discs_start = agent.discs(start)
    discs_goal = agent.discs(goal)
    
    start_ok = True
    goal_ok = True
    
    # Check each robot individually
    print(f"  Checking {len(discs_start)} robot(s) at start...")
    for i, (p, r) in enumerate(discs_start):
        if static_grid.disc_collides(p[0], p[1], r):
            print(f"    ❌ Robot {i} COLLISION at start: pos=({p[0]:.2f}, {p[1]:.2f}), radius={r:.2f}")
            start_ok = False
        else:
            print(f"    ✓ Robot {i} OK at start: pos=({p[0]:.2f}, {p[1]:.2f}), radius={r:.2f}")
    
    print(f"  Checking {len(discs_goal)} robot(s) at goal...")
    for i, (p, r) in enumerate(discs_goal):
        if static_grid.disc_collides(p[0], p[1], r):
            print(f"    ❌ Robot {i} COLLISION at goal: pos=({p[0]:.2f}, {p[1]:.2f}), radius={r:.2f}")
            goal_ok = False
        else:
            print(f"    ✓ Robot {i} OK at goal: pos=({p[0]:.2f}, {p[1]:.2f}), radius={r:.2f}")
    
    if not start_ok:
        print(f"  ❌ START configuration has collision(s)!")
        continue
    if not goal_ok:
        print(f"  ❌ GOAL configuration has collision(s)!")
        continue
    
    print(f"  ✅ All robots collision-free at start and goal")



    if agent_type == "heterogeneous-formation" and HETERO_AVAILABLE:
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
    planner = SIRRT(
        agent_model=agent,
        max_velocity=1.2,                # ⬅️ Match robot speeds
        workspace_bounds=bounds,
        static_grid=static_grid,
        time_horizon=120.0,              # ⬅️ Increased from 100
        max_iter=1500,                   # ⬅️ Increased from 1200
        d_max=0.18,                      # ⬅️ Increased from 0.15
        goal_sample_rate=0.35,           # ⬅️ Increased from 0.30
        neighbor_radius=2.0,             # ⬅️ Increased from 1.8
        precision=2,
        seed=201, #2010
        debug=True,
        use_kinodynamic=use_kino,
        kinodynamic_params=kino_params,
    )

    if use_hetero and HETERO_AVAILABLE:
        print(f"  Using heterogeneous formation steering")

    # TIME THE PLANNING
    t_start = time.time()
    traj = planner.plan(start, goal, dynamic_obstacles)
    t_elapsed = time.time() - t_start
    
    if traj is None:
        print(f"  ❌ Planning FAILED after {t_elapsed:.1f}s")
        print(f"     Tree size: {len(planner.V)} vertices")
        print(f"     Try: Check collisions, increase v_max, or disable kinodynamic")
        continue

    print(f"  ✅ Planning SUCCESS in {t_elapsed:.1f}s")
    print(f"     Final time: {traj[-1][1]:.1f}s")
    print(f"     Waypoints: {len(traj)}")
    print(f"     Tree size: {len(planner.V)} vertices")

    trajectories[name] = traj

    # Extract control trajectories (per-segment)
    control_traj_list = []
    for item in traj:
        if len(item) >= 4 and item[3] is not None:
            control_traj_list.append(item[3])
    control_trajectories[name] = control_traj_list

    traj_for_obstacle = [(item[0], item[1]) for item in traj]

    # 🔒 IMPORTANT: keep robot stationary until time horizon
    traj_for_obstacle = extend_traj_to_T(traj_for_obstacle, 60.0)

    dynamic_obstacles.append({
        "trajectory": traj_for_obstacle,
        "agent": agent
    })

    agent_info[name] = {
        'agent': agent,
        'type': agent_type,
        'use_kinodynamic': use_kino
    }

    if control_traj_list:
        print(f"     Control trajectory segments: {len(control_traj_list)}")

print("\n✓ All agents planned!")

# ============================================================
# Save control trajectories (your existing logic, unchanged)
# ============================================================

print("\n" + "="*70)
print("SAVING CONTROL TRAJECTORIES (SEGMENTS)")
print("="*70)

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

    csv_file = os.path.join(CONTROL_DIR, f"{name}_controls.csv")

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)

        if agent_type == 'diff-drive':
            writer.writerow(['time', 'x', 'y', 'theta', 'v', 'omega'])
            n_states = len(all_x)
            n_controls = len(all_v)

            for i in range(n_states):
                t_val = all_t[i] if i < len(all_t) else 0.0
                x_val = all_x[i]
                y_val = all_y[i]
                theta_val = all_theta[i] if i < len(all_theta) else 0.0
                v_val = all_v[i] if i < n_controls else 0.0
                omega_val = all_omega[i] if i < len(all_omega) else 0.0
                writer.writerow([f"{t_val:.4f}", f"{x_val:.4f}", f"{y_val:.4f}",
                                 f"{theta_val:.4f}", f"{v_val:.4f}", f"{omega_val:.4f}"])
            print(f"  Saved {n_states} states to {csv_file}")

        elif agent_type == 'holonomic':
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

        else:
            # formation controls are implementation-dependent; still save if present
            writer.writerow(['time', 'x', 'y', 'theta', 'u1', 'u2'])
            n_states = min(len(all_x), len(all_y))
            for i in range(n_states):
                t_val = all_t[i] if i < len(all_t) else 0.0
                theta_val = all_theta[i] if i < len(all_theta) else 0.0
                u1 = all_v[i] if i < len(all_v) else 0.0
                u2 = all_omega[i] if i < len(all_omega) else 0.0
                writer.writerow([f"{t_val:.4f}", f"{all_x[i]:.4f}", f"{all_y[i]:.4f}",
                                 f"{theta_val:.4f}", f"{u1:.4f}", f"{u2:.4f}"])
            print(f"  Saved {n_states} rows to {csv_file}")

print("\n✓ Control logs saved!")

# ============================================================
# Extract robot trajectories (states) for ALL robots
# ============================================================

def _wrap_angle(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

robot_trajectories = {}
centroid_trajectories = {}  # only for formations: save xc,yc,th,sx,sy vs time

for name, traj in trajectories.items():
    agent = agent_info[name]['agent']
    agent_type = agent_info[name]['type']

    if agent_type == "heterogeneous-formation":
        Nr = len(agent.P_star)
        robot_paths = [[] for _ in range(Nr)]
        centroid_path = []

        for item in traj:
            q = item[0]
            t = float(item[1])
            psi = item[2] if len(item) >= 3 else None

            q = np.asarray(q, dtype=float).flatten()
            xc, yc, th, sx, sy = q[0], q[1], q[2], q[3], q[4]
            centroid_path.append(((xc, yc, th, sx, sy), t))

            poses = agent.robot_poses(q)
            for i, (x, y, theta) in enumerate(poses):
                if psi is not None and i < len(psi) and psi[i] is not None:
                    theta_use = float(psi[i])
                else:
                    theta_use = float(theta)
                robot_paths[i].append(((float(x), float(y), float(theta_use)), t))

        robot_trajectories[name] = robot_paths
        centroid_trajectories[name] = centroid_path
    else:
        path = []
        for item in traj:
            q = item[0]
            t = float(item[1])
            poses = agent.robot_poses(q)
            x, y, theta = poses[0]
            path.append(((float(x), float(y), float(theta)), t))
        robot_trajectories[name] = [path]

# ============================================================
# Save trajectories of ALL robots to CSV
# ============================================================

print("\n" + "="*70)
print("SAVING STATE TRAJECTORIES (ALL ROBOTS)")
print("="*70)

def safe_float(x):
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")

def write_robot_state_csv(csv_path, rows, header):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

for name, paths in robot_trajectories.items():
    agent = agent_info[name]['agent']
    agent_type = agent_info[name]['type']

    if agent_type == "heterogeneous-formation":
        # Save centroid trajectory (xc,yc,th,sx,sy)
        cfile = os.path.join(OUTPUT_DIR, f"{name}_centroid.csv")
        crows = []
        for (xc, yc, th, sx, sy), t in centroid_trajectories[name]:
            crows.append([f"{t:.6f}", f"{xc:.6f}", f"{yc:.6f}", f"{th:.6f}", f"{sx:.6f}", f"{sy:.6f}"])
        write_robot_state_csv(cfile, crows, ["time", "xc", "yc", "theta_c", "sx", "sy"])
        print(f"{name}: saved centroid -> {cfile}")

        # Save each robot in formation
        for i, path in enumerate(paths):
            rtype = agent.robot_types[i]
            rfile = os.path.join(OUTPUT_DIR, f"{name}_robot{i}_{rtype}.csv")
            rows = []
            for (x, y, th), t in path:
                rows.append([f"{t:.6f}", f"{x:.6f}", f"{y:.6f}", f"{th:.6f}"])
            write_robot_state_csv(rfile, rows, ["time", "x", "y", "theta"])
            print(f"{name}: saved robot {i} ({rtype}) -> {rfile}")

    elif agent_type == "diff-drive":
        ctrl_csv = os.path.join(CONTROL_DIR, f"{name}_controls.csv")
        ctrl_t, ctrl_v, ctrl_w = [], [], []
        if os.path.exists(ctrl_csv):
            with open(ctrl_csv, "r") as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    ctrl_t.append(float(row["time"]))
                    ctrl_v.append(float(row.get("v", 0.0)))
                    ctrl_w.append(float(row.get("omega", 0.0)))

        def lookup_controls(tq):
            if not ctrl_t:
                return float("nan"), float("nan")
            j = int(np.argmin(np.abs(np.asarray(ctrl_t) - tq)))
            return ctrl_v[j], ctrl_w[j]

        rfile = os.path.join(OUTPUT_DIR, f"{name}.csv")
        rows = []
        for (x, y, th), t in paths[0]:
            v, w = lookup_controls(t)
            rows.append([f"{t:.6f}", f"{x:.6f}", f"{y:.6f}", f"{th:.6f}",
                         f"{safe_float(v):.6f}", f"{safe_float(w):.6f}"])
        write_robot_state_csv(rfile, rows, ["time", "x", "y", "theta", "v", "omega"])
        print(f"{name}: saved -> {rfile}")

    elif agent_type == "holonomic":
        ctrl_csv = os.path.join(CONTROL_DIR, f"{name}_controls.csv")
        ctrl_t, ctrl_vx, ctrl_vy = [], [], []
        if os.path.exists(ctrl_csv):
            with open(ctrl_csv, "r") as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    ctrl_t.append(float(row["time"]))
                    ctrl_vx.append(float(row.get("vx", 0.0)))
                    ctrl_vy.append(float(row.get("vy", 0.0)))

        def lookup_controls(tq):
            if not ctrl_t:
                return float("nan"), float("nan")
            j = int(np.argmin(np.abs(np.asarray(ctrl_t) - tq)))
            return ctrl_vx[j], ctrl_vy[j]

        rfile = os.path.join(OUTPUT_DIR, f"{name}.csv")
        rows = []
        for (x, y, th), t in paths[0]:
            vx, vy = lookup_controls(t)
            rows.append([f"{t:.6f}", f"{x:.6f}", f"{y:.6f}",
                         f"{safe_float(vx):.6f}", f"{safe_float(vy):.6f}"])
        write_robot_state_csv(rfile, rows, ["time", "x", "y", "vx", "vy"])
        print(f"{name}: saved -> {rfile}")

print("\n✓ Trajectory CSVs saved!")

# ============================================================
# Visualization
# ============================================================

print("\n" + "="*70)
print("Creating visualization...")
print("="*70)

fig, ax = plt.subplots(figsize=(16, 10))
plt.subplots_adjust(bottom=0.15, right=0.80)

ax.imshow(grid[::-1], cmap="gray_r", extent=[0, width, 0, height], alpha=0.85)

# Build time vector
all_times = sorted(set(
    t for paths in robot_trajectories.values()
    for rp in paths
    for (_, t) in rp
))
Tmax = all_times[-1] if all_times else 0.0

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
    col = colors.get(name, "tab:gray")

    if agent_type == "heterogeneous-formation":
        for i in range(len(robot_paths)):
            base_r = agent.radii[i]
            r = base_r + agent.inflation if DRAW_INFLATED_FOOTPRINTS else base_r
            is_dd = (agent.robot_types[i] == 'diff-drive')
            circ, arr = draw_robot(0, 0, 0, r, is_dd, col)
            robot_artists.append((name, i, circ, arr))
    else:
        base_r = agent.radius
        infl = getattr(agent, "inflation", 0.0)
        r = base_r + infl if DRAW_INFLATED_FOOTPRINTS else base_r
        is_dd = (agent_type == "diff-drive")
        circ, arr = draw_robot(0, 0, 0, r, is_dd, col)
        robot_artists.append((name, 0, circ, arr))

ax_time = plt.axes([0.2, 0.05, 0.6, 0.03])
time_slider = Slider(ax_time, "Time", 0.0, Tmax, valinit=0.0)
# ============================================================
# Play / Pause Button
# ============================================================

is_playing = False
dt_play = 0.05  # seconds per frame (controls playback speed)

ax_play = plt.axes([0.83, 0.05, 0.08, 0.04])
btn_play = Button(ax_play, "▶ Play")

def on_play_clicked(event):
    global is_playing
    is_playing = not is_playing
    btn_play.label.set_text("❚❚ Pause" if is_playing else "▶ Play")

btn_play.on_clicked(on_play_clicked)

# Timer for playback
timer = fig.canvas.new_timer(interval=int(dt_play * 1000))

def advance_time():
    if not is_playing:
        return

    t = time_slider.val + dt_play
    if t >= Tmax:
        t = Tmax
        time_slider.set_val(t)
        # auto-stop at end
        on_play_clicked(None)
    else:
        time_slider.set_val(t)

timer.add_callback(advance_time)
timer.start()

def interp_pose(path, t):
    for i in range(len(path)-1):
        (x1, y1, th1), t1 = path[i]
        (x2, y2, th2), t2 = path[i+1]
        if t1 <= t <= t2:
            a = (t - t1) / (t2 - t1 + 1e-9)
            x = x1 + a*(x2 - x1)
            y = y1 + a*(y2 - y1)
            th = th1 + a*((th2 - th1 + np.pi)%(2*np.pi)-np.pi)
            return x, y, th
    return path[-1][0]

def update(_val):
    t = time_slider.val
    for name, idx, circ, arr in robot_artists:
        path = robot_trajectories[name][idx]
        x, y, th = interp_pose(path, t)
        circ.center = (x, y)
        if arr is not None:
            L = circ.radius * 1.8
            arr.set_positions((x, y), (x + L*np.cos(th), y + L*np.sin(th)))
    fig.canvas.draw_idle()

time_slider.on_changed(update)
update(0.0)

# Draw static trajectories
for name, robot_paths in robot_trajectories.items():
    agent = agent_info[name]['agent']
    agent_type = agent_info[name]['type']
    col = colors.get(name, "tab:gray")

    if agent_type == "heterogeneous-formation":
        for i, path in enumerate(robot_paths):
            pts = np.array([pos[:2] for pos, _ in path])
            rtype = agent.robot_types[i]
            style = '-' if rtype == 'diff-drive' else ':'
            ax.plot(pts[:, 0], pts[:, 1], style, lw=1.8, color=col, alpha=0.6)

        # centroid path
        centroid_pts = np.array([item[0][:2] for item in trajectories[name]])
        ax.plot(centroid_pts[:, 0], centroid_pts[:, 1], lw=3, color=col, alpha=0.9)

    else:
        path = robot_paths[0]
        pts = np.array([pos[:2] for pos, _ in path])
        style = '-' if agent_type == "diff-drive" else ':'
        ax.plot(pts[:, 0], pts[:, 1], style, lw=2.5, color=col, alpha=0.8)

ax.set_xlim(0, width)
ax.set_ylim(0, height)
ax.set_aspect("equal", "box")
ax.set_title(f"Heterogeneous Planning (Inflation={INFLATION_RADIUS}, draw_inflated={DRAW_INFLATED_FOOTPRINTS})")
plt.show()
