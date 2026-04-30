#!/usr/bin/env python3
"""
Complete Test: Heterogeneous Formations + Individual Agents  —  PATCHED

Velocity-limit fix
------------------
Previously every SIRRT planner was created with max_velocity=1.2 regardless
of the agent's own speed limits, causing all waypoint timestamps to be
computed at 1.2 m/s even for agents capped at e.g. 0.26 m/s.

The patch (two changes):

1. SIRRT is now called with `max_velocity` set to the agent's actual
   maximum speed:
     - DD / holonomic individual  →  agent.v_max
     - Heterogeneous formation    →  min(agent.v_max_list)  [slowest robot]

2. Kinematics.travel_time() in si_rrt_enhanced_individual_kinodynamic.py
   was updated to derive speed from agent.v_max / agent.v_max_list
   *inside* the planner as well, so the two layers are consistent.
   (See the patched si_rrt file for details.)

Everything else is identical to the original script.
"""

import os as _os_locate; print(f"[LOCATE] Running main script: {_os_locate.path.abspath(__file__)}")

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
print(f"Heterogeneous steering available: {HETERO_AVAILABLE}")
try:
    from scipy.spatial import cKDTree
    print(f"scipy KD-tree available: True")
except ImportError:
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

INFLATION_RADIUS = 0.0
DRAW_INFLATED_FOOTPRINTS = True

OUTPUT_DIR = "trajectory_logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONTROL_DIR = "control_logs"
os.makedirs(CONTROL_DIR, exist_ok=True)

# ===========================================================
# Agent Classes  (unchanged from original)
# ===========================================================

class DifferentialDriveAgent:
    def __init__(self, radius=0.40, v_max=0.26, omega_max=2.0,
                 a_max=2.0, alpha_max=4.0, inflation=0.0):
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
        dx = float(q2[0] - q1[0]); dy = float(q2[1] - q1[1])
        dth = abs((q2[2] - q1[2] + np.pi) % (2*np.pi) - np.pi)
        return np.hypot(dx, dy) + 0.3 * dth


class HolonomicAgent:
    def __init__(self, radius=0.40, v_max=1.0, a_max=2.0, inflation=0.0):
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
    def __init__(self, P_star, robot_types, radius=0.25, v_max=0.22,
                 omega_max=2.0, a_max=2.0, alpha_max=4.0,
                 sx_range=(1.1, 3.0), sy_range=(1.1, 3.0), inflation=0.0):
        self.P_star = np.array(P_star, dtype=float)
        self.robot_types = robot_types
        self.Nr = len(P_star)
        self.sx_range = sx_range
        self.sy_range = sy_range
        self.inflation = float(inflation)

        def _broadcast(val, name):
            if isinstance(val, (int, float)):
                return [float(val)] * self.Nr
            lst = [float(v) for v in val]
            if len(lst) != self.Nr:
                raise ValueError(f"{name} length ({len(lst)}) != Nr ({self.Nr})")
            return lst

        self.radii = _broadcast(radius, 'radius')
        self.radius = self.radii
        self.v_max_list = _broadcast(v_max, 'v_max')
        self.omega_max_list = _broadcast(omega_max, 'omega_max')
        self.a_max_list = _broadcast(a_max, 'a_max')
        self.alpha_max_list = _broadcast(alpha_max, 'alpha_max')

        self._disc_cache = {}

        from si_rrt_enhanced_individual_kinodynamic import precompute_constants
        self.Nx, self.Ny, self.Nxy = precompute_constants(self.P_star)

        print(f"Heterogeneous formation: {robot_types}")
        print(f"  radii      = {self.radii}")
        print(f"  v_max      = {self.v_max_list}")
        import os as _osha; print(f"  [LOCATE] HeterogeneousFormationAgent constructed in: {_osha.path.abspath(__file__)}")
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
            discs.append((p, self.radii[i] + self.inflation))
        self._disc_cache[qkey] = discs
        return discs

    def centroid_xy(self, q):
        return np.asarray(q, dtype=float)[:2].copy()

    def robot_poses(self, q):
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
        dsx = abs(q2[3] - q1[3]); dsy = abs(q2[4] - q1[4])
        return np.hypot(dx, dy) + 0.3 * dth + 0.2 * (dsx + dsy)

    def clear_cache(self):
        self._disc_cache.clear()


# ===========================================================
# Workspace + Map
# ===========================================================

grid = np.load('10_103_a_outside_1.npy')
res = 0.1
height, width = grid.shape
height, width = height * res, width * res
print(f"Grid width {width} ")
print(f"Grid height {height} ")

bounds = (np.array([0.0, 0.0]), np.array([height, width]))
W = int(width / res)
H = int(height / res)
static_grid = OccupancyGrid(grid, res)

# ===========================================================
# Agents
# ===========================================================


burger1 = DifferentialDriveAgent(
    radius=0.2, v_max=0.22, omega_max=2.84,
    a_max=2.5, alpha_max=3.2, inflation=INFLATION_RADIUS,
)

burger2 = DifferentialDriveAgent(
    radius=0.2, v_max=0.22, omega_max=2.84,
    a_max=2.5, alpha_max=3.2, inflation=INFLATION_RADIUS,
)

burger3 = DifferentialDriveAgent(
    radius=0.2, v_max=0.22, omega_max=2.84,
    a_max=2.5, alpha_max=3.2, inflation=INFLATION_RADIUS,
)


waffle = DifferentialDriveAgent(
    radius=0.25, v_max=0.26, omega_max=1.82,
    a_max=2.5, alpha_max=3.2, inflation=INFLATION_RADIUS,
)



agents = [



    ("waffle", waffle,
    np.array([4.56, 4.375, 3.14]),
    np.array([1.14, 4.375, 3.14]),
     "diff-drive"),

    ("burger1", burger1,
    np.array([1.14, 4.375, 0.0]),
    np.array([4.56, 4.375, 0.0]),
     "diff-drive"),

    ("burger2", burger2,
    np.array([2.85, 1.75, 1.57]),
    np.array([2.85, 7.0,  1.57]),
     "diff-drive"),

    ("burger3", burger3,
    np.array([2.85, 7.00, -1.57]),
    np.array([2.85, 1.75, -1.57]),
     "diff-drive"),

]

colors = {
    "DD1": "tab:blue",
    "DD2": "tab:cyan",
    "Holo1": "tab:green",
    "HeteroForm": "tab:purple",
    "HeteroForm1": "tab:purple",
    "HeteroForm2": "tab:orange",
    "HeteroForm3": "tab:brown",
    "Holo2": "tab:red",
    "waffle": "tab:red",
    "burger1": "tab:blue",
    "burger2": "tab:cyan",
    "burger3": "tab:brown",
}


def extend_traj_to_T(traj, T):
    if not traj:
        return traj
    q_last, t_last = traj[-1]
    if t_last >= T - 1e-9:
        return traj
    traj = list(traj)
    traj.append((np.asarray(q_last, dtype=float).copy(), float(T)))
    return traj


# ===========================================================
# PATCH HELPER: resolve per-agent max velocity
# ===========================================================

def _resolve_agent_vmax(agent, agent_type: str) -> float:
    """
    Return the physically correct maximum speed for this agent.

    Used to set max_velocity in SIRRT so that the planner's internal
    Kinematics and the neighbour-radius heuristics are consistent with
    the actual robot limits.

      - Heterogeneous formation  →  min(v_max_list)
      - DD / holonomic individual →  agent.v_max
    """
    if agent_type == "heterogeneous-formation" and hasattr(agent, 'v_max_list'):
        return float(min(agent.v_max_list))
    if hasattr(agent, 'v_max') and not isinstance(getattr(agent, 'v_max'), (list, tuple, np.ndarray)):
        return float(agent.v_max)
    # Fallback — should not normally be reached
    return 1.2


# ===========================================================
# WARMUP
# ===========================================================

def _complete_warmup(formation_agent, use_kinodynamic=False, kinodynamic_params=None):
    print("\n" + "="*70)
    print("WARMING UP ALL SYSTEMS")
    print("="*70)
    total_start = time.time()

    if NUMBA_AVAILABLE:
        print("Warming up Numba JIT compilation...")
        t0 = time.time()
        try:
            from si_rrt_enhanced_individual_kinodynamic import (
                _nb_dist_sq_xy, _nb_disc_collides,
                _nb_compute_formation_discs, _nb_compute_robot_poses,
                _nb_max_robot_displacement, _nb_formation_nn_distance,
            )
            _ = _nb_dist_sq_xy(0.0, 0.0, 1.0, 1.0)
            dummy_grid = np.zeros((10, 10), dtype=np.uint8)
            _ = _nb_disc_collides(5.0, 5.0, 0.5, dummy_grid, 0.1, 0.0, 0.0, 10, 10)

            if hasattr(formation_agent, 'P_star'):
                dummy_P_star = np.array(formation_agent.P_star, dtype=np.float64)
                dummy_radii  = np.array(formation_agent.radii,  dtype=np.float64)
                _ = _nb_compute_formation_discs(dummy_P_star, dummy_radii, 0.0, 0.0, 0.0, 1.0, 1.0)
                _ = _nb_compute_robot_poses(dummy_P_star, 0.0, 0.0, 0.0, 1.0, 1.0)
                _ = _nb_max_robot_displacement(dummy_P_star, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0)
                q1 = np.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64)
                q2 = np.array([1.0, 1.0, 0.5, 1.0, 1.0], dtype=np.float64)
                _ = _nb_formation_nn_distance(
                    q1, q2, dummy_P_star,
                    formation_agent.Nx, formation_agent.Ny, formation_agent.Nxy,
                    formation_agent.Nr, 0.7, 6.0, 0.6
                )
            print(f"   Numba warmup: {time.time()-t0:.2f}s")
        except Exception as e:
            print(f"   Numba warmup warning: {e}")

    if use_kinodynamic and HETERO_AVAILABLE and kinodynamic_params:
        print("Warming up CasADi/Ipopt solver...")
        t0 = time.time()
        try:
            from heterogeneous_kinodynamic_formation_steering import HeterogeneousKinodynamicFormationSteering
            dummy_steerer = HeterogeneousKinodynamicFormationSteering(
                P_star=formation_agent.P_star,
                robot_types=formation_agent.robot_types,
                v_max=kinodynamic_params.get('v_max', formation_agent.v_max_list),
                w_max=kinodynamic_params.get('w_max', formation_agent.omega_max_list),
                a_max=kinodynamic_params.get('a_max', formation_agent.a_max_list),
                alpha_max=kinodynamic_params.get('alpha_max', formation_agent.alpha_max_list),
                N_steer=kinodynamic_params.get('N_steer', 8),
                T_steer=kinodynamic_params.get('T_steer', 0.8),
                max_iter=50,
            )
            q_s = np.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64)
            q_g = np.array([2.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64)
            psi0 = np.zeros(formation_agent.Nr, dtype=np.float64)
            _ = dummy_steerer.steer(q_s, q_g, psi0)
            print(f"   Kinodynamic warmup: {time.time()-t0:.2f}s")
        except Exception as e:
            print(f"   Kinodynamic warmup warning: {e}")

    if hasattr(formation_agent, 'P_star'):
        q_test = np.array([5.0, 5.0, 0.0, 1.0, 1.0], dtype=np.float64)
        _ = formation_agent.discs(q_test)
        _ = formation_agent.robot_poses(q_test)

    total_time = time.time() - total_start
    print(f"WARMUP COMPLETE: {total_time:.2f}s total")
    print("="*70 + "\n")
    return total_time


# warmup_kino_params = {
#     'robot_types': hetero_form2.robot_types,
#     'v_max': hetero_form2.v_max_list,
#     'w_max': hetero_form2.omega_max_list,
#     'a_max': hetero_form2.a_max_list,
#     'alpha_max': hetero_form2.alpha_max_list,
#     'N_steer': 8,
#     'T_steer': 0.8,
# }

# warmup_time = _complete_warmup(
#     formation_agent=hetero_form2,
#     use_kinodynamic=True,
#     kinodynamic_params=warmup_kino_params
# )

# ===========================================================
# Planning
# ===========================================================

print("="*70)
print("HETEROGENEOUS KINODYNAMIC RRT* (VELOCITY-LIMIT PATCH ACTIVE)")
print("="*70)
print(f"Inflation radius : {INFLATION_RADIUS}")
# print(f"Warmup completed : {warmup_time:.2f}s\n")

dynamic_obstacles  = []
trajectories       = {}
agent_info         = {}
control_trajectories = {}

for name, agent, start, goal, agent_type in agents:
    print(f"\nPlanning {name} ({agent_type})...")

    # ------------------------------------------------------------------
    # PATCH: derive per-agent max_velocity from the agent object itself
    # ------------------------------------------------------------------
    agent_vmax = _resolve_agent_vmax(agent, agent_type)
    print(f"  max_velocity for planner: {agent_vmax:.4f} m/s  (from agent limits)")
    # ------------------------------------------------------------------

    if agent_type == "heterogeneous-formation":
        print(f"  Formation centroid start: {start[:3]}")
        print(f"  Formation centroid goal:  {goal[:3]}")
    else:
        print(f"  Start: {start[:3]}")
        print(f"  Goal:  {goal[:3]}")

    discs_start = agent.discs(start)
    discs_goal  = agent.discs(goal)
    start_ok = True; goal_ok = True

    print(f"  Checking {len(discs_start)} robot(s) at start...")
    for i, (p, r) in enumerate(discs_start):
        if static_grid.disc_collides(p[0], p[1], r):
            print(f"    COLLISION at start: robot {i} pos=({p[0]:.2f},{p[1]:.2f}) r={r:.2f}")
            start_ok = False
        else:
            print(f"    OK at start: robot {i} pos=({p[0]:.2f},{p[1]:.2f}) r={r:.2f}")

    print(f"  Checking {len(discs_goal)} robot(s) at goal...")
    for i, (p, r) in enumerate(discs_goal):
        if static_grid.disc_collides(p[0], p[1], r):
            print(f"    COLLISION at goal: robot {i} pos=({p[0]:.2f},{p[1]:.2f}) r={r:.2f}")
            goal_ok = False
        else:
            print(f"    OK at goal: robot {i} pos=({p[0]:.2f},{p[1]:.2f}) r={r:.2f}")

    if not start_ok:
        print(f"  START configuration has collision(s)!"); continue
    if not goal_ok:
        print(f"  GOAL configuration has collision(s)!"); continue
    print(f"  All robots collision-free at start and goal")

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
        use_kino = True; use_hetero = True

    elif agent_type == "diff-drive":
        kino_params = {
            'v_max': agent.v_max,
            'omega_max': agent.omega_max,
            'a_max': agent.a_max,
            'alpha_max': agent.alpha_max,
            'dt': 0.05,
        }
        use_kino = True; use_hetero = False

    elif agent_type == "holonomic":
        kino_params = {
            'v_max': agent.v_max,
            'a_max': agent.a_max,
            'dt': 0.05,
        }
        use_kino = True; use_hetero = False

    else:
        use_kino = False; use_hetero = False; kino_params = None

    # ------------------------------------------------------------------
    # PATCH: pass agent_vmax (not hardcoded 1.2) to max_velocity
    # ------------------------------------------------------------------
    print(f"[LOCATE] About to create SIRRT: agent={name}, agent.v_max={getattr(agent,'v_max','N/A')}, v_max_list={getattr(agent,'v_max_list','N/A')}, agent_vmax={agent_vmax}")
    planner = SIRRT(
        agent_model=agent,
        max_velocity=agent_vmax,          # ← PATCHED (was 1.2 for all agents)
        workspace_bounds=bounds,
        static_grid=static_grid,
        time_horizon=120.0,
        max_iter=1500,
        d_max=0.18,
        goal_sample_rate=0.35,
        neighbor_radius=2.0,
        precision=2,
        seed=431, #354 # 361 2F #371 #375 #413 #395 #407 #414 #429
        # seed=481, #422 433 449 459 475 481
        debug=True,
        use_kinodynamic=use_kino,
        kinodynamic_params=kino_params,
    )
    # ------------------------------------------------------------------

    if use_hetero and HETERO_AVAILABLE:
        print(f"  Using heterogeneous formation steering")

    t_start = time.time()
    traj = planner.plan(start, goal, dynamic_obstacles)
    t_elapsed = time.time() - t_start

    if traj is None:
        print(f"  Planning FAILED after {t_elapsed:.1f}s")
        print(f"     Tree size: {len(planner.V)} vertices")
        continue

    print(f"  Planning SUCCESS in {t_elapsed:.1f}s")
    print(f"     Final time: {traj[-1][1]:.1f}s")
    print(f"     Waypoints: {len(traj)}")
    print(f"     Tree size: {len(planner.V)} vertices")

    trajectories[name] = traj

    control_traj_list = []
    for item in traj:
        if len(item) >= 4 and item[3] is not None:
            control_traj_list.append(item[3])
    control_trajectories[name] = control_traj_list

    traj_for_obstacle = [(item[0], item[1]) for item in traj]
    traj_for_obstacle = extend_traj_to_T(traj_for_obstacle, 60.0)
    dynamic_obstacles.append({"trajectory": traj_for_obstacle, "agent": agent})

    agent_info[name] = {
        'agent': agent,
        'type': agent_type,
        'use_kinodynamic': use_kino
    }

    if control_traj_list:
        print(f"     Control trajectory segments: {len(control_traj_list)}")

print("\nAll agents planned!")

# ===========================================================
# Save control trajectories  (unchanged)
# ===========================================================

print("\n" + "="*70)
print("SAVING CONTROL TRAJECTORIES")
print("="*70)

for name in trajectories.keys():
    agent_type   = agent_info[name]['type']
    control_list = control_trajectories[name]

    if not control_list:
        print(f"\n{name}: No control trajectory data"); continue

    print(f"\n{name} ({agent_type}):")
    all_x, all_y, all_t = [], [], []
    all_theta, all_v, all_omega, all_vy = [], [], [], []
    cumulative_time = 0.0

    for ctrl_traj in control_list:
        if ctrl_traj is None:
            continue
        x = ctrl_traj.x; y = ctrl_traj.y
        t = ctrl_traj.t + cumulative_time if ctrl_traj.t is not None else None
        if t is not None and len(t) > 0:
            cumulative_time = t[-1]
        all_x.extend(x.tolist()); all_y.extend(y.tolist())
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
            n_states = len(all_x); n_controls = len(all_v)
            for i in range(n_states):
                t_val    = all_t[i] if i < len(all_t) else 0.0
                theta_v  = all_theta[i] if i < len(all_theta) else 0.0
                v_val    = all_v[i] if i < n_controls else 0.0
                omega_v  = all_omega[i] if i < len(all_omega) else 0.0
                writer.writerow([f"{t_val:.4f}", f"{all_x[i]:.4f}", f"{all_y[i]:.4f}",
                                 f"{theta_v:.4f}", f"{v_val:.4f}", f"{omega_v:.4f}"])
            print(f"  Saved {n_states} states to {csv_file}")
        elif agent_type == 'holonomic':
            writer.writerow(['time', 'x', 'y', 'vx', 'vy'])
            n_states = len(all_x); n_controls = len(all_v)
            for i in range(n_states):
                t_val  = all_t[i] if i < len(all_t) else 0.0
                vx_val = all_v[i] if i < n_controls else 0.0
                vy_val = all_vy[i] if i < len(all_vy) else 0.0
                writer.writerow([f"{t_val:.4f}", f"{all_x[i]:.4f}", f"{all_y[i]:.4f}",
                                 f"{vx_val:.4f}", f"{vy_val:.4f}"])
            print(f"  Saved {n_states} states to {csv_file}")
        else:
            writer.writerow(['time', 'x', 'y', 'theta', 'u1', 'u2'])
            n_states = min(len(all_x), len(all_y))
            for i in range(n_states):
                t_val    = all_t[i] if i < len(all_t) else 0.0
                theta_v  = all_theta[i] if i < len(all_theta) else 0.0
                u1       = all_v[i] if i < len(all_v) else 0.0
                u2       = all_omega[i] if i < len(all_omega) else 0.0
                writer.writerow([f"{t_val:.4f}", f"{all_x[i]:.4f}", f"{all_y[i]:.4f}",
                                 f"{theta_v:.4f}", f"{u1:.4f}", f"{u2:.4f}"])
            print(f"  Saved {n_states} rows to {csv_file}")

print("\nControl logs saved!")

# ===========================================================
# Extract & save robot state trajectories  (unchanged)
# ===========================================================

def _wrap_angle(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

robot_trajectories   = {}
centroid_trajectories = {}

for name, traj in trajectories.items():
    agent      = agent_info[name]['agent']
    agent_type = agent_info[name]['type']

    if agent_type == "heterogeneous-formation":
        Nr = len(agent.P_star)
        robot_paths = [[] for _ in range(Nr)]
        centroid_path = []

        for item in traj:
            q = item[0]; t = float(item[1])
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

        robot_trajectories[name]   = robot_paths
        centroid_trajectories[name] = centroid_path
    else:
        path = []
        for item in traj:
            q = item[0]; t = float(item[1])
            poses = agent.robot_poses(q)
            x, y, theta = poses[0]
            path.append(((float(x), float(y), float(theta)), t))
        robot_trajectories[name] = [path]

print("\n" + "="*70)
print("SAVING STATE TRAJECTORIES (ALL ROBOTS)")
print("="*70)

def safe_float(x):
    try:
        if x is None: return float("nan")
        return float(x)
    except Exception:
        return float("nan")

def write_robot_state_csv(csv_path, rows, header):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows:
            w.writerow(r)

for name, paths in robot_trajectories.items():
    agent      = agent_info[name]['agent']
    agent_type = agent_info[name]['type']

    if agent_type == "heterogeneous-formation":
        cfile = os.path.join(OUTPUT_DIR, f"{name}_centroid.csv")
        crows = []
        for (xc, yc, th, sx, sy), t in centroid_trajectories[name]:
            crows.append([f"{t:.6f}", f"{xc:.6f}", f"{yc:.6f}",
                          f"{th:.6f}", f"{sx:.6f}", f"{sy:.6f}"])
        write_robot_state_csv(cfile, crows, ["time", "xc", "yc", "theta_c", "sx", "sy"])
        print(f"{name}: saved centroid -> {cfile}")

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
            if not ctrl_t: return float("nan"), float("nan")
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
            if not ctrl_t: return float("nan"), float("nan")
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

print("\nTrajectory CSVs saved!")

# ===========================================================
# Visualization  (unchanged)
# ===========================================================

print("\n" + "="*70)
print("Creating visualization...")
print("="*70)

fig, ax = plt.subplots(figsize=(16, 10))
plt.subplots_adjust(bottom=0.15, right=0.80)
ax.imshow(grid[::-1], cmap="gray_r", extent=[0, width, 0, height], alpha=0.85)

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
            (x, y), (x + L*np.cos(theta), y + L*np.sin(theta)),
            arrowstyle='->', lw=2, color=color
        )
        ax.add_patch(arrow)
    return circ, arrow

for name, robot_paths in robot_trajectories.items():
    agent      = agent_info[name]['agent']
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

ax_time   = plt.axes([0.2, 0.05, 0.6, 0.03])
time_slider = Slider(ax_time, "Time", 0.0, Tmax, valinit=0.0)

is_playing = False
dt_play = 0.05
ax_play = plt.axes([0.83, 0.05, 0.08, 0.04])
btn_play = Button(ax_play, "▶ Play")

def on_play_clicked(event):
    global is_playing
    is_playing = not is_playing
    btn_play.label.set_text("❚❚ Pause" if is_playing else "▶ Play")

btn_play.on_clicked(on_play_clicked)
timer = fig.canvas.new_timer(interval=int(dt_play * 1000))

def advance_time():
    if not is_playing: return
    t = time_slider.val + dt_play
    if t >= Tmax:
        t = Tmax; time_slider.set_val(t); on_play_clicked(None)
    else:
        time_slider.set_val(t)

timer.add_callback(advance_time)
timer.start()

def interp_pose(path, t):
    for i in range(len(path)-1):
        (x1,y1,th1), t1 = path[i]
        (x2,y2,th2), t2 = path[i+1]
        if t1 <= t <= t2:
            a = (t - t1) / (t2 - t1 + 1e-9)
            x  = x1 + a*(x2-x1)
            y  = y1 + a*(y2-y1)
            th = th1 + a*((th2-th1+np.pi)%(2*np.pi)-np.pi)
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

for name, robot_paths in robot_trajectories.items():
    agent      = agent_info[name]['agent']
    agent_type = agent_info[name]['type']
    col = colors.get(name, "tab:gray")

    if agent_type == "heterogeneous-formation":
        for i, path in enumerate(robot_paths):
            pts = np.array([pos[:2] for pos, _ in path])
            rtype = agent.robot_types[i]
            style = '-' if rtype == 'diff-drive' else ':'
            ax.plot(pts[:, 0], pts[:, 1], style, lw=1.8, color=col, alpha=0.6)
        centroid_pts = np.array([item[0][:2] for item in trajectories[name]])
        ax.plot(centroid_pts[:, 0], centroid_pts[:, 1], lw=3, color=col, alpha=0.9)
    else:
        path  = robot_paths[0]
        pts   = np.array([pos[:2] for pos, _ in path])
        style = '-' if agent_type == "diff-drive" else ':'
        ax.plot(pts[:, 0], pts[:, 1], style, lw=2.5, color=col, alpha=0.8)

ax.set_xlim(0, width); ax.set_ylim(0, height)
ax.set_aspect("equal", "box")
ax.set_title(f"Heterogeneous Planning — velocity limits enforced (inflation={INFLATION_RADIUS})")
plt.show()