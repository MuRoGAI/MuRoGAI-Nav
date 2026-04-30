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
import json

from si_rrt_enhanced_individual_kinodynamic import (
    SIRRT, OccupancyGrid, NUMBA_AVAILABLE
)

from agents import DifferentialDriveAgent, HolonomicAgent, HeterogeneousFormationAgent

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

map_path = "/home/multi-robot/murogai_nav/src/MuRoGAI-Nav/path_plan/path_planner/path_planner/10_103_a_outside.npy"
res = 0.1

grid = np.load(map_path)
height, width = grid.shape
height, width = height * res, width * res
bounds = (np.array([0.0, 0.0]), np.array([height, width]))
W = int(width / res)
H = int(height / res)
static_grid = OccupancyGrid(grid, res)

# ===========================================================
# CONFIG
# ===========================================================
config_file_path = "/home/multi-robot/murogai_nav/src/MuRoGAI-Nav/chatty/config/robot_config_103_1.json"
config_data = json.load(open(config_file_path, 'r'))
path_planner_cfg = config_data["path_planner"]

# ===========================================================
# DUMMY START POSES — replace with real odometry
# ===========================================================


# dummy_start_poses = {
#     "burger1" : (2.565, 0.875,  1.57),
#     "burger2" : (3.705, 0.875,  1.57),
#     "burger3" : (3.135, 1.75,   1.57),
#     "waffle"  : (5.13,  4.375,  3.14),
#     "tb4_1"   : (3.705, 7.875, -1.57),
#     "firebird": (2.565, 7.875, -1.57),
#     "go2"     : (3.135, 7.0),
# }


# eg_formation = {
#     # "F1": {
#     #     "centroid_x": 3.135,
#     #     "centroid_y": 7.583,
#     #     "formation_yaw": 1.57,
#     #     "desired_radius": 1.0,
#     #     "robots": ["burger1", "burger2", "burger3"]
#     # },
#     # "F2": {
#     #     "centroid_x": 3.135,
#     #     "centroid_y": 1.75,
#     #     "formation_yaw": -1.57,
#     #     "desired_radius": 1.0,
#     #     "robots": ["tb4_1", "firebird", "go2"]
#     # },
#     "R1": { 
#         "robot": "waffle", "x": 1.140, "y": 4.375, "yaw": 3.14
#     },
# }

dummy_start_poses = {
    "burger1" : (1.14, 4.375, 0.00),
    "waffle"  : (4.56, 4.375, 3.14),
    "firebird": (2.85, 1.75, 1.57),
    "go2"     : (2.85, 7.0),
}

eg_formation = {
    "R4": { 
        "robot": "go2", "x": 2.85, "y": 1.75, "yaw": -1.57
    },
    "R3": { 
        "robot": "firebird", "x": 2.85, "y": 7.0, "yaw": 1.57
    },
    "R2": { 
        "robot": "waffle", "x": 1.14, "y": 4.375, "yaw": 3.14
    },
    "R1": { 
        "robot": "burger1", "x": 4.56, "y": 4.375, "yaw": 0.0
    }
}


# ===========================================================
# HELPERS
# ===========================================================
def get_drive_type(robot_name: str) -> str:
    cfg = path_planner_cfg.get(robot_name, {})
    t = cfg.get("type", "Holonomic Drive Robot").lower()
    return "diff-drive" if "differential" in t else "holonomic"


def estimate_centroid_and_p_star(poses, theta, sx, sy, robot_types=None):
    """
    Estimate centroid and P_star from robot poses.
    poses : list of (x, y, yaw) for diff-drive  OR  (x, y) for holonomic
    theta : formation yaw at start
    """
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
        [math.sin(theta),  math.cos(theta)]
    ])
    R_inv = R.T
    S_inv = np.diag([1.0 / sx, 1.0 / sy])

    P_star = []
    for pos in positions:
        p = S_inv @ R_inv @ (pos - centroid)
        P_star.append((float(p[0]), float(p[1])))

    return centroid.tolist(), P_star


def compute_formation_start(robot_names, robot_types, dummy_start_poses):
    """
    Compute formation start centroid and yaw from dummy start poses.
    centroid_x = mean of all robot x
    centroid_y = mean of all robot y
    start_yaw  = mean of yaws of diff-drive robots (holonomic have no yaw)
    """
    xs, ys, yaws = [], [], []
    for i, rname in enumerate(robot_names):
        pose = dummy_start_poses.get(rname, (0.0, 0.0, 0.0))
        xs.append(pose[0])
        ys.append(pose[1])
        if robot_types[i] == "diff-drive":
            yaws.append(pose[2])

    cx   = float(np.mean(xs))
    cy   = float(np.mean(ys))
    yaw  = float(np.mean(yaws)) if yaws else 0.0
    return cx, cy, yaw


def compute_p_star_from_dummy_poses(robot_names, start_yaw, robot_types,
                                     dummy_start_poses, sx=1.0, sy=1.0):
    """
    Compute P_star from actual dummy start poses using start_yaw.
    """
    poses = [dummy_start_poses.get(r, (0.0, 0.0, 0.0)) for r in robot_names]
    _, P_star = estimate_centroid_and_p_star(poses, start_yaw, sx, sy, robot_types)
    return P_star


# ===========================================================
# AUTO-BUILD agents
# ===========================================================
agents = []
colors = {}

for key, val in eg_formation.items():

    # ── FORMATION (F*) ──────────────────────────────────────
    if key.startswith("F"):
        robot_names = val["robots"]
        robot_types, v_max_list, omega_max_list = [], [], []
        a_max_list, alpha_max_list, radius_list  = [], [], []

        for rname in robot_names:
            cfg   = path_planner_cfg.get(rname, {})
            drive = get_drive_type(rname)
            robot_types.append(drive)
            v_max_list.append(cfg.get("max_linear_velocity_x", 0.25))
            omega_max_list.append(cfg.get("max_angular_velocity_z", 0.4) if drive == "diff-drive" else 0.0)
            a_max_list.append(cfg.get("max_acceleration", 0.4))
            alpha_max_list.append(cfg.get("max_angular_acceleration", 2.0) if drive == "diff-drive" else 0.0)
            radius_list.append(cfg.get("radius", 0.25))

        # start centroid and yaw from dummy poses
        start_cx, start_cy, start_yaw = compute_formation_start(
            robot_names, robot_types, dummy_start_poses
        )

        # P_star from start poses using start_yaw
        p_star = compute_p_star_from_dummy_poses(
            robot_names       = robot_names,
            start_yaw         = start_yaw,
            robot_types       = robot_types,
            dummy_start_poses = dummy_start_poses,
        )

        print("P Star: ", p_star)

        agent_obj = HeterogeneousFormationAgent(
            P_star     = p_star,
            robot_types= robot_types,
            v_max      = v_max_list,
            omega_max  = omega_max_list,
            a_max      = a_max_list,
            alpha_max  = alpha_max_list,
            sx_range   = (1.0, 2.5),
            sy_range   = (1.0, 2.5),
            radius     = radius_list,
        )

        start_state = np.array([start_cx, start_cy, start_yaw, 1.0, 1.0])
        goal_state  = np.array([val["centroid_x"], val["centroid_y"],
                                 val["formation_yaw"], 1.0, 1.0])

        first_cfg   = path_planner_cfg.get(robot_names[0], {})
        colors[key] = first_cfg.get("colour", "tab:purple")

        agents.append((key, agent_obj, start_state, goal_state, "heterogeneous-formation"))

    # ── INDIVIDUAL ROBOT (R*) ────────────────────────────────
    elif key.startswith("R"):
        rname = val["robot"]
        cfg   = path_planner_cfg.get(rname, {})
        drive = get_drive_type(rname)

        if drive == "diff-drive":
            agent_obj = DifferentialDriveAgent(
                radius    = cfg.get("radius", 0.25),
                v_max     = cfg.get("max_linear_velocity_x", 0.12),
                omega_max = cfg.get("max_angular_velocity_z", 0.4),
                a_max     = cfg.get("max_acceleration", 0.05),
                alpha_max = cfg.get("max_angular_acceleration", 0.3),
                inflation = INFLATION_RADIUS,
            )
        else:
            agent_obj = HolonomicAgent(
                radius = cfg.get("radius", 0.25),
                v_max  = cfg.get("max_linear_velocity_x", 0.25),
                a_max  = cfg.get("max_acceleration", 0.4),
            )

        raw_start   = dummy_start_poses.get(rname, (0.0, 0.0, 0.0))
        start_state = np.array([raw_start[0], raw_start[1],
                                 raw_start[2] if len(raw_start) == 3 else 0.0])
        goal_state  = np.array([val["x"], val["y"], val["yaw"]])
        colors[key] = cfg.get("colour", "tab:blue")

        agents.append((key, agent_obj, start_state, goal_state, drive))



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
#     'robot_types': hetero_form1.robot_types,
#     'v_max': hetero_form1.v_max_list,
#     'w_max': hetero_form1.omega_max_list,
#     'a_max': hetero_form1.a_max_list,
#     'alpha_max': hetero_form1.alpha_max_list,
#     'N_steer': 8,
#     'T_steer': 0.8,
# }

# warmup_time = _complete_warmup(
#     formation_agent=hetero_form1,
#     use_kinodynamic=True,
#     kinodynamic_params=warmup_kino_params
# )

first_formation = next(
    (agent_obj for label, agent_obj, start, goal, atype in agents
     if atype == "heterogeneous-formation"),
    None
)
warmup_time = 0.0
if first_formation is not None:
    warmup_kino_params = {
        'robot_types': first_formation.robot_types,
        'v_max'      : first_formation.v_max_list,
        'w_max'      : first_formation.omega_max_list,
        'a_max'      : first_formation.a_max_list,
        'alpha_max'  : first_formation.alpha_max_list,
        'N_steer'    : 8,
        'T_steer'    : 0.8,
    }
    print(f"\nRunning warmup for first formation agent")
    warmup_time = _complete_warmup(
        formation_agent    = first_formation,
        use_kinodynamic    = True,
        kinodynamic_params = warmup_kino_params,
    )
    print(f"  warmup_time = {warmup_time:.4f} s")

# ===========================================================
# Planning
# ===========================================================

print("="*70)
print("HETEROGENEOUS KINODYNAMIC RRT* (VELOCITY-LIMIT PATCH ACTIVE)")
print("="*70)
print(f"Inflation radius : {INFLATION_RADIUS}")
print(f"Warmup completed : {warmup_time:.2f}s\n")

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
        seed=411, #101 #66 #616 #51 #352
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