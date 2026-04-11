#!/usr/bin/env python3
"""
File-watching path planner.

Usage
-----
1. Run this script once:
       python3 run_planner_watchfile.py

2. Write (or overwrite) the input file at any time:
       /tmp/formation_input.json

   Example contents:
   {
     "F1": {
       "centroid_x": 3.135,
       "centroid_y": 7.583,
       "formation_yaw": 1.57,
       "desired_radius": 1.0,
       "robots": ["burger1", "burger2", "burger3"]
     },
     "R1": {
       "robot": "waffle",
       "x": 1.140,
       "y": 4.375,
       "yaw": 3.14
     }
   }

3. The planner detects the change, loads the new formation, plans, and
   saves CSVs + a PNG to the output directories.

4. Edit the file again to trigger a fresh plan.

INPUT_FILE, OUTPUT_DIR, CONTROL_DIR, MAP_PATH, CONFIG_PATH
can all be overridden via environment variables (see CONFIGURATION below).
"""

import os
import sys
import json
import math
import time
import csv
import hashlib

import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display needed — saves PNG only
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

# ── locate this file so relative imports work when run from any cwd ──
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from si_rrt_enhanced_individual_kinodynamic import (
    SIRRT, OccupancyGrid, NUMBA_AVAILABLE,
)
from agents import DifferentialDriveAgent, HolonomicAgent, HeterogeneousFormationAgent

try:
    from heterogeneous_kinodynamic_formation_steering import (
        HeterogeneousKinodynamicFormationSteering,
    )
    HETERO_AVAILABLE = True
except ImportError:
    HETERO_AVAILABLE = False
    print("Warning: heterogeneous formation steering not available")

# ================================================================
# CONFIGURATION  (override with env vars)
# ================================================================

INPUT_FILE   = os.environ.get("PLANNER_INPUT",
    "/home/multi-robot/murogai_nav/src/MuRoGAI-Nav/"
    "path_plan/path_planner/data/formation_input.json")

MAP_PATH     = os.environ.get("PLANNER_MAP",
    "/home/multi-robot/murogai_nav/src/MuRoGAI-Nav/"
    "path_plan/path_planner/path_planner/10_103_a_outside_1.npy")

CONFIG_PATH  = os.environ.get("PLANNER_CONFIG",
    "/home/multi-robot/murogai_nav/src/MuRoGAI-Nav/"
    "chatty/config/robot_config_103.json")

OUTPUT_DIR   = os.environ.get("PLANNER_OUTPUT",  
    "/home/multi-robot/murogai_nav/src/MuRoGAI-Nav/"
    "path_plan/path_planner/trajectory_logs")

CONTROL_DIR  = os.environ.get("PLANNER_CONTROL",
    "/home/multi-robot/murogai_nav/src/MuRoGAI-Nav/"
    "path_plan/path_planner/control_logs")

IMAGE_DIR    = os.environ.get("PLANNER_IMAGES",
    "/home/multi-robot/murogai_nav/src/MuRoGAI-Nav/"
    "path_plan/path_planner/path_images")

POLL_INTERVAL = float(os.environ.get("PLANNER_POLL", "1.0"))   # seconds

# planner hyper-parameters
INFLATION_RADIUS = 0.0
RES              = 0.1
TIME_HORIZON     = 120.0
MAX_ITER         = 5000
D_MAX            = 0.3
GOAL_SAMPLE_RATE = 0.40
NEIGHBOR_RADIUS  = 2.0
PRECISION        = 2
# SEED             = 138
# SEED             = 163
SEED             = 163

# default start poses (used when real odometry is unavailable)
DEFAULT_START_POSES = {
    "burger1" : (2.565, 0.875,  1.57),
    "burger2" : (3.705, 0.875,  1.57),
    "burger3" : (3.135, 1.75,   1.57),
    "waffle"  : (5.13,  4.375,  3.14),
    "tb4_1"   : (3.705, 7.875, -1.57),
    "firebird": (2.565, 7.875, -1.57),
    "go2"     : (3.135, 7.0,   -1.57),
}

# ================================================================
# One-time setup — map + config
# ================================================================

os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(CONTROL_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR,   exist_ok=True)

print(f"Loading map  : {MAP_PATH}")
_grid_raw        = np.load(MAP_PATH)
_grid            = (_grid_raw > 0).astype(np.uint8)
H_CELLS, W_CELLS = _grid.shape
HEIGHT           = H_CELLS * RES     # y-axis metres
WIDTH            = W_CELLS * RES     # x-axis metres
BOUNDS           = (np.array([0.0, 0.0]), np.array([WIDTH, HEIGHT]))
STATIC_GRID      = OccupancyGrid(_grid, RES)
print(f"Map loaded   : {W_CELLS}x{H_CELLS} cells  ({WIDTH:.1f}x{HEIGHT:.1f} m)")

print(f"Loading config: {CONFIG_PATH}")
with open(CONFIG_PATH, "r") as _f:
    _cfg             = json.load(_f)
PATH_PLANNER_CFG = _cfg.get("path_planner", {})

# ================================================================
# Helpers
# ================================================================

def _get_drive_type(robot_name: str) -> str:
    cfg = PATH_PLANNER_CFG.get(robot_name, {})
    t   = cfg.get("type", "Holonomic Drive Robot").lower()
    return "diff-drive" if "differential" in t else "holonomic"


def _resolve_vmax(agent, agent_type: str) -> float:
    if agent_type == "heterogeneous-formation" and hasattr(agent, "v_max_list"):
        return float(min(agent.v_max_list))
    raw = getattr(agent, "v_max", None)
    if raw is not None and not isinstance(raw, (list, tuple, np.ndarray)):
        return float(raw)
    return 1.2


def _estimate_centroid_and_p_star(poses, theta, sx, sy, robot_types=None):
    positions = []
    for i, pose in enumerate(poses):
        if robot_types and robot_types[i] == "diff-drive":
            x, y, _ = pose
        else:
            x, y = pose[:2]
        positions.append([x, y])
    positions = np.array(positions)
    centroid  = np.mean(positions, axis=0)
    R_inv     = np.array([[ math.cos(theta), math.sin(theta)],
                           [-math.sin(theta), math.cos(theta)]])
    S_inv     = np.diag([1.0 / sx, 1.0 / sy])
    P_star    = []
    for pos in positions:
        p = S_inv @ R_inv @ (pos - centroid)
        P_star.append((float(p[0]), float(p[1])))
    return centroid.tolist(), P_star


def _compute_formation_start(robot_names, robot_types, start_poses):
    xs, ys, yaws = [], [], []
    for i, rname in enumerate(robot_names):
        pose = start_poses.get(rname, (0.0, 0.0, 0.0))
        xs.append(pose[0]); ys.append(pose[1])
        if robot_types[i] == "diff-drive":
            yaws.append(pose[2])
    return (float(np.mean(xs)),
            float(np.mean(ys)),
            float(np.mean(yaws)) if yaws else 0.0)


def _compute_p_star(robot_names, start_yaw, robot_types, start_poses):
    poses = [start_poses.get(r, (0.0, 0.0, 0.0)) for r in robot_names]
    _, P_star = _estimate_centroid_and_p_star(
        poses, start_yaw, 1.0, 1.0, robot_types)
    return P_star


def _extend_traj(traj, T):
    if not traj:
        return traj
    q_last, t_last = traj[-1]
    if t_last >= T - 1e-9:
        return traj
    traj = list(traj)
    traj.append((np.asarray(q_last, dtype=float).copy(), float(T)))
    return traj


def _safe_float(x):
    try:
        return float("nan") if x is None else float(x)
    except Exception:
        return float("nan")


def _write_csv(path, rows, header):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def _file_hash(path: str) -> str:
    """MD5 of file contents — used to detect real changes."""
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except FileNotFoundError:
        return ""


# ================================================================
# Numba warmup (run once at startup)
# ================================================================

def _warmup(formation_agent):
    print("\n" + "="*60)
    print("WARMING UP NUMBA + CASADI")
    print("="*60)
    t0 = time.time()

    if NUMBA_AVAILABLE:
        try:
            from si_rrt_enhanced_individual_kinodynamic import (
                _nb_dist_sq_xy, _nb_disc_collides,
                _nb_compute_formation_discs, _nb_compute_robot_poses,
                _nb_max_robot_displacement, _nb_formation_nn_distance,
            )
            _nb_dist_sq_xy(0.0, 0.0, 1.0, 1.0)
            dummy = np.zeros((10, 10), dtype=np.uint8)
            _nb_disc_collides(5.0, 5.0, 0.5, dummy, 0.1, 0.0, 0.0, 10, 10)
            if hasattr(formation_agent, "P_star"):
                P  = np.array(formation_agent.P_star, dtype=np.float64)
                rr = np.array(formation_agent.radii,  dtype=np.float64)
                _nb_compute_formation_discs(P, rr, 0.0, 0.0, 0.0, 1.0, 1.0)
                _nb_compute_robot_poses(P, 0.0, 0.0, 0.0, 1.0, 1.0)
                _nb_max_robot_displacement(P, 0.0,0.0,0.0,1.0,1.0, 1.0,1.0,0.5,1.0,1.0)
                q1 = np.array([0.0,0.0,0.0,1.0,1.0], dtype=np.float64)
                q2 = np.array([1.0,1.0,0.5,1.0,1.0], dtype=np.float64)
                _nb_formation_nn_distance(
                    q1, q2, P,
                    formation_agent.Nx, formation_agent.Ny,
                    formation_agent.Nxy, formation_agent.Nr,
                    0.7, 6.0, 0.6,
                )
            print(f"  Numba OK  ({time.time()-t0:.2f}s)")
        except Exception as e:
            print(f"  Numba warning: {e}")

    if HETERO_AVAILABLE and hasattr(formation_agent, "P_star"):
        try:
            t1 = time.time()
            dummy_steerer = HeterogeneousKinodynamicFormationSteering(
                P_star      = formation_agent.P_star,
                robot_types = formation_agent.robot_types,
                v_max       = formation_agent.v_max_list,
                w_max       = formation_agent.omega_max_list,
                a_max       = formation_agent.a_max_list,
                alpha_max   = formation_agent.alpha_max_list,
                N_steer=8, T_steer=0.8, max_iter=50,
            )
            dummy_steerer.steer(
                np.array([0.0,0.0,0.0,1.0,1.0]),
                np.array([2.0,0.0,0.0,1.0,1.0]),
                np.zeros(formation_agent.Nr),
            )
            print(f"  CasADi OK ({time.time()-t1:.2f}s)")
        except Exception as e:
            print(f"  CasADi warning: {e}")

    print(f"Warmup total: {time.time()-t0:.2f}s")
    print("="*60 + "\n")


# ================================================================
# Core planning function
# ================================================================

def run_plan(eg_formation: dict, run_tag: str):
    """Build agents, plan, save CSVs and PNG."""

    start_poses = DEFAULT_START_POSES   # swap for real odometry dict here

    # ── build agent list ──────────────────────────────────────────────
    agents = []

    for key, val in eg_formation.items():
        if key.startswith("F"):
            robot_names = val["robots"]
            robot_types, v_max_list, omega_max_list = [], [], []
            a_max_list, alpha_max_list, radius_list  = [], [], []

            for rname in robot_names:
                cfg   = PATH_PLANNER_CFG.get(rname, {})
                drive = _get_drive_type(rname)
                robot_types.append(drive)
                v_max_list.append(cfg.get("max_linear_velocity_x", 0.25))
                omega_max_list.append(
                    cfg.get("max_angular_velocity_z", 0.4)
                    if drive == "diff-drive" else 0.0)
                a_max_list.append(cfg.get("max_acceleration", 0.4))
                alpha_max_list.append(
                    cfg.get("max_angular_acceleration", 2.0)
                    if drive == "diff-drive" else 0.0)
                radius_list.append(cfg.get("radius", 0.25))

            cx, cy, yaw = _compute_formation_start(
                robot_names, robot_types, start_poses)
            p_star = _compute_p_star(
                robot_names, yaw, robot_types, start_poses)

            agent_obj = HeterogeneousFormationAgent(
                P_star      = p_star,
                robot_types = robot_types,
                v_max       = v_max_list,
                omega_max   = omega_max_list,
                a_max       = a_max_list,
                alpha_max   = alpha_max_list,
                sx_range    = (0.5, 2.5),
                sy_range    = (0.5, 2.5),
                radius      = radius_list,
            )
            start_state = np.array([cx, cy, yaw, 1.0, 1.0])
            goal_state  = np.array([val["centroid_x"], val["centroid_y"],
                                    val["formation_yaw"], 1.0, 1.0])
            agents.append((key, agent_obj, start_state, goal_state,
                           "heterogeneous-formation", robot_names))

        elif key.startswith("R"):
            rname = val["robot"]
            cfg   = PATH_PLANNER_CFG.get(rname, {})
            drive = _get_drive_type(rname)

            if drive == "diff-drive":
                agent_obj = DifferentialDriveAgent(
                    radius    = cfg.get("radius", 0.25),
                    v_max     = cfg.get("max_linear_velocity_x", 0.22),
                    omega_max = cfg.get("max_angular_velocity_z", 0.4),
                    a_max     = cfg.get("max_acceleration", 0.4),
                    alpha_max = cfg.get("max_angular_acceleration", 2.0),
                    inflation = INFLATION_RADIUS,
                )
            else:
                agent_obj = HolonomicAgent(
                    radius = cfg.get("radius", 0.25),
                    v_max  = cfg.get("max_linear_velocity_x", 0.25),
                    a_max  = cfg.get("max_acceleration", 0.4),
                )

            raw   = start_poses.get(rname, (0.0, 0.0, 0.0))
            start_state = np.array([raw[0], raw[1],
                                    raw[2] if len(raw) >= 3 else 0.0])
            goal_state  = np.array([val["x"], val["y"], val["yaw"]])
            agents.append((key, agent_obj, start_state, goal_state,
                           drive, [rname]))
        else:
            print(f"  Skipping unknown key '{key}'")

    if not agents:
        print("No valid agents found in input — nothing to plan.")
        return

    # ── warmup on first formation found ───────────────────────────────
    first_formation = next(
        (a for _, a, _s, _g, atype, _rn in agents
         if atype == "heterogeneous-formation"), None)
    if first_formation is not None:
        _warmup(first_formation)

    # ── plan sequentially ─────────────────────────────────────────────
    dynamic_obstacles    = []
    trajectories         = {}
    agent_info           = {}
    control_trajectories = {}

    for name, agent, start, goal, agent_type, robot_names in agents:
        print(f"\n{'='*50}")
        print(f"Planning {name} ({agent_type})")
        print(f"  start : {np.round(start[:3], 3)}")
        print(f"  goal  : {np.round(goal[:3],  3)}")

        agent_vmax = _resolve_vmax(agent, agent_type)
        print(f"  vmax  : {agent_vmax:.4f} m/s")

        # collision check
        ok_start = ok_goal = True
        for i, (p, r) in enumerate(agent.discs(start)):
            if STATIC_GRID.disc_collides(p[0], p[1], r):
                print(f"  COLLISION at start robot {i}  pos=({p[0]:.2f},{p[1]:.2f})")
                ok_start = False
        for i, (p, r) in enumerate(agent.discs(goal)):
            if STATIC_GRID.disc_collides(p[0], p[1], r):
                print(f"  COLLISION at goal  robot {i}  pos=({p[0]:.2f},{p[1]:.2f})")
                ok_goal = False
        if not ok_start:
            print("  Skipping — start in collision"); continue
        if not ok_goal:
            print("  Skipping — goal in collision");  continue

        # kino params
        if agent_type == "heterogeneous-formation" and HETERO_AVAILABLE:
            kino_params = {
                'robot_types': agent.robot_types,
                'v_max':       agent.v_max_list,
                'w_max':       agent.omega_max_list,
                'a_max':       agent.a_max_list,
                'alpha_max':   agent.alpha_max_list,
                'N_steer': 8, 'T_steer': 0.8, 'max_iter': 200,
            }
            use_kino = True
        elif agent_type == "diff-drive":
            kino_params = {
                'v_max': agent.v_max, 'omega_max': agent.omega_max,
                'a_max': agent.a_max, 'alpha_max': agent.alpha_max,
                'dt': 0.05,
            }
            use_kino = True
        elif agent_type == "holonomic":
            kino_params = {'v_max': agent.v_max, 'a_max': agent.a_max, 'dt': 0.05}
            use_kino = True
        else:
            kino_params = None; use_kino = False

        planner = SIRRT(
            agent_model        = agent,
            max_velocity       = agent_vmax,
            workspace_bounds   = BOUNDS,
            static_grid        = STATIC_GRID,
            time_horizon       = TIME_HORIZON,
            max_iter           = MAX_ITER,
            d_max              = D_MAX,
            goal_sample_rate   = GOAL_SAMPLE_RATE,
            neighbor_radius    = NEIGHBOR_RADIUS,
            precision          = PRECISION,
            seed               = SEED,
            debug              = False,
            use_kinodynamic    = use_kino,
            kinodynamic_params = kino_params,
        )

        t0   = time.time()
        traj = planner.plan(start, goal, dynamic_obstacles)
        dt   = time.time() - t0

        if traj is None:
            print(f"  FAILED after {dt:.1f}s  (tree={len(planner.V)})")
            continue

        print(f"  SUCCESS in {dt:.1f}s  waypoints={len(traj)}"
              f"  final_t={traj[-1][1]:.1f}s  tree={len(planner.V)}")

        trajectories[name] = traj
        control_trajectories[name] = [
            item[3] for item in traj
            if len(item) >= 4 and item[3] is not None
        ]
        traj_obs = _extend_traj([(item[0], item[1]) for item in traj], 60.0)
        dynamic_obstacles.append({"trajectory": traj_obs, "agent": agent})
        agent_info[name] = {
            'agent': agent, 'type': agent_type,
            'robot_names': robot_names,
        }

    if not trajectories:
        print("\nNo trajectories produced.")
        return

    # ── save CSVs ─────────────────────────────────────────────────────
    _save_control_csvs(trajectories, agent_info, control_trajectories, run_tag)
    robot_traj, centroid_traj = _extract_robot_traj(trajectories, agent_info)
    _save_state_csvs(robot_traj, centroid_traj, agent_info, trajectories, run_tag)
    _save_image(robot_traj, agent_info, trajectories, run_tag)
    print(f"\nAll outputs saved under tag '{run_tag}'")


# ================================================================
# Save helpers
# ================================================================

def _save_control_csvs(trajectories, agent_info, control_trajectories, tag):
    for name in trajectories:
        agent_type   = agent_info[name]['type']
        control_list = control_trajectories.get(name, [])
        if not control_list:
            continue

        all_x, all_y, all_t         = [], [], []
        all_theta, all_v, all_w, all_vy = [], [], [], []
        cum_t = 0.0
        for seg in control_list:
            if seg is None: continue
            t_arr = seg.t + cum_t if seg.t is not None else None
            if t_arr is not None and len(t_arr): cum_t = t_arr[-1]
            all_x.extend(seg.x.tolist()); all_y.extend(seg.y.tolist())
            if t_arr is not None: all_t.extend(t_arr.tolist())
            if getattr(seg, "theta", None) is not None: all_theta.extend(seg.theta.tolist())
            if getattr(seg, "v",     None) is not None: all_v.extend(seg.v.tolist())
            if getattr(seg, "omega", None) is not None: all_w.extend(seg.omega.tolist())
            if getattr(seg, "vy",    None) is not None: all_vy.extend(seg.vy.tolist())

        path = os.path.join(CONTROL_DIR, f"{tag}_{name}_controls.csv")
        n = len(all_x)
        if agent_type == "diff-drive":
            rows = [[f"{all_t[i] if i<len(all_t) else 0:.4f}",
                     f"{all_x[i]:.4f}", f"{all_y[i]:.4f}",
                     f"{all_theta[i] if i<len(all_theta) else 0:.4f}",
                     f"{all_v[i] if i<len(all_v) else 0:.4f}",
                     f"{all_w[i] if i<len(all_w) else 0:.4f}"] for i in range(n)]
            _write_csv(path, rows, ['time','x','y','theta','v','omega'])
        elif agent_type == "holonomic":
            rows = [[f"{all_t[i] if i<len(all_t) else 0:.4f}",
                     f"{all_x[i]:.4f}", f"{all_y[i]:.4f}",
                     f"{all_v[i] if i<len(all_v) else 0:.4f}",
                     f"{all_vy[i] if i<len(all_vy) else 0:.4f}"] for i in range(n)]
            _write_csv(path, rows, ['time','x','y','vx','vy'])
        else:
            rows = [[f"{all_t[i] if i<len(all_t) else 0:.4f}",
                     f"{all_x[i]:.4f}", f"{all_y[i]:.4f}",
                     f"{all_theta[i] if i<len(all_theta) else 0:.4f}",
                     f"{all_v[i] if i<len(all_v) else 0:.4f}",
                     f"{all_w[i] if i<len(all_w) else 0:.4f}"] for i in range(n)]
            _write_csv(path, rows, ['time','x','y','theta','u1','u2'])
        print(f"  Saved control CSV: {path}")


def _extract_robot_traj(trajectories, agent_info):
    robot_traj    = {}
    centroid_traj = {}
    for name, traj in trajectories.items():
        agent      = agent_info[name]['agent']
        agent_type = agent_info[name]['type']
        if agent_type == "heterogeneous-formation":
            Nr    = len(agent.P_star)
            paths = [[] for _ in range(Nr)]
            cpath = []
            for item in traj:
                q   = np.asarray(item[0], dtype=float).flatten()
                t   = float(item[1])
                psi = item[2] if len(item) >= 3 else None
                xc,yc,th,sx,sy = q
                cpath.append(((xc,yc,th,sx,sy), t))
                for i, (x,y,theta) in enumerate(agent.robot_poses(q)):
                    th_use = (float(psi[i])
                              if psi is not None and i < len(psi) and psi[i] is not None
                              else float(theta))
                    paths[i].append(((float(x),float(y),th_use), t))
            robot_traj[name]    = paths
            centroid_traj[name] = cpath
        else:
            path = []
            for item in traj:
                q = item[0]; t = float(item[1])
                x,y,theta = agent.robot_poses(q)[0]
                path.append(((float(x),float(y),float(theta)), t))
            robot_traj[name] = [path]
    return robot_traj, centroid_traj


def _save_state_csvs(robot_traj, centroid_traj, agent_info, trajectories, tag):
    for name, paths in robot_traj.items():
        agent      = agent_info[name]['agent']
        agent_type = agent_info[name]['type']
        if agent_type == "heterogeneous-formation":
            cfile = os.path.join(OUTPUT_DIR, f"{tag}_{name}_centroid.csv")
            rows  = [[f"{t:.6f}",f"{xc:.6f}",f"{yc:.6f}",f"{th:.6f}",f"{sx:.6f}",f"{sy:.6f}"]
                     for (xc,yc,th,sx,sy),t in centroid_traj[name]]
            _write_csv(cfile, rows, ["time","xc","yc","theta_c","sx","sy"])
            print(f"  Saved centroid CSV: {cfile}")
            for i, path in enumerate(paths):
                rtype = agent.robot_types[i]
                rfile = os.path.join(OUTPUT_DIR, f"{tag}_{name}_robot{i}_{rtype}.csv")
                rows  = [[f"{t:.6f}",f"{x:.6f}",f"{y:.6f}",f"{th:.6f}"]
                         for (x,y,th),t in path]
                _write_csv(rfile, rows, ["time","x","y","theta"])
                print(f"  Saved robot CSV : {rfile}")
        elif agent_type == "diff-drive":
            rfile = os.path.join(OUTPUT_DIR, f"{tag}_{name}.csv")
            rows  = [[f"{t:.6f}",f"{x:.6f}",f"{y:.6f}",f"{th:.6f}","nan","nan"]
                     for (x,y,th),t in paths[0]]
            _write_csv(rfile, rows, ["time","x","y","theta","v","omega"])
            print(f"  Saved state CSV : {rfile}")
        elif agent_type == "holonomic":
            rfile = os.path.join(OUTPUT_DIR, f"{tag}_{name}.csv")
            rows  = [[f"{t:.6f}",f"{x:.6f}",f"{y:.6f}","nan","nan"]
                     for (x,y,_th),t in paths[0]]
            _write_csv(rfile, rows, ["time","x","y","vx","vy"])
            print(f"  Saved state CSV : {rfile}")


def _save_image(robot_traj, agent_info, trajectories, tag):
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.imshow(_grid[::-1], cmap="gray_r",
              extent=[0, WIDTH, 0, HEIGHT], alpha=0.85)

    colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple",
              "tab:brown","tab:pink","tab:cyan"]
    for ci, (name, paths) in enumerate(robot_traj.items()):
        col        = colors[ci % len(colors)]
        agent_type = agent_info[name]['type']
        if agent_type == "heterogeneous-formation":
            for path in paths:
                pts = np.array([p[:2] for p,_ in path])
                ax.plot(pts[:,0], pts[:,1], lw=1.5, color=col, alpha=0.5)
            c_pts = np.array([item[0][:2] for item in trajectories[name]])
            ax.plot(c_pts[:,0], c_pts[:,1], lw=3, color=col, alpha=0.9, label=name)
        else:
            pts = np.array([p[:2] for p,_ in paths[0]])
            ax.plot(pts[:,0], pts[:,1], lw=2.5, color=col, alpha=0.8, label=name)

    ax.set_xlim(0, WIDTH); ax.set_ylim(0, HEIGHT)
    ax.set_aspect("equal","box")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Planned paths — {tag}")
    img_path = os.path.join(IMAGE_DIR, f"{tag}_paths.png")
    fig.savefig(img_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved image     : {img_path}")


# ================================================================
# File watcher — main loop
# ================================================================

def _load_input_file(path: str):
    """Load and validate the JSON input file. Returns dict or None."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print(f"[WARN] {path} does not contain a JSON object — skipping")
            return None
        return data
    except json.JSONDecodeError as e:
        print(f"[WARN] JSON parse error in {path}: {e} — skipping")
        return None
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}")
        return None


def main():
    print("\n" + "="*60)
    print("FILE-WATCHING PATH PLANNER")
    print("="*60)
    print(f"Watching : {INPUT_FILE}")
    print(f"Poll     : every {POLL_INTERVAL}s")
    print(f"Output   : {OUTPUT_DIR} / {CONTROL_DIR} / {IMAGE_DIR}")
    print("Write or overwrite the input file to trigger planning.")
    print("Ctrl-C to stop.\n")

    last_hash  = ""
    run_count  = 0

    while True:
        current_hash = _file_hash(INPUT_FILE)

        if current_hash and current_hash != last_hash:
            print(f"\n[{time.strftime('%H:%M:%S')}] Input file changed — loading...")
            formation = _load_input_file(INPUT_FILE)

            if formation is not None:
                run_count += 1
                tag = f"run{run_count:03d}_{time.strftime('%H%M%S')}"
                print(f"Starting plan  tag={tag}  agents={list(formation.keys())}")
                try:
                    run_plan(formation, tag)
                except Exception as e:
                    import traceback
                    print(f"[ERROR] Planning crashed: {e}")
                    traceback.print_exc()

            last_hash = current_hash

        elif not current_hash and last_hash:
            print(f"[{time.strftime('%H:%M:%S')}] Input file removed — waiting...")
            last_hash = ""

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")