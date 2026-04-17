#!/usr/bin/env python3
"""
File-watching path planner  —  abort-aware, no threads.

HOW THE INTERRUPT WORKS (no threading required)
------------------------------------------------
1. Main loop detects a changed / non-empty input file.
2. File is read, then IMMEDIATELY WIPED (written to "{}").
   The wipe is the "consumed" signal — the file is now empty again.
3. run_plan() is called with:
      abort_flag  = [False]   - a 1-element list shared by reference
      check_input = <lambda>  - callable that peeks at the file
4. Inside run_plan() a small wrapper is passed down to every
   SIRRT.plan() call as its abort_flag argument.  plan() already
   polls abort_flag[0] every ABORT_CHECK_INTERVAL iterations AND
   at every greedy-extend step (see si_rrt_enhanced_individual_kinodynamic.py).
5. Separately, every time we are BETWEEN two planner calls (i.e. one
   agent finished, next hasn't started) we also poll the file.
6. When new content appears in the file while planning is in progress:
      • check_input() sets abort_flag[0] = True
      • plan() returns None on its next poll
      • run_plan() notices abort_flag[0] and returns early
      • main loop re-reads the (now non-empty) file and starts fresh

The file therefore acts as a single-slot mailbox:
   empty  ({})  →  no pending work
   non-empty    →  new job waiting
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

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
    "path_plan/path_planner/path_planner/restaurant_5.npy")

CONFIG_PATH  = os.environ.get("PLANNER_CONFIG",
    "/home/multi-robot/murogai_nav/src/MuRoGAI-Nav/"
    "chatty/config/robot_config_restaurant2.json")

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

# How often (seconds) to poll the file from INSIDE a running plan.
# Keep this at 1-2 s so we react quickly without hammering the disk.
INPLAN_POLL_INTERVAL = float(os.environ.get("PLANNER_INPLAN_POLL", "1.0"))

# planner hyper-parameters
INFLATION_RADIUS = 0.0
RES              = 0.1
TIME_HORIZON     = 120.0
MAX_ITER         = 5000
D_MAX            = 0.3
GOAL_SAMPLE_RATE = 0.40
NEIGHBOR_RADIUS  = 2.0
PRECISION        = 2
SEED             = 163

# default start poses
DEFAULT_START_POSES = {
    # "burger1" : (2.565, 0.875,  1.57),
    # "burger2" : (3.705, 0.875,  1.57),
    # "burger3" : (3.135, 1.75,   1.57),
    # "waffle"  : (5.13,  4.375,  3.14),
    # "tb4_1"   : (3.705, 7.875, -1.57),
    # "firebird": (2.565, 7.875, -1.57),
    # "go2"     : (3.135, 7.0,   -1.57),

    "delivery_bot1"  : (5.0, 7.5, 0.0),
    "delivery_bot2"  : (5.0, 4.5, 0.0),
    "delivery_bot3"  : (7.0, 6.0, 0.0),
    "cleaning_bot"   : (19.0, 15.0,  3.14),

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
HEIGHT           = H_CELLS * RES
WIDTH            = W_CELLS * RES
BOUNDS           = (np.array([0.0, 0.0]), np.array([WIDTH, HEIGHT]))
STATIC_GRID      = OccupancyGrid(_grid, RES)
print(f"Map loaded   : {W_CELLS}x{H_CELLS} cells  ({WIDTH:.1f}x{HEIGHT:.1f} m)")

print(f"Loading config: {CONFIG_PATH}")
with open(CONFIG_PATH, "r") as _f:
    _cfg             = json.load(_f)
PATH_PLANNER_CFG = _cfg.get("path_planner", {})

# ================================================================
# File helpers
# ================================================================

def _file_hash(path: str) -> str:
    """MD5 of file contents — used to detect real changes."""
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except FileNotFoundError:
        return ""


def _wipe_input_file(path: str) -> None:
    """
    Overwrite the input file with '{}' so it is considered 'empty /
    consumed'.  We use '{}' rather than '' so the file remains valid
    JSON — external writers that check the file before writing won't
    see parse errors.
    """
    try:
        with open(path, "w") as f:
            f.write("{}")
        print(f"  [watcher] Input file wiped (consumed): {path}")
    except Exception as e:
        print(f"  [watcher] WARNING — could not wipe input file: {e}")


def _file_has_new_job(path: str) -> bool:
    try:
        with open(path, "r") as f:
            raw = f.read().strip()
        if not raw:
            return False
        data = json.loads(raw)
        return (isinstance(data, dict)
                and isinstance(data.get("goal_pose"), dict)
                and len(data["goal_pose"]) > 0)
    except Exception:
        return False


def _load_input_file(path: str):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict) or len(data) == 0:
            print(f"[WARN] {path} is empty or not a JSON object — skipping")
            return None, None

        # Extract current_pose — fall back to defaults if missing
        raw_poses = data.get("current_pose", {})
        start_poses = dict(DEFAULT_START_POSES)
        for rname, p in raw_poses.items():
            start_poses[rname] = tuple(p)   # [x, y, yaw] → (x, y, yaw)

        # Extract goal_pose
        formation = data.get("goal_pose", {})
        if not formation:
            print(f"[WARN] No 'goal_pose' key found in {path} — skipping")
            return None, None

        return formation, start_poses

    except json.JSONDecodeError as e:
        print(f"[WARN] JSON parse error in {path}: {e} — skipping")
        return None, None
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}")
        return None, None
# ================================================================
# Helpers (unchanged from original)
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


# ================================================================
# Numba warmup
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




def _resolve_run_dir() -> tuple[int, str, str, str]:
    """
    ONLY READ from published_paths directory.
    DO NOT create anything inside published_paths.

    If published_paths or run dirs don't exist:
    start from run_001
    """

    published = os.path.join(
        OUTPUT_DIR,
        "published_paths"
    )

    # ---- DO NOT create published_paths ----
    if not os.path.exists(published):
        run_num = 1
    else:
        existing = [
            d for d in os.listdir(published)
            if os.path.isdir(os.path.join(published, d))
            and d.startswith("run_")
        ]

        if not existing:
            run_num = 1
        else:
            nums = []
            for d in existing:
                try:
                    nums.append(int(d[4:]))
                except ValueError:
                    pass

            run_num = (max(nums) + 1) if nums else 1

    tag = f"run_{run_num:03d}"

    # These are allowed output dirs (NOT published_paths)
    traj_dir    = os.path.join(OUTPUT_DIR, tag)
    control_dir = os.path.join(CONTROL_DIR, tag)
    image_dir   = os.path.join(IMAGE_DIR, tag)

    # Create only allowed directories
    os.makedirs(traj_dir, exist_ok=True)
    os.makedirs(control_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    return run_num, traj_dir, control_dir, image_dir

def _save_raw_map_image(image_run_dir: str, run_num: int):
    """Save the occupancy grid as a plain PNG at the start of each run."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(_grid, origin="lower", cmap="gray_r",
              extent=[0, WIDTH, 0, HEIGHT])
    ax.set_title(f"Occupancy Map — run_{run_num:03d}")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    path = os.path.join(image_run_dir, f"map_run_{run_num:03d}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved raw map : {path}")


# ================================================================
# Core planning function  —  now abort-aware
# ================================================================

def run_plan(eg_formation: dict, run_tag: str,
             abort_flag: list,
             check_input_fn,
             traj_dir: str, control_dir: str,
             image_dir: str, req_num: int,
             start_poses: dict = None):
    """
    Build agents, plan, save CSVs and PNG.

    Parameters
    ----------
    eg_formation   : formation dict loaded from the input file
    run_tag        : string tag for output filenames
    abort_flag     : [False]  — set to [True] externally to abort
    check_input_fn : callable() → bool
        Returns True if a NEW job has appeared in the input file.
        When it returns True we set abort_flag[0] = True so that
        plan() stops at its next poll point.
    """

    # ── helper: poll the file and flip abort_flag if needed ──────────
    def _poll_for_new_input(context: str = ""):
        if abort_flag[0]:
            return True                      # already aborted
        if check_input_fn():
            print(f"\n  [watcher] New input detected "
                  f"{'(' + context + ') ' if context else ''}"
                  f"— aborting current plan.")
            abort_flag[0] = True
            return True
        return False

    start_poses = start_poses or DEFAULT_START_POSES

    req_tag     = f"req_{req_num:03d}"
    # req_traj    = os.path.join(traj_dir,    req_tag);  os.makedirs(req_traj,    exist_ok=True)
    # req_control = os.path.join(control_dir, req_tag);  os.makedirs(req_control, exist_ok=True)
    # req_image   = os.path.join(image_dir,   req_tag);  os.makedirs(req_image,   exist_ok=True)
    req_traj    = os.path.join(traj_dir,    req_tag)
    req_control = os.path.join(control_dir, req_tag)
    req_image   = os.path.join(image_dir,   req_tag)

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

            raw         = start_poses.get(rname, (0.0, 0.0, 0.0))
            if drive == "diff-drive":

                start_state = np.array([raw[0], raw[1], raw[2] if len(raw) >= 3 else 0.0])
                goal_state  = np.array([val["x"], val["y"], val["yaw"]])
                agent_obj = DifferentialDriveAgent(
                    radius    = cfg.get("radius", 0.25),
                    v_max     = cfg.get("max_linear_velocity_x", 0.22),
                    omega_max = cfg.get("max_angular_velocity_z", 0.4),
                    a_max     = cfg.get("max_acceleration", 0.4),
                    alpha_max = cfg.get("max_angular_acceleration", 2.0),
                    inflation = INFLATION_RADIUS,
                )
            else:
                start_state = np.array([raw[0], raw[1]])
                goal_state  = np.array([val["x"], val["y"]])
                agent_obj = HolonomicAgent(
                    radius = cfg.get("radius", 0.25),
                    v_max  = cfg.get("max_linear_velocity_x", 0.25),
                    a_max  = cfg.get("max_acceleration", 0.4),
                )

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

    # Check abort after (potentially long) warmup
    if _poll_for_new_input("after warmup"):
        return

    # ── plan sequentially ─────────────────────────────────────────────
    dynamic_obstacles    = []
    trajectories         = {}
    agent_info           = {}
    control_trajectories = {}

    for name, agent, start, goal, agent_type, robot_names in agents:

        # ── Check for new input BEFORE starting each agent ────────────
        if _poll_for_new_input(f"before agent {name}"):
            return                           # abandon entire run

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
            debug              = True,
            use_kinodynamic    = use_kino,
            kinodynamic_params = kino_params,
        )


        t0   = time.time()

        class _ActiveAbortFlag(list):
            """
            Drop-in replacement for [False].
            When plan() reads self[0] to check for abort, we also
            poll the input file — so the file check piggybacks on
            plan()'s own polling cadence (every ABORT_CHECK_INTERVAL
            iterations) at zero extra cost.
            """
            def __init__(self, check_fn):
                super().__init__([False])
                self._check_fn      = check_fn
                self._last_poll     = time.monotonic()

            def __getitem__(self, idx):
                # Only do the file I/O every INPLAN_POLL_INTERVAL seconds
                # to avoid hammering the disk on every iteration check.
                now = time.monotonic()
                if now - self._last_poll >= INPLAN_POLL_INTERVAL:
                    self._last_poll = now
                    if not super().__getitem__(0) and self._check_fn():
                        print(f"\n  [watcher] New input detected "
                              f"inside plan() — aborting.")
                        super().__setitem__(0, True)
                return super().__getitem__(idx)

        active_flag = _ActiveAbortFlag(check_input_fn)

        traj = planner.plan(start, goal, dynamic_obstacles,
                            abort_flag=active_flag)
        dt   = time.time() - t0

        # Propagate abort state back to the shared flag
        if active_flag[0]:
            abort_flag[0] = True

        if abort_flag[0]:
            print(f"\n  [watcher] Aborting run '{run_tag}' — "
                  f"new input arrived while planning {name}.")
            return                           # discard partial results

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

    # ── final abort check before writing outputs ───────────────────────
    if _poll_for_new_input("before saving outputs"):
        return

    expected_agents = [name for name, _, _, _, _, _ in agents]
    if not trajectories:
        print("\nNo trajectories produced.")
        return

    os.makedirs(req_traj,    exist_ok=True)
    os.makedirs(req_control, exist_ok=True)
    os.makedirs(req_image,   exist_ok=True)

    # ── save CSVs ─────────────────────────────────────────────────────

    robot_traj, centroid_traj = _extract_robot_traj(trajectories, agent_info)
    _save_image(robot_traj, agent_info, trajectories, run_tag, req_num, req_image)
    if len(trajectories) < len(expected_agents):
        missing = [n for n in expected_agents if n not in trajectories]
        print(f"\nPartial failure — missing trajectories for: {missing}. Skipping save.")
        return

    _save_control_csvs(trajectories, agent_info, control_trajectories, req_control)
    _save_state_csvs(robot_traj, centroid_traj, agent_info, trajectories, req_traj)
    print(f"\nAll outputs saved under tag '{run_tag}'")


# ================================================================
# Save helpers  (unchanged from original)
# ================================================================



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


def _save_state_csvs(robot_traj, centroid_traj, agent_info, trajectories, out_dir):
    for name, paths in robot_traj.items():
        agent       = agent_info[name]['agent']
        agent_type  = agent_info[name]['type']
        robot_names = agent_info[name]['robot_names']

        if agent_type == "heterogeneous-formation":
            cfile = os.path.join(out_dir, f"{name}_centroid.csv")
            rows  = [[f"{t:.6f}",f"{xc:.6f}",f"{yc:.6f}",f"{th:.6f}",f"{sx:.6f}",f"{sy:.6f}"]
                     for (xc,yc,th,sx,sy),t in centroid_traj[name]]
            _write_csv(cfile, rows, ["time","xc","yc","theta_c","sx","sy"])
            print(f"  Saved centroid CSV: {cfile}")
            for i, path in enumerate(paths):
                rname = robot_names[i] if i < len(robot_names) else f"robot{i}"
                rfile = os.path.join(out_dir, f"{name}_{rname}.csv")
                rows  = [[f"{t:.6f}",f"{x:.6f}",f"{y:.6f}",f"{th:.6f}"]
                         for (x,y,th),t in path]
                _write_csv(rfile, rows, ["time","x","y","theta"])
                print(f"  Saved robot CSV : {rfile}")

        elif agent_type == "diff-drive":
            rname = robot_names[0] if robot_names else name
            rfile = os.path.join(out_dir, f"{name}_{rname}.csv")
            rows  = [[f"{t:.6f}",f"{x:.6f}",f"{y:.6f}",f"{th:.6f}","nan","nan"]
                     for (x,y,th),t in paths[0]]
            _write_csv(rfile, rows, ["time","x","y","theta","v","omega"])
            print(f"  Saved state CSV : {rfile}")

        elif agent_type == "holonomic":
            rname = robot_names[0] if robot_names else name
            rfile = os.path.join(out_dir, f"{name}_{rname}.csv")
            rows  = [[f"{t:.6f}",f"{x:.6f}",f"{y:.6f}","nan","nan"]
                     for (x,y,_th),t in paths[0]]
            _write_csv(rfile, rows, ["time","x","y","vx","vy"])
            print(f"  Saved state CSV : {rfile}")


def _save_control_csvs(trajectories, agent_info, control_trajectories, out_dir):
    for name in trajectories:
        agent_type   = agent_info[name]['type']
        robot_names  = agent_info[name]['robot_names']
        control_list = control_trajectories.get(name, [])
        if not control_list:
            continue

        all_x, all_y, all_t             = [], [], []
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

        if agent_type == "heterogeneous-formation":
            robots_str = "_".join(robot_names)
            fname      = f"{name}_{robots_str}_controls.csv"
        else:
            rname = robot_names[0] if robot_names else name
            fname = f"{name}_{rname}_controls.csv"

        path = os.path.join(out_dir, fname)
        n    = len(all_x)

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


def _save_image(robot_traj, agent_info, trajectories, run_tag, req_num, out_dir):
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
    ax.set_aspect("equal", "box")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Planned paths — {run_tag}  req_{req_num:03d}")
    img_path = os.path.join(out_dir, f"{run_tag}_req{req_num:03d}_paths.png")
    fig.savefig(img_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved image     : {img_path}")


# ================================================================
# File watcher  —  main loop
# ================================================================

def main():
    print("\n" + "="*60)
    print("FILE-WATCHING PATH PLANNER  (abort-aware)")
    print("="*60)
    print(f"Watching : {INPUT_FILE}")
    print(f"Poll     : every {POLL_INTERVAL}s (idle) / "
          f"every {INPLAN_POLL_INTERVAL}s (in-plan)")
    print(f"Output   : {OUTPUT_DIR} / {CONTROL_DIR} / {IMAGE_DIR}")
    print()
    print("Protocol:")
    print("  • Write a non-empty JSON object to the input file to start planning.")
    print("  • The file is wiped to '{}' immediately after reading.")
    print("  • Write a new job at any time — the current plan aborts within")
    print(f"    ~{INPLAN_POLL_INTERVAL}s and the new job starts.")
    print("  • Ctrl-C to stop.\n")

    run_num, traj_dir, control_dir, image_dir = _resolve_run_dir()
    _save_raw_map_image(image_dir, run_num)
    req_count = 0
    run_tag = f"run_{run_num:03d}"

    while True:
        if not _file_has_new_job(INPUT_FILE):
            time.sleep(POLL_INTERVAL)
            continue

        print(f"\n[{time.strftime('%H:%M:%S')}] New input detected — reading...")
        formation, start_poses = _load_input_file(INPUT_FILE)

        if formation is None:
            _wipe_input_file(INPUT_FILE)
            continue

        _wipe_input_file(INPUT_FILE)

        req_count += 1   # ← INCREMENT HERE
        print(f"Starting plan  run={run_tag}  req={req_count:03d}  agents={list(formation.keys())}")

        abort_flag = [False]

        try:
            run_plan(
                formation, run_tag,
                abort_flag     = abort_flag,
                check_input_fn = lambda: _file_has_new_job(INPUT_FILE),
                traj_dir    = traj_dir,
                control_dir = control_dir,
                image_dir   = image_dir,
                req_num     = req_count,
                start_poses = start_poses,
            )
        except Exception as e:
            import traceback
            print(f"[ERROR] Planning crashed: {e}")
            traceback.print_exc()

        if abort_flag[0]:
            print(f"\n[{time.strftime('%H:%M:%S')}] "
                  f"Plan '{run_tag}' req {req_count:03d} aborted.")
        else:
            print(f"\n[{time.strftime('%H:%M:%S')}] "
                  f"Plan '{run_tag}' req {req_count:03d} complete.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")