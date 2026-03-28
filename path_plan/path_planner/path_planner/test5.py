import numpy as np
import json
import os
import math

from si_rrt_enhanced_individual_kinodynamic import SIRRT, OccupancyGrid, NUMBA_AVAILABLE
from agents import DifferentialDriveAgent, HolonomicAgent, HeterogeneousFormationAgent

# ===========================================================
# GLOBAL SETTINGS
# ===========================================================
INFLATION_RADIUS = 0.0
OUTPUT_DIR = "trajectory_logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CONTROL_DIR = "control_logs"
os.makedirs(CONTROL_DIR, exist_ok=True)

map_path = "/home/multi-robot/murogai_nav/src/MuRoGAI-Nav/path_plan/path_planner/path_planner/restaurant_5.npy"
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
config_file_path = "/home/multi-robot/murogai_nav/src/MuRoGAI-Nav/chatty/config/robot_config_restaurant.json"
config_data = json.load(open(config_file_path, 'r'))
path_planner_cfg = config_data["path_planner"]

# ===========================================================
# DUMMY START POSES — replace with real odometry
# ===========================================================
dummy_start_poses = {
    "robot1":  (3.705, 7.875, -1.57),
    "robot2":  (2.565, 7.875, -1.57),
    "robot3":  (2.0,   1.0,    0.0),
    "robot4":  (2.5,   1.0,    0.0),
    "robot5":  (3.0,   1.0,    0.0),
    "robot6":  (3.5,   1.0,    0.0),
    "robot7":  (4.0,   1.0,    0.0),
    "robot8":  (4.5,   1.0,    0.0),
    "robot9":  (3.135, 7.000),           # holonomic — no yaw
    "robot10": (5.5,   1.0,    0.0),
    "robot11": (6.0,   1.0,    0.0),
}

# ===========================================================
# INPUT — goal poses
# ===========================================================
eg_formation = {
    "F1": {
        "centroid_x": 3.5,
        "centroid_y": 6.3,
        "formation_yaw": 1.2,
        "desired_radius": 1.0,
        "robots": ["robot1", "robot2", "robot9"]
    },
    "R1": {
        "robot": "robot7", "x": 3.5, "y": 3.2, "yaw": 2.6
    },
    "R2": {
        "robot": "robot10", "x": 9.4, "y": 7.3, "yaw": 2.6
    },
    "F2": {
        "centroid_x": 4.2,
        "centroid_y": 5.0,
        "formation_yaw": 1.5,
        "desired_radius": 1.5,
        "robots": ["robot5", "robot4", "robot6"]
    },
    "R3": {
        "robot": "robot15", "x": 6.4, "y": 2.5, "yaw": 0.0
    },
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


# ===========================================================
# PRINT SUMMARY
# ===========================================================
print("\n" + "="*70)
print(f"{'LABEL':<10} {'TYPE':<28} {'COLOUR':<14} {'START':<30} GOAL")
print("="*70)
for label, agent_obj, start, goal, atype in agents:
    print(f"{label:<10} {atype:<28} {colors[label]:<14} "
          f"{str(np.round(start, 3)):<30} {np.round(goal, 3)}")

print("\n--- Detailed agent attributes ---")
for label, agent_obj, start, goal, atype in agents:
    print(f"\n[{label}]  type={atype}  colour={colors[label]}")
    if atype == "heterogeneous-formation":
        print(f"  robot_types  : {agent_obj.robot_types}")
        print(f"  v_max_list   : {agent_obj.v_max_list}")
        print(f"  omega_max    : {agent_obj.omega_max_list}")
        print(f"  a_max        : {agent_obj.a_max_list}")
        print(f"  alpha_max    : {agent_obj.alpha_max_list}")
        print(f"  P_star       : {agent_obj.P_star}")
        print(f"  start        : {np.round(start, 4)}")
        print(f"  goal         : {np.round(goal, 4)}")
    else:
        print(f"  radius       : {agent_obj.radius}")
        print(f"  v_max        : {agent_obj.v_max}")
        print(f"  a_max        : {agent_obj.a_max}")
        print(f"  start        : {np.round(start, 4)}")
        print(f"  goal         : {np.round(goal, 4)}")

first_formation = next(
    (agent_obj for label, agent_obj, start, goal, atype in agents
     if atype == "heterogeneous-formation"),
    None
)

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