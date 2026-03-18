import os
import json
import numpy as np
import math

from si_rrt_enhanced_individual_kinodynamic import (
    SIRRT,
    OccupancyGrid,
    NUMBA_AVAILABLE,
)
from agents import DifferentialDriveAgent, HolonomicAgent, HeterogeneousFormationAgent

# ===========================================================
# GLOBAL
# ===========================================================
INFLATION_RADIUS = 0.0

OUTPUT_DIR = "trajectory_logs"
CONTROL_DIR = "control_logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CONTROL_DIR, exist_ok=True)

# ===========================================================
# INPUT (UNCHANGED)
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

map_path = "/home/suraj/murogai_nav/src/MuRoGAI-Nav/path_plan/path_planner/path_planner/10_103_a_outside_1.npy"
config_file_path = "/home/suraj/murogai_nav/src/MuRoGAI-Nav/chatty/config/robot_config_restaurant.json"

res = 0.1

# ===========================================================
# LOAD CONFIG & MAP
# ===========================================================
config_data = json.load(open(config_file_path, 'r'))

grid = np.load(map_path)
height, width = grid.shape
height, width = height * res, width * res

bounds = (np.array([0.0, 0.0]), np.array([height, width]))
static_grid = OccupancyGrid(grid, res)

# ===========================================================
# HELPERS
# ===========================================================
def get_robot_type(cfg):
    t = cfg["type"].lower()
    if "differential" in t:
        return "diff-drive"
    elif "holonomic" in t:
        return "holonomic"
    else:
        return None

# ===========================================================
# CREATE SINGLE ROBOT AGENT
# ===========================================================
def create_single_agent(name, val):
    if name not in config_data["path_planner"]:
        print(f"[WARN] {name} missing")
        return None

    cfg = config_data["path_planner"][name]
    r_type = get_robot_type(cfg)

    if r_type == "diff-drive":
        agent = DifferentialDriveAgent(
            radius=cfg["radius"],
            v_max=cfg["max_linear_velocity_x"],
            omega_max=cfg["max_angular_velocity_z"],
            a_max=cfg["max_acceleration"],
            alpha_max=cfg["max_angular_acceleration"],
            inflation=INFLATION_RADIUS,
        )
        mode = "diff-drive"

    elif r_type == "holonomic":
        agent = HolonomicAgent(
            radius=cfg["radius"],
            v_max=max(cfg["max_linear_velocity_x"], cfg["max_linear_velocity_y"]),
            a_max=cfg["max_acceleration"],
        )
        mode = "holonomic"

    else:
        return None

    start = np.array([val["x"], val["y"], val["yaw"]])
    goal = start.copy()

    return (name, agent, start, goal, mode)

# ===========================================================
# CREATE FORMATION AGENT
# ===========================================================
def create_formation_agent(key, val):

    robot_list = val["robots"]
    theta = val["formation_yaw"]
    sx = val["desired_radius"]
    sy = val["desired_radius"]

    robot_types = []
    v_max = []
    omega_max = []
    a_max = []
    alpha_max = []
    radii = []

    valid_robots = []

    for r in robot_list:
        if r not in config_data["path_planner"]:
            print(f"[WARN] {r} missing")
            continue

        cfg = config_data["path_planner"][r]
        r_type = get_robot_type(cfg)

        robot_types.append(r_type)
        v_max.append(cfg["max_linear_velocity_x"])
        a_max.append(cfg["max_acceleration"])
        radii.append(cfg["radius"])
        valid_robots.append(r)

        if r_type == "diff-drive":
            omega_max.append(cfg["max_angular_velocity_z"])
            alpha_max.append(cfg["max_angular_acceleration"])
        else:
            omega_max.append(0.0)
            alpha_max.append(0.0)

    N = len(valid_robots)
    if N == 0:
        return None

    # 🔥 Generate P_star as regular polygon
    P_star = []
    for i in range(N):
        angle = 2 * math.pi * i / N
        P_star.append([math.cos(angle), math.sin(angle)])

    agent = HeterogeneousFormationAgent(
        P_star=P_star,
        robot_types=robot_types,
        v_max=v_max,
        omega_max=omega_max,
        a_max=a_max,
        alpha_max=alpha_max,
        sx_range=(sx, sx),
        sy_range=(sy, sy),
        radius=radii,
    )

    start = np.array([val["centroid_x"], val["centroid_y"], theta, 0.0, 0.0])
    goal = np.array([val["centroid_x"], val["centroid_y"], theta, sx, sy])

    return (key, agent, start, goal, "heterogeneous-formation")

# ===========================================================
# BUILD AGENTS
# ===========================================================
agents = []

for key, val in eg_formation.items():
    if key.startswith("F"):
        a = create_formation_agent(key, val)
        if a:
            agents.append(a)
    elif key.startswith("R"):
        a = create_single_agent(val["robot"], val)
        if a:
            agents.append(a)

# ===========================================================
# PRINT EACH AGENT INDIVIDUALLY
# ===========================================================
print("\n===== INDIVIDUAL AGENTS =====\n")

for name, agent, start, goal, mode in agents:

    if mode == "diff-drive":
        print(f"# {name} = DifferentialDriveAgent(")
        print(f"#     radius={agent.radius}, v_max={agent.v_max}, omega_max={agent.omega_max},")
        print(f"#     a_max={agent.a_max}, alpha_max={agent.alpha_max}, inflation={INFLATION_RADIUS},")
        print(f"# )\n")

    elif mode == "holonomic":
        print(f"# {name} = HolonomicAgent(")
        print(f"#     radius={agent.radius}, v_max={agent.v_max}, a_max={agent.a_max},")
        print(f"# )\n")

    elif mode == "heterogeneous-formation":
        print(f"# {name} = HeterogeneousFormationAgent(")
        print(f"#     P_star={agent.P_star},")
        print(f"#     robot_types={agent.robot_types},")
        print(f"#     v_max={agent.v_max_list}, omega_max={agent.omega_max_list},")
        print(f"#     a_max={agent.a_max_list}, alpha_max={agent.alpha_max_list},")
        print(f"#     sx_range={agent.sx_range}, sy_range={agent.sy_range},")
        print(f"#     radius={agent.radius},")
        print(f"# )\n")

# ===========================================================
# PRINT FULL AGENTS LIST LIKE COPY-PASTE BLOCK
# ===========================================================
print("\nagents = [\n")

for name, agent, start, goal, mode in agents:
    if mode == "diff-drive" or mode == "holonomic":
        print(f"    ('{name}', {name},")
        print(f"     np.array([{start[0]:.3f}, {start[1]:.3f}, {start[2]:.3f}]),")
        print(f"     np.array([{goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f}]),")
        print(f"     '{mode}'),\n")
    elif mode == "heterogeneous-formation":
        print(f"    ('{name}', {name},")
        print(f"     np.array([{start[0]:.3f}, {start[1]:.3f}, {start[2]:.3f}, {start[3]:.1f}, {start[4]:.1f}]),")
        print(f"     np.array([{goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f}, {goal[3]:.1f}, {goal[4]:.1f}]),")
        print(f"     '{mode}'),\n")

print("]\n")