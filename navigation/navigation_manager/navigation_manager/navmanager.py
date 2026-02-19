#!/usr/bin/env python3
import json
import math
import base64
import numpy as np
from openai import OpenAI

# ============================================================
# USER COMMAND
# ============================================================

user_command = "burger robots gather around the round table to discuss strategy."

STATE_PATH = "state.json"
MAP_JSON_PATH = "map_v2_full_without_robot.json"
MAP_PNG_PATH = "map_v2_full_without_robot.png"

RES = 0.1
ROBOT_RADIUS = 0.35
CLEARANCE = 0.12
MAX_REPAIR = 4

MODEL = "gpt-4.1"

client = OpenAI()


# ============================================================
# Load files
# ============================================================

def load_json(p):
    with open(p) as f:
        return json.load(f)

state = load_json(STATE_PATH)
map_json = load_json(MAP_JSON_PATH)


# ============================================================
# PNG to vision input
# ============================================================

def png_to_dataurl(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{b64}"


# ============================================================
# Build occupancy grid from map.json
# ============================================================

W = int(map_json["map_info"]["width"] / RES)
H = int(map_json["map_info"]["height"] / RES)

grid = np.zeros((H, W), dtype=np.uint8)

def draw_rect(c, w, h):
    cx, cy = c
    for i in range(H):
        y = (i + 0.5) * RES
        if abs(y - cy) > h/2:
            continue
        for j in range(W):
            x = (j + 0.5) * RES
            if abs(x - cx) <= w/2:
                grid[i, j] = 1

def draw_circle(c, r):
    cx, cy = c
    for i in range(H):
        y = (i + 0.5) * RES
        for j in range(W):
            x = (j + 0.5) * RES
            if (x-cx)**2 + (y-cy)**2 <= r*r:
                grid[i, j] = 1

for obj in map_json["objects"]:
    s = obj["shape"]
    if s["type"] == "rectangle":
        draw_rect(s["center"], s["width"], s["height"])
    elif s["type"] == "circle":
        draw_circle(s["center"], s["radius"])


# ============================================================
# Disc collision checking (planner style)
# ============================================================

def world_to_grid(x, y):
    return int(y/RES), int(x/RES)

def disc_cells(center, r):
    cx, cy = center
    i0, j0 = world_to_grid(cx-r, cy-r)
    i1, j1 = world_to_grid(cx+r, cy+r)

    i0 = max(0, i0); j0 = max(0, j0)
    i1 = min(H-1, i1); j1 = min(W-1, j1)

    cells = []
    for i in range(i0, i1+1):
        y = (i + 0.5) * RES
        for j in range(j0, j1+1):
            x = (j + 0.5) * RES
            if (x-cx)**2 + (y-cy)**2 <= r*r:
                cells.append((i,j))
    return cells

def safe_pose(p):
    for i,j in disc_cells(p, ROBOT_RADIUS + CLEARANCE):
        if grid[i,j] == 1:
            return False
    return True


# ============================================================
# LLM SYSTEM PROMPT (GENERIC)
# ============================================================

SYSTEM_PROMPT = """
You are a Navigation Manager for a multi-robot system.

You receive:
- user_command (task)
- state (robot poses and formations)
- map_json (semantic objects)
- map_png (visual layout of the map)

Your job: assign SAFE goal poses.

CRITICAL:
Never place goals on or inside obstacles.
Maintain safe standoff distance.

Method:

Robot:
 - interpret task
 - locate relevant object
 - choose approach pose in free space
 - output x,y,yaw

Formation:
 - output centroid_x, centroid_y, centroid_yaw
 - output desired_radius
 - compute scale sx,sy

If surround task:
 - centroid may be at object
 - members must be collision free

If assembly:
 - only assign those robots

Output ONLY valid JSON:

{
 "robots": { "name": {"x":..,"y":..,"yaw":..,"reason":""} },
 "formations": {
   "name":{
     "centroid_x":..,
     "centroid_y":..,
     "centroid_yaw":..,
     "desired_radius":..,
     "scale":{"sx":..,"sy":..},
     "reason":""
   }
 },
 "notes":""
}
"""


# ============================================================
# Call LLM (vision + text)
# ============================================================

def call_llm(payload):
    img = png_to_dataurl(MAP_PNG_PATH)

    r = client.responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=[{
            "role":"user",
            "content":[
                {"type":"input_text","text":json.dumps(payload)},
                {"type":"input_image","image_url":img}
            ]
        }]
    )

    text = r.output_text.strip()

    if text.startswith("```"):
        text = text.split("```")[1]

    return json.loads(text)


# ============================================================
# Validate goals
# ============================================================

def validate(goals):
    errors = {}

    for r,g in goals["robots"].items():
        if not safe_pose((g["x"], g["y"])):
            errors[r] = "robot collides with obstacle"

    for f,g in goals["formations"].items():
        cx, cy = g["centroid_x"], g["centroid_y"]
        R = g["desired_radius"] + ROBOT_RADIUS + CLEARANCE
        for i,j in disc_cells((cx,cy), R):
            if grid[i,j] == 1:
                errors[f] = "formation intersects obstacle"
                break

    return errors


# ============================================================
# Closed loop Navigation Manager
# ============================================================

def navigation_manager():

    if not user_command.strip():
        return {"robots":{}, "formations":{}, "notes":"no task"}

    payload = {
        "user_command": user_command,
        "state": state,
        "map_json": map_json
    }

    goals = call_llm(payload)

    for k in range(MAX_REPAIR):

        errors = validate(goals)

        if not errors:
            goals["notes"] = f"validated after {k} repair rounds"
            return goals

        print("Repairing:", errors)

        repair_payload = {
            "invalid": errors,
            "current_goals": goals,
            "user_command": user_command,
            "state": state,
            "map_json": map_json,
            "instruction": "Fix only invalid goals and move to nearest free space."
        }

        goals = call_llm(repair_payload)

    raise RuntimeError("Could not produce safe goals")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    goals = navigation_manager()

    print("\nUSER COMMAND:")
    print(user_command)

    print("\nFINAL SAFE GOALS:")
    print(json.dumps(goals, indent=2))
