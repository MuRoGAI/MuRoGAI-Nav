#!/usr/bin/env python3
"""
WORLD-FRAME trajectory replay with correct robot radii
======================================================

✔ Uses ONLY per-robot CSVs (world frame)
✔ No origin shifting
✔ No axis swapping
✔ Correct physical + inflated radii
✔ Slider-based time replay
✔ Diff-drive orientation arrows

This script is CONSISTENT with the provided planning + logging code.

Author: ChatGPT (for Suraj)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.widgets import Slider
from pathlib import Path

# =========================================================
# USER CONFIG
# =========================================================

TRAJ_DIR = "trajectory_logs"
DRAW_INFLATED = True

# MUST match planner values
INFLATION_RADIUS = 0.1

# radius map MUST match agents in planner
RADIUS_MAP = {
    "DD1": 0.40,
    "DD2": 0.35,
    "Holo1": 0.40,
    "Holo2": 0.30,

    # formation robots (match naming!)
    "HeteroForm_robot0_diff-drive": 0.35,
    "HeteroForm_robot1_holonomic": 0.30,
    "HeteroForm_robot2_diff-drive": 0.35,

    "HeteroForm2_robot0_diff-drive": 0.40,
    "HeteroForm2_robot1_holonomic": 0.35,
    "HeteroForm2_robot2_diff-drive": 0.40,
    "HeteroForm2_robot3_holonomic": 0.30,

    "HeteroForm3_robot0_diff-drive": 0.30,
    "HeteroForm3_robot1_holonomic": 0.25,
    "HeteroForm3_robot2_diff-drive": 0.30,
}

# =========================================================
# Load trajectories (WORLD FRAME ONLY)
# =========================================================

robots = []

traj_dir = Path(TRAJ_DIR)
assert traj_dir.exists()

for csv in sorted(traj_dir.glob("*.csv")):
    name = csv.stem

    # skip centroids
    if "centroid" in name:
        continue

    df = pd.read_csv(csv)

    if not {"time", "x", "y"}.issubset(df.columns):
        continue

    T = df["time"].values
    X = df["x"].values
    Y = df["y"].values
    TH = df["theta"].values if "theta" in df.columns else None

    robots.append(dict(
        name=name,
        T=T,
        X=X,
        Y=Y,
        TH=TH,
        is_diff=("theta" in df.columns)
    ))

print(f"Loaded {len(robots)} robot trajectories")

# =========================================================
# Visualization
# =========================================================

fig, ax = plt.subplots(figsize=(14, 10))
plt.subplots_adjust(bottom=0.15)

# static paths
for r in robots:
    ax.plot(r["X"], r["Y"], lw=2, alpha=0.6)

# robot artists
artists = []

def make_robot(x, y, th, r, is_dd):
    circ = Circle((x, y), r, fill=False, lw=2)
    ax.add_patch(circ)

    arrow = None
    if is_dd:
        L = r * 1.8
        arrow = FancyArrowPatch(
            (x, y),
            (x + L*np.cos(th), y + L*np.sin(th)),
            arrowstyle="->",
            lw=2
        )
        ax.add_patch(arrow)

    return circ, arrow

for r in robots:
    base_r = RADIUS_MAP.get(r["name"], 0.3)
    rad = base_r + INFLATION_RADIUS if DRAW_INFLATED else base_r
    circ, arr = make_robot(0, 0, 0, rad, r["is_diff"])
    artists.append((r, circ, arr))

# bounds
all_x = np.concatenate([r["X"] for r in robots])
all_y = np.concatenate([r["Y"] for r in robots])
ax.set_xlim(all_x.min() - 1, all_x.max() + 1)
ax.set_ylim(all_y.min() - 1, all_y.max() + 1)
ax.set_aspect("equal", "box")

# time slider
Tmax = max(r["T"][-1] for r in robots)
ax_time = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_time, "Time", 0.0, Tmax, valinit=0.0)

def interp(r, t):
    idx = np.searchsorted(r["T"], t)
    if idx <= 0:
        return r["X"][0], r["Y"][0], (r["TH"][0] if r["TH"] is not None else 0.0)
    if idx >= len(r["T"]):
        return r["X"][-1], r["Y"][-1], (r["TH"][-1] if r["TH"] is not None else 0.0)

    t1, t2 = r["T"][idx-1], r["T"][idx]
    a = (t - t1) / (t2 - t1 + 1e-9)

    x = r["X"][idx-1] + a*(r["X"][idx] - r["X"][idx-1])
    y = r["Y"][idx-1] + a*(r["Y"][idx] - r["Y"][idx-1])

    if r["TH"] is not None:
        th1 = r["TH"][idx-1]
        th2 = r["TH"][idx]
        th = th1 + a*((th2 - th1 + np.pi)%(2*np.pi)-np.pi)
    else:
        th = 0.0

    return x, y, th

def update(val):
    t = slider.val
    for r, circ, arr in artists:
        x, y, th = interp(r, t)
        circ.center = (x, y)
        if arr is not None:
            L = circ.radius * 1.8
            arr.set_positions((x, y), (x + L*np.cos(th), y + L*np.sin(th)))
    fig.canvas.draw_idle()

slider.on_changed(update)
update(0.0)

ax.set_title(f"WORLD-FRAME REPLAY | Inflation={INFLATION_RADIUS}")
plt.show()
