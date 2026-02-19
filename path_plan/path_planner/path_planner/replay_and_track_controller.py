#!/usr/bin/env python3
"""
Smooth controller tracking using CONTROL CSV *state columns* as reference
========================================================================

Fixes the jagged/bad replay:
- DO NOT re-integrate v/omega or vx/vy (control CSV already contains x,y,theta states)
- Use control CSV's (time,x,y,theta) as the curved reference for individual robots
- Resample reference onto a uniform dt for smooth sim + slider
- Formation members usually have no controls -> use trajectory_logs state CSV (already smooth)

Controllers:
- Diff-drive: stable polar controller to track (x_r,y_r)
- Holonomic : proportional velocity controller to track (x_r,y_r)

World frame throughout. Correct radii (+ inflation).

Suraj-ready.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.widgets import Slider
from pathlib import Path

# ==========================
# CONFIG
# ==========================
TRAJ_DIR = "trajectory_logs"
CTRL_DIR = "control_logs"

DT = 0.05                   # uniform simulation / visualization dt
INFLATION_RADIUS = 0.1
DRAW_INFLATED = True

# Controller gains (stable defaults)
K_RHO   = 1.2
K_ALPHA = 3.0
K_HOLO  = 2.0

# Saturations (optional)
V_MAX  = 1.5
W_MAX  = 3.5
VX_MAX = 1.5
VY_MAX = 1.5

# Radii map (physical radius; inflation added if DRAW_INFLATED)
RADIUS_MAP = {
    "DD1": 0.40,
    "DD2": 0.35,
    "Holo1": 0.40,
    "Holo2": 0.30,

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

# ==========================
# Helpers
# ==========================
def wrap(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def resample_xy_theta(t_raw, x_raw, y_raw, th_raw, t_uniform):
    """Resample x,y,theta onto uniform time grid with angle wrap."""
    x = np.interp(t_uniform, t_raw, x_raw)
    y = np.interp(t_uniform, t_raw, y_raw)
    if th_raw is None:
        return x, y, None

    # unwrap -> interp -> wrap
    th_unwrap = np.unwrap(th_raw)
    th_u = np.interp(t_uniform, t_raw, th_unwrap)
    th = np.array([wrap(a) for a in th_u])
    return x, y, th

# ==========================
# Controllers (executed motion)
# ==========================
class DiffDriveExec:
    def __init__(self, x, y, th):
        self.x, self.y, self.th = float(x), float(y), float(th)

    def step_track_point(self, xr, yr, dt):
        dx = xr - self.x
        dy = yr - self.y
        rho = np.hypot(dx, dy)
        alpha = wrap(np.arctan2(dy, dx) - self.th)

        v = np.clip(K_RHO * rho, -V_MAX, V_MAX)
        w = np.clip(K_ALPHA * alpha, -W_MAX, W_MAX)

        self.x += v * np.cos(self.th) * dt
        self.y += v * np.sin(self.th) * dt
        self.th = wrap(self.th + w * dt)
        return self.x, self.y, self.th

class HoloExec:
    def __init__(self, x, y):
        self.x, self.y = float(x), float(y)

    def step_track_point(self, xr, yr, dt):
        vx = np.clip(K_HOLO * (xr - self.x), -VX_MAX, VX_MAX)
        vy = np.clip(K_HOLO * (yr - self.y), -VY_MAX, VY_MAX)
        self.x += vx * dt
        self.y += vy * dt
        return self.x, self.y

# ==========================
# Reference builders
# ==========================
def build_reference_for_robot(name: str, traj_csv: Path, ctrl_csv: Path | None):
    """
    Priority:
    1) If ctrl_csv exists: use its (time,x,y,theta) as reference (already curved)
    2) Else: use traj_csv (time,x,y,theta) as reference (piecewise linear between waypoints)
    Returns: t_uniform, ref_xy(Nx2), ref_th(N) or None, rtype, used_ctrl(bool)
    """
    used_ctrl = False
    if ctrl_csv is not None and ctrl_csv.exists():
        df = pd.read_csv(ctrl_csv)
        if {"time","x","y"}.issubset(df.columns):
            used_ctrl = True
            t_raw = df["time"].astype(float).values
            x_raw = df["x"].astype(float).values
            y_raw = df["y"].astype(float).values
            th_raw = df["theta"].astype(float).values if "theta" in df.columns else None
            rtype = "diff" if th_raw is not None else "holo"

            t0, tf = float(t_raw[0]), float(t_raw[-1])
            t_uniform = np.arange(t0, tf + DT, DT)

            x_u, y_u, th_u = resample_xy_theta(t_raw, x_raw, y_raw, th_raw, t_uniform)
            ref_xy = np.column_stack([x_u, y_u])
            return t_uniform, ref_xy, th_u, rtype, used_ctrl

    # Fallback to trajectory state CSV
    df = pd.read_csv(traj_csv)
    if not {"time","x","y"}.issubset(df.columns):
        raise RuntimeError(f"{traj_csv.name}: missing time/x/y")

    t_raw = df["time"].astype(float).values
    x_raw = df["x"].astype(float).values
    y_raw = df["y"].astype(float).values
    th_raw = df["theta"].astype(float).values if "theta" in df.columns else None
    rtype = "diff" if th_raw is not None else "holo"

    t0, tf = float(t_raw[0]), float(t_raw[-1])
    t_uniform = np.arange(t0, tf + DT, DT)

    x_u, y_u, th_u = resample_xy_theta(t_raw, x_raw, y_raw, th_raw, t_uniform)
    ref_xy = np.column_stack([x_u, y_u])
    return t_uniform, ref_xy, th_u, rtype, used_ctrl

# ==========================
# Load robots + simulate
# ==========================
traj_dir = Path(TRAJ_DIR)
ctrl_dir = Path(CTRL_DIR)

robots = []

for traj_csv in sorted(traj_dir.glob("*.csv")):
    name = traj_csv.stem
    if "centroid" in name:
        continue

    ctrl_csv = ctrl_dir / f"{name}_controls.csv"
    ctrl_csv = ctrl_csv if ctrl_csv.exists() else None

    t, ref_xy, ref_th, rtype, used_ctrl = build_reference_for_robot(name, traj_csv, ctrl_csv)

    # Execute controller on uniform DT
    if rtype == "diff":
        th0 = ref_th[0] if ref_th is not None else 0.0
        exec_sim = DiffDriveExec(ref_xy[0,0], ref_xy[0,1], th0)
        exec_xy = np.zeros_like(ref_xy)
        exec_th = np.zeros(len(t))
        for k in range(len(t)):
            xr, yr = ref_xy[k]
            x, y, th = exec_sim.step_track_point(xr, yr, DT)
            exec_xy[k] = [x, y]
            exec_th[k] = th
    else:
        exec_sim = HoloExec(ref_xy[0,0], ref_xy[0,1])
        exec_xy = np.zeros_like(ref_xy)
        exec_th = None
        for k in range(len(t)):
            xr, yr = ref_xy[k]
            x, y = exec_sim.step_track_point(xr, yr, DT)
            exec_xy[k] = [x, y]

    base_r = RADIUS_MAP.get(name, 0.3)
    rad = base_r + INFLATION_RADIUS if DRAW_INFLATED else base_r

    robots.append(dict(
        name=name,
        rtype=rtype,
        t=t,
        ref_xy=ref_xy,
        ref_th=ref_th,
        exec_xy=exec_xy,
        exec_th=exec_th,
        radius=rad,
        used_ctrl=used_ctrl
    ))

print(f"Loaded {len(robots)} robots.")
print("Using control-state reference for:")
for r in robots:
    if r["used_ctrl"]:
        print(" ", r["name"])

# ==========================
# Global time grid for slider
# ==========================
t0 = min(r["t"][0] for r in robots)
tf = max(r["t"][-1] for r in robots)
t_global = np.arange(t0, tf + DT, DT)

def idx_at_time(r, t):
    # uniform DT arrays => fast index
    k = int(round((t - r["t"][0]) / DT))
    return int(np.clip(k, 0, len(r["t"]) - 1))

# ==========================
# Visualization
# ==========================
fig, ax = plt.subplots(figsize=(14, 10))
plt.subplots_adjust(bottom=0.16)

# Plot references (dashed) and executed (solid)
for r in robots:
    ax.plot(r["ref_xy"][:,0], r["ref_xy"][:,1], "--", alpha=0.35)
    ax.plot(r["exec_xy"][:,0], r["exec_xy"][:,1], "-", alpha=0.65)

artists = []
def make_robot(rad, is_dd):
    c = Circle((0,0), rad, fill=False, lw=2)
    ax.add_patch(c)
    a = None
    if is_dd:
        a = FancyArrowPatch((0,0),(1,0), arrowstyle="->", lw=2)
        ax.add_patch(a)
    return c, a

for r in robots:
    c, a = make_robot(r["radius"], r["rtype"]=="diff")
    artists.append((r, c, a))

# Bounds
all_x = np.concatenate([r["ref_xy"][:,0] for r in robots] + [r["exec_xy"][:,0] for r in robots])
all_y = np.concatenate([r["ref_xy"][:,1] for r in robots] + [r["exec_xy"][:,1] for r in robots])
ax.set_xlim(all_x.min()-1, all_x.max()+1)
ax.set_ylim(all_y.min()-1, all_y.max()+1)
ax.set_aspect("equal", "box")
ax.set_title("Controller tracking (control-state reference, uniform dt, smooth)")

# Slider
ax_slider = plt.axes([0.18, 0.06, 0.64, 0.03])
slider = Slider(ax_slider, "Time", t_global[0], t_global[-1], valinit=t_global[0])

def update(val):
    t = slider.val
    for r, c, a in artists:
        k = idx_at_time(r, t)
        x, y = r["exec_xy"][k]
        c.center = (x, y)
        if a is not None:
            th = r["exec_th"][k] if r["exec_th"] is not None else 0.0
            L = c.radius * 1.8
            a.set_positions((x,y),(x + L*np.cos(th), y + L*np.sin(th)))
    fig.canvas.draw_idle()

slider.on_changed(update)
update(t_global[0])

plt.show()
