#!/usr/bin/env python3
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from si_rrt_enhanced_individual_kinodynamic import OccupancyGrid


# ============================================================
# Formation collision check
# ============================================================

def formation_collides(P_star, radius, centroid, theta, sx, sy, occupancy):
    """
    P_star   : list of [x, y]
    radius   : list of floats
    centroid : [x, y]
    theta    : float (rad)
    sx, sy   : floats
    occupancy: OccupancyGrid
    """
    c = math.cos(theta)
    s = math.sin(theta)

    for i in range(len(P_star)):
        px, py = P_star[i]

        # R(theta) * S(sx, sy) * p + centroid
        x = centroid[0] + c * (sx * px) - s * (sy * py)
        y = centroid[1] + s * (sx * px) + c * (sy * py)

        if occupancy.disc_collides(x, y, radius[i]):
            return True   # collision

    return False  # collision-free


# ============================================================
# Find best (sx, sy) in given ranges (FULL exploration)
# ============================================================

def find_percentile_scaling(
    P_star,
    radius,
    centroid,
    theta,
    grid,
    resolution,
    origin=(0.0, 0.0),
    sx_range=(0.6, 0.6),
    sy_range=(0.6, 0.6),
    step=0.02,
    percentile=20
):
    occupancy = OccupancyGrid(grid, resolution, origin)

    feasible = []   # (sx, sy, area)

    sx_vals = np.arange(sx_range[0], sx_range[1] + 1e-9, step)
    sy_vals = np.arange(sy_range[0], sy_range[1] + 1e-9, step)

    for sx in sx_vals:
        for sy in sy_vals:
            if not formation_collides(
                P_star, radius, centroid, theta, sx, sy, occupancy
            ):
                feasible.append((sx, sy, sx * sy))

    if not feasible:
        print("[WARN] No collision-free scaling found.")
        return None, None, None

    areas = np.array([a for _, _, a in feasible])
    target_area = np.percentile(areas, percentile)

    idx = np.argmin(np.abs(areas - target_area))
    best_sx, best_sy, best_area = feasible[idx]

    return best_sx, best_sy, target_area


def find_best_scaling(
    P_star,
    radius,
    centroid,
    theta,
    grid,
    resolution,
    origin=(0.0, 0.0),
    sx_range=(0.8, 4.0),
    sy_range=(0.8, 4.0),
    step=0.02
):
    """
    Returns the (sx, sy) with maximum area sx*sy
    that is collision-free.
    """

    occupancy = OccupancyGrid(grid, resolution, origin)

    best_sx = sx_range[0]
    best_sy = sy_range[0]
    best_area = 100000

    sx_vals = np.arange(sx_range[0], sx_range[1] + 1e-9, step)
    sy_vals = np.arange(sy_range[0], sy_range[1] + 1e-9, step)
    print(sx_vals)
    for sx in sx_vals:
        for sy in sy_vals:
            if not formation_collides(
                P_star, radius, centroid, theta, sx, sy, occupancy
            ):
                area = sx * sy
                if area < best_area:
                    best_area = area
                    best_sx = sx
                    best_sy = sy
                    print(f"[NEW BEST] sx={sx:.2f}, sy={sy:.2f}, area={area:.3f}")

    if best_area < 0:
        print("[WARN] No collision-free scaling found in given range.")
        return sx_range[0], sy_range[0]

    return best_sx, best_sy


# ============================================================
# Visualization: map + formation
# ============================================================

def visualize_formation(
    grid,
    resolution,
    origin,
    P_star,
    radius,
    centroid,
    theta,
    sx,
    sy,
    title="Formation visualization"
):
    H, W = grid.shape
    fig, ax = plt.subplots(figsize=(8, 8))

    # Occupancy grid
    ax.imshow(
        grid,
        origin="lower",
        cmap="gray_r",
        extent=[
            origin[0],
            origin[0] + W * resolution,
            origin[1],
            origin[1] + H * resolution,
        ],
        alpha=0.8
    )

    # Centroid
    ax.plot(centroid[0], centroid[1], "ro", markersize=8, label="Centroid")

    # Robots
    c = math.cos(theta)
    s = math.sin(theta)

    for i, (px, py) in enumerate(P_star):
        x = centroid[0] + c * (sx * px) - s * (sy * py)
        y = centroid[1] + s * (sx * px) + c * (sy * py)

        circ = Circle(
            (x, y),
            radius[i],
            edgecolor="blue",
            facecolor="none",
            linewidth=2
        )
        ax.add_patch(circ)
        ax.text(x, y, f"{i}", color="blue", fontsize=12)

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()


# ============================================================
# Optional: visualize feasible (sx, sy) region
# ============================================================

def visualize_feasible_region(
    P_star,
    radius,
    centroid,
    theta,
    grid,
    resolution,
    origin,
    sx_range,
    sy_range,
    step=0.02
):
    occupancy = OccupancyGrid(grid, resolution, origin)

    sx_vals = np.arange(sx_range[0], sx_range[1] + 1e-9, step)
    sy_vals = np.arange(sy_range[0], sy_range[1] + 1e-9, step)

    Z = np.zeros((len(sx_vals), len(sy_vals)))

    for i, sx in enumerate(sx_vals):
        for j, sy in enumerate(sy_vals):
            Z[i, j] = not formation_collides(
                P_star, radius, centroid, theta, sx, sy, occupancy
            )

    plt.figure(figsize=(6, 5))
    plt.imshow(
        Z.T,
        origin="lower",
        extent=[sx_vals[0], sx_vals[-1], sy_vals[0], sy_vals[-1]],
        aspect="auto",
        cmap="Greens"
    )
    plt.xlabel("sx")
    plt.ylabel("sy")
    plt.title("Feasible (sx, sy) region (green = free)")
    plt.colorbar()
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # -------- GIVEN DATA --------
    P_star = [
        [-0.33333333333333304,  1.0],
        [-0.33333333333333304, -1.0],
        [ 0.666666666666667,    0.0]
    ]

    radius = [0.25, 0.25, 0.25]

    centroid = [9.0, 7.0]
    theta = 0.0

    grid = np.load("restaurant.npy")

    resolution = 0.1
    origin = (0.0, 0.0)

    sx_range = (0.3, 0.3)
    sy_range = (0.5, 0.5)

    # -------- RUN SEARCH --------
    best_sx, best_sy, _= find_percentile_scaling(
        P_star=P_star,
        radius=radius,
        centroid=centroid,
        theta=theta,
        grid=grid,
        resolution=resolution,
        origin=origin,
        sx_range=sx_range,
        sy_range=sy_range,
        step=0.02
    )

    print("\n=== FINAL RESULT ===")
    print(f"Best collision-free scaling:")
    print(f"  sx = {best_sx:.3f}")
    print(f"  sy = {best_sy:.3f}")
    print(f"  area = {best_sx * best_sy:.3f}")

    # -------- VISUALIZE --------
    visualize_formation(
        grid=grid,
        resolution=resolution,
        origin=origin,
        P_star=P_star,
        radius=radius,
        centroid=centroid,
        theta=theta,
        sx=best_sx,
        sy=best_sy,
        title=f"Best scaling: sx={best_sx:.2f}, sy={best_sy:.2f}"
    )

    # OPTIONAL (comment out if not needed)
    visualize_feasible_region(
        P_star=P_star,
        radius=radius,
        centroid=centroid,
        theta=theta,
        grid=grid,
        resolution=resolution,
        origin=origin,
        sx_range=sx_range,
        sy_range=sy_range,
        step=0.03
    )
