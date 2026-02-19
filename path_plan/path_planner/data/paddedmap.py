import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

# =============================
# Load map
# =============================
occ_map = np.load("restaurant_5.npy")   # 0 = free, 1 = obstacle

# =============================
# Parameters
# =============================
resolution = 0.1      # meters per cell
robot_radius = 0.3    # meters

radius_cells = int(np.ceil(robot_radius / resolution))
print(f"Robot radius = {robot_radius} m → {radius_cells} cells")

# =============================
# Circular structuring element
# =============================
y, x = np.ogrid[-radius_cells:radius_cells+1,
                -radius_cells:radius_cells+1]
disk = (x**2 + y**2) <= radius_cells**2

# =============================
# Inflate map (C-space)
# =============================
inflated_map = binary_dilation(
    occ_map.astype(bool),
    structure=disk
).astype(np.uint8)

# =============================
# Save inflated map
# =============================
np.save("restaurant_5_padded.npy", inflated_map)
print("Saved: restaurant_inflated.npy")

# =============================
# Plot
# =============================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Original Occupancy Map")
plt.imshow(occ_map, cmap="gray", origin="lower")
plt.xlabel("x (cells)")
plt.ylabel("y (cells)")
plt.axis("equal")

plt.subplot(1, 2, 2)
plt.title("Inflated Map (Robot Radius = 0.3 m)")
plt.imshow(inflated_map, cmap="gray", origin="lower")
plt.xlabel("x (cells)")
plt.ylabel("y (cells)")
plt.axis("equal")

plt.tight_layout()
plt.show()
