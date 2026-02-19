import numpy as np
from scipy.ndimage import binary_dilation

# Load map
occ_map = np.load('restaurant_map.npy')

# ----- PARAMETERS -----
inflation_radius_cells = 5   # radius in cells (change as needed)

# Convert occupancy grid to binary obstacle map
# Assuming:
# 0 = free
# 1 or 100 = obstacle (modify if your map uses different values)
obstacles = occ_map > 0

# Create circular structuring element
y, x = np.ogrid[-inflation_radius_cells:inflation_radius_cells+1,
                -inflation_radius_cells:inflation_radius_cells+1]
struct = x**2 + y**2 <= inflation_radius_cells**2

# Inflate obstacles
inflated = binary_dilation(obstacles, structure=struct)

# Convert back to occupancy grid
inflated_map = occ_map.copy()
inflated_map[inflated] = 1   # or 100 if using ROS occupancy style

# Save new map
np.save('restaurant_map_inflated.npy', inflated_map)
