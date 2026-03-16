import numpy as np
import math


def compute_robot_poses(centroid, theta, P_star, sx, sy, robot_types=None):
    """
    Compute individual robot poses from formation state.

    Args:
        centroid : [xc, yc]
        theta    : formation orientation
        P_star   : list of [px, py] formation offsets
        sx, sy   : formation scaling
        robot_types : optional list ["diff-drive", "holonomic"]

    Returns:
        List of robot poses:
            diff-drive -> (x, y, theta)
            holonomic  -> (x, y)
    """

    xc, yc = centroid

    # Rotation matrix
    R = np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta),  math.cos(theta)]
    ])

    # Scaling matrix
    S = np.diag([sx, sy])

    robot_poses = []

    for i, p in enumerate(P_star):
        p = np.array(p)

        # Global robot position
        pos = np.array([xc, yc]) + R @ S @ p

        # Orientation handling
        if robot_types and robot_types[i] == "diff-drive":
            robot_poses.append((float(pos[0]), float(pos[1]), float(theta)))
        else:
            robot_poses.append((float(pos[0]), float(pos[1])))

    return robot_poses



#########################

P_star = [[-0.19, 0.57], [-0.19, -0.57], [0.38, 0.00]]
centroid = [2.28, 1.33]
theta = 0.0
sx = 1.0
sy = 1.0

robot_types = ["diff-drive", "diff-drive", "holonomic",]

poses = compute_robot_poses(
    centroid, theta, P_star, sx, sy, robot_types
)

for i, p in enumerate(poses):
    print(f"Robot {i} :{p}: : {robot_types[i]}")


##################
print("")
###################

P_star = [
    [-0.19, 0.57],
    [-0.19, -0.57],
    [0.38, 0.0]
]

centroid = [2.28, 1.33]
theta = 1.57
sx = 1.0
sy = 1.0

robot_types = ["diff-drive", "holonomic", "diff-drive"]

poses = compute_robot_poses(
    centroid, theta, P_star, sx, sy, robot_types
)

for i, p in enumerate(poses):
    print(f"Robot {i} :{p}: : {robot_types[i]}")