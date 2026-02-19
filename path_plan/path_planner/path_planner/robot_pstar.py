import numpy as np
import math


def estimate_centroid_and_p_star(poses, theta, sx, sy, robot_types=None):
    """
    Estimate centroid and P_star from robot poses
    """

    # ---- Extract positions ----
    positions = []

    for i, pose in enumerate(poses):
        if robot_types and robot_types[i] == "diff-drive":
            x, y, _ = pose
        else:
            x, y = pose

        positions.append([x, y])

    positions = np.array(positions)

    # ---- Estimate centroid ----
    centroid = np.mean(positions, axis=0)

    # ---- Rotation matrix ----
    R = np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta),  math.cos(theta)]
    ])

    R_inv = R.T

    # ---- Inverse scaling ----
    S_inv = np.diag([1.0/sx, 1.0/sy])

    # ---- Compute P_star ----
    P_star = []

    for pos in positions:
        p = S_inv @ R_inv @ (pos - centroid)
        P_star.append((float(p[0]), float(p[1])))

    return centroid.tolist(), P_star



poses = [
    (1.71, 1.14, 1.57),
    (2.85, 1.14, 1.57),
    (2.28, 1.71, 1.57)
]

theta = 1.57
sx = 1.0
sy = 1.0

robot_types = ["diff-drive", "diff-drive", "diff-drive"]

centroid, p_star = estimate_centroid_and_p_star(
    poses, theta, sx, sy, robot_types
)

print("Estimated centroid:", centroid)

for i, p in enumerate(p_star):
    print(f"Robot {i} P_star: {p}")