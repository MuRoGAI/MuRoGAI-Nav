import numpy as np

# ===========================================================
# Agent Classes  (every robot stores its own limits)
# ===========================================================

class DifferentialDriveAgent:
    """
    Individual differential-drive robot.
    
    Stores per-robot limits:
        radius, v_max, omega_max, a_max, alpha_max
    """
    def __init__(
        self,
        radius: float = 0.40,
        v_max: float = 1.0,
        omega_max: float = 2.0,
        a_max: float = 2.0,
        alpha_max: float = 4.0,
    ):
        self.radius = float(radius)
        self.is_holonomic = False
        # Kinodynamic limits stored on the agent
        self.v_max = float(v_max)
        self.omega_max = float(omega_max)
        self.a_max = float(a_max)
        self.alpha_max = float(alpha_max)
    
    def sample_q(self, bounds_xy):
        lo, hi = bounds_xy
        q = np.zeros(3, dtype=float)
        q[:2] = np.random.uniform(lo, hi)
        q[2] = np.random.uniform(-np.pi, np.pi)
        return q
    
    def interpolate_q(self, q1, q2, a):
        q = q1.copy()
        q[:2] = q1[:2] + a * (q2[:2] - q1[:2])
        th1, th2 = float(q1[2]), float(q2[2])
        dth = (th2 - th1 + np.pi) % (2*np.pi) - np.pi
        q[2] = th1 + a * dth
        return q
    
    def discs(self, q):
        p = np.asarray(q, dtype=float)[:2]
        return [(p, self.radius)]
    
    def centroid_xy(self, q):
        return np.asarray(q, dtype=float)[:2].copy()
    
    def robot_poses(self, q):
        q = np.asarray(q, dtype=float)
        return [(float(q[0]), float(q[1]), float(q[2]))]
    
    def max_robot_displacement(self, q1, q2):
        return np.hypot(float(q2[0] - q1[0]), float(q2[1] - q1[1]))
    
    def dist_for_nn(self, q1, q2):
        dx = float(q2[0] - q1[0])
        dy = float(q2[1] - q1[1])
        dth = abs((q2[2] - q1[2] + np.pi) % (2*np.pi) - np.pi)
        return np.hypot(dx, dy) + 0.3 * dth


class HolonomicAgent:
    """
    Individual holonomic (omnidirectional) robot.
    
    Stores per-robot limits:
        radius, v_max, a_max
    """
    def __init__(
        self,
        radius: float = 0.40,
        v_max: float = 1.0,
        a_max: float = 2.0,
    ):
        self.radius = float(radius)
        self.is_holonomic = True
        # Kinodynamic limits stored on the agent
        self.v_max = float(v_max)
        self.a_max = float(a_max)
    
    def sample_q(self, bounds_xy):
        lo, hi = bounds_xy
        return np.random.uniform(lo, hi)
    
    def interpolate_q(self, q1, q2, a):
        return q1 + a * (q2 - q1)
    
    def discs(self, q):
        p = np.asarray(q, dtype=float)[:2]
        return [(p, self.radius)]
    
    def centroid_xy(self, q):
        return np.asarray(q, dtype=float)[:2].copy()
    
    def robot_poses(self, q):
        q = np.asarray(q, dtype=float)
        return [(float(q[0]), float(q[1]), 0.0)]
    
    def max_robot_displacement(self, q1, q2):
        return np.hypot(float(q2[0] - q1[0]), float(q2[1] - q1[1]))
    
    def dist_for_nn(self, q1, q2):
        return np.hypot(float(q2[0] - q1[0]), float(q2[1] - q1[1]))


class HeterogeneousFormationAgent:
    """
    Formation with MIXED robot types.
    
    Every robot has its own:
        radius, v_max, omega_max, a_max, alpha_max
    
    All limit arrays are length Nr (one per robot).
    Scalar values are broadcast to all robots.
    """
    def __init__(
        self,
        P_star: list,
        robot_types: list,          # ['diff-drive', 'holonomic', ...]
        radius=0.35,                # float OR list[float]
        v_max=1.0,                  # float OR list[float]
        omega_max=2.0,              # float OR list[float]  (DD only; ignored for holonomic)
        a_max=2.0,                  # float OR list[float]
        alpha_max=4.0,              # float OR list[float]  (DD only; ignored for holonomic)
        sx_range: tuple = (0.7, 3.0),
        sy_range: tuple = (0.7, 3.0),
    ):
        self.P_star = np.array(P_star, dtype=float)
        self.robot_types = robot_types
        self.Nr = len(P_star)
        self.sx_range = sx_range
        self.sy_range = sy_range
        
        if len(robot_types) != self.Nr:
            raise ValueError(f"robot_types length ({len(robot_types)}) != Nr ({self.Nr})")
        
        # --- Helper: scalar → per-robot list ---
        def _broadcast(val, name):
            if isinstance(val, (int, float)):
                return [float(val)] * self.Nr
            lst = [float(v) for v in val]
            if len(lst) != self.Nr:
                raise ValueError(f"{name} length ({len(lst)}) != Nr ({self.Nr})")
            return lst
        
        self.radii = _broadcast(radius, 'radius')       # per-robot radii
        self.radius = self.radii                         # alias (list)
        self.v_max_list = _broadcast(v_max, 'v_max')
        self.omega_max_list = _broadcast(omega_max, 'omega_max')
        self.a_max_list = _broadcast(a_max, 'a_max')
        self.alpha_max_list = _broadcast(alpha_max, 'alpha_max')
        
        # For disc decomposition cache
        self._disc_cache = {}
        
        print(f"Heterogeneous formation: {robot_types}")
        print(f"  radii      = {self.radii}")
        print(f"  v_max      = {self.v_max_list}")
        print(f"  omega_max  = {self.omega_max_list}")
        print(f"  a_max      = {self.a_max_list}")
        print(f"  alpha_max  = {self.alpha_max_list}")
    
    def sample_q(self, bounds_xy):
        lo, hi = bounds_xy
        q = np.zeros(5, dtype=float)
        q[:2] = np.random.uniform(lo, hi)
        q[2] = np.random.uniform(-np.pi, np.pi)
        q[3] = np.random.uniform(*self.sx_range)
        q[4] = np.random.uniform(*self.sy_range)
        return q
    
    def interpolate_q(self, q1, q2, a):
        q = q1.copy()
        q[:2] = q1[:2] + a * (q2[:2] - q1[:2])
        dth = (q2[2] - q1[2] + np.pi) % (2*np.pi) - np.pi
        q[2] = q1[2] + a * dth
        q[3:5] = q1[3:5] + a * (q2[3:5] - q1[3:5])
        return q
    
    def discs(self, q):
        """Decompose formation into discs for collision checking (per-robot radius)."""
        q = np.asarray(q, dtype=float).flatten()
        
        # Use cache
        qkey = tuple(np.round(q, decimals=2))
        if qkey in self._disc_cache:
            return self._disc_cache[qkey]
        
        xc, yc, th, sx, sy = q[0], q[1], q[2], q[3], q[4]
        
        R = np.array([[np.cos(th), -np.sin(th)],
                      [np.sin(th), np.cos(th)]])
        D = np.diag([sx, sy])
        
        discs = []
        for i, p_star in enumerate(self.P_star):
            p = np.array([xc, yc]) + R @ D @ p_star
            discs.append((p, self.radii[i]))       # ← per-robot radius
        
        self._disc_cache[qkey] = discs
        return discs
    
    def centroid_xy(self, q):
        return np.asarray(q, dtype=float)[:2].copy()
    
    def robot_poses(self, q):
        """Return (x, y, heading) for each robot"""
        q = np.asarray(q, dtype=float).flatten()
        xc, yc, th, sx, sy = q[0], q[1], q[2], q[3], q[4]
        
        R = np.array([[np.cos(th), -np.sin(th)],
                      [np.sin(th), np.cos(th)]])
        D = np.diag([sx, sy])
        
        poses = []
        for i, p_star in enumerate(self.P_star):
            p = np.array([xc, yc]) + R @ D @ p_star
            # For DD robots, use formation heading; for holonomic, 0.0
            theta = th if self.robot_types[i] == 'diff-drive' else 0.0
            poses.append((float(p[0]), float(p[1]), float(theta)))
        
        return poses
    
    def max_robot_displacement(self, q1, q2):
        poses1 = self.robot_poses(q1)
        poses2 = self.robot_poses(q2)
        max_d = 0.0
        for (x1, y1, _), (x2, y2, _) in zip(poses1, poses2):
            d = np.hypot(x2 - x1, y2 - y1)
            max_d = max(max_d, d)
        return max_d
    
    def dist_for_nn(self, q1, q2):
        """Distance metric for nearest neighbor (formations)"""
        q1, q2 = np.asarray(q1, dtype=float), np.asarray(q2, dtype=float)
        dx, dy = q2[0] - q1[0], q2[1] - q1[1]
        dth = abs((q2[2] - q1[2] + np.pi) % (2*np.pi) - np.pi)
        dsx = abs(q2[3] - q1[3])
        dsy = abs(q2[4] - q1[4])
        
        # Weighted distance
        return np.hypot(dx, dy) + 0.3 * dth + 0.2 * (dsx + dsy)
    
    def clear_cache(self):
        self._disc_cache.clear()

