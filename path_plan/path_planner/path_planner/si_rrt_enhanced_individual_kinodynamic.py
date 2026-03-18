#!/usr/bin/env python3
"""
si_rrt_enhanced_individual_kinodynamic.py  —  PATCHED

Key change vs original
----------------------
Kinematics.travel_time() now derives the maximum speed directly from the
agent object instead of the global planner max_velocity.

  • HeterogeneousFormationAgent  →  max_i( d_i / v_max_list[i] )
      Every individual robot's displacement is divided by *its own* speed
      limit; the bottleneck robot determines the edge travel time.
  • DifferentialDriveAgent / HolonomicAgent  →  distance / agent.v_max
  • Legacy FlexibleFormationAgent (no v_max_list)  →  max_displacement / min(v_max)
  • Ultimate fallback  →  distance / planner max_velocity  (unchanged behaviour
      for agents that expose no velocity information)

A new helper _agent_effective_vmax() is also added for callers that only
need a scalar speed bound (e.g. neighbour-radius heuristics).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Any

import numpy as np

# Import individual agent kinodynamic steering
try:
    from individual_kinodynamic_steering import (
        DifferentialDriveSteering,
        HolonomicSteering,
        ControlTrajectory,
        create_steering_for_agent
    )
    INDIVIDUAL_KINO_AVAILABLE = True
except ImportError:
    INDIVIDUAL_KINO_AVAILABLE = False
    print("Warning: individual_kinodynamic_steering not available")

# Import heterogeneous formation steering
try:
    from heterogeneous_kinodynamic_formation_steering import (
        HeterogeneousKinodynamicFormationSteering
    )
    HETEROGENEOUS_KINO_AVAILABLE = True
except ImportError:
    HETEROGENEOUS_KINO_AVAILABLE = False

# Optional imports with fallbacks
try:
    from scipy.spatial import cKDTree
    KDTREE_AVAILABLE = True
except ImportError:
    KDTREE_AVAILABLE = False

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


# ============================================================
# Numba-accelerated functions
# ============================================================

@njit(cache=True, fastmath=True)
def _nb_blocked_interval_segment(
    px: float, py: float, R: float,
    p0x: float, p0y: float, p1x: float, p1y: float,
    t0: float, t1: float, T: float
) -> Tuple[float, float, bool]:
    """Compute blocked interval for one trajectory segment."""
    if t1 <= 0.0 or t0 >= T:
        return 0.0, 0.0, False

    s0 = max(0.0, t0)
    s1 = min(T, t1)
    if s1 <= s0 + 1e-12:
        return 0.0, 0.0, False

    dt = t1 - t0
    if dt <= 1e-12:
        dx = p1x - px
        dy = p1y - py
        if dx*dx + dy*dy <= R*R:
            return s0, s1, True
        return 0.0, 0.0, False

    vx = (p1x - p0x) / dt
    vy = (p1y - p0y) / dt
    dx = p0x - px
    dy = p0y - py

    a = vx*vx + vy*vy
    b = 2.0 * (dx*vx + dy*vy)
    c = dx*dx + dy*dy - R*R

    tau_lo = s0 - t0
    tau_hi = s1 - t0

    if a <= 1e-12:
        if c <= 0.0:
            return s0, s1, True
        return 0.0, 0.0, False

    disc = b*b - 4.0*a*c
    if disc < 0.0:
        return 0.0, 0.0, False

    sqrt_disc = math.sqrt(disc)
    r1 = (-b - sqrt_disc) / (2.0*a)
    r2 = (-b + sqrt_disc) / (2.0*a)

    lo = max(tau_lo, min(r1, r2))
    hi = min(tau_hi, max(r1, r2))

    if hi > lo + 1e-12:
        return t0 + lo, t0 + hi, True
    return 0.0, 0.0, False


@njit(cache=True, fastmath=True)
def _nb_compute_blocked_intervals(
    px: float, py: float, R: float,
    traj_x: np.ndarray, traj_y: np.ndarray, traj_t: np.ndarray,
    T: float
) -> np.ndarray:
    """Compute all blocked intervals for a trajectory. Returns Nx2 array."""
    n = len(traj_t) - 1
    if n <= 0:
        return np.empty((0, 2), dtype=np.float64)

    result = np.empty((n, 2), dtype=np.float64)
    count = 0

    for i in range(n):
        lo, hi, valid = _nb_blocked_interval_segment(
            px, py, R,
            traj_x[i], traj_y[i], traj_x[i+1], traj_y[i+1],
            traj_t[i], traj_t[i+1], T
        )
        if valid:
            result[count, 0] = lo
            result[count, 1] = hi
            count += 1

    return result[:count]


@njit(cache=True, fastmath=True)
def _nb_disc_collides(
    x: float, y: float, radius: float,
    grid: np.ndarray, resolution: float,
    origin_x: float, origin_y: float, H: int, W: int
) -> bool:
    """Check if disc collides with grid obstacles."""
    r = radius

    j0 = int((x - r - origin_x) / resolution) - 1
    i0 = int((y - r - origin_y) / resolution) - 1
    j1 = int((x + r - origin_x) / resolution) + 2
    i1 = int((y + r - origin_y) / resolution) + 2

    rr = r * r

    for i in range(i0, i1):
        for j in range(j0, j1):
            if i < 0 or i >= H or j < 0 or j >= W:
                return True
            if grid[i, j] == 0:
                continue
            cx = origin_x + (j + 0.5) * resolution
            cy = origin_y + (i + 0.5) * resolution
            dx = cx - x
            dy = cy - y
            if dx*dx + dy*dy <= rr:
                return True
    return False


@njit(cache=True, fastmath=True)
def _nb_discs_overlap(
    p1x: float, p1y: float, r1: float,
    p2x: float, p2y: float, r2: float
) -> bool:
    """Check if two discs overlap."""
    dx = p1x - p2x
    dy = p1y - p2y
    rsum = r1 + r2
    return dx*dx + dy*dy < rsum*rsum*1.5


@njit(cache=True, fastmath=True)
def _nb_dist_sq_xy(x1: float, y1: float, x2: float, y2: float) -> float:
    """Fast squared distance between two 2D points."""
    dx = x2 - x1
    dy = y2 - y1
    return dx * dx + dy * dy


@njit(cache=True, fastmath=True)
def _nb_ff_dist_sq(
    x1: float, y1: float, th1: float, sx1: float, sy1: float,
    x2: float, y2: float, th2: float, sx2: float, sy2: float,
    Nx: float, Ny: float, Nxy: float, Nr: int
) -> float:
    """FF-RRT* squared distance metric."""
    dth = th2 - th1
    cosd = math.cos(dth)
    sind = math.sin(dth)
    dxc = x1 - x2
    dyc = y1 - y2

    term1 = Nx * (sx1*sx1 - 2.0*sx1*sx2*cosd + sx2*sx2)
    term2 = Ny * (sy1*sy1 - 2.0*sy1*sy2*cosd + sy2*sy2)
    term3 = -2.0 * Nxy * (sy1*sx2 - sy2*sx1) * sind
    term4 = Nr * (dxc*dxc + dyc*dyc)
    return term1 + term2 + term3 + term4


@njit(cache=True, fastmath=True)
def _nb_compute_formation_discs(
    P_star: np.ndarray,
    radii: np.ndarray,
    xc: float, yc: float, th: float, sx: float, sy: float
) -> np.ndarray:
    Nr = P_star.shape[0]
    result = np.empty((Nr, 3), dtype=np.float64)
    c = math.cos(th)
    s = math.sin(th)
    for i in range(Nr):
        px_star = P_star[i, 0]
        py_star = P_star[i, 1]
        px_scaled = sx * px_star
        py_scaled = sy * py_star
        x = c * px_scaled - s * py_scaled + xc
        y = s * px_scaled + c * py_scaled + yc
        result[i, 0] = x
        result[i, 1] = y
        result[i, 2] = radii[i]
    return result


@njit(cache=True, fastmath=True)
def _nb_compute_robot_poses(
    P_star: np.ndarray,
    xc: float, yc: float, th: float, sx: float, sy: float
) -> np.ndarray:
    Nr = P_star.shape[0]
    result = np.empty((Nr, 3), dtype=np.float64)
    c = math.cos(th)
    s = math.sin(th)
    for i in range(Nr):
        px_star = P_star[i, 0]
        py_star = P_star[i, 1]
        px_scaled = sx * px_star
        py_scaled = sy * py_star
        x = c * px_scaled - s * py_scaled + xc
        y = s * px_scaled + c * py_scaled + yc
        result[i, 0] = x
        result[i, 1] = y
        result[i, 2] = th
    return result


@njit(cache=True, fastmath=True)
def _nb_max_robot_displacement(
    P_star: np.ndarray,
    q1_xc: float, q1_yc: float, q1_th: float, q1_sx: float, q1_sy: float,
    q2_xc: float, q2_yc: float, q2_th: float, q2_sx: float, q2_sy: float
) -> float:
    Nr = P_star.shape[0]
    c1 = math.cos(q1_th)
    s1 = math.sin(q1_th)
    c2 = math.cos(q2_th)
    s2 = math.sin(q2_th)
    max_disp = 0.0
    for i in range(Nr):
        px_star = P_star[i, 0]
        py_star = P_star[i, 1]
        px1_scaled = q1_sx * px_star
        py1_scaled = q1_sy * py_star
        x1 = c1 * px1_scaled - s1 * py1_scaled + q1_xc
        y1 = s1 * px1_scaled + c1 * py1_scaled + q1_yc
        px2_scaled = q2_sx * px_star
        py2_scaled = q2_sy * py_star
        x2 = c2 * px2_scaled - s2 * py2_scaled + q2_xc
        y2 = s2 * px2_scaled + c2 * py2_scaled + q2_yc
        dx = x2 - x1
        dy = y2 - y1
        disp = math.sqrt(dx*dx + dy*dy)
        if disp > max_disp:
            max_disp = disp
    return max_disp


@njit(cache=True, fastmath=True)
def _nb_interpolate_formation_5d(
    q1: np.ndarray,
    q2: np.ndarray,
    alpha: float
) -> np.ndarray:
    result = np.empty(5, dtype=np.float64)
    result[0] = q1[0] + alpha * (q2[0] - q1[0])
    result[1] = q1[1] + alpha * (q2[1] - q1[1])
    th1 = q1[2]
    th2 = q2[2]
    dth = th2 - th1
    if dth > math.pi:
        dth -= 2.0 * math.pi
    elif dth < -math.pi:
        dth += 2.0 * math.pi
    result[2] = th1 + alpha * dth
    result[3] = q1[3] + alpha * (q2[3] - q1[3])
    result[4] = q1[4] + alpha * (q2[4] - q1[4])
    return result


@njit(cache=True, fastmath=True)
def _nb_interpolate_individual_2d(
    q1: np.ndarray,
    q2: np.ndarray,
    alpha: float
) -> np.ndarray:
    result = np.empty(2, dtype=np.float64)
    result[0] = q1[0] + alpha * (q2[0] - q1[0])
    result[1] = q1[1] + alpha * (q2[1] - q1[1])
    return result


@njit(cache=True, fastmath=True)
def _nb_rs_like_proxy(
    x1: float, y1: float, th1: float,
    x2: float, y2: float, th2: float,
    lateral_weight: float,
    turn_weight: float
) -> float:
    dx = x2 - x1
    dy = y2 - y1
    c = math.cos(th1)
    s = math.sin(th1)
    df = c * dx + s * dy
    dl = -s * dx + c * dy
    dth = th2 - th1
    if dth > math.pi:
        dth -= 2.0 * math.pi
    elif dth < -math.pi:
        dth += 2.0 * math.pi
    trans = math.sqrt(df * df + lateral_weight * dl * dl)
    turn = turn_weight * abs(dth)
    return trans + turn


@njit(cache=True, fastmath=True)
def _nb_max_rs_like_distance(
    P_star: np.ndarray,
    q1_xc: float, q1_yc: float, q1_th: float, q1_sx: float, q1_sy: float,
    q2_xc: float, q2_yc: float, q2_th: float, q2_sx: float, q2_sy: float,
    lateral_weight: float,
    turn_weight: float
) -> float:
    Nr = P_star.shape[0]
    c1 = math.cos(q1_th)
    s1 = math.sin(q1_th)
    c2 = math.cos(q2_th)
    s2 = math.sin(q2_th)
    max_rs = 0.0
    for i in range(Nr):
        px_star = P_star[i, 0]
        py_star = P_star[i, 1]
        px1 = q1_sx * px_star
        py1 = q1_sy * py_star
        x1 = c1 * px1 - s1 * py1 + q1_xc
        y1 = s1 * px1 + c1 * py1 + q1_yc
        th1_robot = q1_th
        px2 = q2_sx * px_star
        py2 = q2_sy * py_star
        x2 = c2 * px2 - s2 * py2 + q2_xc
        y2 = s2 * px2 + c2 * py2 + q2_yc
        th2_robot = q2_th
        rs_dist = _nb_rs_like_proxy(
            x1, y1, th1_robot,
            x2, y2, th2_robot,
            lateral_weight, turn_weight
        )
        if rs_dist > max_rs:
            max_rs = rs_dist
    return max_rs


@njit(cache=True, fastmath=True)
def _nb_formation_nn_distance(
    q1: np.ndarray,
    q2: np.ndarray,
    P_star: np.ndarray,
    Nx: float, Ny: float, Nxy: float, Nr: int,
    alpha: float,
    lateral_weight: float,
    turn_weight: float
) -> float:
    d2_aff = _nb_ff_dist_sq(
        q1[0], q1[1], q1[2], q1[3], q1[4],
        q2[0], q2[1], q2[2], q2[3], q2[4],
        Nx, Ny, Nxy, Nr
    )
    d_aff = math.sqrt(d2_aff)
    d_nh = _nb_max_rs_like_distance(
        P_star,
        q1[0], q1[1], q1[2], q1[3], q1[4],
        q2[0], q2[1], q2[2], q2[3], q2[4],
        lateral_weight, turn_weight
    )
    return alpha * d_aff + (1.0 - alpha) * d_nh


# ============================================================
# Occupancy Grid
# ============================================================

class OccupancyGrid:
    def __init__(self, grid: np.ndarray, resolution: float, origin: Tuple[float, float] = (0.0, 0.0)):
        self.grid = np.ascontiguousarray(grid, dtype=np.uint8)
        self.resolution = float(resolution)
        self.origin = (float(origin[0]), float(origin[1]))
        self.H, self.W = self.grid.shape

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        j = int((x - self.origin[0]) / self.resolution)
        i = int((y - self.origin[1]) / self.resolution)
        return i, j

    def in_bounds(self, i: int, j: int) -> bool:
        return (0 <= i < self.H) and (0 <= j < self.W)

    def disc_collides(self, x: float, y: float, radius: float) -> bool:
        if NUMBA_AVAILABLE:
            return _nb_disc_collides(
                x, y, radius, self.grid, self.resolution,
                self.origin[0], self.origin[1], self.H, self.W
            )
        r = float(radius)
        i0, j0 = self.world_to_cell(x - r, y - r)
        i1, j1 = self.world_to_cell(x + r, y + r)
        i0 -= 1; j0 -= 1; i1 += 1; j1 += 1
        rr = r * r
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                if not self.in_bounds(i, j):
                    return True
                if self.grid[i, j] == 0:
                    continue
                cx = self.origin[0] + (j + 0.5) * self.resolution
                cy = self.origin[1] + (i + 0.5) * self.resolution
                if (cx - x) ** 2 + (cy - y) ** 2 <= rr:
                    return True
        return False


# ============================================================
# Safe intervals + helper interval ops
# ============================================================

@dataclass(frozen=True)
class SafeInterval:
    low: float
    high: float


def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ls, le = merged[-1]
        if s <= le + 1e-9:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def _complement_intervals(blocked: List[Tuple[float, float]], T: float) -> List[SafeInterval]:
    blocked = _merge_intervals([(max(0.0, s), min(T, e)) for s, e in blocked if e > s + 1e-12])
    safe: List[SafeInterval] = []
    cur = 0.0
    for s, e in blocked:
        if s > cur + 1e-12:
            safe.append(SafeInterval(cur, s))
        cur = max(cur, e)
    if cur < T - 1e-12:
        safe.append(SafeInterval(cur, T))
    return safe


def _blocked_intervals_fixed_disc_vs_moving_disc_traj(
    p_fixed: np.ndarray,
    R: float,
    traj: List[Tuple[np.ndarray, float]],
    T: float,
) -> List[Tuple[float, float]]:
    if not traj or len(traj) < 2:
        return []

    p_fixed = np.asarray(p_fixed, dtype=np.float64)[:2]
    R = float(R)

    n = len(traj)
    traj_x = np.empty(n, dtype=np.float64)
    traj_y = np.empty(n, dtype=np.float64)
    traj_t = np.empty(n, dtype=np.float64)

    for i, (pos, t) in enumerate(traj):
        pos = np.asarray(pos, dtype=np.float64)
        traj_x[i] = pos[0]
        traj_y[i] = pos[1]
        traj_t[i] = float(t)

    if NUMBA_AVAILABLE:
        result = _nb_compute_blocked_intervals(
            p_fixed[0], p_fixed[1], R,
            traj_x, traj_y, traj_t, T
        )
        return [(float(result[i, 0]), float(result[i, 1])) for i in range(result.shape[0])]

    blocked: List[Tuple[float, float]] = []
    for i in range(n - 1):
        lo, hi, valid = _nb_blocked_interval_segment(
            p_fixed[0], p_fixed[1], R,
            traj_x[i], traj_y[i], traj_x[i+1], traj_y[i+1],
            traj_t[i], traj_t[i+1], T
        )
        if valid:
            blocked.append((lo, hi))
    return blocked


# ============================================================
# FF metric constants
# ============================================================

def precompute_constants(P_star: np.ndarray) -> Tuple[float, float, float]:
    P_star = np.asarray(P_star, dtype=float)
    Nx = float(np.sum(P_star[:, 0] ** 2))
    Ny = float(np.sum(P_star[:, 1] ** 2))
    Nxy = float(np.sum(P_star[:, 0] * P_star[:, 1]))
    return Nx, Ny, Nxy


def ff_dist_sq(q1: np.ndarray, q2: np.ndarray, Nx: float, Ny: float, Nxy: float, Nr: int) -> float:
    q1 = np.asarray(q1, dtype=float).ravel()
    q2 = np.asarray(q2, dtype=float).ravel()
    if NUMBA_AVAILABLE:
        return _nb_ff_dist_sq(
            q1[0], q1[1], q1[2], q1[3], q1[4],
            q2[0], q2[1], q2[2], q2[3], q2[4],
            Nx, Ny, Nxy, Nr
        )
    x1, y1, th1, sx1, sy1 = q1[0], q1[1], q1[2], q1[3], q1[4]
    x2, y2, th2, sx2, sy2 = q2[0], q2[1], q2[2], q2[3], q2[4]
    dth = th2 - th1
    cosd = math.cos(dth)
    sind = math.sin(dth)
    dxc = x1 - x2
    dyc = y1 - y2
    term1 = Nx * (sx1**2 - 2.0*sx1*sx2*cosd + sx2**2)
    term2 = Ny * (sy1**2 - 2.0*sy1*sy2*cosd + sy2**2)
    term3 = -2.0 * Nxy * (sy1*sx2 - sy2*sx1) * sind
    term4 = Nr * (dxc**2 + dyc**2)
    return float(term1 + term2 + term3 + term4)


# ============================================================
# Agent models
# ============================================================

class IndividualAgent:
    def __init__(self, radius: float):
        self.radius = float(radius)

    def sample_q(self, bounds_xy: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        lo, hi = bounds_xy
        return np.random.uniform(lo, hi)

    def interpolate_q(self, q1: np.ndarray, q2: np.ndarray, a: float) -> np.ndarray:
        q1 = np.asarray(q1, dtype=float)
        q2 = np.asarray(q2, dtype=float)
        if NUMBA_AVAILABLE and q1.shape[0] == 2:
            return _nb_interpolate_individual_2d(q1, q2, float(a))
        q = q1.copy()
        q[:2] = q1[:2] + a * (q2[:2] - q1[:2])
        return q

    def discs(self, q: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        p = np.asarray(q, dtype=float)[:2]
        return [(p, self.radius)]

    def centroid_xy(self, q: np.ndarray) -> np.ndarray:
        return np.asarray(q, dtype=float)[:2].copy()

    def robot_poses(self, q: np.ndarray) -> List[Tuple[float, float, float]]:
        q = np.asarray(q, dtype=float)
        x, y = float(q[0]), float(q[1])
        return [(x, y, 0.0)]

    def max_robot_displacement(self, q1: np.ndarray, q2: np.ndarray) -> float:
        q1 = np.asarray(q1, dtype=float)
        q2 = np.asarray(q2, dtype=float)
        dx = float(q2[0] - q1[0])
        dy = float(q2[1] - q1[1])
        return math.hypot(dx, dy)

    def dist_for_nn(self, q1: np.ndarray, q2: np.ndarray) -> float:
        if NUMBA_AVAILABLE:
            return math.sqrt(_nb_dist_sq_xy(q1[0], q1[1], q2[0], q2[1]))
        dx = float(q2[0] - q1[0])
        dy = float(q2[1] - q1[1])
        return math.hypot(dx, dy)


class FlexibleFormationAgent:
    def __init__(
        self,
        P_star: List[Iterable[float]],
        radius,
        sx_range: Tuple[float, float] = (0.85, 1.25),
        sy_range: Tuple[float, float] = (0.85, 1.25),
    ):
        self.P_star = np.asarray(P_star, dtype=float)
        self.Nr = int(self.P_star.shape[0])
        if isinstance(radius, (int, float)):
            self.radii = np.array([float(radius)] * self.Nr, dtype=np.float64)
        else:
            self.radii = np.array([float(r) for r in radius], dtype=np.float64)
            if len(self.radii) != self.Nr:
                raise ValueError(f"radius length ({len(self.radii)}) != Nr ({self.Nr})")

        self.radius = self.radii.tolist()
        self.Nx, self.Ny, self.Nxy = precompute_constants(self.P_star)
        self.sx_range = (float(sx_range[0]), float(sx_range[1]))
        self.sy_range = (float(sy_range[0]), float(sy_range[1]))

        self._disc_cache: Dict[Tuple[float, ...], List[Tuple[np.ndarray, float]]] = {}
        self._disc_cache_precision = 3
        self._robot_poses_cache: Dict[Tuple[float, ...], np.ndarray] = {}

    def clear_cache(self):
        self._disc_cache.clear()
        self._robot_poses_cache.clear()

    def _qkey(self, q: np.ndarray) -> Tuple[float, ...]:
        return tuple(np.round(np.asarray(q, dtype=float), self._disc_cache_precision))

    def sample_q(self, bounds_xy: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        lo, hi = bounds_xy
        q = np.zeros(5, dtype=float)
        q[0:2] = np.random.uniform(lo, hi)
        q[2] = np.random.uniform(-math.pi, math.pi)
        q[3] = np.random.uniform(self.sx_range[0], self.sx_range[1])
        q[4] = np.random.uniform(self.sy_range[0], self.sy_range[1])
        return q

    def interpolate_q(self, q1: np.ndarray, q2: np.ndarray, a: float) -> np.ndarray:
        q1 = np.asarray(q1, dtype=float)
        q2 = np.asarray(q2, dtype=float)
        if NUMBA_AVAILABLE and q1.shape[0] == 5:
            return _nb_interpolate_formation_5d(q1, q2, float(a))
        q = q1.copy()
        q[0:2] = q1[0:2] + a * (q2[0:2] - q1[0:2])
        q[3] = q1[3] + a * (q2[3] - q1[3])
        q[4] = q1[4] + a * (q2[4] - q1[4])
        th1, th2 = float(q1[2]), float(q2[2])
        dth = (th2 - th1 + math.pi) % (2.0 * math.pi) - math.pi
        q[2] = th1 + a * dth
        return q

    def discs(self, q: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        key = self._qkey(q)
        cached = self._disc_cache.get(key, None)
        if cached is not None:
            return [(cached[i, :2].copy(), cached[i, 2]) for i in range(cached.shape[0])]

        q = np.asarray(q, dtype=float)
        if NUMBA_AVAILABLE:
            discs_array = _nb_compute_formation_discs(
                self.P_star, self.radii,
                q[0], q[1], q[2], q[3], q[4]
            )
            self._disc_cache[key] = discs_array
            return [(discs_array[i, :2].copy(), discs_array[i, 2]) for i in range(discs_array.shape[0])]

        xc, yc, th, sx, sy = q[0], q[1], q[2], q[3], q[4]
        c, s = math.cos(th), math.sin(th)
        Rm = np.array([[c, -s], [s, c]], dtype=float)
        Sm = np.array([[sx, 0.0], [0.0, sy]], dtype=float)
        center = np.array([xc, yc], dtype=float)
        P = (Rm @ (Sm @ self.P_star.T)).T + center
        out = [(P[i, :].copy(), self.radii[i]) for i in range(self.Nr)]
        return out

    def centroid_xy(self, q: np.ndarray) -> np.ndarray:
        return np.asarray(q, dtype=float)[:2].copy()

    def robot_poses(self, q: np.ndarray) -> List[Tuple[float, float, float]]:
        q = np.asarray(q, dtype=float)
        key = self._qkey(q)
        cached = self._robot_poses_cache.get(key, None)
        if cached is not None:
            return [(cached[i, 0], cached[i, 1], cached[i, 2]) for i in range(cached.shape[0])]

        if NUMBA_AVAILABLE:
            poses_array = _nb_compute_robot_poses(
                self.P_star, q[0], q[1], q[2], q[3], q[4]
            )
            if len(self._robot_poses_cache) > 500:
                self._robot_poses_cache.clear()
            self._robot_poses_cache[key] = poses_array
            return [(poses_array[i, 0], poses_array[i, 1], poses_array[i, 2]) for i in range(poses_array.shape[0])]

        xc, yc, th, sx, sy = float(q[0]), float(q[1]), float(q[2]), float(q[3]), float(q[4])
        c, s = math.cos(th), math.sin(th)
        Rm = np.array([[c, -s], [s, c]], dtype=float)
        Sm = np.array([[sx, 0.0], [0.0, sy]], dtype=float)
        center = np.array([xc, yc], dtype=float)
        P = (Rm @ (Sm @ self.P_star.T)).T + center
        result = [(float(P[i, 0]), float(P[i, 1]), th) for i in range(self.Nr)]
        if len(self._robot_poses_cache) > 500:
            self._robot_poses_cache.clear()
        result_array = np.array(result, dtype=float)
        self._robot_poses_cache[key] = result_array
        return result

    def max_robot_displacement(self, q1: np.ndarray, q2: np.ndarray) -> float:
        q1 = np.asarray(q1, dtype=float)
        q2 = np.asarray(q2, dtype=float)
        dx_c = float(q2[0] - q1[0])
        dy_c = float(q2[1] - q1[1])
        centroid_dist = math.hypot(dx_c, dy_c)
        if centroid_dist < 1e-6:
            return 0.0
        if NUMBA_AVAILABLE:
            return _nb_max_robot_displacement(
                self.P_star,
                q1[0], q1[1], q1[2], q1[3], q1[4],
                q2[0], q2[1], q2[2], q2[3], q2[4]
            )
        poses1 = self.robot_poses(q1)
        poses2 = self.robot_poses(q2)
        md = 0.0
        for i in range(self.Nr):
            dx = poses2[i][0] - poses1[i][0]
            dy = poses2[i][1] - poses1[i][1]
            d = math.hypot(dx, dy)
            if d > md:
                md = d
        return md

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    @classmethod
    def _rs_like_proxy(cls, pose1, pose2, lateral_weight=4.0, turn_weight=0.35):
        if NUMBA_AVAILABLE:
            return _nb_rs_like_proxy(
                pose1[0], pose1[1], pose1[2],
                pose2[0], pose2[1], pose2[2],
                lateral_weight, turn_weight
            )
        x1, y1, th1 = pose1
        x2, y2, th2 = pose2
        dx = x2 - x1; dy = y2 - y1
        c, s = math.cos(th1), math.sin(th1)
        df = c * dx + s * dy
        dl = -s * dx + c * dy
        dth = abs(cls._wrap_angle(th2 - th1))
        return math.sqrt(df * df + lateral_weight * dl * dl) + turn_weight * dth

    def max_rs_like_distance(self, q1: np.ndarray, q2: np.ndarray) -> float:
        q1 = np.asarray(q1, dtype=float)
        q2 = np.asarray(q2, dtype=float)
        lat_w = float(getattr(self, "rs_lateral_weight", 6.0))
        turn_w = float(getattr(self, "rs_turn_weight", 0.6))
        if NUMBA_AVAILABLE:
            return _nb_max_rs_like_distance(
                self.P_star,
                q1[0], q1[1], q1[2], q1[3], q1[4],
                q2[0], q2[1], q2[2], q2[3], q2[4],
                lat_w, turn_w
            )
        poses1 = self.robot_poses(q1)
        poses2 = self.robot_poses(q2)
        best = 0.0
        for i in range(self.Nr):
            d = self._rs_like_proxy(poses1[i], poses2[i], lat_w, turn_w)
            if d > best:
                best = d
        return best

    def dist_for_nn(self, q1: np.ndarray, q2: np.ndarray) -> float:
        q1 = np.asarray(q1, dtype=float)
        q2 = np.asarray(q2, dtype=float)
        alpha = float(getattr(self, "nn_affine_weight", 0.7))
        alpha = max(0.0, min(1.0, alpha))
        lat_w = float(getattr(self, "rs_lateral_weight", 6.0))
        turn_w = float(getattr(self, "rs_turn_weight", 0.6))
        if NUMBA_AVAILABLE:
            return _nb_formation_nn_distance(
                q1, q2, self.P_star,
                self.Nx, self.Ny, self.Nxy, self.Nr,
                alpha, lat_w, turn_w
            )
        d2 = ff_dist_sq(q1, q2, self.Nx, self.Ny, self.Nxy, self.Nr)
        d_aff = float(math.sqrt(max(float(d2), 0.0)))
        d_nh = float(self.max_rs_like_distance(q1, q2))
        return alpha * d_aff + (1.0 - alpha) * d_nh


# ============================================================
# SafeIntervalMap
# ============================================================

class SafeIntervalMap:
    def __init__(self, precision: int = 3):
        self.precision = precision
        self._cache: Dict[Tuple[float, ...], List[SafeInterval]] = {}

    def clear(self):
        self._cache.clear()

    def key(self, q: np.ndarray) -> Tuple[float, ...]:
        q = np.asarray(q, dtype=float)
        return tuple(np.round(q, self.precision))

    @staticmethod
    def _disc_trajs_from_obstacle(obs: dict) -> List[List[Tuple[np.ndarray, float]]]:
        agent = obs["agent"]
        traj = obs["trajectory"]
        if not traj or len(traj) < 2:
            return []
        discs0 = agent.discs(np.asarray(traj[0][0], dtype=float))
        M = len(discs0)
        per_disc: List[List[Tuple[np.ndarray, float]]] = [[] for _ in range(M)]
        for (qk, tk, *_) in traj:
            qk = np.asarray(qk, dtype=float)
            tk = float(tk)
            dlist = agent.discs(qk)
            if len(dlist) != M:
                raise RuntimeError("Obstacle disc count changed across trajectory.")
            for j in range(M):
                per_disc[j].append((np.asarray(dlist[j][0], dtype=float)[:2], tk))
        return per_disc

    def compute(
        self,
        q: np.ndarray,
        agent,
        static_grid: OccupancyGrid,
        dynamic_obstacles: List[dict],
        T: float,
    ) -> List[SafeInterval]:
        k = self.key(q)
        if k in self._cache:
            return self._cache[k]

        my_discs = agent.discs(np.asarray(q, dtype=float))

        for (p, r) in my_discs:
            if static_grid.disc_collides(float(p[0]), float(p[1]), float(r)):
                self._cache[k] = []
                return []

        if not dynamic_obstacles:
            self._cache[k] = [SafeInterval(0.0, T)]
            return self._cache[k]

        blocked: List[Tuple[float, float]] = []

        for obs in dynamic_obstacles:
            per_disc_trajs = self._disc_trajs_from_obstacle(obs)
            if not per_disc_trajs:
                continue
            agent_obs = obs["agent"]
            obs_discs0 = agent_obs.discs(np.asarray(obs["trajectory"][0][0], dtype=float))
            obs_radii = [float(rr) for (_, rr) in obs_discs0]

            for (p_fixed, r_fixed) in my_discs:
                p_fixed = np.asarray(p_fixed, dtype=float)[:2]
                r_fixed = float(r_fixed)
                for j, traj_j in enumerate(per_disc_trajs):
                    Rsum = r_fixed + obs_radii[j]
                    blocked.extend(_blocked_intervals_fixed_disc_vs_moving_disc_traj(p_fixed, Rsum, traj_j, T))

        safe = _complement_intervals(blocked, T)
        self._cache[k] = safe
        return safe


# ============================================================
# SI-RRT core
# ============================================================

@dataclass(eq=False)
class Vertex:
    q: np.ndarray
    t: float
    si: SafeInterval
    parent: Optional["Vertex"] = None
    psi: Optional[np.ndarray] = None
    control_traj: Optional[Any] = None

    def __hash__(self):
        return id(self)


def _as_xy(q: np.ndarray) -> np.ndarray:
    return np.asarray(q, dtype=float)[:2].copy()


def interpolate_traj_q(traj: List[Tuple[np.ndarray, float]], t: float) -> Optional[np.ndarray]:
    if not traj:
        return None
    if t < traj[0][1]:
        return None
    if t >= traj[-1][1]:
        return np.asarray(traj[-1][0], dtype=float).copy()

    for i in range(len(traj) - 1):
        item0 = traj[i]
        item1 = traj[i + 1]
        q0, t0 = item0[0], item0[1]
        q1, t1 = item1[0], item1[1]
        t0, t1 = float(t0), float(t1)
        if t0 <= t <= t1:
            q0 = np.asarray(q0, dtype=float)
            q1 = np.asarray(q1, dtype=float)
            if t1 <= t0 + 1e-12:
                return q1.copy()
            a = (t - t0) / (t1 - t0)
            q = q0.copy()
            q[:2] = q0[:2] + a * (q1[:2] - q0[:2])
            if q.shape[0] >= 3:
                th0, th1 = float(q0[2]), float(q1[2])
                dth = (th1 - th0 + math.pi) % (2.0 * math.pi) - math.pi
                q[2] = th0 + a * dth
            if q.shape[0] >= 5:
                q[3] = float(q0[3] + a * (q1[3] - q0[3]))
                q[4] = float(q0[4] + a * (q1[4] - q0[4]))
            return q

    return np.asarray(traj[-1][0], dtype=float).copy()


# ============================================================
# Kinematics  —  PATCHED
# ============================================================

class Kinematics:
    def __init__(self, max_velocity: float):
        # max_velocity is kept as a fallback upper bound.
        # Actual travel times are derived from the agent's own limits.
        self.vmax = float(max_velocity)
        import os as _osk; print(f'[DIAG] Kinematics.__init__: self.vmax={self.vmax}  file={_osk.path.abspath(__file__)}')

    # ------------------------------------------------------------------
    # Helper: resolve the correct effective speed for an agent
    # ------------------------------------------------------------------
    def _agent_effective_vmax(self, agent) -> float:
        """
        Resolve effective maximum speed from the agent object.

        Priority:
          1. agent.v_max_list  (HeterogeneousFormationAgent)
                → min(v_max_list)  [slowest member governs]
          2. agent.v_max  (DifferentialDriveAgent / HolonomicAgent)
          3. self.vmax    (planner-level fallback — only if agent
                           exposes no velocity information)
        """
        if hasattr(agent, 'v_max_list'):
            lst = agent.v_max_list
            return max(min(float(v) for v in lst), 1e-9)

        raw = getattr(agent, 'v_max', None)
        if raw is not None and not isinstance(raw, (list, tuple, np.ndarray)):
            return max(float(raw), 1e-9)

        # Fallback: use planner-level bound
        return max(self.vmax, 1e-9)

    # ------------------------------------------------------------------
    # Core method: travel_time
    # ------------------------------------------------------------------
    def travel_time(self, agent, q_from: np.ndarray, q_to: np.ndarray) -> float:
        """
        Compute the minimum time to travel between two configurations,
        respecting per-robot (or per-formation-member) velocity limits.

        Branch selection uses hasattr(agent, 'P_star') as the discriminator
        between formation agents and individual robots.  This is critical
        because DifferentialDriveAgent and HolonomicAgent both define
        max_robot_displacement() and robot_poses(), so attribute presence
        alone cannot distinguish them from formation agents.

        Formations (agent has P_star):
            With v_max_list  →  t = max_i( d_i / v_max_list[i] )
            Without v_max_list → t = max_robot_displacement / effective_vmax

        Individual robots (no P_star):
            t = euclidean_distance / agent.v_max
        """
        q_from = np.asarray(q_from, dtype=float)
        q_to   = np.asarray(q_to,   dtype=float)

        is_formation = hasattr(agent, 'P_star')

        if is_formation:
            # --- Heterogeneous formation: per-robot displacement / per-robot v_max ---
            if hasattr(agent, 'v_max_list') and hasattr(agent, 'robot_poses'):
                try:
                    poses1 = agent.robot_poses(q_from)
                    poses2 = agent.robot_poses(q_to)
                    t_needed = 0.0
                    for i, ((x1, y1, _), (x2, y2, _)) in enumerate(zip(poses1, poses2)):
                        d_i    = math.hypot(x2 - x1, y2 - y1)
                        vmax_i = max(float(agent.v_max_list[i]), 1e-9)
                        t_needed = max(t_needed, d_i / vmax_i)
                    return max(t_needed, 0.0)
                except Exception:
                    pass  # fall through to centroid-based fallback

            # --- Homogeneous / legacy formation ---
            if hasattr(agent, "max_robot_displacement"):
                d = float(agent.max_robot_displacement(q_from, q_to))
                return d / self._agent_effective_vmax(agent)

        # --- Individual robot (DifferentialDrive, Holonomic, or unknown) ---
        # Always reached for non-formation agents regardless of which
        # helper methods they define.
        p0 = q_from[:2]
        p1 = q_to[:2]
        d  = float(np.linalg.norm(p1 - p0))
        return d / self._agent_effective_vmax(agent)


# ============================================================
# SIRRT planner  (unchanged from original — only Kinematics changed above)
# ============================================================

class SIRRT:
    def __init__(
        self,
        agent_model,
        max_velocity: float,
        workspace_bounds: Tuple[np.ndarray, np.ndarray],
        static_grid: OccupancyGrid,
        time_horizon: float = 60.0,
        max_iter: int = 1200,
        d_max: float = 0.05,
        goal_sample_rate: float = 0.2,
        neighbor_radius: float = 1.2,
        precision: int = 2,
        seed: Optional[int] = None,
        debug: bool = False,
        use_kinodynamic: bool = False,
        kinodynamic_params: Optional[dict] = None,
    ):
        self.agent = agent_model
        self.kin = Kinematics(max_velocity)
        self.bounds_xy = (
            np.asarray(workspace_bounds[0], dtype=float),
            np.asarray(workspace_bounds[1], dtype=float),
        )
        self.static_grid = static_grid
        self.T = float(time_horizon)
        self.max_iter = int(max_iter)
        self.d_max = float(d_max)
        self.goal_sample_rate = float(goal_sample_rate)
        self.neighbor_radius = float(neighbor_radius)
        self.si_map = SafeIntervalMap(precision=precision)
        self.precision = precision
        self.V: List[Vertex] = []
        self.debug = bool(debug)

        if seed is not None:
            np.random.seed(seed)

        self._kdtree = None
        self._V_xy = None
        self._kdtree_size = 0
        self._kdtree_rebuild_threshold = 300
        self._by_qkey: Dict[Tuple[float, ...], List[Vertex]] = {}
        self._static_ok_cache: Dict[Tuple[float, ...], bool] = {}

        _raw_radius = getattr(self.agent, "radius", 0.1)
        if isinstance(_raw_radius, (list, tuple, np.ndarray)):
            self._agent_radius = float(max(_raw_radius))
        else:
            self._agent_radius = float(_raw_radius)

        self.use_kinodynamic = False
        self.kino_steerer = None
        self.individual_kino_steerer = None

        if use_kinodynamic and hasattr(agent_model, 'P_star'):
            is_heterogeneous = hasattr(agent_model, 'robot_types')

            if is_heterogeneous and HETEROGENEOUS_KINO_AVAILABLE:
                try:
                    kino_defaults = {
                        'robot_types': agent_model.robot_types,
                        'v_max': max_velocity,
                        'w_max': 2.0,
                        'a_max': 2.0,
                        'alpha_max': 4.0,
                        'vc_max': max_velocity,
                        'wth_max': 1.2,
                        'ds_max': 0.5,
                        'N_steer': 8,
                        'T_steer': 0.8,
                        'w_reach': 10.0,
                        'w_energy': 1.0,
                        'w_smooth': 0.1,
                        'max_iter': 200,
                        'warm_start': True,
                    }
                    if hasattr(agent_model, 'v_max_list'):
                        kino_defaults['v_max'] = agent_model.v_max_list
                    if hasattr(agent_model, 'omega_max_list'):
                        kino_defaults['w_max'] = agent_model.omega_max_list
                    if hasattr(agent_model, 'a_max_list'):
                        kino_defaults['a_max'] = agent_model.a_max_list
                    if hasattr(agent_model, 'alpha_max_list'):
                        kino_defaults['alpha_max'] = agent_model.alpha_max_list
                    if kinodynamic_params:
                        kino_defaults.update(kinodynamic_params)
                    if 'robot_types' not in kino_defaults:
                        kino_defaults['robot_types'] = agent_model.robot_types

                    self.kino_steerer = HeterogeneousKinodynamicFormationSteering(
                        P_star=agent_model.P_star,
                        **kino_defaults
                    )
                    self.use_kinodynamic = True
                    if self.debug:
                        print(f"✓ Heterogeneous formation steering enabled")
                except Exception as e:
                    if self.debug:
                        print(f"⚠ Failed to initialize heterogeneous formation steering: {e}")
                    is_heterogeneous = False

            if not is_heterogeneous or not HETEROGENEOUS_KINO_AVAILABLE:
                try:
                    import sys, os
                    try:
                        from kinodynamic_steering_with_full_output import KinodynamicFormationSteering
                        KINO_AVAILABLE = True
                    except ImportError:
                        KINO_AVAILABLE = False

                    if KINO_AVAILABLE:
                        kino_defaults = {
                            'v_max': max_velocity,
                            'w_max': 2.0,
                            'a_max': 2.0,
                            'alpha_max': 4.0,
                            'vc_max': max_velocity,
                            'wth_max': 1.2,
                            'ds_max': 0.5,
                            'N_steer': 8,
                            'T_steer': 0.8,
                            'w_reach': 10.0,
                            'w_energy': 1.0,
                            'w_smooth': 0.1,
                            'max_iter': 200,
                            'warm_start': True,
                        }
                        if kinodynamic_params:
                            kino_defaults.update(kinodynamic_params)
                            kino_defaults.pop('robot_types', None)
                        self.kino_steerer = KinodynamicFormationSteering(
                            P_star=agent_model.P_star,
                            **kino_defaults
                        )
                        self.use_kinodynamic = True
                except Exception as e:
                    if self.debug:
                        print(f"⚠ Failed to initialize formation kinodynamic steering: {e}")

        elif use_kinodynamic and not hasattr(agent_model, 'P_star'):
            if INDIVIDUAL_KINO_AVAILABLE:
                agent_type = None
                if hasattr(agent_model, 'is_holonomic'):
                    agent_type = 'holonomic' if agent_model.is_holonomic else 'diff-drive'
                else:
                    sample_q = agent_model.sample_q(self.bounds_xy)
                    agent_type = 'holonomic' if len(sample_q) < 3 else 'diff-drive'

                individual_kino_params = {
                    'v_max': self.kin._agent_effective_vmax(agent_model),
                    'a_max': 2.0,
                    'dt': 0.05,
                }
                if hasattr(agent_model, 'a_max'):
                    individual_kino_params['a_max'] = agent_model.a_max
                if agent_type == 'diff-drive':
                    individual_kino_params['v_min'] = -self.kin._agent_effective_vmax(agent_model) * 0.5
                    individual_kino_params['omega_max'] = getattr(agent_model, 'omega_max', 2.0)
                    individual_kino_params['alpha_max'] = getattr(agent_model, 'alpha_max', 4.0)
                    individual_kino_params['max_time'] = 5.0
                if kinodynamic_params:
                    individual_kino_params.update(kinodynamic_params)

                try:
                    self.individual_kino_steerer = create_steering_for_agent(agent_type, **individual_kino_params)
                    self.use_kinodynamic = True
                    if self.debug:
                        print(f"✓ Individual agent kinodynamic steering enabled ({agent_type})")
                except Exception as e:
                    if self.debug:
                        print(f"⚠ Failed to initialize individual kinodynamic steering: {e}")

    def _reset_per_plan_caches(self):
        self.si_map.clear()
        self._by_qkey = {}
        self._static_ok_cache = {}
        if hasattr(self.agent, "clear_cache") and callable(getattr(self.agent, "clear_cache")):
            self.agent.clear_cache()

    def _rebuild_kdtree(self):
        if not KDTREE_AVAILABLE or len(self.V) < 20:
            self._kdtree = None
            return
        self._V_xy = np.array([self.agent.centroid_xy(v.q) for v in self.V], dtype=np.float64)
        self._kdtree = cKDTree(self._V_xy)
        self._kdtree_size = len(self.V)

    def _maybe_rebuild_kdtree(self):
        if KDTREE_AVAILABLE and len(self.V) - self._kdtree_size >= self._kdtree_rebuild_threshold:
            self._rebuild_kdtree()

    def _qkey(self, q: np.ndarray) -> Tuple[float, ...]:
        return tuple(np.round(np.asarray(q, dtype=float), self.precision))

    def _si_key(self, si: SafeInterval, nd: int = 9) -> Tuple[float, float]:
        return (round(float(si.low), nd), round(float(si.high), nd))

    def debug_check_same_q_multiple_SIs(self) -> bool:
        from collections import defaultdict
        by_q = defaultdict(set)
        for v in self.V:
            by_q[self._qkey(v.q)].add(self._si_key(v.si))
        any_multi = any(len(sis) >= 2 for sis in by_q.values())
        print(f"[DEBUG] Any configuration with >=2 distinct SIs? {any_multi}")
        return any_multi

    def debug_dump_vertices(self, top_k: int = 30) -> None:
        from collections import defaultdict
        by_q = defaultdict(list)
        for v in self.V:
            by_q[self._qkey(v.q)].append(v)
        multi = [(k, vs) for k, vs in by_q.items() if len(vs) > 1]
        multi.sort(key=lambda kv: len(kv[1]), reverse=True)
        print(f"\n[DEBUG] Total vertices: {len(self.V)}")
        print(f"[DEBUG] Unique qkeys: {len(by_q)}")
        print(f"[DEBUG] qkeys with >1 vertex: {len(multi)}")
        for idx, (k, vs) in enumerate(multi[:top_k]):
            vs_sorted = sorted(vs, key=lambda v: v.t)
            si_groups = defaultdict(list)
            for v in vs_sorted:
                si_groups[self._si_key(v.si)].append(v)
            print(f"\n[DEBUG] qkey #{idx+1} = {k}  |  vertices={len(vs)}  |  distinct_SIs={len(si_groups)}")
            for sik, g in sorted(si_groups.items(), key=lambda x: x[0]):
                times = [vv.t for vv in g]
                times_sorted = sorted(times)
                med = times_sorted[len(times_sorted)//2]
                print(f"  SI={sik}  count={len(g)}  t(min/med/max)={min(times):.3f}/{med:.3f}/{max(times):.3f}")

    def debug_check_near_duplicate_SIs(self, eps: float = 1e-8) -> None:
        from collections import defaultdict
        by_q = defaultdict(list)
        for v in self.V:
            by_q[self._qkey(v.q)].append(v.si)
        print("\n[DEBUG] Near-duplicate SI scan:")
        for k, sis in by_q.items():
            if len(sis) < 2:
                continue
            sis_sorted = sorted(sis, key=lambda s: (s.low, s.high))
            for a, b in zip(sis_sorted[:-1], sis_sorted[1:]):
                if (abs(float(a.low) - float(b.low)) < eps and abs(float(a.high) - float(b.high)) < eps
                        and (a.low != b.low or a.high != b.high)):
                    print(f"  qkey={k} near-duplicate SIs: ({a.low:.12f},{a.high:.12f}) vs ({b.low:.12f},{b.high:.12f})")
                    break

    def _nn_dist(self, q1: np.ndarray, q2: np.ndarray) -> float:
        if hasattr(self.agent, "dist_for_nn"):
            return self.agent.dist_for_nn(q1, q2)
        if NUMBA_AVAILABLE:
            return math.sqrt(_nb_dist_sq_xy(q1[0], q1[1], q2[0], q2[1]))
        dx = q2[0] - q1[0]; dy = q2[1] - q1[1]
        return math.hypot(dx, dy)

    def _sample(self, q_goal: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.goal_sample_rate:
            return q_goal.copy()
        return self.agent.sample_q(self.bounds_xy)

    def _nearest(self, q: np.ndarray) -> Vertex:
        if self._kdtree is not None and len(self.V) >= 20:
            q_xy = self.agent.centroid_xy(q)
            k = min(len(self.V), max(30, len(self.V) // 4))
            _, indices = self._kdtree.query(q_xy, k=k)
            if np.isscalar(indices):
                indices = [indices]
            best_v = None; best_d = float("inf")
            for i in indices:
                if i < len(self.V):
                    v = self.V[i]
                    d = self._nn_dist(v.q, q)
                    if d < best_d:
                        best_d = d; best_v = v
            if best_v is not None:
                return best_v
        return min(self.V, key=lambda v: self._nn_dist(v.q, q))

    def _neighbors(self, q: np.ndarray) -> List[Vertex]:
        if self._kdtree is not None and len(self.V) >= 20:
            q_xy = self.agent.centroid_xy(q)
            indices = self._kdtree.query_ball_point(q_xy, self.neighbor_radius * 2.0)
            out = []
            for i in indices:
                if i < len(self.V):
                    v = self.V[i]
                    if self._nn_dist(v.q, q) <= self.neighbor_radius:
                        out.append(v)
            return out
        return [v for v in self.V if self._nn_dist(v.q, q) <= self.neighbor_radius]

    def _steer(self, q_from: np.ndarray, q_to: np.ndarray,
               psi_from: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        q_from = np.asarray(q_from, dtype=float)
        q_to   = np.asarray(q_to,   dtype=float)

        if self.use_kinodynamic and self.kino_steerer is not None and q_from.shape[0] >= 5:
            try:
                q_new, psi_new = self.kino_steerer.steer(q_from, q_to, psi_from)
                if q_new is not None:
                    return q_new, psi_new
            except Exception as e:
                if self.debug:
                    print(f"⚠ Kinodynamic steering failed: {e}, using fallback")

        if q_from.shape[0] < 5 or not hasattr(self.agent, "max_robot_displacement"):
            if self.use_kinodynamic and self.individual_kino_steerer is not None:
                try:
                    q_new, control_traj = self.individual_kino_steerer.steer(
                        q_from, q_to, step_size=self.d_max
                    )
                    return q_new, control_traj
                except Exception as e:
                    if self.debug:
                        print(f"⚠ Individual kinodynamic steering failed: {e}, using fallback")
            p = self.agent.centroid_xy(q_from)
            g = self.agent.centroid_xy(q_to)
            if NUMBA_AVAILABLE:
                L = math.sqrt(_nb_dist_sq_xy(p[0], p[1], g[0], g[1]))
            else:
                L = float(np.linalg.norm(g - p))
            if L <= self.d_max + 1e-12:
                return q_to.copy(), None
            alpha = self.d_max / (L + 1e-12)
            return self.agent.interpolate_q(q_from, q_to, alpha), None

        d_eff = float(self.agent.max_robot_displacement(q_from, q_to))
        if d_eff <= self.d_max + 1e-12:
            psi_fallback = psi_from.copy() if psi_from is not None else None
            return q_to.copy(), psi_fallback

        a = self.d_max / (d_eff + 1e-12)
        a = max(0.0, min(1.0, a))

        xc, yc, th, sx, sy = float(q_from[0]), float(q_from[1]), float(q_from[2]), float(q_from[3]), float(q_from[4])
        xg, yg, thg, sxg, syg = float(q_to[0]), float(q_to[1]), float(q_to[2]), float(q_to[3]), float(q_to[4])

        def wrap(a_):
            return (a_ + math.pi) % (2.0 * math.pi) - math.pi

        dth = wrap(thg - th)
        kappa_max = float(getattr(self.agent, "kappa_max", 1.0))
        ds_eff = self.d_max
        max_dtheta = kappa_max * ds_eff
        dth_step = max(-max_dtheta, min(max_dtheta, dth))
        th_new = wrap(th + dth_step)

        dx = xg - xc; dy = yg - yc
        dist_xy = math.hypot(dx, dy)
        sign = 1.0
        if dist_xy > 1e-9:
            desired = math.atan2(dy, dx)
            if math.cos(wrap(desired - th_new)) < 0.0:
                sign = -1.0

        step_xy = min(self.d_max, dist_xy)
        xc_new = xc + sign * step_xy * math.cos(th_new) * a
        yc_new = yc + sign * step_xy * math.sin(th_new) * a
        sx_new = sx + a * (sxg - sx)
        sy_new = sy + a * (syg - sy)

        q_new = np.array([xc_new, yc_new, th_new, sx_new, sy_new], dtype=float)
        psi_fallback = None
        if psi_from is not None:
            psi_fallback = psi_from + a * (th_new - psi_from)
        return q_new, psi_fallback

    def _static_ok(self, q: np.ndarray) -> bool:
        k = self._qkey(q)
        cached = self._static_ok_cache.get(k, None)
        if cached is not None:
            return cached
        ok = True
        for (p, r) in self.agent.discs(q):
            if self.static_grid.disc_collides(float(p[0]), float(p[1]), float(r)):
                ok = False
                break
        self._static_ok_cache[k] = ok
        return ok

    def _edge_samples_count(self, q_from: np.ndarray, q_to: np.ndarray, step: float, min_n: int = 3) -> int:
        p0 = self.agent.centroid_xy(q_from)
        p1 = self.agent.centroid_xy(q_to)
        if NUMBA_AVAILABLE:
            L = math.sqrt(_nb_dist_sq_xy(p0[0], p0[1], p1[0], p1[1]))
        else:
            L = float(np.linalg.norm(p1 - p0))
        if step <= 1e-12:
            return 14
        return max(min_n, int(L / step))

    def _edge_static_ok(self, q_from: np.ndarray, q_to: np.ndarray) -> bool:
        step = 0.5 * max(self._agent_radius, 1e-6)
        n = self._edge_samples_count(q_from, q_to, step=step, min_n=3)
        for k in range(n + 1):
            a = k / n
            q_mid = self.agent.interpolate_q(q_from, q_to, a)
            if not self._static_ok(q_mid):
                return False
        return True

    def _dynamic_ok(self, q: np.ndarray, t: float, dynamic_obstacles: List[dict]) -> bool:
        my_discs = self.agent.discs(q)
        for obs in dynamic_obstacles:
            oq = interpolate_traj_q(obs["trajectory"], t)
            if oq is None:
                continue
            other_discs = obs["agent"].discs(oq)
            for (p, r) in my_discs:
                px, py = float(p[0]), float(p[1])
                for (op, orad) in other_discs:
                    if NUMBA_AVAILABLE:
                        if _nb_discs_overlap(px, py, float(r), float(op[0]), float(op[1]), float(orad)):
                            return False
                    else:
                        if float(np.linalg.norm(p - op)) < (float(r) + float(orad)):
                            return False
        return True

    def _edge_dynamic_ok(self, q_from, q_to, t_depart, travel, dynamic_obstacles):
        if not dynamic_obstacles:
            return True
        step = 0.5 * max(self._agent_radius, 1e-6)
        n = self._edge_samples_count(q_from, q_to, step=step, min_n=3)
        for k in range(n + 1):
            a = k / n
            t = t_depart + a * travel
            q_mid = self.agent.interpolate_q(q_from, q_to, a)
            if not self._dynamic_ok(q_mid, t, dynamic_obstacles):
                return False
        return True

    def _dominates(self, v_old: Vertex, v_new: Vertex) -> bool:
        if self._qkey(v_old.q) != self._qkey(v_new.q):
            return False
        if abs(v_old.si.low - v_new.si.low) > 1e-9:
            return False
        if abs(v_old.si.high - v_new.si.high) > 1e-9:
            return False
        return v_old.t <= v_new.t + 1e-9

    def _is_dominated(self, v_new: Vertex) -> bool:
        k = self._qkey(v_new.q)
        bucket = self._by_qkey.get(k, None)
        if not bucket:
            return False
        for v in bucket:
            if self._dominates(v, v_new):
                return True
        return False

    def _register_vertex(self, v: Vertex) -> None:
        k = self._qkey(v.q)
        if k in self._by_qkey:
            self._by_qkey[k].append(v)
        else:
            self._by_qkey[k] = [v]

    def _earliest_arrival(self, v_from, q_to, si_to, dynamic_obstacles):
        travel = self.kin.travel_time(self.agent, v_from.q, q_to)
        dep_low  = max(v_from.t, v_from.si.low, si_to.low - travel)
        dep_high = min(v_from.si.high, si_to.high - travel)
        if dep_low >= dep_high - 1e-9:
            return None
        t_depart = dep_low
        t_arr = t_depart + travel
        if t_arr > self.T:
            return None
        if not self._edge_dynamic_ok(v_from.q, q_to, t_depart, travel, dynamic_obstacles):
            return None
        return t_arr

    def plan(self, q_start, q_goal, dynamic_obstacles):
        q_start = np.asarray(q_start, dtype=float)
        q_goal  = np.asarray(q_goal,  dtype=float)

        self._reset_per_plan_caches()

        if not self._static_ok(q_start):
            if self.debug: print("[DEBUG] start statically invalid")
            return None
        if not self._static_ok(q_goal):
            if self.debug: print("[DEBUG] goal statically invalid")
            return None

        si_start_list = self.si_map.compute(q_start, self.agent, self.static_grid, dynamic_obstacles, self.T)
        if not si_start_list:
            if self.debug: print("[DEBUG] start has no safe intervals")
            return None

        si0 = None
        for si in si_start_list:
            if si.low <= 0.0 < si.high:
                si0 = si
                break
        if si0 is None:
            if self.debug: print("[DEBUG] start not safe at t=0")
            return None

        psi_start = None
        if q_start.shape[0] >= 5 and hasattr(self.agent, 'P_star'):
            psi_start = np.full(self.agent.Nr, q_start[2], dtype=float)

        v0 = Vertex(q=q_start.copy(), t=0.0, si=si0, parent=None, psi=psi_start)
        self.V = [v0]
        self._register_vertex(v0)
        self._kdtree = None
        self._kdtree_size = 0

        best_goal: Optional[Vertex] = None
        best_goal_t = float("inf")
        q_goal_xy = _as_xy(q_goal)

        for _it in range(self.max_iter):
            q_rand = self._sample(q_goal)
            v_near = self._nearest(q_rand)

            if q_rand.shape[0] >= 5:
                dx = float(q_rand[0] - v_near.q[0])
                dy = float(q_rand[1] - v_near.q[1])
                if abs(dx) + abs(dy) > 1e-9:
                    th_hint = math.atan2(dy, dx)
                    th_hint += np.random.normal(0.0, 0.35)
                    q_rand = np.asarray(q_rand, dtype=float).copy()
                    q_rand[2] = (th_hint + math.pi) % (2.0 * math.pi) - math.pi

            q_new, psi_new = self._steer(v_near.q, q_rand, psi_from=v_near.psi)

            control_traj = None
            if psi_new is not None and not isinstance(psi_new, np.ndarray):
                control_traj = psi_new
                psi_new = None

            if not self._edge_static_ok(v_near.q, q_new):
                continue

            si_list = self.si_map.compute(q_new, self.agent, self.static_grid, dynamic_obstacles, self.T)
            if not si_list:
                continue

            neighbors = self._neighbors(q_new) or [v_near]

            for si_to in si_list:
                best_parent = None; best_arr = None

                for v in neighbors:
                    if not self._edge_static_ok(v.q, q_new):
                        continue
                    t_arr = self._earliest_arrival(v, q_new, si_to, dynamic_obstacles)
                    if t_arr is None:
                        continue
                    if best_arr is None or t_arr < best_arr:
                        best_arr = t_arr; best_parent = v

                if best_parent is None or best_arr is None:
                    continue

                v_new = Vertex(q=q_new.copy(), t=best_arr, si=si_to, parent=best_parent,
                               psi=psi_new, control_traj=control_traj)

                if self._is_dominated(v_new):
                    continue

                self.V.append(v_new)
                self._register_vertex(v_new)
                self._maybe_rebuild_kdtree()

                if not dynamic_obstacles:
                    v_cur = v_new
                    q_cur = q_new.copy()
                    psi_cur = psi_new
                    t_cur = float(v_new.t)
                    max_extend = int(max(5, min(80, math.ceil(self._nn_dist(q_cur, q_goal) / max(self.d_max, 1e-6)) + 5)))

                    for _j in range(max_extend):
                        if self._nn_dist(q_cur, q_goal) <= 1e-6:
                            break
                        q_next, psi_next = self._steer(q_cur, q_goal, psi_from=psi_cur)
                        control_traj_next = None
                        if psi_next is not None and not isinstance(psi_next, np.ndarray):
                            control_traj_next = psi_next; psi_next = None

                        if np.allclose(q_next, q_cur, atol=1e-9, rtol=0.0):
                            break
                        if not self._edge_static_ok(q_cur, q_next):
                            break

                        travel = self.kin.travel_time(self.agent, q_cur, q_next)
                        t_next = t_cur + travel
                        if t_next > self.T:
                            break

                        si_list_next = self.si_map.compute(q_next, self.agent, self.static_grid, [], self.T)
                        if not si_list_next:
                            break
                        si_next = si_list_next[0]
                        for si_cand in si_list_next:
                            if si_cand.low - 1e-9 <= t_next < si_cand.high + 1e-9:
                                si_next = si_cand; break

                        v_next = Vertex(q=q_next.copy(), t=float(t_next), si=si_next, parent=v_cur,
                                        psi=psi_next, control_traj=control_traj_next)
                        q_cur = q_next; psi_cur = psi_next
                        self.V.append(v_next)
                        self._register_vertex(v_next)
                        self._maybe_rebuild_kdtree()
                        v_cur = v_next; t_cur = float(t_next)

                        if self._nn_dist(q_cur, q_goal) <= self.d_max + 1e-9:
                            q_next2, _ = self._steer(q_cur, q_goal)
                            if q_next2 is not None and self._edge_static_ok(q_cur, q_next2):
                                travel2 = self.kin.travel_time(self.agent, q_cur, q_goal)
                                t_goal = t_cur + travel2
                                if t_goal <= self.T:
                                    si_goal_list = self.si_map.compute(q_goal, self.agent, self.static_grid, [], self.T)
                                    if si_goal_list:
                                        si_g = si_goal_list[0]
                                        for si_cand in si_goal_list:
                                            if si_cand.low - 1e-9 <= t_goal < si_cand.high + 1e-9:
                                                si_g = si_cand; break
                                        if t_goal < best_goal_t:
                                            best_goal_t = t_goal
                                            best_goal = Vertex(q=q_goal.copy(), t=float(t_goal), si=si_g,
                                                               parent=v_cur, psi=psi_cur, control_traj=None)
                            break
                else:
                    q_new_xy = _as_xy(q_new)
                    if NUMBA_AVAILABLE:
                        close_enough = (_nb_dist_sq_xy(q_new_xy[0], q_new_xy[1], q_goal_xy[0], q_goal_xy[1]) <= 0.49)
                    else:
                        close_enough = (np.linalg.norm(q_new_xy - q_goal_xy) <= 0.7)

                    if close_enough and self._edge_static_ok(q_new, q_goal):
                        si_goal_list = self.si_map.compute(q_goal, self.agent, self.static_grid, dynamic_obstacles, self.T)
                        for si_g in si_goal_list:
                            t_goal = self._earliest_arrival(v_new, q_goal, si_g, dynamic_obstacles)
                            if t_goal is not None and t_goal < best_goal_t:
                                best_goal_t = t_goal
                                best_goal = Vertex(q=q_goal.copy(), t=float(t_goal), si=si_g,
                                                   parent=v_new, psi=psi_new, control_traj=None)
                                if _it > 500 and t_goal < self.T * 0.5:
                                    break

            if best_goal is not None and _it > 500 and best_goal_t < self.T * 0.5:
                break

        if best_goal is None:
            if self.debug:
                self.debug_check_same_q_multiple_SIs()
                self.debug_dump_vertices(top_k=20)
                self.debug_check_near_duplicate_SIs(eps=1e-8)
            return None

        out: List[Tuple[np.ndarray, float]] = []
        v = best_goal
        while v is not None:
            psi_to_save = None
            control_traj_to_save = v.control_traj
            if v.psi is not None:
                if isinstance(v.psi, np.ndarray):
                    psi_to_save = np.array(v.psi, dtype=float).copy()
                elif INDIVIDUAL_KINO_AVAILABLE and isinstance(v.psi, ControlTrajectory):
                    control_traj_to_save = v.psi; psi_to_save = None
                elif hasattr(v.psi, 'x'):
                    control_traj_to_save = v.psi; psi_to_save = None
                else:
                    try:
                        psi_to_save = np.array(v.psi, dtype=float).copy()
                    except:
                        psi_to_save = None
            out.append((v.q.copy(), float(v.t), psi_to_save, control_traj_to_save))
            v = v.parent
        out.reverse()

        if self.debug:
            self.debug_check_same_q_multiple_SIs()
            self.debug_dump_vertices(top_k=20)
            self.debug_check_near_duplicate_SIs(eps=1e-8)

        return out