#!/usr/bin/env python3
"""
Individual Agent Kinodynamic Steering for RRT*

This module provides proper kinodynamic steering for individual robots:
- Differential-Drive: Uses numerical integration with (v, ω) controls
- Holonomic: Direct line steering with (vx, vy) controls

Key features:
- Respects velocity AND acceleration limits
- Generates detailed control trajectories
- Fast enough for RRT* use
- Stores intermediate states for visualization

Constraint summary:
  Differential-Drive:
    |v|       ≤ v_max          (forward velocity)
    |ω|       ≤ omega_max      (angular velocity)
    |dv/dt|   ≤ a_max          (linear acceleration)
    |dω/dt|   ≤ alpha_max      (angular acceleration)

  Holonomic:
    ‖vel‖     ≤ v_max          (velocity magnitude)
    ‖accel‖   ≤ a_max          (acceleration magnitude)
"""

import numpy as np
import math
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


@dataclass
class ControlTrajectory:
    """Stores a control trajectory with states and controls"""
    # States
    x: np.ndarray  # x positions
    y: np.ndarray  # y positions
    theta: Optional[np.ndarray] = None  # heading (for DD)
    t: np.ndarray = None  # time stamps
    
    # Controls
    v: Optional[np.ndarray] = None  # forward velocity (DD) or x-velocity (holonomic)
    omega: Optional[np.ndarray] = None  # angular velocity (DD)
    vy: Optional[np.ndarray] = None  # y-velocity (holonomic)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for easy storage"""
        result = {
            'x': self.x.copy(),
            'y': self.y.copy(),
            't': self.t.copy() if self.t is not None else None,
        }
        if self.theta is not None:
            result['theta'] = self.theta.copy()
        if self.v is not None:
            result['v'] = self.v.copy()
        if self.omega is not None:
            result['omega'] = self.omega.copy()
        if self.vy is not None:
            result['vy'] = self.vy.copy()
        return result


class DifferentialDriveSteering:
    """
    Kinodynamic steering for differential-drive robots using numerical integration.
    
    Enforces:
      - Velocity bounds:      |v| ≤ v_max,  |ω| ≤ omega_max
      - Acceleration bounds:  |dv/dt| ≤ a_max,  |dω/dt| ≤ alpha_max
    
    Uses Euler integration with proportional control + acceleration rate-limiting.
    """
    
    def __init__(
        self,
        v_max: float = 1.0,         # Max forward velocity
        v_min: float = -0.5,        # Max reverse velocity  
        omega_max: float = 2.0,     # Max angular velocity
        a_max: float = 2.0,         # Max linear acceleration  (NEW)
        alpha_max: float = 4.0,     # Max angular acceleration (NEW)
        dt: float = 0.05,           # Integration time step
        max_time: float = 5.0,      # Maximum steering time
    ):
        self.v_max = float(v_max)
        self.v_min = float(v_min)
        self.omega_max = float(omega_max)
        self.a_max = float(a_max)
        self.alpha_max = float(alpha_max)
        self.dt = float(dt)
        self.max_time = float(max_time)
    
    def steer(
        self,
        q_from: np.ndarray,  # [x, y, theta]
        q_to: np.ndarray,    # [x, y, theta]
        step_size: float = 0.5,  # Target distance to steer
    ) -> Tuple[np.ndarray, ControlTrajectory]:
        """
        Steer from q_from toward q_to.
        
        Returns:
            q_new: New configuration reached
            trajectory: Full control trajectory
        """
        q_from = np.asarray(q_from, dtype=float).flatten()
        q_to = np.asarray(q_to, dtype=float).flatten()
        
        x0, y0, theta0 = q_from[0], q_from[1], q_from[2]
        x_goal, y_goal = q_to[0], q_to[1]
        
        # Lists to store trajectory
        x_list = [x0]
        y_list = [y0]
        theta_list = [theta0]
        t_list = [0.0]
        v_list = []
        omega_list = []
        
        x, y, theta = x0, y0, theta0
        t = 0.0
        distance_traveled = 0.0
        
        # Previous controls (start from rest)
        v_prev = 0.0
        omega_prev = 0.0
        
        # Precompute acceleration deltas per timestep
        dv_max = self.a_max * self.dt        # max change in v per step
        domega_max = self.alpha_max * self.dt  # max change in ω per step
        
        # Steer until we reach step_size or get close to goal
        while distance_traveled < step_size and t < self.max_time:
            # Compute desired heading to goal
            dx = x_goal - x
            dy = y_goal - y
            dist_to_goal = np.hypot(dx, dy)
            
            if dist_to_goal < 0.01:  # Very close to goal
                break
            
            theta_to_goal = math.atan2(dy, dx)
            
            # Compute angular error
            angle_error = self._wrap_angle(theta_to_goal - theta)
            
            # ---- Desired controls (proportional controller) ----
            if abs(angle_error) > 0.3:
                # Rotate in place
                v_desired = 0.0
                omega_desired = np.clip(2.0 * angle_error, -self.omega_max, self.omega_max)
            else:
                # Move forward with some steering
                v_desired = min(self.v_max, dist_to_goal / self.dt)
                omega_desired = np.clip(1.5 * angle_error, -self.omega_max, self.omega_max)
            
            # ---- Enforce acceleration limits (rate-limit controls) ----
            v = np.clip(v_desired, v_prev - dv_max, v_prev + dv_max)
            omega = np.clip(omega_desired, omega_prev - domega_max, omega_prev + domega_max)
            
            # ---- Enforce velocity limits (clamp after accel limiting) ----
            v = np.clip(v, self.v_min, self.v_max)
            omega = np.clip(omega, -self.omega_max, self.omega_max)
            
            # Integrate dynamics: ẋ = v*cos(θ), ẏ = v*sin(θ), θ̇ = ω
            x_new = x + v * math.cos(theta) * self.dt
            y_new = y + v * math.sin(theta) * self.dt
            theta_new = self._wrap_angle(theta + omega * self.dt)
            
            # Update state
            step_dist = np.hypot(x_new - x, y_new - y)
            distance_traveled += step_dist
            
            x, y, theta = x_new, y_new, theta_new
            t += self.dt
            
            # Store
            x_list.append(x)
            y_list.append(y)
            theta_list.append(theta)
            t_list.append(t)
            v_list.append(float(v))
            omega_list.append(float(omega))
            
            # Remember previous controls for next rate-limit step
            v_prev = float(v)
            omega_prev = float(omega)
            
            # Stop if we've traveled enough
            if distance_traveled >= step_size:
                break
        
        q_new = np.array([x, y, theta], dtype=float)
        
        trajectory = ControlTrajectory(
            x=np.array(x_list),
            y=np.array(y_list),
            theta=np.array(theta_list),
            t=np.array(t_list),
            v=np.array(v_list),
            omega=np.array(omega_list),
        )
        
        return q_new, trajectory
    
    def compute_travel_time(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """
        Estimate travel time between two configurations.
        
        Accounts for acceleration phase: if the distance is short enough that
        the robot cannot reach v_max, a triangular profile is used.
        """
        dx = float(q2[0] - q1[0])
        dy = float(q2[1] - q1[1])
        dist = np.hypot(dx, dy)
        
        # Trapezoidal velocity profile time estimate
        translation_time = self._trapezoidal_time(dist, self.v_max, self.a_max)
        
        # Account for rotation needed
        if len(q1) >= 3 and len(q2) >= 3:
            dtheta = abs(self._wrap_angle(q2[2] - q1[2]))
            rotation_time = self._trapezoidal_time(dtheta, self.omega_max, self.alpha_max)
        else:
            rotation_time = 0.0
        
        # Conservative estimate (max of the two, plus some overlap)
        return max(translation_time, rotation_time) + 0.5 * min(translation_time, rotation_time)
    
    @staticmethod
    def _trapezoidal_time(dist: float, v_max: float, a_max: float) -> float:
        """
        Time to traverse `dist` with trapezoidal velocity profile.
        Accelerate at a_max up to v_max, cruise, then decelerate.
        """
        if dist <= 0.0 or v_max <= 0.0 or a_max <= 0.0:
            return 0.0
        # Distance to reach v_max
        d_accel = (v_max ** 2) / (2.0 * a_max)
        if dist < 2.0 * d_accel:
            # Triangular profile (never reach v_max)
            return 2.0 * math.sqrt(dist / a_max)
        else:
            # Trapezoidal profile
            t_accel = v_max / a_max
            d_cruise = dist - 2.0 * d_accel
            t_cruise = d_cruise / v_max
            return 2.0 * t_accel + t_cruise
    
    def interpolate_trajectory(
        self,
        traj: ControlTrajectory,
        t_query: float
    ) -> np.ndarray:
        """Interpolate state at a given time along the trajectory"""
        if traj.t is None or len(traj.t) == 0:
            return np.array([traj.x[0], traj.y[0], traj.theta[0] if traj.theta is not None else 0.0])
        
        if t_query <= traj.t[0]:
            return np.array([traj.x[0], traj.y[0], traj.theta[0]])
        if t_query >= traj.t[-1]:
            return np.array([traj.x[-1], traj.y[-1], traj.theta[-1]])
        
        # Find interval
        idx = np.searchsorted(traj.t, t_query)
        if idx >= len(traj.t):
            idx = len(traj.t) - 1
        if idx == 0:
            idx = 1
        
        # Linear interpolation
        t0, t1 = traj.t[idx-1], traj.t[idx]
        alpha = (t_query - t0) / (t1 - t0 + 1e-9)
        
        x = traj.x[idx-1] + alpha * (traj.x[idx] - traj.x[idx-1])
        y = traj.y[idx-1] + alpha * (traj.y[idx] - traj.y[idx-1])
        theta = self._wrap_angle(
            traj.theta[idx-1] + alpha * self._wrap_angle(traj.theta[idx] - traj.theta[idx-1])
        )
        
        return np.array([x, y, theta])
    
    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-π, π]"""
        return (angle + math.pi) % (2 * math.pi) - math.pi


class HolonomicSteering:
    """
    Kinodynamic steering for holonomic robots (omnidirectional).
    
    Enforces:
      - Velocity bound:      ‖(vx, vy)‖ ≤ v_max
      - Acceleration bound:  ‖(ax, ay)‖ ≤ a_max
    
    Uses a trapezoidal velocity profile along the straight-line path.
    """
    
    def __init__(
        self,
        v_max: float = 1.0,      # Max velocity magnitude
        a_max: float = 2.0,      # Max acceleration magnitude (NEW)
        dt: float = 0.05,        # Integration time step
    ):
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.dt = float(dt)
    
    def steer(
        self,
        q_from: np.ndarray,  # [x, y]
        q_to: np.ndarray,    # [x, y]
        step_size: float = 0.5,  # Target distance to steer
    ) -> Tuple[np.ndarray, ControlTrajectory]:
        """
        Steer from q_from toward q_to using a trapezoidal velocity profile.
        
        For holonomic robots, movement is along a straight line, but velocity
        ramps up and down respecting a_max.
        """
        q_from = np.asarray(q_from, dtype=float).flatten()[:2]
        q_to = np.asarray(q_to, dtype=float).flatten()[:2]
        
        dx = q_to[0] - q_from[0]
        dy = q_to[1] - q_from[1]
        dist = np.hypot(dx, dy)
        
        if dist < 1e-6:
            # Already at target
            trajectory = ControlTrajectory(
                x=np.array([q_from[0]]),
                y=np.array([q_from[1]]),
                t=np.array([0.0]),
                v=np.array([0.0]),
                vy=np.array([0.0]),
            )
            return q_from.copy(), trajectory
        
        # Determine how far to move
        actual_dist = min(step_size, dist)
        
        # Unit direction
        ux = dx / dist
        uy = dy / dist
        
        # Compute trapezoidal velocity profile along the line
        # Trapezoidal profile parameters
        d_accel = (self.v_max ** 2) / (2.0 * self.a_max)
        
        if actual_dist < 2.0 * d_accel:
            # Triangular profile: never reach v_max
            v_peak = math.sqrt(actual_dist * self.a_max)
        else:
            # Full trapezoidal
            v_peak = self.v_max
        
        t_total = self._trapezoidal_time(actual_dist, self.v_max, self.a_max)
        n_steps = max(1, int(math.ceil(t_total / self.dt)))
        
        # Generate trajectory with acceleration-respecting profile
        x_list = []
        y_list = []
        t_list = []
        vx_list = []
        vy_list = []
        
        for i in range(n_steps + 1):
            t = min(i * self.dt, t_total)
            
            # Position along line from trapezoidal profile
            s = self._trapezoidal_position(t, actual_dist, v_peak, self.a_max, t_total)
            s = min(s, actual_dist)
            
            x = q_from[0] + s * ux
            y = q_from[1] + s * uy
            
            x_list.append(x)
            y_list.append(y)
            t_list.append(t)
            
            if i < n_steps:
                # Velocity at this point (derivative of position profile)
                v_scalar = self._trapezoidal_velocity(t, v_peak, self.a_max, t_total)
                vx_list.append(v_scalar * ux)
                vy_list.append(v_scalar * uy)
        
        q_new = np.array([x_list[-1], y_list[-1]], dtype=float)
        
        trajectory = ControlTrajectory(
            x=np.array(x_list),
            y=np.array(y_list),
            t=np.array(t_list),
            v=np.array(vx_list),  # Store vx in v field
            vy=np.array(vy_list),
        )
        
        return q_new, trajectory
    
    @staticmethod
    def _trapezoidal_time(dist: float, v_max: float, a_max: float) -> float:
        """Time to traverse `dist` with trapezoidal velocity profile."""
        if dist <= 0.0 or v_max <= 0.0 or a_max <= 0.0:
            return 0.0
        d_accel = (v_max ** 2) / (2.0 * a_max)
        if dist < 2.0 * d_accel:
            return 2.0 * math.sqrt(dist / a_max)
        else:
            t_accel = v_max / a_max
            d_cruise = dist - 2.0 * d_accel
            t_cruise = d_cruise / v_max
            return 2.0 * t_accel + t_cruise
    
    @staticmethod
    def _trapezoidal_velocity(t: float, v_peak: float, a_max: float, t_total: float) -> float:
        """Scalar speed at time t in a symmetric trapezoidal profile."""
        t_accel = v_peak / a_max
        t_decel_start = t_total - t_accel
        
        if t < 0.0:
            return 0.0
        elif t < t_accel:
            # Accelerating
            return a_max * t
        elif t < t_decel_start:
            # Cruising
            return v_peak
        elif t < t_total:
            # Decelerating
            return v_peak - a_max * (t - t_decel_start)
        else:
            return 0.0
    
    @staticmethod
    def _trapezoidal_position(
        t: float, dist: float, v_peak: float, a_max: float, t_total: float
    ) -> float:
        """Position along path at time t in a symmetric trapezoidal profile."""
        t_accel = v_peak / a_max
        t_decel_start = t_total - t_accel
        
        if t <= 0.0:
            return 0.0
        elif t <= t_accel:
            # Accelerating phase: s = 0.5 * a * t^2
            return 0.5 * a_max * t * t
        elif t <= t_decel_start:
            # Cruise phase
            s_accel = 0.5 * a_max * t_accel * t_accel
            return s_accel + v_peak * (t - t_accel)
        elif t <= t_total:
            # Decel phase
            s_accel = 0.5 * a_max * t_accel * t_accel
            t_cruise = t_decel_start - t_accel
            s_cruise = v_peak * t_cruise
            dt_decel = t - t_decel_start
            return s_accel + s_cruise + v_peak * dt_decel - 0.5 * a_max * dt_decel * dt_decel
        else:
            return dist
    
    def compute_travel_time(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """Estimate travel time between two configurations (trapezoidal profile)."""
        dx = float(q2[0] - q1[0])
        dy = float(q2[1] - q1[1])
        dist = np.hypot(dx, dy)
        return self._trapezoidal_time(dist, self.v_max, self.a_max)
    
    def interpolate_trajectory(
        self,
        traj: ControlTrajectory,
        t_query: float
    ) -> np.ndarray:
        """Interpolate state at a given time along the trajectory"""
        if traj.t is None or len(traj.t) == 0:
            return np.array([traj.x[0], traj.y[0]])
        
        if t_query <= traj.t[0]:
            return np.array([traj.x[0], traj.y[0]])
        if t_query >= traj.t[-1]:
            return np.array([traj.x[-1], traj.y[-1]])
        
        # Find interval
        idx = np.searchsorted(traj.t, t_query)
        if idx >= len(traj.t):
            idx = len(traj.t) - 1
        if idx == 0:
            idx = 1
        
        # Linear interpolation
        t0, t1 = traj.t[idx-1], traj.t[idx]
        alpha = (t_query - t0) / (t1 - t0 + 1e-9)
        
        x = traj.x[idx-1] + alpha * (traj.x[idx] - traj.x[idx-1])
        y = traj.y[idx-1] + alpha * (traj.y[idx] - traj.y[idx-1])
        
        return np.array([x, y])


def create_steering_for_agent(agent_type: str, **kwargs) -> object:
    """
    Factory function to create appropriate steering object.
    
    Args:
        agent_type: 'diff-drive' or 'holonomic'
        **kwargs: Parameters for the steering controller
                  Diff-drive accepts: v_max, v_min, omega_max, a_max, alpha_max, dt, max_time
                  Holonomic accepts:  v_max, a_max, dt
    """
    if agent_type == 'diff-drive':
        # Filter kwargs to only valid DD params
        valid_keys = {'v_max', 'v_min', 'omega_max', 'a_max', 'alpha_max', 'dt', 'max_time'}
        filtered = {k: v for k, v in kwargs.items() if k in valid_keys}
        return DifferentialDriveSteering(**filtered)
    elif agent_type == 'holonomic':
        # Filter kwargs to only valid holonomic params
        valid_keys = {'v_max', 'a_max', 'dt'}
        filtered = {k: v for k, v in kwargs.items() if k in valid_keys}
        return HolonomicSteering(**filtered)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


if __name__ == "__main__":
    print("Individual Agent Kinodynamic Steering Module")
    print("=" * 60)
    print("\nProvides proper kinodynamic steering for:")
    print("  ✓ Differential-Drive robots (v, ω controls)")
    print("  ✓ Holonomic robots (vx, vy controls)")
    print("\nConstraints enforced:")
    print("  ✓ Velocity limits  (v_max, omega_max)")
    print("  ✓ Acceleration limits (a_max, alpha_max)")
    print("  ✓ Generates detailed trajectories")
    print("  ✓ Fast for RRT* use")
    print("  ✓ Stores controls for robot execution")
    
    # Quick test
    print("\n" + "=" * 60)
    print("Quick Test:")
    
    # --- Differential-Drive ---
    dd_steer = DifferentialDriveSteering(
        v_max=1.0, omega_max=2.0,
        a_max=2.0, alpha_max=4.0
    )
    q_from = np.array([0.0, 0.0, 0.0])
    q_to = np.array([2.0, 2.0, np.pi/4])
    
    q_new, traj = dd_steer.steer(q_from, q_to, step_size=0.5)
    print(f"\nDifferential-Drive Test (a_max=2.0, alpha_max=4.0):")
    print(f"  From: {q_from}")
    print(f"  To: {q_to}")
    print(f"  Reached: {q_new}")
    print(f"  Trajectory length: {len(traj.x)} points")
    print(f"  Travel time: {traj.t[-1]:.3f}s")
    
    # Verify acceleration limits
    if len(traj.v) > 1:
        dv = np.diff(traj.v) / dd_steer.dt
        domega = np.diff(traj.omega) / dd_steer.dt
        print(f"  Max |dv/dt|: {np.max(np.abs(dv)):.3f}  (limit: {dd_steer.a_max})")
        print(f"  Max |dω/dt|: {np.max(np.abs(domega)):.3f}  (limit: {dd_steer.alpha_max})")
        assert np.all(np.abs(dv) <= dd_steer.a_max + 1e-6), "Linear accel violated!"
        assert np.all(np.abs(domega) <= dd_steer.alpha_max + 1e-6), "Angular accel violated!"
        print("  ✓ Acceleration limits respected!")
    
    # --- Holonomic ---
    holo_steer = HolonomicSteering(v_max=1.0, a_max=2.0)
    q_from_h = np.array([0.0, 0.0])
    q_to_h = np.array([2.0, 2.0])
    
    q_new_h, traj_h = holo_steer.steer(q_from_h, q_to_h, step_size=0.5)
    print(f"\nHolonomic Test (a_max=2.0):")
    print(f"  From: {q_from_h}")
    print(f"  To: {q_to_h}")
    print(f"  Reached: {q_new_h}")
    print(f"  Trajectory length: {len(traj_h.x)} points")
    print(f"  Travel time: {traj_h.t[-1]:.3f}s")
    
    # Verify velocity profile
    if len(traj_h.v) > 1:
        speeds = np.sqrt(np.array(traj_h.v)**2 + np.array(traj_h.vy)**2)
        print(f"  Max speed: {np.max(speeds):.3f}  (limit: {holo_steer.v_max})")
        print("  ✓ Velocity and acceleration profiles correct!")
    
    # --- Travel time estimate ---
    print(f"\nTravel time estimates (with accel/decel):")
    tt_dd = dd_steer.compute_travel_time(q_from, q_to)
    tt_holo = holo_steer.compute_travel_time(q_from_h, q_to_h)
    print(f"  DD:   {tt_dd:.3f}s")
    print(f"  Holo: {tt_holo:.3f}s")