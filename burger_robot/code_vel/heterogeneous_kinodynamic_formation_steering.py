#!/usr/bin/env python3
"""
Heterogeneous Kinodynamic Formation Steering

Supports MIXED robot types within a single formation:
- Differential-drive robots: (v, ω) controls with heading ψ
- Holonomic robots: (vx, vy) controls, no heading state

Each robot has its own:
- Type (diff-drive or holonomic)
- Velocity limits (v_max, ω_max)
- Acceleration limits (a_max, α_max)   ← NEW
- Kinematic constraints in the optimizer

Constraint summary per robot type:
  Differential-Drive:
    |v_i|     ≤ v_max_i        (forward velocity)
    |ω_i|     ≤ w_max_i        (angular velocity)
    |Δv_i|    ≤ a_max_i * dt   (linear acceleration)
    |Δω_i|    ≤ alpha_max_i*dt (angular acceleration)

  Holonomic:
    |vx_i|    ≤ v_max_i        (velocity per-component)
    |vy_i|    ≤ v_max_i
    |Δvx_i|   ≤ a_max_i * dt   (acceleration per-component)
    |Δvy_i|   ≤ a_max_i * dt

Formation constraint: pᵢ = c + R(θ) * diag(sx,sy) * pᵢ*
"""

import numpy as np
import math
from typing import Optional, Tuple, List, Union, Dict
try:
    import casadi as ca
    CASADI_AVAILABLE = True
except ImportError:
    CASADI_AVAILABLE = False
    print("Warning: CasADi not available. Kinodynamic steering disabled.")


class HeterogeneousKinodynamicFormationSteering:
    """
    Kinodynamic steering for formations with MIXED robot types.
    
    Each robot can be:
    - 'diff-drive': Nonholonomic, uses (v, ω) controls, has heading ψ
    - 'holonomic': Omnidirectional, uses (vx, vy) controls, no heading
    
    The optimizer automatically applies correct velocity AND acceleration
    constraints based on robot type.
    """
    
    def __init__(
        self,
        P_star: np.ndarray,  # Reference formation points (Nr x 2)
        robot_types: List[str],  # ['diff-drive', 'holonomic', ...] for each robot
        # Per-robot velocity limits
        v_max: Union[float, List[float]] = 1.0,   # Forward velocity (DD) or magnitude (holonomic)
        w_max: Union[float, List[float]] = 2.0,    # Angular velocity (DD only)
        # Per-robot acceleration limits (NEW)
        a_max: Union[float, List[float]] = 2.0,    # Linear acceleration
        alpha_max: Union[float, List[float]] = 4.0, # Angular acceleration (DD only)
        # Formation rate limits
        vc_max: float = 1.0,  # Centroid velocity limit
        wth_max: float = 1.0, # Formation rotation rate
        ds_max: float = 0.5,  # Scaling rate
        # Steering parameters
        N_steer: int = 8,     # Discretization steps
        T_steer: float = 1.0, # Time horizon
        # Weights
        w_reach: float = 10.0,   # Weight on reaching target
        w_energy: float = 1.0,   # Weight on control effort
        w_smooth: float = 0.1,   # Weight on smoothness
        # Optimization
        max_iter: int = 200,
        warm_start: bool = True,
    ):
        if not CASADI_AVAILABLE:
            raise ImportError("CasADi required for kinodynamic steering")
        
        self.P_star = np.array(P_star, dtype=float)
        self.Nr = self.P_star.shape[0]
        
        # Validate robot types
        self.robot_types = robot_types
        if len(robot_types) != self.Nr:
            raise ValueError(f"robot_types length ({len(robot_types)}) must match Nr ({self.Nr})")
        
        for rt in robot_types:
            if rt not in ['diff-drive', 'holonomic']:
                raise ValueError(f"Unknown robot type: {rt}. Must be 'diff-drive' or 'holonomic'")
        
        # --- Convert scalar → per-robot lists ---
        
        def _to_list(val, name):
            if isinstance(val, (int, float)):
                return [float(val)] * self.Nr
            lst = [float(v) for v in val]
            if len(lst) != self.Nr:
                raise ValueError(f"{name} length must match Nr ({self.Nr})")
            return lst
        
        self.v_max = _to_list(v_max, 'v_max')
        self.w_max = _to_list(w_max, 'w_max')
        self.a_max = _to_list(a_max, 'a_max')          # NEW
        self.alpha_max = _to_list(alpha_max, 'alpha_max')  # NEW
        
        # Formation limits
        self.vc_max = float(vc_max)
        self.wth_max = float(wth_max)
        self.ds_max = float(ds_max)
        
        # Steering params
        self.N = int(N_steer)
        self.T = float(T_steer)
        self.dt = self.T / (self.N - 1)
        
        # Weights
        self.w_reach = float(w_reach)
        self.w_energy = float(w_energy)
        self.w_smooth = float(w_smooth)
        
        # Optimization
        self.max_iter = int(max_iter)
        self.warm_start = warm_start
        
        # Count robot types
        self.dd_indices = [i for i, rt in enumerate(robot_types) if rt == 'diff-drive']
        self.holo_indices = [i for i, rt in enumerate(robot_types) if rt == 'holonomic']
        self.n_dd = len(self.dd_indices)
        self.n_holo = len(self.holo_indices)
        
        print(f"Formation: {self.n_dd} diff-drive + {self.n_holo} holonomic robots")
        print(f"  Accel limits: a_max={self.a_max}, alpha_max={self.alpha_max}")
        
        # Build the solver
        self._build_solver()
        
        # Warm start storage
        self.last_solution = None
    
    def _build_solver(self):
        """Build CasADi NLP solver with mixed robot type constraints."""
        
        N = self.N
        Nr = self.Nr
        dt = self.dt
        
        # ===================================================================
        # Decision Variables (type-dependent)
        # ===================================================================
        
        # Formation state (same for all)
        q = ca.SX.sym("q", 5, N)  # [xc, yc, θ, sx, sy]
        uq = ca.SX.sym("uq", 5, N - 1)  # Formation rates
        
        # Per-robot variables (separated by type)
        # Differential-drive robots
        psi_dd = []  # Headings
        v_dd = []    # Forward velocities
        om_dd = []   # Angular velocities
        
        for i in self.dd_indices:
            psi_dd.append(ca.SX.sym(f"psi_{i}", N))
            v_dd.append(ca.SX.sym(f"v_{i}", N-1))
            om_dd.append(ca.SX.sym(f"om_{i}", N-1))
        
        # Holonomic robots
        vx_holo = []  # X-velocities
        vy_holo = []  # Y-velocities
        
        for i in self.holo_indices:
            vx_holo.append(ca.SX.sym(f"vx_{i}", N-1))
            vy_holo.append(ca.SX.sym(f"vy_{i}", N-1))
        
        # ===================================================================
        # Parameters (initial and target states)
        # ===================================================================
        
        q0_param = ca.SX.sym("q0", 5)  # Initial formation config
        q1_param = ca.SX.sym("q1", 5)  # Target formation config
        
        # Initial robot headings (only for DD robots)
        psi0_params = []
        for i in self.dd_indices:
            psi0_params.append(ca.SX.sym(f"psi0_{i}"))
        
        # ===================================================================
        # Helper functions
        # ===================================================================
        
        S = ca.DM([[0, -1], [1, 0]])  # Rotation derivative
        
        def R_ca(th):
            return ca.vertcat(
                ca.horzcat(ca.cos(th), -ca.sin(th)),
                ca.horzcat(ca.sin(th),  ca.cos(th))
            )
        
        # ===================================================================
        # Constraints and Cost
        # ===================================================================
        
        g = []
        lbg = []
        ubg = []
        J = 0
        
        # Initial conditions - Formation
        g.append(q[:, 0] - q0_param)
        lbg.extend([0.0] * 5)
        ubg.extend([0.0] * 5)
        
        # Initial conditions - DD robot headings
        for idx, i in enumerate(self.dd_indices):
            g.append(psi_dd[idx][0] - psi0_params[idx])
            lbg.append(0.0)
            ubg.append(0.0)
        
        # ===================================================================
        # Dynamics and Kinematic Feasibility
        # ===================================================================
        
        for k in range(N - 1):
            # Formation state at time k
            xc, yc, th, sx, sy = q[0, k], q[1, k], q[2, k], q[3, k], q[4, k]
            xcd, ycd, thd, sxd, syd = uq[0, k], uq[1, k], uq[2, k], uq[3, k], uq[4, k]
            
            # Integrate formation state
            g.append(q[:, k+1] - (q[:, k] + dt * uq[:, k]))
            lbg.extend([0.0] * 5)
            ubg.extend([0.0] * 5)
            
            # Formation motion matrices
            Rk = R_ca(th)
            Dk = ca.diag(ca.vertcat(sx, sy))
            dD = ca.diag(ca.vertcat(sxd, syd))
            cdot = ca.vertcat(xcd, ycd)
            
            # ===============================================================
            # Per-robot constraints (TYPE-DEPENDENT)
            # ===============================================================
            
            dd_counter = 0
            holo_counter = 0
            
            for i in range(Nr):
                # Reference position
                p_star_i = ca.vertcat(float(self.P_star[i, 0]), float(self.P_star[i, 1]))
                
                # Induced velocity from formation motion (same for all robot types)
                pdot = cdot + thd * (Rk @ (S @ (Dk @ p_star_i))) + (Rk @ (dD @ p_star_i))
                
                if self.robot_types[i] == 'diff-drive':
                    # -------------------------------------------------------
                    # DIFFERENTIAL-DRIVE CONSTRAINT
                    # Kinematic constraint: pdot = v * [cos(ψ), sin(ψ)]
                    # -------------------------------------------------------
                    
                    psi_i = psi_dd[dd_counter][k]
                    v_i = v_dd[dd_counter][k]
                    om_i = om_dd[dd_counter][k]
                    
                    # Velocity must align with heading
                    dir_v = ca.vertcat(ca.cos(psi_i), ca.sin(psi_i))
                    g.append(pdot - v_i * dir_v)
                    lbg.extend([0.0, 0.0])
                    ubg.extend([0.0, 0.0])
                    
                    # Integrate heading: ψ(k+1) = ψ(k) + ω * dt
                    g.append(psi_dd[dd_counter][k+1] - (psi_i + dt * om_i))
                    lbg.append(0.0)
                    ubg.append(0.0)
                    
                    dd_counter += 1
                
                elif self.robot_types[i] == 'holonomic':
                    # -------------------------------------------------------
                    # HOLONOMIC CONSTRAINT
                    # Kinematic constraint: pdot = [vx, vy]
                    # -------------------------------------------------------
                    
                    vx_i = vx_holo[holo_counter][k]
                    vy_i = vy_holo[holo_counter][k]
                    
                    # Velocity can be in any direction
                    robot_vel = ca.vertcat(vx_i, vy_i)
                    g.append(pdot - robot_vel)
                    lbg.extend([0.0, 0.0])
                    ubg.extend([0.0, 0.0])
                    
                    holo_counter += 1
            
            # ===============================================================
            # Acceleration Constraints (NEW)
            # ===============================================================
            # Applied between consecutive timesteps: |u(k+1) - u(k)| ≤ a * dt
            
            if k < N - 2:
                dd_counter2 = 0
                holo_counter2 = 0
                
                for i in range(Nr):
                    if self.robot_types[i] == 'diff-drive':
                        # Linear acceleration: |v(k+1) - v(k)| ≤ a_max * dt
                        dv = v_dd[dd_counter2][k+1] - v_dd[dd_counter2][k]
                        g.append(dv)
                        lbg.append(-self.a_max[i] * dt)
                        ubg.append( self.a_max[i] * dt)
                        
                        # Angular acceleration: |ω(k+1) - ω(k)| ≤ alpha_max * dt
                        dom = om_dd[dd_counter2][k+1] - om_dd[dd_counter2][k]
                        g.append(dom)
                        lbg.append(-self.alpha_max[i] * dt)
                        ubg.append( self.alpha_max[i] * dt)
                        
                        dd_counter2 += 1
                    
                    elif self.robot_types[i] == 'holonomic':
                        # X-acceleration: |vx(k+1) - vx(k)| ≤ a_max * dt
                        dvx = vx_holo[holo_counter2][k+1] - vx_holo[holo_counter2][k]
                        g.append(dvx)
                        lbg.append(-self.a_max[i] * dt)
                        ubg.append( self.a_max[i] * dt)
                        
                        # Y-acceleration: |vy(k+1) - vy(k)| ≤ a_max * dt
                        dvy = vy_holo[holo_counter2][k+1] - vy_holo[holo_counter2][k]
                        g.append(dvy)
                        lbg.append(-self.a_max[i] * dt)
                        ubg.append( self.a_max[i] * dt)
                        
                        holo_counter2 += 1
            
            # ===============================================================
            # Energy Cost (type-dependent)
            # ===============================================================
            
            # Formation control effort
            J += dt * self.w_energy * ca.sumsqr(uq[:, k])
            
            # DD robot control effort
            for idx in range(self.n_dd):
                J += dt * self.w_energy * (
                    ca.sumsqr(v_dd[idx][k]) + 
                    ca.sumsqr(om_dd[idx][k])
                )
            
            # Holonomic robot control effort
            for idx in range(self.n_holo):
                J += dt * self.w_energy * (
                    ca.sumsqr(vx_holo[idx][k]) + 
                    ca.sumsqr(vy_holo[idx][k])
                )
            
            # ===============================================================
            # Smoothness (if not last interval)
            # ===============================================================
            
            if k < N - 2:
                # Formation smoothness
                J += dt * self.w_smooth * ca.sumsqr(uq[:, k+1] - uq[:, k])
                
                # DD robot smoothness
                for idx in range(self.n_dd):
                    J += dt * self.w_smooth * (
                        ca.sumsqr(v_dd[idx][k+1] - v_dd[idx][k]) +
                        ca.sumsqr(om_dd[idx][k+1] - om_dd[idx][k])
                    )
                
                # Holonomic robot smoothness
                for idx in range(self.n_holo):
                    J += dt * self.w_smooth * (
                        ca.sumsqr(vx_holo[idx][k+1] - vx_holo[idx][k]) +
                        ca.sumsqr(vy_holo[idx][k+1] - vy_holo[idx][k])
                    )
        
        # Terminal cost: reach target
        q_final = q[:, -1]
        J += self.w_reach * ca.sumsqr(q_final - q1_param)
        
        # ===================================================================
        # Pack Decision Variables with Bounds
        # ===================================================================
        
        w_list = []
        lbw = []
        ubw = []
        
        # Formation state q
        for k in range(N):
            w_list.append(q[:, k])
            if k == 0:
                lbw.extend([-ca.inf] * 5)
                ubw.extend([ca.inf] * 5)
            else:
                lbw.extend([-ca.inf, -ca.inf, -ca.inf, 0.4, 0.4])  # sx, sy > 0
                ubw.extend([ca.inf, ca.inf, ca.inf, 3.0, 3.0])
        
        # Formation rates uq
        for k in range(N - 1):
            w_list.append(uq[:, k])
            lbw.extend([-self.vc_max, -self.vc_max, -self.wth_max, -self.ds_max, -self.ds_max])
            ubw.extend([self.vc_max, self.vc_max, self.wth_max, self.ds_max, self.ds_max])
        
        # DD robots: psi, v, omega
        for idx, i in enumerate(self.dd_indices):
            # Heading psi
            w_list.append(psi_dd[idx])
            lbw.extend([-ca.inf] * N)
            ubw.extend([ca.inf] * N)
            
            # Forward velocity v  (bounded by v_max)
            w_list.append(v_dd[idx])
            lbw.extend([-self.v_max[i]] * (N-1))
            ubw.extend([self.v_max[i]] * (N-1))
            
            # Angular velocity omega  (bounded by w_max)
            w_list.append(om_dd[idx])
            lbw.extend([-self.w_max[i]] * (N-1))
            ubw.extend([self.w_max[i]] * (N-1))
        
        # Holonomic robots: vx, vy
        for idx, i in enumerate(self.holo_indices):
            # X-velocity  (bounded by v_max)
            w_list.append(vx_holo[idx])
            lbw.extend([-self.v_max[i]] * (N-1))
            ubw.extend([self.v_max[i]] * (N-1))
            
            # Y-velocity  (bounded by v_max)
            w_list.append(vy_holo[idx])
            lbw.extend([-self.v_max[i]] * (N-1))
            ubw.extend([self.v_max[i]] * (N-1))
        
        # ===================================================================
        # Create NLP
        # ===================================================================
        
        w = ca.vertcat(*[ca.vec(wi) for wi in w_list])
        p = ca.vertcat(q0_param, q1_param, *psi0_params)  # Parameters
        
        nlp = {
            "x": w,
            "p": p,
            "f": J,
            "g": ca.vertcat(*g)
        }
        
        opts = {
            "ipopt.print_level": 0,
            "print_time": False,
            "ipopt.max_iter": self.max_iter,
            "ipopt.warm_start_init_point": "yes" if self.warm_start else "no",
            "ipopt.tol": 1e-4,
        }
        
        self.solver = ca.nlpsol("hetero_steer_solver", "ipopt", nlp, opts)
        self.lbw = np.array(lbw)
        self.ubw = np.array(ubw)
        self.lbg = np.array(lbg)
        self.ubg = np.array(ubg)
        
        g_vec = ca.vertcat(*g)
        
        print(f"Solver built: {w.numel()} decision vars, {g_vec.numel()} constraints")
        print(f"  (includes {Nr * (N-2)} acceleration constraints per component)")

    
    def steer(
        self,
        q_from: np.ndarray,
        q_to: np.ndarray,
        psi_from: Optional[np.ndarray] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Steer from q_from toward q_to.
        
        Args:
            q_from: Initial formation config [xc, yc, θ, sx, sy]
            q_to: Target formation config [xc, yc, θ, sx, sy]
            psi_from: Initial robot states (Nr,)
                      - For DD robots: heading ψ
                      - For holonomic: not used (can be None or ignored)
        
        Returns:
            q_new: New formation config
            psi_new: New robot states (Nr,)
                     - For DD robots: heading ψ
                     - For holonomic: 0.0 (placeholder)
        """
        q_from = np.array(q_from, dtype=float).flatten()
        q_to = np.array(q_to, dtype=float).flatten()
        
        # Initialize psi for DD robots
        if psi_from is None:
            psi_from = np.full(self.Nr, q_from[2], dtype=float)
        else:
            psi_from = np.array(psi_from, dtype=float).flatten()
        
        # Pack parameters: q0, q1, then psi0 for each DD robot
        p_val = [q_from, q_to]
        for i in self.dd_indices:
            p_val.append(np.array([psi_from[i]]))
        p_val = np.concatenate(p_val)
        
        # Initial guess
        if self.warm_start and self.last_solution is not None:
            x0 = self.last_solution
        else:
            x0 = self._make_initial_guess(q_from, q_to, psi_from)
        
        try:
            sol = self.solver(
                x0=x0,
                lbx=self.lbw,
                ubx=self.ubw,
                lbg=self.lbg,
                ubg=self.ubg,
                p=p_val
            )
            
            X = sol["x"].full().flatten()
            
            if self.warm_start:
                self.last_solution = X
            
            # Unpack solution
            q_traj, psi_traj = self._unpack_solution(X)
            
            # Return final state
            q_new = q_traj[:, -1]
            psi_new = psi_traj[:, -1]
            
            return q_new, psi_new
            
        except Exception as e:
            print(f"Steering optimization failed: {e}")
            return None, None
    
    def _make_initial_guess(self, q0, q1, psi0):
        """Create initial guess for optimization."""
        N = self.N
        
        x0 = []
        
        # Formation state q: linear interpolation
        for k in range(N):
            alpha = k / (N - 1)
            q_k = (1 - alpha) * q0 + alpha * q1
            x0.extend(q_k.tolist())
        
        # Formation rates uq: from finite difference
        for k in range(N - 1):
            alpha_k = k / (N - 1)
            alpha_kp1 = (k + 1) / (N - 1)
            q_k = (1 - alpha_k) * q0 + alpha_k * q1
            q_kp1 = (1 - alpha_kp1) * q0 + alpha_kp1 * q1
            uq_k = (q_kp1 - q_k) / self.dt
            x0.extend(uq_k.tolist())
        
        # DD robots: psi (constant), v=0, omega=0
        for i in self.dd_indices:
            x0.extend([psi0[i]] * N)  # psi
            x0.extend([0.0] * (N-1))  # v
            x0.extend([0.0] * (N-1))  # omega
        
        # Holonomic robots: vx=0, vy=0
        for i in self.holo_indices:
            x0.extend([0.0] * (N-1))  # vx
            x0.extend([0.0] * (N-1))  # vy
        
        return np.array(x0)
    
    def _unpack_solution(self, X):
        """Unpack solution vector into trajectories."""
        N = self.N
        Nr = self.Nr
        
        ptr = 0
        
        # Formation state q
        q_traj = np.zeros((5, N))
        for k in range(N):
            q_traj[:, k] = X[ptr:ptr+5]
            ptr += 5
        
        # Skip formation rates uq
        ptr += 5 * (N - 1)
        
        # Robot states (psi for all robots, placeholder for holonomic)
        psi_traj = np.zeros((Nr, N))
        
        # DD robots: extract psi, skip v and omega
        for idx, i in enumerate(self.dd_indices):
            psi_traj[i, :] = X[ptr:ptr+N]
            ptr += N
            ptr += (N-1)  # Skip v
            ptr += (N-1)  # Skip omega
        
        # Holonomic robots: psi=0 (placeholder), skip vx and vy
        for idx, i in enumerate(self.holo_indices):
            psi_traj[i, :] = 0.0  # Placeholder
            ptr += (N-1)  # Skip vx
            ptr += (N-1)  # Skip vy
        
        return q_traj, psi_traj


if __name__ == "__main__":
    print("Heterogeneous Kinodynamic Formation Steering")
    print("=" * 60)
    print("\nSupports MIXED robot types in formations:")
    print("  ✓ Differential-drive: (v, ω) controls, heading ψ")
    print("  ✓ Holonomic: (vx, vy) controls, no heading")
    print("\nConstraints enforced:")
    print("  ✓ Per-robot velocity limits  (v_max, ω_max)")
    print("  ✓ Per-robot acceleration limits (a_max, alpha_max)")
    print("  ✓ Formation rate limits (vc_max, wth_max, ds_max)")
    
    # Test with mixed formation
    if CASADI_AVAILABLE:
        print("\n" + "=" * 60)
        print("Test: 2 DD + 1 Holonomic formation")
        
        P_star = np.array([
            [-0.6, 0.0],  # Robot 0: DD
            [0.6, 0.0],   # Robot 1: Holonomic
            [0.0, 0.6]    # Robot 2: DD
        ])
        
        steerer = HeterogeneousKinodynamicFormationSteering(
            P_star=P_star,
            robot_types=['diff-drive', 'holonomic', 'diff-drive'],
            v_max=[0.22, 0.22, 0.22],      # Different speeds
            w_max=[2.84, 2.84, 2.84],      # omega only for DD
            a_max=[2.5, 2.5, 2.5],      # Per-robot accel limits (NEW)
            alpha_max=[3.2, 3.2, 3.2],  # Angular accel for DD (NEW)
            N_steer=6,
            T_steer=0.8,
            max_iter=150,
        )
        
        q_from = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        q_to = np.array([2.0, 2.0, 0.5, 1.0, 1.0])
        psi_from = np.array([0.0, 0.0, 0.0])
        
        q_new, psi_new = steerer.steer(q_from, q_to, psi_from)
        
        if q_new is not None:
            print(f"\n✓ Steering successful!")
            print(f"  From: {q_from}")
            print(f"  To: {q_to}")
            print(f"  Reached: {q_new}")
            print(f"  Robot headings: {psi_new}")
        else:
            print("\n✗ Steering failed")