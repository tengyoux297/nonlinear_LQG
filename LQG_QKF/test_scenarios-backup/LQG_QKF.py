import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
from typing import Literal
from tqdm import tqdm
from stateDynamics import *
import pickle as pkl
from typing import Literal
import time
small_value = 1e-6  # Small value to prevent numerical issues

perf_dir = 'D:/AC/UCLA/ECE/UCLA_LEMUR/nonlinear_LQG/LQG_QKF/test_scenarios/perf'
os.makedirs(perf_dir, exist_ok=True)

pkl_dir = 'D:/AC/UCLA/ECE/UCLA_LEMUR/nonlinear_LQG/LQG_QKF/test_scenarios/pkl/'
os.makedirs(pkl_dir, exist_ok=True)

# Path tracking related functions
def generate_reference_path(path_type: Literal['figure8', 'circle', 'straight', 'sine_wave', 'racetrack'] = 'figure8', 
                            anchor_points: int = 100, dt: float = 0.1, scale: float = 10.0, 
                            path_length: float = 100.0, start_point: tuple = (0.0, 0.0), end_point: tuple = (10.0, 0.0)):
    """
    Generate reference paths for point-to-point navigation with fixed start and end points.
    
    Args:
        path_type: Type of path ('figure8', 'circle', 'straight', 'sine_wave', 'racetrack')
        anchor_points: Number of anchor points on the path (controls resolution)
        dt: Time step
        scale: Scaling factor for path size
        path_length: Total length of the path in time units (constant for all anchor point counts)
        start_point: Fixed starting point (x, y) - always at origin (0, 0)
        end_point: Fixed ending point (x, y) - constant for all controllers
    
    Returns:
        path: Dictionary containing x, y, vx, vy, ax, ay reference trajectories
    """
    # Create fixed sequence of anchor points between start and end points
    # Always start at origin (0, 0) and end at the specified end point
    start_x, start_y = start_point
    end_x, end_y = end_point
    
    # Generate path shape first
    t = np.linspace(0, 1, anchor_points)  # Normalized parameter from 0 to 1
    
    if path_type == 'figure8':
        # Figure-8 path between start and end points
        freq = 2 * np.pi  # One complete figure-8
        x_shape = np.sin(freq * t)
        y_shape = np.sin(2 * freq * t) * 0.5
        # Scale and shift to connect start and end points
        x_ref = start_x + (end_x - start_x) * t + scale * 0.3 * x_shape
        y_ref = start_y + (end_y - start_y) * t + scale * 0.3 * y_shape
        
        # Ensure first point is exactly at start_point
        x_ref[0] = start_x
        y_ref[0] = start_y
        
    elif path_type == 'circle':
        # Circular arc between start and end points
        # Create a semicircle connecting start and end, ensuring start at origin
        center_x = (start_x + end_x) / 2
        center_y = (start_y + end_y) / 2
        radius = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2) / 2
        
        # Calculate the angle range to ensure we start at start_point and end at end_point
        start_angle = np.arctan2(start_y - center_y, start_x - center_x)
        end_angle = np.arctan2(end_y - center_y, end_x - center_x)
        
        # Ensure we go the shorter way around the circle
        angle_diff = end_angle - start_angle
        if angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        elif angle_diff < -np.pi:
            angle_diff += 2 * np.pi
            
        angle = start_angle + angle_diff * t
        x_ref = center_x + radius * np.cos(angle)
        y_ref = center_y + radius * np.sin(angle)
        
    elif path_type == 'straight':
        # Straight line path between start and end points
        x_ref = start_x + (end_x - start_x) * t
        y_ref = start_y + (end_y - start_y) * t
        
    elif path_type == 'sine_wave':
        # Sinusoidal path between start and end points
        freq = 2 * np.pi  # One complete cycle
        x_ref = start_x + (end_x - start_x) * t
        y_ref = start_y + (end_y - start_y) * t + scale * 0.3 * np.sin(freq * t)
        
        # Ensure first point is exactly at start_point
        x_ref[0] = start_x
        y_ref[0] = start_y
        
    elif path_type == 'racetrack':
        # Racetrack-like path between start and end points
        x_ref = np.zeros_like(t)
        y_ref = np.zeros_like(t)
        
        for i, tn in enumerate(t):
            if tn < 0.25:  # First quarter: straight line
                x_ref[i] = start_x + (end_x - start_x) * (4 * tn)
                y_ref[i] = start_y
            elif tn < 0.5:  # Second quarter: curve
                angle = np.pi * (tn - 0.25) / 0.25
                x_ref[i] = end_x
                y_ref[i] = start_y + scale * 0.5 * (1 - np.cos(angle))
            elif tn < 0.75:  # Third quarter: straight line back
                x_ref[i] = end_x - (end_x - start_x) * (4 * (tn - 0.5))
                y_ref[i] = start_y + scale * 0.5
            else:  # Fourth quarter: curve back to start
                angle = np.pi * (tn - 0.75) / 0.25
                x_ref[i] = start_x
                y_ref[i] = start_y + scale * 0.5 * (1 + np.cos(angle))
        
        # Ensure first point is exactly at start_point
        x_ref[0] = start_x
        y_ref[0] = start_y
    else:
        raise ValueError(f"Unknown path type: {path_type}")
    
    # Compute velocities and accelerations using finite differences
    vx_ref = np.gradient(x_ref, dt)
    vy_ref = np.gradient(y_ref, dt)
    ax_ref = np.gradient(vx_ref, dt)
    ay_ref = np.gradient(vy_ref, dt)
    
    path = {
        't': t,
        'x': x_ref,
        'y': y_ref,
        'vx': vx_ref,
        'vy': vy_ref,
        'ax': ax_ref,
        'ay': ay_ref
    }
    
    return path

def create_vehicle_dynamics_matrices(dt=0.1, process_noise_scale=0.01):
    """
    Create vehicle dynamics matrices for path tracking.
    State: [x, y, vx, vy] - position and velocity in 2D
    Control: [ax, ay, steering_angle] - acceleration inputs and steering
    
    Args:
        dt: Time step
        process_noise_scale: Scale of process noise
    
    Returns:
        A_E, A_S, B_S, W: System matrices
    """
    # Earth state: [x, y] - position
    # Sensor state: [vx, vy] - velocity  
    n1, n2 = 2, 2  # earth_size, sensor_size
    p = 3  # control inputs: [ax, ay, steering_angle]
    
    # State transition matrices - very stable
    A_E = np.eye(2) * 0.99  # Position with slight damping
    A_S = np.eye(2) * 0.98  # Velocity with strong damping for stability
    
    # Control input matrix (only affects sensor/velocity states) - reasonable gains
    B_S = np.array([
        [dt * 0.5, 0, 0],      # vx affected by ax (reasonable gain)
        [0, dt * 0.5, 0.1*dt]  # vy affected by ay and steering (reasonable gains)
    ])
    
    # Process noise covariance
    W = generate_random_symmetric_matrix(n1+n2, scale=process_noise_scale)
    # W[:2, :2] *= 0.001  # Extremely low noise for position
    # W[2:, 2:] *= 0.01   # Very low noise for velocity
    
    return A_E, A_S, B_S, W

def create_sensor_matrices_for_tracking(n=4, m=2, measurement_noise_scale=0.1, nonlinearity_scale=1.0):
    """
    Create sensor matrices for vehicle tracking.
    Measurements: [range, bearing] or [x_gps, y_gps]
    
    Args:
        n: State size
        m: Measurement size  
        measurement_noise_scale: Scale of measurement noise
        nonlinearity_scale: Scale of nonlinear measurement terms
        
    Returns:
        C, M, V: Measurement matrices
    """
    # GPS-like measurements of position (realistic measurement model)
    C = np.array([
        [1.0, 0.0, 0.0, 0.0],    # x position measurement
        [0.0, 1.0, 0.0, 0.0]     # y position measurement
    ])
    
    # Quadratic measurement matrices (small nonlinear terms)
    M = np.zeros((m, n, n))
    for i in range(m):
        M[i] = generate_random_symmetric_matrix(n, scale=nonlinearity_scale * 0.1)
    
    # Measurement noise covariance
    V = generate_random_symmetric_matrix(m, scale=measurement_noise_scale)
    
    return C, M, V

class PathTrackingDynamics(StateDynamics):
    """
    Extended state dynamics for path tracking with proper position-velocity coupling.
    State: [x, y, vx, vy] where position is integrated from velocity.
    """
    def __init__(self, n1, n2, p, W, A_E, A_S, B_S, dt=0.1):
        super().__init__(n1, n2, p, W, A_E, A_S, B_S)
        self.dt = dt
        
        # Proper vehicle dynamics: position integrated from velocity
        # State: [x, y, vx, vy] - double integrator model
        self.A = np.array([
            [1, 0, self.dt, 0],        # x = x + dt*vx (proper integration)
            [0, 1, 0, self.dt],        # y = y + dt*vy (proper integration)
            [0, 0, 1, 0],              # vx = vx + dt*ax (will be set by B)
            [0, 0, 0, 1]               # vy = vy + dt*ay (will be set by B)
        ])
        
        # Proper control input matrix - accelerations affect velocity
        self.B = np.array([
            [0, 0, 0],                 # x not directly affected by control
            [0, 0, 0],                 # y not directly affected by control
            [self.dt, 0, 0],           # vx affected by ax
            [0, self.dt, 0]            # vy affected by ay
        ])
        
    def get_current_reference_state(self, path_index, reference_path):
        """Get the reference state at the current time step."""
        if reference_path is None or path_index >= len(reference_path['x']):
            return np.zeros((4, 1))
        
        ref_state = np.array([
            [reference_path['x'][path_index]], 
            [reference_path['y'][path_index]],
            [reference_path['vx'][path_index]],
            [reference_path['vy'][path_index]]
        ])
        return ref_state

def generate_random_symmetric_matrix(size, scale=1.0):
    """"Generate a random symmetric positive definite matrix."""
    A = np.random.randn(size, size)
    return scale * (A.T @ A) + np.eye(size) * 1e-3  # Ensure it's positive definite

def detect_convergence(values, window_size=50, tolerance=1e-3, min_steps=100):
    """
    Detect convergence of a time series using multiple criteria:
    1. Stabilization: variance over recent window is small
    2. Trend: slope of linear fit over recent window is near zero
    3. Threshold: absolute value is below tolerance
    
    Args:
        values: list/array of values over time
        window_size: number of recent steps to analyze
        tolerance: absolute threshold for convergence
        min_steps: minimum number of steps before convergence can be declared
    
    Returns:
        convergence_step: step at which convergence occurred (None if not converged)
        convergence_metrics: dict with convergence information
    """
    if len(values) < min_steps:
        return None, {}
    
    for i in range(min_steps, len(values)):
        # Get recent window
        start_idx = max(0, i - window_size + 1)
        window = values[start_idx:i+1]
        window = np.array(window)
        
        # Criterion 1: Absolute threshold
        current_val = abs(values[i])
        below_threshold = current_val < tolerance
        
        # Criterion 2: Stabilization (low variance)
        if len(window) >= 10:  # Need sufficient points for variance
            normalized_variance = np.var(window) / (np.mean(np.abs(window)) + 1e-10)
            is_stable = normalized_variance < 1e-4
        else:
            is_stable = False
        
        # Criterion 3: Trend analysis (slope near zero)
        if len(window) >= 10:
            time_points = np.arange(len(window))
            slope = np.polyfit(time_points, window, 1)[0]
            is_trending_zero = abs(slope) < tolerance / window_size
        else:
            is_trending_zero = False
        
        # Convergence if at least 2 out of 3 criteria are met
        criteria_met = sum([below_threshold, is_stable, is_trending_zero])
        
        if criteria_met >= 2:
            return i, {
                'below_threshold': below_threshold,
                'is_stable': is_stable,
                'is_trending_zero': is_trending_zero,
                'current_value': current_val,
                'variance': np.var(window) if len(window) >= 10 else None,
                'slope': slope if len(window) >= 10 else None
            }
    
    return None, {}

def finite_horizon_lqr(A, B, Q, R, N=100, Qf=None):
    if Qf is None:
        Qf = Q.copy()
    P = Qf.copy()
    # backward recursion
    for k in reversed(range(N)):
        P = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.pinv(R + B.T @ P @ B) @ B.T @ P @ A
    return P

def update_lqr_one_step(A, B, Q, R, P):
    # Compute the LQR gain
    K = -np.linalg.pinv(R + B.T @ P @ B) @ B.T @ P @ A
    P_new = A.T @ P @ A - A.T @ P @ B @ K + Q + K.T @ R @ K  # update the cost-to-go matrix
    
    return K, P_new

def generate_goal_state(goal_state_E, state_S_size): 
        goal_state_S = np.random.randn(state_S_size, 1)
        goal_state = np.vstack((goal_state_E, goal_state_S))
        return goal_state

def generate_stable_system_parameters(n1, n2, p, m, noise_scale=1e-1, m_scale=1e2):
    """
    Generate stable system parameters similar to Julia example.
    Ensures eigenvalues are within unit circle for stability.
    """
    n = n1 + n2
    
    # Generate stable state transition matrices
    A_E = np.random.randn(n1, n1) * 0.1
    A_S = np.random.randn(n2, n2) * 0.1
    
    # Ensure stability by scaling eigenvalues
    eig_E, _ = np.linalg.eig(A_E)
    eig_S, _ = np.linalg.eig(A_S)
    
    # Scale to ensure eigenvalues are within unit circle
    max_eig_E = np.max(np.abs(eig_E))
    max_eig_S = np.max(np.abs(eig_S))
    
    if max_eig_E > 0.8:
        A_E = A_E * 0.8 / max_eig_E
    if max_eig_S > 0.8:
        A_S = A_S * 0.8 / max_eig_S
    
    B_S = np.random.randn(n2, p) * 0.1
    
    # Generate measurement matrices
    C = np.random.randn(m, n)
    
    # Generate quadratic measurement matrices
    M = []
    for i in range(m):
        M_i = generate_random_symmetric_matrix(n, scale=m_scale)
        M.append(M_i)
    M = np.array(M)
    
    # Generate noise matrices
    W = generate_random_symmetric_matrix(n, scale=noise_scale)
    V = generate_random_symmetric_matrix(m, scale=noise_scale)
    
    return A_E, A_S, B_S, C, M, W, V

def validate_stable_parameters(A_E, A_S):
    """
    Validate that the generated parameters are stable.
    Returns True if stable, False otherwise.
    """
    eig_E, _ = np.linalg.eig(A_E)
    eig_S, _ = np.linalg.eig(A_S)
    
    max_eig_E = np.max(np.abs(eig_E))
    max_eig_S = np.max(np.abs(eig_S))
    
    is_stable = max_eig_E < 1.0 and max_eig_S < 1.0
    
    if not is_stable:
        print(f"Warning: Unstable parameters detected!")
        print(f"Max eigenvalue A_E: {max_eig_E:.4f}")
        print(f"Max eigenvalue A_S: {max_eig_S:.4f}")
    
    return is_stable

class LQG_PathTracking:
    def __init__(self, n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, goal_state, H=50, 
             filter_type: Literal['qkf', 'ekf', 'kf', 'ukf'] = 'qkf',
             lqr_type: Literal['orig', 'aug', 'aug_analytic', 'aug_numeric', 'None'] = 'orig',
             reference_path=None, dt=0.1):

        self.filter_type = filter_type
        self.lqr_type = lqr_type
        self.dt = dt

        # --- dynamics / plant ---
        self.F = PathTrackingDynamics(n1, n2, p, W, A_E, A_S, B_S, dt=self.dt)
        n = n1 + n2

        # --- sensor ---
        self.sensor = sensor(C, M, V)
        self.V = self.sensor.get_V()

        # --- state sizes / matrices ---
        self.A = self.F.get_A()
        self.B = self.F.get_B()
        self.n1 = n1
        self.n2 = n2
        self.n  = n
        self.p  = self.F.get_input_size()
        self.W  = self.F.get_W()

        # --- augmented state initialisation (QKF) ---
        mu_tilde_u   = (np.eye(n + n**2) - self.F.get_A_tilde()).T @ self.F.get_mu_tilde()
        self.Z_est   = mu_tilde_u
        I_big        = np.eye(n**2 * (n+1)**2)
        Phi_tilde    = self.F.get_A_tilde()
        Sigma_tilde  = self.F.get_Sigma_tilde()
        vec_sigma    = (I_big - np.kron(Phi_tilde, Phi_tilde)) @ Vec(Sigma_tilde)
        self.Pz_est  = invVec(vec_sigma)

        # --- horizon ---
        self.H = H

        # --- tracking flags / path ---
        self.hierarchical_mode = False
        self.reference_path   = reference_path
        self.path_index       = 0  # kept for your helper methods that use index

        # --- state estimates / goal ---
        if self.reference_path is not None:
            # initial estimate seeded from the first reference (assumes [x,y,vx,vy] leading states)
            self.x_hat = np.zeros((self.n, 1))
            for i, key in enumerate(['x','y','vx','vy']):
                if key in reference_path and i < self.n:
                    self.x_hat[i,0] = float(reference_path[key][0])
        else:
            self.x_hat = np.zeros((self.n, 1))

        self.z_hat  = np.zeros((self.n + self.n**2, 1))
        self.x_goal = np.zeros((self.n, 1))  # will be set below

        # --- LQR cost / Riccati seed ---
        self.Q      = Q.astype(np.float64) 
        self.R      = R.astype(np.float64)
        self.P_lqr  = Q.copy()[:self.n, :self.n]

        # --- LQE covariance init ---
        if self.reference_path is not None:
            # reasonable uncertainties for [x,y,vx,vy], rest small
            self.P_est = np.eye(self.n) * 1e-3
            base = [1.0, 1.0, 0.5, 0.5]
            for i, v in enumerate(base[:self.n]):
                self.P_est[i,i] = v
        else:
            self.P_est = np.eye(self.n) * small_value

        # --- convergence / divergence bookkeeping ---
        self.convergence_history = {
            'cost': [],
            'estimation_error': [],
            'control_effort': [],
            'tracking_error': [],
            'is_converged': False,
            'convergence_step': None,
            'convergence_metrics': {}
        }
        self.divergence_threshold = 50.0  # Distance threshold for divergence detection
        self.is_diverged = False
        self.divergence_step = None

        # --- set initial goal ---
        if self.reference_path is not None:
            # Use your built-in path helpers to manage the moving goal (lookahead blend)
            # Reduced advance_thresh and lookahead for better progress tracking
            self.attach_path(self.reference_path, lookahead=3, advance_thresh=3.0)
            # attach_path sets self.x_goal via _blend_goal(...)
        else:
            self.x_goal = goal_state
    
    # iLQR related
    def get_A_hat(self, x, u):
        '''A_hat = d Z(t+1) / d x(t)'''
        I_n = np.eye(self.n) # shape (n, n)
        B = self.F.B # shape (n, p)
        Bu = B @ u # shape (n, 1)
        A = self.F.A # shape (n, n)
        A2 = np.kron(A, A) @ (np.kron(I_n, x) + np.kron(x, I_n)) + np.kron(Bu, A) + np.kron(A, Bu) # shape (n^2, n)
        A_hat = np.vstack((A, A2)) # shape (n^2 + n, n)
        return A_hat.astype(np.float64) # shape (n^2 + n, n)

    def get_B_hat(self, x, u):
        '''B_hat = d Z(t+1) / d u(t)'''
        B = self.F.B # shape (n, p)
        I_p = np.eye(self.p) # shape (p, p)
        Ax = self.F.A @ x # shape (n, 1)
        B2 = np.kron(B, B) @ (np.kron(I_p, u) + np.kron(u, I_p)) + np.kron(Ax, B) + np.kron(B, Ax) # shape (n^2, p)
        B_hat = np.vstack((B, B2)) # shape (n^2 + n, p)
        return B_hat.astype(np.float64) # shape (n^2 + n, p)
    
    def linearise(self, x_nom, u_nom):
        """Return A_hat, B_hat for current nominal (no noise)."""
        z = self.Z_est
        A_hat = self.get_A_hat(x_nom, u_nom)   # df/dx
        B_hat = self.get_B_hat(x_nom, u_nom)   # df/du
        return z, A_hat, B_hat
    
    def line_search(self, u_nom, d, K, x_nom, x_now, x_goal, A, B, alpha_init=1.0):
        alpha = alpha_init
        for _ in range(10):
            u_try = u_nom + alpha*d + K @ (x_now - x_nom)
            x_try = A @ x_nom + B @ u_try
            
            # Let Q matrix handle position vs velocity error weighting
            z_goal = np.vstack((x_goal, Vec(x_goal @ x_goal.T))) # shape (n+n^2, 1)
            z_try = np.vstack((x_try, Vec(x_try @ x_try.T))) # shape (n+n^2, 1)
            z_nom = np.vstack((x_nom, Vec(x_nom @ x_nom.T))) # shape (n+n^2, 1)
            cost_try = (z_try - z_goal).T @ self.Q @ (z_try - z_goal) + u_try.T @ self.R @ u_try # cost function
            cost_nom = (z_nom - z_goal).T @ self.Q @ (z_nom - z_goal) + u_nom.T @ self.R @ u_nom # cost function
            if cost_try < cost_nom:            # cost improved?
                return u_try, x_try, alpha
            alpha *= 0.5                       # shrink step
        return u_nom, x_nom, 0.0               # no progress
    
    def update_aug_ilqr(self, goal_state, alpha = 1, verbose=False):
        x_nominal = self.x_hat
        u_nominal = np.zeros((self.p, 1)) # nominal control input vector
         
        max_iter = 50  # Reduced from 1000 to prevent getting stuck
        iteration = 0
        diff_u = float('inf') 
        diff_cost = float('inf')
        epsilon_u = 1e-4  # Relaxed convergence threshold
        epsilon_cost = 1e-6  # Relaxed convergence threshold
        prev_cost = float('inf')
        
        while iteration < max_iter:
            iteration += 1
            # F: first-order derivative of f   
            #   f(x, u) = A_tilde z + B_tilde u + noise = ...
            z_curr, A_hat, B_hat = self.linearise(x_nominal, u_nominal) # shape (n+n^2, 1), (n+n^2, n), (n+n^2, p)
            # l: cost function
                # l = Σ z.T Q z + u.T R u
            A_tilde = self.F.get_A_tilde()
            mu_tilde = self.F.get_mu_tilde() 
            # print(f'A_tilde: {A_tilde.shape}, z_curr: {z_curr.shape}, mu_tilde: {mu_tilde.shape}, w_tilde: {w_tilde.shape}')
            z_next = A_tilde @ z_curr + mu_tilde # shape (n+n^2, 1)
            
            # Let Q matrix handle position vs velocity error weighting
            z_goal = (np.concatenate([goal_state.T, Vec(goal_state @ goal_state.T).T], axis=1)).T # shape (n+n^2, 1)
            dz = z_next - z_goal # shape (n+n^2, 1)
            Q = self.Q # shape (n+n^2, n+n^2)
            R = self.R # shape (p, p)
            u = self.F.get_u() # shape (p, 1)
            
            # c: first-order derivative of cost function
            l_x = A_hat.T @ Q @ dz # shape (n, 1)
            l_u = 2 * B_hat.T @ Q @ dz + 2 * R @ u_nominal # shape (p, 1)
            c = np.vstack((l_x, l_u)) # shape (n+p, 1)  
             
            # cc: second-order derivative of cost function
            l_xx = A_hat.T @ self.Q @ A_hat
            l_uu = 2*B_hat.T @ self.Q @ B_hat + 2*self.R
            l_ux = B_hat.T @ self.Q @ A_hat
            
            # run backward pass
            #   compute the feedback gain matrix K

            #   regularize Q_uu for numerical stability
            reg = 1e-8 * np.eye(self.p)
            Q_uu = l_uu + reg

            #   solve for gains
            K = -np.linalg.solve(Q_uu, l_ux)  # feedback gain: shape (p, n)
            d = -np.linalg.solve(Q_uu, l_u)   # feedforward term: shape (p, 1)
            
            # run forward pass
            #   compute the new control input
            x_cur = self.F.get_x() # shape (n, 1)
            A = self.F.get_A() # shape (n, n)
            B = self.F.get_B()
            u_new, x_new, alpha = self.line_search(u_nominal, d, K, x_nominal, x_cur, goal_state, A, B)
            
            # compute current cost for convergence checking
            z_cur = np.vstack((x_cur, Vec(x_cur @ x_cur.T)))
            z_goal = np.vstack((goal_state, Vec(goal_state @ goal_state.T)))
            current_cost = (z_cur - z_goal).T @ self.Q @ (z_cur - z_goal) + u_new.T @ self.R @ u_new
            current_cost = current_cost.item()
            
            # check convergence using multiple criteria
            diff_u = np.linalg.norm(u_new - u_nominal)
            diff_cost = abs(current_cost - prev_cost) / (abs(prev_cost) + 1e-10)  # relative cost change
            
            if verbose:
                print(f"iter {iteration:3d} | step α={alpha:.3f} | Δu={diff_u:.2e} | Δcost={diff_cost:.2e}")
            
            # Check multiple convergence criteria
            if diff_u < epsilon_u and diff_cost < epsilon_cost:
                if verbose:
                    print(f"Converged: Δu={diff_u:.2e} < {epsilon_u:.2e}, Δcost={diff_cost:.2e} < {epsilon_cost:.2e}")
                break
            elif iteration > 10 and diff_u < 1e-3:  # Early termination for reasonable convergence
                if verbose:
                    print(f"Early termination: Δu={diff_u:.2e} < 1e-3 after {iteration} iterations")
                break
            if alpha == 0.0:  # line search failed
                if verbose:
                    print("Line search failed, stopping iteration")
                break
                
            u_nominal = u_new
            x_nominal = x_new
            prev_cost = current_cost
            
        self.F.set_u(u_new)
        return
    
    def update_aug_lqr_analytic(self, goal_state):
        # LQG update only with augmented state
        I_p = np.eye(self.p)  # shape (p, p)
        I_p2 = np.eye(self.p ** 2)  # shape (p^2, p^2)
        I_n = np.eye(self.n) # shape (n, n)
        B = self.F.get_B()  # shape (n, p)
        A = self.F.get_A()  # shape (n, n)
        
        
        ####################################################
        x_actual = self.F.get_x()  # shape (n, 1), current state vector
        x_estimated = self.x_hat
        x_tilde = x_estimated - x_actual # shape (n, 1)
        
        # Let Q matrix handle position vs velocity error weighting
        x = x_estimated - goal_state # shape (n, 1)
        
        z = np.vstack((x, Vec(x @ x.T))) # shape (n+n^2, 1)
        
        ####################################################
        
        # commutation matrix for I_p kron u
        T = np.zeros((self.p * self.p, self.p * self.p)) # shape (p^2, p^2)
        for i in range(self.p):
            for j in range(self.p):
                e_ij = np.zeros((self.p, self.p))
                e_ij[i, j] = 1
                vec_e_ij = e_ij.T.flatten()  # transpose before vec
                T[:, i * self.p + j] = vec_e_ij

        M = np.kron(B, B) @ (I_p2 + T) # shape (n^2, p^2)
        q = Vec(self.Q[:self.n, :self.n]) # shape (n^2, 1)
        # print(f'q: {q.shape}')
        # print(f'Q: {self.Q.shape}')
        
        S = np.zeros((self.p, self.p))  # shape (p, p)
        for i in range(self.p):
            e_i = np.zeros((self.p, 1)) # shape (p, 1)
            e_i[i] = 1
            term1 = (M @ np.kron(e_i, I_p)) # shape (n^2, p)
            # print(f'term1: {term1.shape}')
            # print(f'q: {q.shape}')
            # print(f'e_i: {e_i.shape}')
            term2 = term1.T @ q @ e_i.T  # shape (p, p)
            S += term2  # accumulate over p columns
        
        # Z = np.kron(A, B) @ np.kron(x, I_p)   + np.kron(B, A) @ np.kron(I_p, x) # shape (n^2, p)
        Z = np.kron(B, A @ x) + np.kron(A @ x, B) + np.kron(B, A @ x_tilde) + np.kron(A @ x_tilde, B)
        u_new = -np.linalg.inv(S + 2 * self.R) @ Z.T @ q # shape (p, 1)
        self.F.set_u(u_new)


    def update_lqr_orig(self, goal_state):
        # LQR update only with original state, no augmented state
        # P_lqr = scipy.linalg.solve_discrete_are(self.A, self.B, self.Q[:self.n, :self.n], self.R)  # P is the fixed-point
        P_lqr = finite_horizon_lqr(self.A, self.B, self.Q[:self.n, :self.n], self.R, N=1, Qf=self.P_lqr)
        self.P_lqr = P_lqr.copy() # update cost-to-go matrix
        # feedback_gain = -np.linalg.pinv(self.R + self.B.T @ P_lqr @ self.B) @ self.B.T @ P_lqr @ self.A
        G = self.R + self.B.T @ P_lqr @ self.B
        feedback_gain = -np.linalg.solve(G, self.B.T @ P_lqr @ self.A)
        
        # Use time-varying reference for path tracking
        # Let Q matrix handle position vs velocity error weighting
        state_error = self.x_hat - goal_state
        
        u_new = feedback_gain @ state_error  # control input
            
        self.F.set_u(u_new)
        return
    
    def update_ilqr(self, goal_state, alpha=1, verbose=False):
        """iLQR implementation matching 2025_summer version"""
        x_nominal = self.x_hat
        u_nominal = np.zeros((self.p, 1))  # nominal control input vector
         
        max_iter = 1000
        iter = 0
        diff_u = 1e10  
        diff_cost = 1e10
        epsilon_u = 1e-4  # convergence threshold for control input change (relaxed for path tracking)
        epsilon_cost = 1e-6  # convergence threshold for cost change (relaxed for path tracking)
        prev_cost = 1e10
        
        while iter < max_iter:
            iter += 1
            # F: first-order derivative of f   
            #   f(x, u) = A_tilde z + B_tilde u + noise = ...
            z_curr, A_hat, B_hat = self.linearise(x_nominal, u_nominal)  # shape (n+n^2, 1), (n+n^2, n), (n+n^2, p)
            # l: cost function
            #   l = Σ z.T Q z + u.T R u
            A_tilde = self.F.get_A_tilde()
            mu_tilde = self.F.get_mu_tilde() 
            z_next = A_tilde @ z_curr + mu_tilde  # shape (n+n^2, 1)
            
            z_goal = (np.concatenate([goal_state.T, Vec(goal_state @ goal_state.T).T], axis=1)).T  # shape (n+n^2, 1)
            dz = z_next - z_goal  # shape (n+n^2, 1)
            Q = self.Q  # shape (n+n^2, n+n^2)
            R = self.R  # shape (p, p)
            u = self.F.get_u()  # shape (p, 1)
            
            # c: first-order derivative of cost function
            l_x = A_hat.T @ Q @ dz  # shape (n, 1)
            l_u = 2 * B_hat.T @ Q @ dz + 2 * R @ u_nominal  # shape (p, 1)
            c = np.vstack((l_x, l_u))  # shape (n+p, 1)  
             
            # cc: second-order derivative of cost function
            l_xx = A_hat.T @ self.Q @ A_hat
            l_uu = 2*B_hat.T @ self.Q @ B_hat + 2*self.R
            l_ux = B_hat.T @ self.Q @ A_hat
            
            # run backward pass
            #   compute the feedback gain matrix K
            #   regularize Q_uu for numerical stability
            reg = 1e-8 * np.eye(self.p)
            Q_uu = l_uu + reg

            #   solve for gains
            K = -np.linalg.solve(Q_uu, l_ux)  # feedback gain: shape (p, n)
            d = -np.linalg.solve(Q_uu, l_u)   # feedforward term: shape (p, 1)
            
            # run forward pass
            #   compute the new control input
            x_cur = self.F.get_x()  # shape (n, 1)
            A = self.F.get_A()  # shape (n, n)
            B = self.F.get_B()
            u_new, x_new, alpha = self.line_search(u_nominal, d, K, x_nominal, x_cur, goal_state, A, B)
            
            # compute current cost for convergence checking
            z_cur = np.vstack((x_cur, Vec(x_cur @ x_cur.T)))
            z_goal = np.vstack((goal_state, Vec(goal_state @ goal_state.T)))
            current_cost = (z_cur - z_goal).T @ self.Q @ (z_cur - z_goal) + u_new.T @ self.R @ u_new
            current_cost = current_cost.item()
            
            # check convergence using multiple criteria
            diff_u = np.linalg.norm(u_new - u_nominal)
            diff_cost = abs(current_cost - prev_cost) / (abs(prev_cost) + 1e-10)  # relative cost change
            
            if verbose:
                print(f"iter {iter:3d} | step α={alpha:.3f} | Δu={diff_u:.2e} | Δcost={diff_cost:.2e}")
            
            # Check multiple convergence criteria
            if diff_u < epsilon_u and diff_cost < epsilon_cost:
                if verbose:
                    print(f"Converged: Δu={diff_u:.2e} < {epsilon_u:.2e}, Δcost={diff_cost:.2e} < {epsilon_cost:.2e}")
                break
            if alpha == 0.0:  # line search failed
                if verbose:
                    print("Line search failed, stopping iteration")
                break
            
            # Early termination for path tracking (don't over-optimize)
            if iter > 20 and diff_u < 1e-3:
                if verbose:
                    print(f"Early termination: iter={iter}, Δu={diff_u:.2e}")
                break
                
            # update nominal trajectory
            u_nominal = u_new.copy()
            x_nominal = x_cur.copy()
            prev_cost = current_cost
            
        if verbose:
            print(f"iLQR converged in {iter} iterations, final Δu: {diff_u:.2e}, final Δcost: {diff_cost:.2e}")
            
        self.F.set_u(u_new)
        return
        
    def update_lqr(self):
        if self.lqr_type == 'None':
            # No LQR update, no control input
            # self.F.set_u(np.ones((self.p, 1)))
            self.F.set_u(np.random.randn(self.p, 1))  # small random noise
            return 
        else:
            goal_state = self.x_goal
            if self.filter_type == 'ekf' or self.filter_type == 'kf' or self.filter_type == 'ukf':
                self.update_lqr_orig(goal_state)
            elif self.filter_type == 'qkf':
                if self.lqr_type == 'aug_numeric':
                    self.update_ilqr(goal_state, alpha=1)
                elif self.lqr_type == 'aug':
                    self.update_ilqr(goal_state, alpha=1)
                elif self.lqr_type == 'aug_analytic':
                    self.update_ilqr(goal_state, alpha=1)
                elif self.lqr_type == 'orig':
                    self.update_lqr_orig(goal_state)
            else:
                raise ValueError("Invalid filter type. Choose 'qkf', 'ekf', 'kf', or 'ukf'.")
        return
        
    ######################################
    # LQE related
    ######################################
    def update_lqe_qkf(self):
        Phi_tilde  = self.F.get_A_tilde()
        Sigma_tilde = self.F.get_Sigma_tilde()
        # print('sigma_tilde', Sigma_tilde)   
        mu_tilde = self.F.get_mu_tilde()
        
        # state prediction     Z_{t|t‑1} ,  P⁽ᶻ⁾_{t|t‑1}
        Z_pred = Phi_tilde @ self.Z_est + mu_tilde
        Pz_pred = Phi_tilde @ self.Pz_est @ Phi_tilde.T + Sigma_tilde

        # measurement prediction Y_{t|t‑1} , innovation cov  M
        measA = self.sensor.get_measA() # shape (m, 1)
        measB_tilde = self.sensor.get_aug_measB() # shape (m, n+n^2)
        Y_pred = measA + measB_tilde @ Z_pred # shape (m, n+n^2)
        M = measB_tilde @ Pz_pred @ measB_tilde.T + self.V

        # Kalman gain          Kₜ
        K = Pz_pred @ measB_tilde.T @ np.linalg.inv(M)

        # state update         Z_{t|t} ,  P⁽ᶻ⁾_{t|t}
        Z, _, _ = self.F.get_z()
        Y_meas = self.sensor.aug_measure(Z)
        innovation = Y_meas - Y_pred
        self.Z_est = Z_pred + K @ innovation
        Pz_1 = Pz_pred - K @ M @ K.T
        
        self.Pz_est = Pz_1
        self.x_hat = self.Z_est[:self.n, :]
        return K
    
    def update_lqe_ekf(self):
        mu = self.F.B @ self.F.u
        Phi = self.F.A
        Sigma = self.F.W
        
        # state prediction
        X_pred = mu + Phi @ self.x_hat
        P_pred = Phi @ self.P_est @ Phi.T + Sigma
        
        # measurement prediction
        Y_pred = self.sensor.measure_pred(X_pred)
        g = self.sensor.g(X_pred)
        M = g @ P_pred @ g.T + self.sensor.V
        
        # gain
        K = P_pred @ g.T @ np.linalg.inv(M)
        
        # state update
        Y_meas = self.sensor.measure(self.F.get_x())
        innov = Y_meas - Y_pred
        self.x_hat = X_pred + K @ innov
        self.P_est = P_pred - K @ M @ K.T
        return K
    
    def update_lqe_kf(self):
        C = self.sensor.C
        # priori estimate 
        x_hat_pri = self.A @ self.x_hat + self.B @ self.F.get_u()   
        
        # P_k-1
        p0 = self.A @ self.P_est @ self.A.T + self.W
        
        # kalman gain
            # K = P- @ C^T @ inv(C @ P- @ C^T + V)
        kalman_gain =(p0 @ C.T @ np.linalg.pinv(C @ p0 @ C.T + self.V))
        self.kalman_gain = (kalman_gain)
        
        # measurement
        y = self.sensor.measure(self.F.get_x())
        
        # innovation
        innov = y - self.sensor.measure_pred(x_hat_pri)
        
        # posterior estimate
        x_hat_post = x_hat_pri + kalman_gain @ innov
        self.x_hat = (x_hat_post)
        
        # P_k - Propagation of the estimation error covariance matrix
        self.P_est = (np.eye(self.n) - kalman_gain @ C) @ p0
        return kalman_gain
    
    def update_lqe_ukf(self):
        # UKF parameters
        alpha = 1e-3
        beta = 2
        kappa = 0
        n = self.x_hat.shape[0]
        lambda_ = alpha**2 * (n + kappa) - n
        
        # Compute sigma points
        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = self.x_hat.flatten()
        
        # Cholesky decomposition for numerical stability
        try:
            P_reg = self.P_est + np.eye(n) * 1e-8  # Add regularization
            sqrt_P = np.linalg.cholesky((n + lambda_) * P_reg)
        except np.linalg.LinAlgError:
            # If not positive definite, use eigendecomposition
            eigenvals, eigenvecs = np.linalg.eigh(self.P_est)
            eigenvals = np.maximum(eigenvals, 1e-6)  # Ensure positive
            sqrt_P = eigenvecs @ np.diag(np.sqrt(eigenvals))
            sqrt_P = np.sqrt(n + lambda_) * sqrt_P
        
        for i in range(n):
            sigma_points[i + 1] = self.x_hat.flatten() + sqrt_P[i]
            sigma_points[n + i + 1] = self.x_hat.flatten() - sqrt_P[i]
        
        # Predict sigma points through state dynamics (consistent with EKF)
        sigma_points_pred = np.zeros_like(sigma_points)
        for i in range(2 * n + 1):
            # Use the state dynamics to predict (consistent with EKF approach)
            x_pred = self.F.A @ sigma_points[i].reshape(-1, 1) + self.F.B @ self.F.u
            sigma_points_pred[i] = x_pred.flatten()
        
        # Compute state mean
        weights_mean = np.full(2 * n + 1, 1 / (2 * (n + lambda_)))
        weights_mean[0] = lambda_ / (n + lambda_)
        x_predicted = np.sum(weights_mean[:, np.newaxis] * sigma_points_pred, axis=0).reshape(-1, 1)
        
        # Compute state covariance (consistent with EKF)
        weights_cov = np.full(2 * n + 1, 1 / (2 * (n + lambda_)))
        weights_cov[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
        sigma_0 = self.F.W.copy()  # Use F.W like EKF
        for i in range(2 * n + 1):
            diff = sigma_points_pred[i] - x_predicted.flatten()
            sigma_0 += weights_cov[i] * np.outer(diff, diff)
        
        # Predict measurements using sigma points (consistent with EKF)
        sigma_points_meas = np.zeros((2 * n + 1, self.sensor.m))
        for i in range(2 * n + 1):
            # Use measure_pred for prediction like EKF
            sigma_points_meas[i] = self.sensor.measure_pred(sigma_points_pred[i].reshape(-1, 1)).flatten()
        
        # Predict measurement mean
        y_predicted = np.sum(weights_mean[:, np.newaxis] * sigma_points_meas, axis=0).reshape(-1, 1)
        
        # Predict measurement covariance (consistent with EKF)
        S = self.sensor.V.copy()  # Use sensor.V like EKF
        for i in range(2 * n + 1):
            diff = sigma_points_meas[i] - y_predicted.flatten()
            S += weights_cov[i] * np.outer(diff, diff)
        
        # Cross covariance
        C_tilde = np.zeros((n, self.sensor.m))
        for i in range(2 * n + 1):
            diff_state = sigma_points_pred[i] - x_predicted.flatten()
            diff_meas = sigma_points_meas[i] - y_predicted.flatten()
            C_tilde += weights_cov[i] * np.outer(diff_state, diff_meas)
        
        # Kalman gain with numerical stability
        try:
            # Add stronger regularization for numerical stability
            S_reg = S + np.eye(S.shape[0]) * 1e-3
            # Use lstsq for more robust computation
            K = np.linalg.lstsq(S_reg.T, C_tilde.T, rcond=1e-6)[0].T
        except np.linalg.LinAlgError:
            # Ultimate fallback: use identity gain (no correction)
            raise ValueError("UKF Kalman gain computation failed")
        
        # Measurement residual (consistent with EKF)
        y = self.sensor.measure(self.F.get_x())
        delta_y = y - y_predicted
        
        # Update the state estimate
        self.x_hat = x_predicted + K @ delta_y
        
        # Update the covariance estimate
        self.P_est = sigma_0 - K @ S @ K.T
        return K
    
    def update_lqe(self):
        if self.filter_type == 'qkf':
            K = self.update_lqe_qkf()
        elif self.filter_type == 'ekf':
            K = self.update_lqe_ekf()
        elif self.filter_type == 'kf':
            K = self.update_lqe_kf()
        elif self.filter_type == 'ukf':
            K = self.update_lqe_ukf()
        else:
            raise ValueError("Invalid filter type. Choose 'qkf', 'ekf', 'kf', or 'ukf'.")
        t = self.F.t
        # print(f'  t={t:4d}', f'‖K_{self.filter_type}‖₂=', np.linalg.norm(K),) if t % 100 == 0 else None
        return


    ######################################
    # helper functions
    ######################################
    def forward_state(self):
        self.F.forward()
    
    def check_divergence(self):
        """
        Check if the system has diverged based on distance from actual position to current goal point.
        
        Returns:
            bool: True if diverged
        """
        if self.is_diverged:
            return True
        
        # Check for NaN or infinite values (keep this safety check)
        u = self.F.get_u()
        if (np.any(np.isnan(self.x_hat)) or np.any(np.isinf(self.x_hat)) or
            np.any(np.isnan(u)) or np.any(np.isinf(u))):
            self.is_diverged = True
            self.divergence_step = self.F.t
            print(f"  {self.filter_type}-{self.lqr_type}: DIVERGED at step {self.F.t} (NaN/Inf values)")
            return True
        
        # Check distance from actual position to current goal point
        if hasattr(self, 'x_goal') and self.x_goal is not None:
            # Get actual vehicle position (first 2 states: x, y)
            actual_pos = self.F.get_x()[:2, :].flatten()
            # Get current goal position (first 2 states: x, y)
            goal_pos = self.x_goal[:2, :].flatten()
            
            # Calculate distance to goal
            distance_to_goal = np.linalg.norm(actual_pos - goal_pos)
            
            # Check if distance exceeds threshold
            if distance_to_goal > self.divergence_threshold:
                self.is_diverged = True
                self.divergence_step = self.F.t
                print(f"  {self.filter_type}-{self.lqr_type}: DIVERGED at step {self.F.t} (distance to goal: {distance_to_goal:.2f} > {self.divergence_threshold})")
                return True
            
        return False

    def check_system_convergence(self, tolerance_factor=0.01, window_size=50):
        """
        Check if the system has converged based on recent performance history.
        
        Args:
            tolerance_factor: convergence threshold as fraction of initial cost
            window_size: number of recent steps to analyze
        
        Returns:
            bool: True if converged
        """
        if len(self.convergence_history['cost']) < 100:  # Need minimum history
            return False
            
        if self.convergence_history['is_converged']:
            return True
            
        # Dynamic tolerance based on initial cost
        initial_cost = np.mean(self.convergence_history['cost'][:10])
        tolerance = initial_cost * tolerance_factor
        
        # Use improved convergence detection
        conv_step, conv_metrics = detect_convergence(
            self.convergence_history['cost'], 
            window_size=window_size, 
            tolerance=tolerance,
            min_steps=100
        )
        
        if conv_step is not None:
            self.convergence_history['is_converged'] = True
            self.convergence_history['convergence_step'] = conv_step
            self.convergence_history['convergence_metrics'] = conv_metrics
            return True
            
        return False
    
    ######################################
    # path tracking related functions
    ######################################
    def attach_path(self, path, map_to_state=None, lookahead=5, advance_thresh=0.5):
        """
        path: dict from your generate_reference_path (contains arrays like 'x','y','vx','vy',...)
        map_to_state: function k -> (n,1) numpy column that maps anchor k into full system state
        lookahead: how many anchors to look ahead when picking the goal (smoothing)
        advance_thresh: distance threshold to advance to the next anchor
        """
        self.ref = {
            "path": path,
            "N": len(path["x"]),
            "k": 0,                        # current anchor index
            "lookahead": lookahead,
            "advance_thresh": advance_thresh,
            "last_advance_step": 0,        # track when goal was last advanced
            "stuck_threshold": 50,         # max steps before forced advancement
            "accessed_points": set(),      # track which anchor points have been accessed
        }
        # default mapper: [x, y] into the first 2 states, rest zeros (position-only goals)
        def _default_map(k):
            xg = np.zeros((self.n, 1))
            # Only set position targets, not velocity targets
            xg[0, 0] = float(path["x"][k])  # target x position
            xg[1, 0] = float(path["y"][k])  # target y position
            # velocity and acceleration targets remain zero (let controller decide)
            return xg
        self.ref["map"] = _default_map if map_to_state is None else map_to_state
        # initialize goal
        self.x_goal = self._blend_goal(self.ref["k"])
        self.path_index = self.ref["k"]            # <-- keep path_index in sync

    def _nearest_anchor(self):
        """Find index of nearest anchor to current *estimated* position (first 2 states assumed x,y)."""
        # if your state layout differs, change how pos is extracted
        pos = self.x_hat[:2, :].flatten()
        X = np.vstack([self.ref["path"]["x"], self.ref["path"]["y"]]).T
        d2 = np.sum((X - pos[None,:])**2, axis=1)
        return int(np.argmin(d2))
    
    def _next_anchor_in_sequence(self):
        """Get the next anchor point in the fixed sequence."""
        # Find the next unaccessed anchor point in sequence
        for i in range(self.ref["N"]):
            if i not in self.ref["accessed_points"]:
                return i
        
        # All points accessed, return the last point
        return self.ref["N"] - 1

    def _blend_goal(self, k0):
        """
        Return a smoothed goal as a convex combo of k0..k0+lookahead.
        This keeps your *single-step* LQR happy while looking slightly ahead.
        """
        L = self.ref["lookahead"]
        idxs = np.arange(k0, min(k0+L+1, self.ref["N"]))
        # exponential weights (heavier weight for near-term anchors)
        lamb = 0.7
        w = np.array([lamb**i for i in range(len(idxs))], dtype=float)
        w /= w.sum()
        # weighted blend of anchor states
        xg = np.zeros((self.n, 1))
        for wi, kk in zip(w, idxs):
            xg += wi * self.ref["map"](kk)
        return xg

    def update_goal_from_path(self):
        if not hasattr(self, "ref"):
            return
        
        # Initialize with first anchor point in sequence on first step
        if self.ref["k"] == 0 and self.F.t == 0:
            self.ref["k"] = 0  # Start with first anchor point

        # Get current vehicle position
        x = self.x_hat[:2, :].ravel()
        
        # Check if current anchor point has been accessed
        current_anchor = self.ref["k"]
        p = self.ref["map"](current_anchor)[:2, :].ravel()
        distance_to_current = np.linalg.norm(x - p)
        
        # Mark current anchor as accessed if close enough
        if distance_to_current < self.ref["advance_thresh"]:
            self.ref["accessed_points"].add(current_anchor)
            self.ref["last_advance_step"] = self.F.t
        
        # Get the next anchor point in sequence
        next_anchor = self._next_anchor_in_sequence()
        
        # Update current anchor to next anchor in sequence
        if next_anchor != current_anchor:
            self.ref["k"] = next_anchor
        
        # Time-based advancement (prevent getting stuck)
        steps_since_advance = self.F.t - self.ref["last_advance_step"]
        if steps_since_advance > self.ref["stuck_threshold"]:
            # Force advancement to next anchor in sequence
            self.ref["k"] = self._next_anchor_in_sequence()
            self.ref["last_advance_step"] = self.F.t

        self.x_goal = self._blend_goal(self.ref["k"])
        self.path_index = self.ref["k"]

    def get_current_reference_state(self, idx=None, path=None):
        """
        Return a full state vector made from a path entry.
        If a path is attached, prefer the blended goal.
        """
        if hasattr(self, "ref"):
            # Use the blended goal for consistency with control
            return self.x_goal

        # Fallback: build from provided (idx, path) like your old code expects
        if (idx is not None) and (path is not None):
            k = max(0, min(idx, len(path['x']) - 1))
            xg = np.zeros((self.n, 1))
            vals = []
            for key in ['x','y','vx','vy']:
                if key in path: vals.append(float(path[key][k]))
            if vals:
                vals = np.array(vals).reshape(-1,1)
                xg[:len(vals), :] = vals
            return xg

        # Last-resort: current goal
        return self.x_goal

    def get_hierarchical_reference_point(self):
        """Stub for hierarchical mode; return current goal unless you add waypoints."""
        return self.x_goal
    
    ######################################
    # main function
    ######################################
    def run_sim(self):
        rmse_list, var_list, cost_list = [], [], []
        tracking_error_list, trajectory_x, trajectory_y = [], [], []

        ref_traj_x, ref_traj_y = [], []
        
        for step in tqdm(range(1, self.H + 1, 1)):
            # Divergence guard
            if self.check_divergence():
                # For diverged controllers, fill remaining steps with NaN to indicate no valid data
                for _ in range(step, self.H + 1):
                    rmse_list.append(float('nan'))
                    var_list.append(float('nan'))
                    cost_list.append(float('nan'))
                    tracking_error_list.append(float('nan'))
                    trajectory_x.append(trajectory_x[-1] if trajectory_x else 0.0)
                    trajectory_y.append(trajectory_y[-1] if trajectory_y else 0.0)
                break

            # 1) Filter update
            self.update_lqe()

            # 2) Refresh moving goal BEFORE control
            if self.reference_path is not None:
                self.update_goal_from_path()
                x_ref_now = self.get_current_reference_state(self.path_index, self.reference_path)
            else:
                x_ref_now = self.x_goal
            
            ref_traj_x.append(x_ref_now[0,0])
            ref_traj_y.append(x_ref_now[1,0])

            # 3) Control + plant step
            if self.lqr_type != 'None':
                self.update_lqr()
                self.forward_state()

            # 4) Log actual trajectory
            x_act = self.F.get_x()
            trajectory_x.append(x_act[0].item())
            trajectory_y.append(x_act[1].item())
            
            # 4.5) Check if end point reached
            if hasattr(self, 'reference_path') and self.reference_path is not None:
                # Check if we're at the last anchor point and close enough to it
                current_anchor = self.ref["k"]
                if current_anchor == self.ref["N"] - 1:  # At the last anchor point
                    # Check distance to the end point
                    end_point = np.array([[self.ref["path"]["x"][-1]], [self.ref["path"]["y"][-1]]])
                    distance_to_end = np.linalg.norm(x_act[:2] - end_point)
                    if distance_to_end < self.ref["advance_thresh"]:
                        print(f"  {self.filter_type}-{self.lqr_type}: Reached end point at step {step}")
                        break
            

            # 5) Estimation RMSE
            rmse_list.append(np.linalg.norm(x_act - self.x_hat).item())

            # 6) Tracking error (use blended goal for consistency)
            if self.reference_path is not None:
                trk_err = np.linalg.norm(x_act - x_ref_now).item()
            else:
                trk_err = 0.0
            tracking_error_list.append(trk_err)
            self.convergence_history['tracking_error'].append(trk_err)

            # 7) Variance
            if self.filter_type == 'qkf':
                var = float(np.trace(self.Pz_est[:self.n, :self.n]))
            elif self.filter_type in ['ekf','kf','ukf']:
                var = float(np.trace(self.P_est))
            else:
                var = 0.0
            var_list.append(var)

            # 8) Stage cost (tracking vs fixed goal)
            u  = self.F.get_u()
            dx = self.x_hat - x_ref_now
            cost = float((dx.T @ self.Q[:self.n, :self.n] @ dx + u.T @ self.R @ u).item())
            cost_list.append(cost)

            # 9) Convergence bookkeeping
            self.convergence_history['cost'].append(cost)
            self.convergence_history['estimation_error'].append(np.linalg.norm(x_act - self.x_hat).item())
            self.convergence_history['control_effort'].append(float(np.linalg.norm(u)))

            if step > 200 and self.check_system_convergence():
                if step % 100 == 0:
                    print(f"  {self.filter_type}-{self.lqr_type}: Converged at step {step}")
                # keep running to fill horizon

        # Cost-to-go
        cost_to_go_list = []
        
        # Check if divergence occurred
        if self.is_diverged and self.divergence_step is not None and self.divergence_step > 0:
            # Calculate cost-to-go for the entire horizon, but with special handling for diverged portion
            
            # First, calculate cost-to-go normally for the entire list
            acc = 0.0
            temp_cost_to_go = []
            for c in reversed(cost_list):
                acc += c
                temp_cost_to_go.append(acc)
            temp_cost_to_go.reverse()
            
            # Now modify the diverged portion to be horizontal
            # The cost-to-go at divergence step should remain constant afterwards
            divergence_cost_to_go = temp_cost_to_go[self.divergence_step - 1] if self.divergence_step > 0 else temp_cost_to_go[0]
            
            cost_to_go_list = temp_cost_to_go.copy()
            # Make the cost-to-go horizontal from divergence point onwards
            for i in range(self.divergence_step, len(cost_to_go_list)):
                cost_to_go_list[i] = divergence_cost_to_go
        else:
            # No divergence, calculate normally
            acc = 0.0
            for c in reversed(cost_list):
                acc += c
                cost_to_go_list.append(acc)
            cost_to_go_list.reverse()

        return rmse_list, var_list, cost_list, cost_to_go_list, tracking_error_list, trajectory_x, trajectory_y, ref_traj_x, ref_traj_y

def one_trial(process_noise_scale=1e-3, measurement_noise_scale=1e-3, nonlinearity_scale=1, Q_scale=1, R_scale=1, rand_seed=None, trial_num=0):
    # --- reproducibility ---
    if rand_seed is not None:
        np.random.seed(rand_seed)

    # --- sim / model config ---
    dt = 0.01  # Small update step for precise control
    n1, n2, p, m = 2, 2, 3, 2     # [x,y] + [vx,vy], 3 inputs, 2 measurements
    n = n1 + n2
    path_type = 'circle'              # horizon = length of reference
    
    # --- reference path ---
    # Define path parameters
    anchor_points = 2  # Just start and end points  # Number of anchor points (reduced for point-to-point navigation)
    path_length = 100.0  # Constant path length in time units
    H = 1000  # Large horizon for point-to-point navigation (timeout limit)
    
    # Fixed start and end points (constant for all controllers)
    start_point = (0.0, 0.0)  # Always start at origin
    end_point = (20.0, 10.0)  # Fixed end point
    
    ref = generate_reference_path(
        path_type=path_type, 
        anchor_points=anchor_points, 
        dt=dt, 
        scale=10.0, 
        path_length=path_length,
        start_point=start_point,
        end_point=end_point
    )

    # --- dynamics / sensor ---
    A_E, A_S, B_S, W = create_vehicle_dynamics_matrices(dt=dt, process_noise_scale=process_noise_scale)
    C, M, V = create_sensor_matrices_for_tracking(n=n, m=m, measurement_noise_scale=measurement_noise_scale, nonlinearity_scale=nonlinearity_scale)

    # --- costs ---
    Q = generate_random_symmetric_matrix(n+n**2, scale=Q_scale)
    
    # Q matrix for position-focused error tracking
    # Much smaller scaling to prevent huge control inputs
    Q[:2, :2] *= 1.0    # Moderate penalty for position errors (x, y)
    Q[2:4, 2:4] *= 0.1  # Low penalty for velocity errors (vx, vy)
    Q[4:, 4:] *= 0.01   # Very low penalty for augmented state part
    Q[:2, 2:4] *= 0.0   # Zero cross-coupling between position and velocity
    Q[2:4, :2] *= 0.0   # Zero cross-coupling between velocity and position
    
    # R matrix for position-focused error tracking
    # Allow more control effort for better tracking
    R = generate_random_symmetric_matrix(p, scale=R_scale)
    R *= 1.0  # Lower control penalty for more aggressive control

    goal_state = np.zeros((n, 1))  # unused in tracking except as a default

    # --- controller A: QKF + Aug-LQR-Analytic ---
    lqg_A = LQG_PathTracking(
        n1, n2, p, W, A_E, A_S, B_S,
        C, M, V, Q, R,
        goal_state=goal_state,
        H=H + 1,
        filter_type='qkf',
        lqr_type='aug_analytic',
        reference_path=ref,
        dt=dt
    )
    start_time_A = time.time()
    rmse_A, var_A, stage_cost_A, cost_to_go_A, err_A, x_A, y_A, rx_A, ry_A = lqg_A.run_sim()
    end_time_A = time.time()
    time_A = end_time_A - start_time_A
    u_A = list(lqg_A.convergence_history["control_effort"].copy())

    # --- controller B (baseline): QKF + Aug-LQR-Numeric ---
    lqg_B = LQG_PathTracking(
        n1, n2, p, W, A_E, A_S, B_S,
        C, M, V, Q, R,
        goal_state=goal_state,
        H=H + 1,
        filter_type='qkf',
        lqr_type='aug_numeric',
        reference_path=ref,
        dt=dt
    )
    start_time_B = time.time()
    rmse_B, var_B, stage_cost_B, cost_to_go_B, err_B, x_B, y_B, rx_B, ry_B = lqg_B.run_sim()
    end_time_B = time.time()
    time_B = end_time_B - start_time_B
    u_B = list(lqg_B.convergence_history["control_effort"].copy())

    # --- controller C (baseline): EKF + orig LQR ---
    lqg_C = LQG_PathTracking(
        n1, n2, p, W, A_E, A_S, B_S,
        C, M, V, Q, R,
        goal_state=goal_state,
        H=H + 1,
        filter_type='ekf',
        lqr_type='orig',
        reference_path=ref,
        dt=dt
    )
    start_time_C = time.time()
    rmse_C, var_C, stage_cost_C, cost_to_go_C, err_C, x_C, y_C, rx_C, ry_C = lqg_C.run_sim()
    end_time_C = time.time()
    time_C = end_time_C - start_time_C
    u_C = list(lqg_C.convergence_history["control_effort"].copy())
    
    # --- controller D (baseline): UKF + orig LQR ---
    lqg_D = LQG_PathTracking(
        n1, n2, p, W, A_E, A_S, B_S,
        C, M, V, Q, R,
        goal_state=goal_state,
        H=H + 1,
        filter_type='ukf',
        lqr_type='orig',
        reference_path=ref,
        dt=dt
    )
    start_time_D = time.time()
    rmse_D, var_D, stage_cost_D, cost_to_go_D, err_D, x_D, y_D, rx_D, ry_D = lqg_D.run_sim()
    end_time_D = time.time()
    time_D = end_time_D - start_time_D
    u_D = list(lqg_D.convergence_history["control_effort"].copy())
    
    
    # --- pad to H if any run diverged early ---
    def pad_to_H(lst, H):
        if len(lst) < H:
            lst.extend([None] * (H - len(lst)))
        return lst

    for series in [u_A, rmse_A, var_A, stage_cost_A, cost_to_go_A, err_A, x_A, y_A, rx_A, ry_A,
                   u_B, rmse_B, var_B, stage_cost_B, cost_to_go_B, err_B, x_B, y_B, rx_B, ry_B,
                   u_C, rmse_C, var_C, stage_cost_C, cost_to_go_C, err_C, x_C, y_C, rx_C, ry_C,
                   u_D, rmse_D, var_D, stage_cost_D, cost_to_go_D, err_D, x_D, y_D, rx_D, ry_D]:
        pad_to_H(series, H)

    # --- pack & save results (all four controllers) ---
    results = {
        "ref": ref, "dt": dt, "path_type": path_type,
        "A": {"label": "QKF + Aug-LQR-Analytic", "rmse": rmse_A, "var": var_A, "stage_cost": stage_cost_A, "cost_to_go": cost_to_go_A,
              "track_err": err_A, "traj_x": x_A, "traj_y": y_A, "ref_traj_x": rx_A, "ref_traj_y": ry_A, "u_norm": u_A,
              "is_diverged": lqg_A.is_diverged, "divergence_step": lqg_A.divergence_step, "execution_time": time_A},
        "B": {"label": "QKF + Aug-LQR-Numeric", "rmse": rmse_B, "var": var_B, "stage_cost": stage_cost_B, "cost_to_go": cost_to_go_B,
              "track_err": err_B, "traj_x": x_B, "traj_y": y_B, "ref_traj_x": rx_B, "ref_traj_y": ry_B, "u_norm": u_B,
              "is_diverged": lqg_B.is_diverged, "divergence_step": lqg_B.divergence_step, "execution_time": time_B},
        "C": {"label": "EKF + Orig-LQR", "rmse": rmse_C, "var": var_C, "stage_cost": stage_cost_C, "cost_to_go": cost_to_go_C,
              "track_err": err_C, "traj_x": x_C, "traj_y": y_C, "ref_traj_x": rx_C, "ref_traj_y": ry_C, "u_norm": u_C,
              "is_diverged": lqg_C.is_diverged, "divergence_step": lqg_C.divergence_step, "execution_time": time_C},
        "D": {"label": "UKF + Orig-LQR", "rmse": rmse_D, "var": var_D, "stage_cost": stage_cost_D, "cost_to_go": cost_to_go_D,
              "track_err": err_D, "traj_x": x_D, "traj_y": y_D, "ref_traj_x": rx_D, "ref_traj_y": ry_D, "u_norm": u_D,
              "is_diverged": lqg_D.is_diverged, "divergence_step": lqg_D.divergence_step, "execution_time": time_D},
    }

    # Print timing information
    print(f"\n=== Execution Times (Trial {trial_num}) ===")
    print(f"QKF + Aug-LQR-Analytic:  {time_A:.4f} seconds")
    print(f"QKF + Aug-LQR-Numeric:  {time_B:.4f} seconds")
    print(f"EKF + Orig-LQR:          {time_C:.4f} seconds")
    print(f"UKF + Orig-LQR:          {time_D:.4f} seconds")
    print("=" * 40)

    pkl_path = os.path.join(pkl_dir, f"tracking_{path_type}-{trial_num}.pkl")
    with open(pkl_path, "wb") as f:
        pkl.dump(results, f)

    # --- plots ---
    # Create time array based on the maximum length of any result array
    max_length = max(len(results["A"]["cost_to_go"]), len(results["B"]["cost_to_go"]), 
                     len(results["C"]["cost_to_go"]), len(results["D"]["cost_to_go"]))
    T = np.arange(max_length) * dt
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.4)

    # Define consistent colors for each controller
    colors = {
        "A": "#1f77b4",  # Blue for QKF + Aug-LQR-Analytic
        "B": "#ff7f0e",  # Orange for QKF + Aug-LQR-Numeric  
        "C": "#2ca02c",  # Green for EKF + Orig-LQR
        "D": "#d62728"   # Red for UKF + Orig-LQR
    }

    # Helper function to get line style based on divergence
    def get_line_style(controller_key):
        return '--' if results[controller_key]["is_diverged"] else '-'

    # 1) XY path
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ref['x'], ref['y'], '--', lw=1.2, alpha=0.7, label="Reference (global)", color='gray')
    # plot the anchor points (show every 20th point to avoid overcrowding)
    step_size = max(1, len(ref['x']) // 100)  # Show at most 20 anchor points
    ax1.plot(ref['x'][::step_size], ref['y'][::step_size], 'o', lw=1.2, alpha=0.7, color='gray', markersize=3)

    # plot reference trajectory with consistent colors and dashed lines for diverged controllers
    ax1.plot(np.array(results["A"]["traj_x"], dtype=float),
             np.array(results["A"]["traj_y"], dtype=float),
             get_line_style("A"), lw=2.0, label=results["A"]["label"], color=colors["A"])
    ax1.plot(np.array(results["B"]["traj_x"], dtype=float),
             np.array(results["B"]["traj_y"], dtype=float),
             get_line_style("B"), lw=2.0, label=results["B"]["label"], color=colors["B"])
    ax1.plot(np.array(results["C"]["traj_x"], dtype=float),
             np.array(results["C"]["traj_y"], dtype=float),
             get_line_style("C"), lw=2.0, label=results["C"]["label"], color=colors["C"])
    ax1.plot(np.array(results["D"]["traj_x"], dtype=float),
             np.array(results["D"]["traj_y"], dtype=float),
             get_line_style("D"), lw=2.0, label=results["D"]["label"], color=colors["D"])

    ax1.set_title("Path Tracking (XY)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.axis("equal")
    ax1.grid(True, ls=":")
    ax1.legend(loc='best', fontsize=9, ncol=1, framealpha=0.9)

    # 2) Cost plot (overall cumulative cost)
    ax2 = fig.add_subplot(gs[0, 1])
    # Truncate arrays to match time array length
    T_trunc = T[:len(results["A"]["cost_to_go"])]
    ax2.plot(T_trunc, results["A"]["cost_to_go"], get_line_style("A"), lw=1.5, label=results["A"]["label"], color=colors["A"])
    T_trunc = T[:len(results["B"]["cost_to_go"])]
    ax2.plot(T_trunc, results["B"]["cost_to_go"], get_line_style("B"), lw=1.5, label=results["B"]["label"], color=colors["B"])
    T_trunc = T[:len(results["C"]["cost_to_go"])]
    ax2.plot(T_trunc, results["C"]["cost_to_go"], get_line_style("C"), lw=1.5, label=results["C"]["label"], color=colors["C"])
    T_trunc = T[:len(results["D"]["cost_to_go"])]
    ax2.plot(T_trunc, results["D"]["cost_to_go"], get_line_style("D"), lw=1.5, label=results["D"]["label"], color=colors["D"])
    ax2.set_title("Overall Cumulative Cost")
    ax2.set_xlabel("time [s]"); ax2.set_ylabel("Cumulative Cost")
    ax2.grid(True, ls=":"); ax2.legend(loc='best', fontsize=9, ncol=1, framealpha=0.9)

    # 3) Tracking error
    ax3 = fig.add_subplot(gs[1, 0])
    # Truncate arrays to match time array length
    T_trunc = T[:len(results["A"]["track_err"])]
    ax3.plot(T_trunc, results["A"]["track_err"], get_line_style("A"), lw=1.5, label=results["A"]["label"], color=colors["A"])
    T_trunc = T[:len(results["B"]["track_err"])]
    ax3.plot(T_trunc, results["B"]["track_err"], get_line_style("B"), lw=1.5, label=results["B"]["label"], color=colors["B"])
    T_trunc = T[:len(results["C"]["track_err"])]
    ax3.plot(T_trunc, results["C"]["track_err"], get_line_style("C"), lw=1.5, label=results["C"]["label"], color=colors["C"])
    T_trunc = T[:len(results["D"]["track_err"])]
    ax3.plot(T_trunc, results["D"]["track_err"], get_line_style("D"), lw=1.5, label=results["D"]["label"], color=colors["D"])
    ax3.set_title("Tracking Error ‖x - x_ref‖")
    ax3.set_xlabel("time [s]"); ax3.set_ylabel("error")
    ax3.grid(True, ls=":"); ax3.legend(loc='best', fontsize=9, ncol=1, framealpha=0.9)

    # 4) Estimation RMSE
    ax4 = fig.add_subplot(gs[1, 1])
    # Truncate arrays to match time array length
    T_trunc = T[:len(results["A"]["rmse"])]
    ax4.plot(T_trunc, results["A"]["rmse"], get_line_style("A"), lw=1.5, label=results["A"]["label"], color=colors["A"])
    T_trunc = T[:len(results["B"]["rmse"])]
    ax4.plot(T_trunc, results["B"]["rmse"], get_line_style("B"), lw=1.5, label=results["B"]["label"], color=colors["B"])
    T_trunc = T[:len(results["C"]["rmse"])]
    ax4.plot(T_trunc, results["C"]["rmse"], get_line_style("C"), lw=1.5, label=results["C"]["label"], color=colors["C"])
    T_trunc = T[:len(results["D"]["rmse"])]
    ax4.plot(T_trunc, results["D"]["rmse"], get_line_style("D"), lw=1.5, label=results["D"]["label"], color=colors["D"])
    ax4.set_title("Estimation RMSE ‖x - x̂‖")
    ax4.set_xlabel("time [s]"); ax4.set_ylabel("RMSE")
    ax4.grid(True, ls=":"); ax4.legend(loc='best', fontsize=9, ncol=1, framealpha=0.9)

    # Add divergence info to title
    divergence_info = []
    for key, controller in [("A", "QKF+Aug-LQR-Analytic"), ("B", "QKF+Aug-LQR-Numeric"), ("C", "EKF+Orig-LQR"), ("D", "UKF+Orig-LQR")]:
        if results[key]["is_diverged"]:
            divergence_info.append(f"{controller} (DIVERGED at step {results[key]['divergence_step']})")
        else:
            divergence_info.append(controller)
    
    title = f"LQG Path Tracking — {path_type}  ({', '.join(divergence_info)})"
    fig.suptitle(title, y=0.95, fontsize=12)
    
    # Use subplots_adjust for more reliable layout without warnings
    fig.subplots_adjust(left=0.08, right=0.95, top=0.8, bottom=0.08, hspace=0.5, wspace=0.4)

    fig_path = os.path.join(perf_dir, f"tracking_{path_type}-{trial_num}.png")
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    # plt.show()


if __name__ == "__main__":
    one_trial()
