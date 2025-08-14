import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
from typing import Literal
from tqdm import tqdm
from stateDynamics import *
import pickle as pkl

small_value = 1e-6  # Small value to prevent numerical issues

perf_dir = 'D:/AC/UCLA/ECE/UCLA_LEMUR/nonlinear_LQG/LQG_QKF/test_scenarios/perf'
os.makedirs(perf_dir, exist_ok=True)

pkl_dir = 'D:/AC/UCLA/ECE/UCLA_LEMUR/nonlinear_LQG/LQG_QKF/test_scenarios/pkl/'
os.makedirs(pkl_dir, exist_ok=True)

# Path tracking related functions
def generate_reference_path(path_type='figure8', num_points=1000, dt=0.1, scale=10.0):
    """
    Generate reference paths for tracking.
    
    Args:
        path_type: Type of path ('figure8', 'circle', 'straight', 'sine_wave', 'racetrack')
        num_points: Number of points in the path
        dt: Time step
        scale: Scaling factor for path size
    
    Returns:
        path: Dictionary containing x, y, vx, vy, ax, ay reference trajectories
    """
    t = np.linspace(0, num_points * dt, num_points)
    
    if path_type == 'figure8':
        # Figure-8 path starting at origin
        x_ref = scale * np.sin(t * 0.5)
        y_ref = scale * np.sin(t) * 0.5
        # Shift to start at origin
        x_ref = x_ref - x_ref[0]
        y_ref = y_ref - y_ref[0]
        
    elif path_type == 'circle':
        # Very slow circular path starting at origin
        radius = scale * 0.5  # smaller radius
        omega = 0.05  # much slower angular velocity
        x_ref = radius * np.cos(omega * t + np.pi)  # Start at (radius, 0) = origin offset
        y_ref = radius * np.sin(omega * t + np.pi)
        # Shift to start at origin
        x_ref = x_ref - x_ref[0]
        y_ref = y_ref - y_ref[0]
        
    elif path_type == 'straight':
        # Straight line path starting at origin
        x_ref = scale * t / num_points
        y_ref = np.zeros_like(t)
        # Already starts at origin
        
    elif path_type == 'sine_wave':
        # Sinusoidal path starting at origin
        x_ref = scale * t / num_points
        y_ref = scale * 0.3 * np.sin(2 * np.pi * t / (num_points * dt * 0.2))
        # Shift to start at origin
        x_ref = x_ref - x_ref[0]
        y_ref = y_ref - y_ref[0]
        
    elif path_type == 'racetrack':
        # Racetrack-like path (rounded rectangle)
        period = num_points * dt
        t_norm = (t % period) / period  # normalize to [0, 1]
        
        x_ref = np.zeros_like(t)
        y_ref = np.zeros_like(t)
        
        for i, tn in enumerate(t_norm):
            if tn < 0.25:  # Bottom straight
                x_ref[i] = scale * (4 * tn)
                y_ref[i] = 0
            elif tn < 0.5:  # Right curve
                angle = np.pi * (tn - 0.25) / 0.25
                x_ref[i] = scale
                y_ref[i] = scale * 0.5 * (1 - np.cos(angle))
            elif tn < 0.75:  # Top straight
                x_ref[i] = scale * (1 - 4 * (tn - 0.5))
                y_ref[i] = scale * 0.5
            else:  # Left curve
                angle = np.pi * (tn - 0.75) / 0.25
                x_ref[i] = 0
                y_ref[i] = scale * 0.5 * (1 + np.cos(angle))
        
        # Shift to start at origin
        x_ref = x_ref - x_ref[0]
        y_ref = y_ref - y_ref[0]
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
    
    # Process noise covariance - extremely small for stability
    W = np.eye(4) * process_noise_scale
    W[:2, :2] *= 0.001  # Extremely low noise for position
    W[2:, 2:] *= 0.01   # Very low noise for velocity
    
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
    # Linear measurement matrix (GPS-like measurements of position)
    C = np.zeros((m, n))
    C[0, 0] = 1.0  # x position
    C[1, 1] = 1.0  # y position
    
    # Quadratic measurement matrices (small but meaningful nonlinear effects)
    M = np.zeros((m, n, n))
    for i in range(m):
        M[i] = generate_random_symmetric_matrix(n, scale=nonlinearity_scale * 0.1)  # Small but meaningful nonlinearity
    
    # Measurement noise covariance
    V = np.eye(m) * measurement_noise_scale
    
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
            [0, 0, 0.95, 0],           # vx with moderate damping 
            [0, 0, 0, 0.95]            # vy with moderate damping
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

class LQG:
    def __init__(self, n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, goal_state, H = 50, 
                 filter_type: Literal['qkf', 'ekf', 'kf', 'ukf'] = 'qkf',
                 lqr_type: Literal['orig', 'aug_analytic', 'aug_numeric', 'None'] = 'orig',
                 reference_path=None, dt=0.1, tracking_mode=False):
        
        self.filter_type = filter_type
        self.lqr_type = lqr_type
        self.tracking_mode = tracking_mode
        self.dt = dt
        
        # Path tracking specific settings
        if tracking_mode and reference_path is not None:
            self.reference_path = reference_path
            self.path_index = 0  # Current index in reference path
            self.tracking_errors = []  # Store tracking errors over time
            # Hierarchical control parameters
            self.hierarchical_mode = False
            self.local_horizon = 50  # H1: Local LQR horizon
            self.global_step = 0     # Current global step
            self.local_step = 0      # Current step within local horizon
            self.current_waypoint = None
            self.next_waypoint = None
        else:
            self.reference_path = None
            self.path_index = 0
            self.tracking_errors = []
            self.hierarchical_mode = False
        
        # dynamics setting - modified for path tracking
        if tracking_mode:
            self.F = PathTrackingDynamics(n1, n2, p, W, A_E, A_S, B_S, dt)
            # Initialize actual system state at reference path starting point
            if reference_path is not None:
                initial_state = np.array([[reference_path['x'][0]], 
                                        [reference_path['y'][0]], 
                                        [reference_path['vx'][0]], 
                                        [reference_path['vy'][0]]])
                self.F.set_x(initial_state)
        else:
            self.F = StateDynamics(n1, n2, p , W, A_E, A_S, B_S)
        n = n1 + n2 # state size
        
        # sensor settings
        self.sensor = sensor(C, M, V)
        self.V = self.sensor.get_V()
        
        # state settings
        self.A = self.F.get_A()
        self.B = self.F.get_B()
        self.n1 = n1 
        self.n2 = n2
        self.n = n
        self.p = self.F.get_input_size() # control input size
        self.W = self.F.get_W()
        
        # augmented state settings
        mu_tilde_u = (np.eye(n+n**2) - self.F.get_A_tilde()).T @ self.F.get_mu_tilde() # shape (n+n^2, 1)
        self.Z_est = mu_tilde_u # initialize estimated augmented state vector
        I = np.eye(n**2 * (n+1)**2) # shape (n^2(n+1)^2, n^2(n+1)^2)
        Phi_tilde = self.F.get_A_tilde() # shape (n+n^2, n+n^2)
        Sigma_tilde = self.F.get_Sigma_tilde() # shape (n+n^2, n+n^2)
        vec_sigma_tilde_u = (I - np.kron(Phi_tilde, Phi_tilde)) @ Vec(Sigma_tilde) # shape (n^2(n+1)^2, 1)
        self.Pz_est = invVec(vec_sigma_tilde_u) # estimation error covariance matrix
        
        # horizon
        self.H = H
        
        # states
        if tracking_mode and reference_path is not None:
            # Initialize state estimate at reference path starting point
            self.x_hat = np.array([[reference_path['x'][0]], 
                                  [reference_path['y'][0]], 
                                  [reference_path['vx'][0]], 
                                  [reference_path['vy'][0]]])
        else:
            self.x_hat = np.zeros((self.n, 1)) # estimated state vector
        self.z_hat = np.zeros((self.n + self.n**2, 1)) # estimated augmented state vector
        self.x_goal = np.zeros((self.n, 1)) # goal state vector
        
        # lqr
        self.x_goal = goal_state
        self.Q = Q.astype(np.float64)
        self.R = R.astype(np.float64)
        self.P_lqr = Q.copy()[:self.n, :self.n] # cost-to-go matrix for LQR
        
        # lqe - better initial covariance
        if self.tracking_mode:
            # For path tracking, initialize with more reasonable uncertainty
            self.P_est = np.diag([1.0, 1.0, 0.5, 0.5])  # [x, y, vx, vy] uncertainties
        else:
            self.P_est = np.eye(self.n) * small_value  # estimation error covariance matrix 
        
        # convergence tracking
        self.convergence_history = {
            'cost': [],
            'estimation_error': [],
            'control_effort': [],
            'tracking_error': [],  # Add tracking error for path following
            'is_converged': False,
            'convergence_step': None,
            'convergence_metrics': {}
        }
        
        # Divergence detection
        self.divergence_threshold = 1e6  # Threshold for divergence detection
        self.is_diverged = False
        self.divergence_step = None 
    
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
            z_goal = np.vstack((x_goal, Vec(x_goal @ x_goal.T))) # shape (n+n^2, 1)
            z_try = np.vstack((x_try, Vec(x_try @ x_try.T))) # shape (n+n^2, 1)
            z_nom = np.vstack((x_nom, Vec(x_nom @ x_nom.T))) # shape (n+n^2, 1)
            cost_try = (z_try - z_goal).T @ self.Q @ (z_try - z_goal) + u_try.T @ self.R @ u_try # cost function
            cost_nom = (z_nom - z_goal).T @ self.Q @ (z_nom - z_goal) + u_nom.T @ self.R @ u_nom # cost function
            if cost_try < cost_nom:            # cost improved?
                return u_try, x_try, alpha
            alpha *= 0.5                       # shrink step
        return u_nom, x_nom, 0.0               # no progress
    
    def update_ilqr(self, goal_state, alpha = 1, verbose=False):
        x_nominal = self.x_hat
        u_nominal = np.zeros((self.p, 1)) # nominal control input vector
         
        max_iter = 1000
        iteration = 0
        diff_u = 1e10  
        diff_cost = 1e10
        epsilon_u = 1e-6  # convergence threshold for control input change
        epsilon_cost = 1e-8  # convergence threshold for cost change
        prev_cost = 1e10
        
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
            if alpha == 0.0:  # line search failed
                if verbose:
                    print("Line search failed, stopping iteration")
                break
                
            u_nominal = u_new
            x_nominal = x_new
            prev_cost = current_cost
            
        self.F.set_u(u_new)
        return
    
    def update_lqr_analytic(self, goal_state, infinite_horizon=False):
        # For hierarchical control, use simpler Q matrix structure
        if self.hierarchical_mode:
            # Use only the state part of Q matrix (not full augmented)
            Q_local = self.Q[:self.n, :self.n]  # n x n matrix
            
            # Simple analytic LQR for point-to-point control
            A = self.F.get_A()  # shape (n, n)
            B = self.F.get_B()  # shape (n, p)
            
            # Get target waypoint
            if self.tracking_mode and self.reference_path is not None:
                target = self.get_hierarchical_reference_point()
            else:
                target = goal_state
            
            # Solve discrete-time LQR 
            try:
                P_lqr = finite_horizon_lqr(A, B, Q_local, self.R, N=self.local_horizon)
                K = -np.linalg.pinv(self.R + B.T @ P_lqr @ B) @ B.T @ P_lqr @ A
                u_new = K @ (target - self.x_hat)
                self.F.set_u(u_new)
                return
            except Exception as e:
                raise ValueError(f"Hierarchical analytic LQR failed: {e}")
        
        # Original analytic LQR for non-hierarchical mode
        I_p = np.eye(self.p)  # shape (p, p)
        I_p2 = np.eye(self.p ** 2)  # shape (p^2, p^2)
        B = self.F.get_B()  # shape (n, p)
        A = self.F.get_A()  # shape (n, n)
        x = self.F.get_x()  # shape (n, 1), current state vector
        
        # Check for NaN or infinite values in inputs
        if np.any(np.isnan(B)) or np.any(np.isinf(B)):
            raise ValueError("B matrix contains NaN or infinite values")
        if np.any(np.isnan(A)) or np.any(np.isinf(A)):
            raise ValueError("A matrix contains NaN or infinite values")
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            raise ValueError("State vector x contains NaN or infinite values")
        
        # commutation matrix for I_p kron u
        T = np.zeros((self.p * self.p, self.p * self.p)) # shape (p^2, p^2)
        for i in range(self.p):
            for j in range(self.p):
                e_ij = np.zeros((self.p, self.p))
                e_ij[i, j] = 1
                vec_e_ij = e_ij.T.flatten()  # transpose before vec
                T[:, i * self.p + j] = vec_e_ij

        M = np.kron(B, B) @ (I_p2 + T) # shape (n^2, p^2)
        q = Vec(self.Q) # shape (n^2, 1)
        
        S = np.zeros((self.p, self.p))  # shape (p, p)
        for i in range(self.p):
            e_i = np.zeros((self.p, 1)) # shape (p, 1)
            e_i[i] = 1
            term1 = (M @ np.kron(e_i, I_p)) # shape (n^2, p)
            term2 = term1 @ q @ e_i.T  # shape (p, p)
            S += term2  # accumulate over p columns

        # Check if S + 2*R is invertible
        S_R = S + 2 * self.R
        det_S_R = np.linalg.det(S_R)
        if abs(det_S_R) < 1e-12:
            raise ValueError(f"Matrix S + 2*R is singular (det = {det_S_R:.2e})")
        
        # Check condition number
        cond_S_R = np.linalg.cond(S_R)
        if cond_S_R > 1e12:
            raise ValueError(f"Matrix S + 2*R is ill-conditioned (cond = {cond_S_R:.2e})")

        Z = np.kron(A, B) @ np.kron(x, I_p)  + np.kron(B, A) @ np.kron(I_p, x) # shape (n^2, p)
        u_new = -np.linalg.inv(S_R) @ Z.T @ q # shape (p, 1)
        
        # Check for NaN or infinite values in result
        if np.any(np.isnan(u_new)) or np.any(np.isinf(u_new)):
            raise ValueError("Control input u_new contains NaN or infinite values")
        
        self.F.set_u(u_new)

    def update_lqr_orig(self, goal_state, ):
        # LQR update only with original state, no augmented state
        # P_lqr = scipy.linalg.solve_discrete_are(self.A, self.B, self.Q[:self.n, :self.n], self.R)  # P is the fixed-point
        P_lqr = finite_horizon_lqr(self.A, self.B, self.Q[:self.n, :self.n], self.R, N=1, Qf=self.P_lqr)
        self.P_lqr = P_lqr.copy() # update cost-to-go matrix
        feedback_gain = -np.linalg.pinv(self.R + self.B.T @ P_lqr @ self.B) @ self.B.T @ P_lqr @ self.A
        
        # Use time-varying reference for path tracking
        if self.tracking_mode and self.reference_path is not None:
            if self.hierarchical_mode:
                current_ref = self.get_hierarchical_reference_point()
            else:
                current_ref = self.get_current_reference_point()
            u_new = feedback_gain @ (current_ref - self.x_hat)
        else:
            u_new = feedback_gain @ (goal_state - self.x_hat)  # control input
            
        self.F.set_u(u_new)
        return
        
    def get_current_reference_point(self):
        """Get the current reference point for path tracking."""
        if self.reference_path is None:
            return self.x_goal
            
        # Ensure path_index doesn't exceed path length
        if self.path_index >= len(self.reference_path['x']):
            self.path_index = len(self.reference_path['x']) - 1
            
        ref_point = np.array([
            [self.reference_path['x'][self.path_index]],
            [self.reference_path['y'][self.path_index]],
            [self.reference_path['vx'][self.path_index]],
            [self.reference_path['vy'][self.path_index]]
        ])
        return ref_point
    
    def advance_reference_path(self):
        """Advance to the next point in the reference path."""
        if self.hierarchical_mode:
            self.advance_hierarchical_path()
        else:
            if self.reference_path is not None and self.path_index < len(self.reference_path['x']) - 1:
                self.path_index += 1
    
    def enable_hierarchical_control(self, local_horizon=50):
        """Enable hierarchical control with specified local horizon."""
        self.hierarchical_mode = True
        self.local_horizon = local_horizon
        self.global_step = 0
        self.local_step = 0
        self.update_waypoints()
    
    def create_global_waypoints(self, num_waypoints=20):
        """Create global waypoints by subsampling the reference path."""
        if self.reference_path is None:
            return None
            
        total_points = len(self.reference_path['x'])
        waypoint_indices = np.linspace(0, total_points-1, num_waypoints, dtype=int)
        
        waypoints = {
            'x': [self.reference_path['x'][i] for i in waypoint_indices],
            'y': [self.reference_path['y'][i] for i in waypoint_indices],
            'vx': [self.reference_path['vx'][i] for i in waypoint_indices],
            'vy': [self.reference_path['vy'][i] for i in waypoint_indices]
        }
        return waypoints
    
    def update_waypoints(self):
        """Update current and next waypoints for hierarchical control."""
        if not self.hierarchical_mode or self.reference_path is None:
            return
            
        # Create waypoints if not done yet
        if not hasattr(self, 'waypoints'):
            self.waypoints = self.create_global_waypoints()
            
        # Update current and next waypoints
        if self.global_step < len(self.waypoints['x']) - 1:
            self.current_waypoint = np.array([
                [self.waypoints['x'][self.global_step]],
                [self.waypoints['y'][self.global_step]],
                [self.waypoints['vx'][self.global_step]],
                [self.waypoints['vy'][self.global_step]]
            ])
            self.next_waypoint = np.array([
                [self.waypoints['x'][self.global_step + 1]],
                [self.waypoints['y'][self.global_step + 1]],
                [self.waypoints['vx'][self.global_step + 1]],
                [self.waypoints['vy'][self.global_step + 1]]
            ])
        else:
            # Use last waypoint as both current and next
            self.current_waypoint = np.array([
                [self.waypoints['x'][-1]],
                [self.waypoints['y'][-1]],
                [self.waypoints['vx'][-1]],
                [self.waypoints['vy'][-1]]
            ])
            self.next_waypoint = self.current_waypoint.copy()
    
    def advance_hierarchical_path(self):
        """Advance hierarchical path (local step within global waypoint)."""
        self.local_step += 1
        
        # Check if we've completed the local horizon
        if self.local_step >= self.local_horizon:
            self.global_step += 1
            self.local_step = 0
            self.update_waypoints()
            # print(f"  Advanced to global step {self.global_step}")
    
    def get_hierarchical_reference_point(self):
        """Get current reference point for hierarchical control."""
        if not self.hierarchical_mode or self.next_waypoint is None:
            return self.get_current_reference_point()
            
        # For hierarchical control, always aim for the next waypoint
        return self.next_waypoint
        
    def update_lqr(self):
        if self.lqr_type == 'None':
            # No LQR update, no control input
            # self.F.set_u(np.ones((self.p, 1)))
            self.F.set_u(np.random.randn(self.p, 0))  # small random noise
            return 
        else:
            goal_state = self.x_goal
            if self.filter_type == 'ekf' or self.filter_type == 'kf' or self.filter_type == 'ukf':
                self.update_lqr_orig(goal_state)
            elif self.filter_type == 'qkf':
                if self.lqr_type == 'aug_numeric':
                    self.update_ilqr(goal_state, alpha=1)
                elif self.lqr_type == 'aug_analytic':
                    try:
                        self.update_lqr_analytic(goal_state, infinite_horizon=False)
                    except Exception as e:
                        # Fallback to numeric method if analytic fails
                        print(f"Warning: Analytic LQR failed with error: {e}")
                        print(f"  At time step: {self.F.t}")
                        self.update_ilqr(goal_state, alpha=1)
                elif self.lqr_type == 'orig':
                    self.update_lqr_orig(goal_state)
            else:
                raise ValueError("Invalid filter type. Choose 'qkf', 'ekf', 'kf', or 'ukf'.")
        return
        
    
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
            K = np.linalg.lstsq(S_reg.T, C_tilde.T, rcond=1e-3)[0].T
        except (np.linalg.LinAlgError, np.linalg.LinAlgWarning):
            # Ultimate fallback: use identity gain (no correction)
            print("Warning: UKF Kalman gain computation failed, using identity")
            K = np.eye(C_tilde.shape[0], S.shape[0]) * 0.1
        
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

    
    def forward_state(self):
        self.F.forward()
    
    def check_divergence(self):
        """
        Check if the system has diverged based on state estimates, control inputs, or covariance.
        
        Returns:
            bool: True if diverged
        """
        if self.is_diverged:
            return True
            
        # Check state estimate magnitude
        if np.any(np.abs(self.x_hat) > self.divergence_threshold):
            self.is_diverged = True
            self.divergence_step = self.F.t
            print(f"  {self.filter_type}-{self.lqr_type}: DIVERGED at step {self.F.t} (large state estimate)")
            return True
            
        # Check control input magnitude
        u = self.F.get_u()
        if np.any(np.abs(u) > self.divergence_threshold):
            self.is_diverged = True
            self.divergence_step = self.F.t
            print(f"  {self.filter_type}-{self.lqr_type}: DIVERGED at step {self.F.t} (large control input)")
            return True
            
        # Check covariance matrix trace (for non-QKF filters)
        if self.filter_type in ['ekf', 'ukf', 'kf']:
            if np.trace(self.P_est) > self.divergence_threshold:
                self.is_diverged = True
                self.divergence_step = self.F.t
                print(f"  {self.filter_type}-{self.lqr_type}: DIVERGED at step {self.F.t} (large covariance)")
                return True
        
        # Check for NaN or infinite values
        if (np.any(np.isnan(self.x_hat)) or np.any(np.isinf(self.x_hat)) or
            np.any(np.isnan(u)) or np.any(np.isinf(u))):
            self.is_diverged = True
            self.divergence_step = self.F.t
            print(f"  {self.filter_type}-{self.lqr_type}: DIVERGED at step {self.F.t} (NaN/Inf values)")
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
        
    def run_sim(self):
        rmse_list = []
        var_list = []
        cost_list = []
        tracking_error_list = []
        trajectory_x = []
        trajectory_y = []
        
        for step in tqdm(range(1, self.H + 1, 1)):
            # Check for divergence before processing
            if self.check_divergence():
                # Fill remaining steps with last known values (marked as diverged)
                for remaining_step in range(step, self.H + 1):
                    rmse_list.append(self.divergence_threshold)  # Large error to indicate divergence
                    var_list.append(self.divergence_threshold)   # Large variance to indicate divergence
                    cost_list.append(self.divergence_threshold)  # Large cost to indicate divergence
                    tracking_error_list.append(self.divergence_threshold)
                    trajectory_x.append(trajectory_x[-1] if trajectory_x else 0.0)  # Repeat last position
                    trajectory_y.append(trajectory_y[-1] if trajectory_y else 0.0)
                break
                
            self.update_lqe()
            if self.lqr_type != 'None':
                self.update_lqr()
                self.forward_state()
            
            # Advance reference path for tracking mode
            if self.tracking_mode:
                self.advance_reference_path()
            
            # Record actual trajectory
            current_state = self.F.get_x()
            trajectory_x.append(current_state[0].item())
            trajectory_y.append(current_state[1].item())
            
            # record error
            estimate_error = np.linalg.norm(current_state - self.x_hat).item() 
            rmse_list.append(estimate_error)
            
            # record tracking error if in tracking mode
            if self.tracking_mode and self.reference_path is not None:
                current_ref = self.get_current_reference_point()
                tracking_error = np.linalg.norm(self.F.get_x() - current_ref).item()
                tracking_error_list.append(tracking_error)
                self.convergence_history['tracking_error'].append(tracking_error)
            else:
                tracking_error_list.append(0.0)
                self.convergence_history['tracking_error'].append(0.0)
            
            # record variance
            if self.filter_type == 'qkf':
                var = np.trace(self.Pz_est[:self.n, :self.n])
            elif self.filter_type == 'ekf' or self.filter_type == 'kf' or self.filter_type == 'ukf':
                var = np.trace(self.P_est)
            else:
                var = 0.0  # Default case
            var_list.append(var)
            
            # record cost   
            if self.tracking_mode and self.reference_path is not None:
                # Use appropriate reference point for cost calculation
                if self.hierarchical_mode:
                    x_ref = self.get_hierarchical_reference_point()
                else:
                    x_ref = self.get_current_reference_point()
                x_est = self.x_hat
                u = self.F.get_u()
                dx = x_est - x_ref
                cost = dx.T @ self.Q[:self.n, :self.n] @ dx + u.T @ self.R @ u
            else:
                # Use fixed goal state
                x_goal = self.x_goal
                x_est = self.x_hat
                u = self.F.get_u()
                dx = x_est - x_goal
                cost = dx.T @ self.Q[:self.n, :self.n] @ dx + u.T @ self.R @ u
                
            cost_value = cost.item()
            cost_list.append(cost_value)
            
            # Update convergence history
            self.convergence_history['cost'].append(cost_value)
            self.convergence_history['estimation_error'].append(estimate_error)
            self.convergence_history['control_effort'].append(np.linalg.norm(u).item())
            
            # Check for early convergence (optional - can save computation)
            if step > 200 and self.check_system_convergence():
                if step % 100 == 0:  # Print occasionally
                    print(f"  {self.filter_type}-{self.lqr_type}: Converged at step {step}")
                # Could break here for early stopping, but continue for full simulation
            
        cost_to_go_list = []
        for i in range(len(cost_list)):
            cost_to_go = np.sum(cost_list[i:])
            cost_to_go_list.append(cost_to_go)  
            
        return rmse_list, var_list, cost_to_go_list, tracking_error_list, trajectory_x, trajectory_y
        
            

def one_trial_hierarchical_path_tracking(H=1000, local_horizon=50, noise_scale=1e-1, m_scale=1e2, 
                                        Q_scale=1.0, R_scale=1.0, path_type='figure8', rand_seed=None, dt=0.1):
    """
    Run one trial of hierarchical path tracking experiment.
    
    Args:
        H: Total simulation horizon 
        local_horizon: Local LQR horizon (H1)
        noise_scale, m_scale, Q_scale, R_scale: System parameters
        path_type: Type of reference path
        rand_seed: Random seed
        dt: Time step
    """
    n1 = 2  # position states [x, y]
    n2 = 2  # velocity states [vx, vy]
    n = n1 + n2  # total state size
    p = 3   # control inputs [ax, ay, steering]
    m = 2   # measurements [x_gps, y_gps]
    
    if rand_seed is not None:
        np.random.seed(rand_seed)
    
    # Generate reference path
    reference_path = generate_reference_path(path_type=path_type, num_points=H, dt=dt, scale=10.0)
    
    # Create vehicle dynamics matrices
    A_E, A_S, B_S, W = create_vehicle_dynamics_matrices(dt=dt, process_noise_scale=noise_scale)
    
    # Create sensor matrices for tracking
    C, M, V = create_sensor_matrices_for_tracking(n=n, m=m, 
                                                 measurement_noise_scale=noise_scale, 
                                                 nonlinearity_scale=m_scale)
    
    # Create cost matrices focused on tracking performance - balanced scaling
    Q = np.eye(n) * Q_scale * 0.1  # Reasonable base scale
    Q[0, 0] = Q_scale * 1.0  # Strong position tracking weight
    Q[1, 1] = Q_scale * 1.0  # Strong position tracking weight
    Q[2, 2] = Q_scale * 0.1  # Light velocity tracking weight  
    Q[3, 3] = Q_scale * 0.1  # Light velocity tracking weight
    
    # Expand Q for augmented state - small but meaningful quadratic terms
    Q_aug = np.eye(n + n**2) * Q_scale * 0.01  # Small quadratic weights
    Q_aug[:n, :n] = Q
    
    R = np.eye(p) * R_scale
    
    # Initial goal state (will be overridden by reference path)
    goal_state = np.array([[reference_path['x'][0]], [reference_path['y'][0]], 
                          [reference_path['vx'][0]], [reference_path['vy'][0]]])
    
    # Run different filter/controller combinations with hierarchical control
    results = {}
    
    for method_name, (filter_type, lqr_type) in [
        ('lqg_ekf', ('ekf', 'orig')),
        ('lqg_ukf', ('ukf', 'orig')), 
        ('lqg_qkf_numeric', ('qkf', 'aug_numeric')),
        ('lqg_qkf_analytic', ('qkf', 'aug_analytic'))
    ]:
        print(f"Running {method_name} with hierarchical control...")
        
        lqg_system = LQG(n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q_aug, R, 
                        goal_state=goal_state, H=H, filter_type=filter_type, lqr_type=lqr_type,
                        reference_path=reference_path, dt=dt, tracking_mode=True)
        
        # Enable hierarchical control
        lqg_system.enable_hierarchical_control(local_horizon=local_horizon)
        
        # Run simulation
        err_list, var_list, cost_list, track_err, traj_x, traj_y = lqg_system.run_sim()
        
        results[method_name] = [err_list, var_list, cost_list, track_err, traj_x, traj_y]

    results['reference_path'] = reference_path
    return results

def one_trial_path_tracking(H=1000, noise_scale=1e-1, m_scale=1e2, Q_scale=1.0, R_scale=1.0, 
                           path_type='figure8', rand_seed=None, dt=0.1):
    """
    Run one trial of path tracking experiment.
    """
    n1 = 2  # position states [x, y]
    n2 = 2  # velocity states [vx, vy]
    n = n1 + n2  # total state size
    p = 3   # control inputs [ax, ay, steering]
    m = 2   # measurements [x_gps, y_gps]
    
    if rand_seed is not None:
        np.random.seed(rand_seed)
    
    # Generate reference path
    reference_path = generate_reference_path(path_type=path_type, num_points=H, dt=dt, scale=10.0)
    
    # Create vehicle dynamics matrices
    A_E, A_S, B_S, W = create_vehicle_dynamics_matrices(dt=dt, process_noise_scale=noise_scale)
    
    # Create sensor matrices for tracking
    C, M, V = create_sensor_matrices_for_tracking(n=n, m=m, 
                                                 measurement_noise_scale=noise_scale, 
                                                 nonlinearity_scale=m_scale)
    
    # Create cost matrices focused on tracking performance - balanced scaling
    Q = np.eye(n) * Q_scale * 0.1  # Reasonable base scale
    Q[0, 0] = Q_scale * 1.0  # Strong position tracking weight
    Q[1, 1] = Q_scale * 1.0  # Strong position tracking weight
    Q[2, 2] = Q_scale * 0.1  # Light velocity tracking weight  
    Q[3, 3] = Q_scale * 0.1  # Light velocity tracking weight
    
    # Expand Q for augmented state - small but meaningful quadratic terms
    Q_aug = np.eye(n + n**2) * Q_scale * 0.01  # Small quadratic weights
    Q_aug[:n, :n] = Q
    
    R = np.eye(p) * R_scale
    
    # Initial goal state (will be overridden by reference path)
    goal_state = np.array([[reference_path['x'][0]], [reference_path['y'][0]], 
                          [reference_path['vx'][0]], [reference_path['vy'][0]]])
    
    # Run different filter/controller combinations
    # LQG-EKF: EKF + LQR with original state
    lqg_ekf = LQG(n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q_aug, R, 
                  goal_state=goal_state, H=H, filter_type='ekf', lqr_type='orig',
                  reference_path=reference_path, dt=dt, tracking_mode=True)
    err_list_ekf, var_list_ekf, cost_list_ekf, track_err_ekf, traj_x_ekf, traj_y_ekf = lqg_ekf.run_sim()
    
    # LQG-UKF: UKF + LQR with original state
    lqg_ukf = LQG(n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q_aug, R, 
                  goal_state=goal_state, H=H, filter_type='ukf', lqr_type='orig',
                  reference_path=reference_path, dt=dt, tracking_mode=True)
    err_list_ukf, var_list_ukf, cost_list_ukf, track_err_ukf, traj_x_ukf, traj_y_ukf = lqg_ukf.run_sim()
    
    # LQG-QKF (Numeric): QKF + LQR with augmented state (numeric iLQR)
    lqg_qkf_numeric = LQG(n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q_aug, R, 
                          goal_state=goal_state, H=H, filter_type='qkf', lqr_type='aug_numeric',
                          reference_path=reference_path, dt=dt, tracking_mode=True)
    err_list_qkf_num, var_list_qkf_num, cost_list_qkf_num, track_err_qkf_num, traj_x_qkf_num, traj_y_qkf_num = lqg_qkf_numeric.run_sim()
    
    # LQG-QKF (Analytic): QKF + LQR with augmented state (analytic)
    lqg_qkf_analytic = LQG(n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q_aug, R, 
                           goal_state=goal_state, H=H, filter_type='qkf', lqr_type='aug_analytic',
                           reference_path=reference_path, dt=dt, tracking_mode=True)
    err_list_qkf_ana, var_list_qkf_ana, cost_list_qkf_ana, track_err_qkf_ana, traj_x_qkf_ana, traj_y_qkf_ana = lqg_qkf_analytic.run_sim()

    return {
        'lqg_ekf': [err_list_ekf, var_list_ekf, cost_list_ekf, track_err_ekf, traj_x_ekf, traj_y_ekf],
        'lqg_ukf': [err_list_ukf, var_list_ukf, cost_list_ukf, track_err_ukf, traj_x_ukf, traj_y_ukf], 
        'lqg_qkf_numeric': [err_list_qkf_num, var_list_qkf_num, cost_list_qkf_num, track_err_qkf_num, traj_x_qkf_num, traj_y_qkf_num],
        'lqg_qkf_analytic': [err_list_qkf_ana, var_list_qkf_ana, cost_list_qkf_ana, track_err_qkf_ana, traj_x_qkf_ana, traj_y_qkf_ana],
        'reference_path': reference_path
    }

def one_trial(H=1000, noise_scale=1e-1, m_scale=1e2, Q_scale=1.0, R_scale=1.0, rand_seed=None):
    n1 = 2
    n2 = 2
    n = n1 + n2 # state size
    p = 3
    m = 2
    
    if rand_seed is not None:
        np.random.seed(rand_seed)
    
    # Use stable parameter generation
    A_E, A_S, B_S, C, M, W, V = generate_stable_system_parameters(
        n1, n2, p, m, noise_scale, m_scale
    )
    
    # Validate stability
    if not validate_stable_parameters(A_E, A_S):
        print("Warning: Generated unstable parameters, but continuing...")
    
    # Q, R must be symmetric positive definite matrices
    Q = generate_random_symmetric_matrix(n+n**2, scale=Q_scale)
    # Q = generate_random_symmetric_matrix(n, scale=1.0)
    R = generate_random_symmetric_matrix(p, scale=R_scale)
    
    goal_state = generate_goal_state(np.zeros((n1, 1)), n2) # goal state vector
    # lqg_kf_sys = LQG(n, p, W, A, B, C, M, V, Q, R, H=1000, filter_type='kf')
    # err_list_kf = lqg_kf_sys.run_sim()
    # plt.plot(err_list_kf, label=f'kf measure error')
    
    lqe_qkf = LQG(n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, H=H, filter_type='qkf', lqr_type='None', goal_state=goal_state)
    err_list_qkf, var_list_qkf, _, _, _, _ = lqe_qkf.run_sim()
    
    lqg_ekf = LQG(n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, H=H, filter_type='ekf', lqr_type='orig', goal_state=goal_state)
    err_list_ekf, var_list_ekf, cost_list_ekf, _, _, _ = lqg_ekf.run_sim()
    
    lqg_qkf_aug_num = LQG(n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, H=H, filter_type='qkf', lqr_type='aug_numeric', goal_state=goal_state)
    err_list_aug_num, var_list_aug_num, cost_list_aug_num, _, _, _ = lqg_qkf_aug_num.run_sim()
    
    lqg_qkf_aug_analytic = LQG(n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, H=H, filter_type='qkf', lqr_type='aug_analytic', goal_state=goal_state)
    err_list_aug_analytic, var_list_aug_analytic, cost_list_aug_analytic, _, _, _ = lqg_qkf_aug_analytic.run_sim()
    
    lqg_ukf = LQG(n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, H=H, filter_type='ukf', lqr_type='orig', goal_state=goal_state)
    err_list_ukf, var_list_ukf, cost_list_ukf, _, _, _ = lqg_ukf.run_sim()

    return [err_list_qkf, var_list_qkf], [err_list_ekf, var_list_ekf, cost_list_ekf], [err_list_aug_num, var_list_aug_num, cost_list_aug_num], [err_list_aug_analytic, var_list_aug_analytic, cost_list_aug_analytic], [err_list_ukf, var_list_ukf, cost_list_ukf]

def test_path_tracking(H=1000, trials=20, plot=True, noise_scale=1e-1, m_scale=1e2, 
                      Q_scale=1.0, R_scale=1.0, path_type='figure8', rand_seed=None):
    """
    Test path tracking performance across different filter types.
    """
    
    # File naming for path tracking results
    pkl_file = pkl_dir + f'path_tracking_results-mscale={int(m_scale)}.pkl'
    
    if os.path.exists(pkl_file):
        print(f"Found existing path tracking results for m_scale={int(m_scale)}. Loading...")
        with open(pkl_file, 'rb') as f:
            all_results = pkl.load(f)
        skip_simulation = True
    else:
        print(f"Running path tracking simulation for m_scale={int(m_scale)}...")
        skip_simulation = False
        all_results = {'lqg_ekf': [], 'lqg_ukf': [], 'lqg_qkf_numeric': [], 'lqg_qkf_analytic': [], 'reference_paths': []}
        
        for i in tqdm(range(trials)):
            seed_i = rand_seed + i if rand_seed is not None else None
            trial_results = one_trial_path_tracking(
                H=H, noise_scale=noise_scale, m_scale=m_scale,
                Q_scale=Q_scale, R_scale=R_scale, path_type=path_type, 
                rand_seed=seed_i
            )
            
            all_results['lqg_ekf'].append(trial_results['lqg_ekf'])
            all_results['lqg_ukf'].append(trial_results['lqg_ukf'])
            all_results['lqg_qkf_numeric'].append(trial_results['lqg_qkf_numeric'])
            all_results['lqg_qkf_analytic'].append(trial_results['lqg_qkf_analytic'])
            if i == 0:  # Store reference path from first trial
                all_results['reference_paths'].append(trial_results['reference_path'])
        
        # Save results
        with open(pkl_file, 'wb') as f:
            pkl.dump(all_results, f)
    
    # Extract and average results
    methods = ['lqg_ekf', 'lqg_ukf', 'lqg_qkf_numeric', 'lqg_qkf_analytic']
    avg_results = {}
    
    for method in methods:
        err_lists = [result[0] for result in all_results[method]]
        var_lists = [result[1] for result in all_results[method]]
        cost_lists = [result[2] for result in all_results[method]]
        track_err_lists = [result[3] for result in all_results[method]]
        traj_x_lists = [result[4] for result in all_results[method]]
        traj_y_lists = [result[5] for result in all_results[method]]
        
        avg_results[method] = {
            'estimation_error': np.mean(err_lists, axis=0),
            'variance': np.mean(var_lists, axis=0),
            'cost': np.mean(cost_lists, axis=0),
            'tracking_error': np.mean(track_err_lists, axis=0),
            'trajectory_x': np.mean(traj_x_lists, axis=0),
            'trajectory_y': np.mean(traj_y_lists, axis=0)
        }
    
    if plot:
        plot_path_tracking_results(avg_results, all_results, methods, H)
    
    return avg_results

def plot_path_tracking_results(avg_results, all_results, methods, H):
    """Create comprehensive plots for path tracking results."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Path Tracking Performance Comparison: LQG-EKF vs LQG-UKF vs LQG-QKF Methods', fontsize=16)
    
    colors = {'lqg_ekf': 'blue', 'lqg_ukf': 'green', 'lqg_qkf_numeric': 'orange', 'lqg_qkf_analytic': 'red'}
    labels = {'lqg_ekf': 'LQG-EKF', 'lqg_ukf': 'LQG-UKF', 'lqg_qkf_numeric': 'LQG-QKF (Numeric)', 'lqg_qkf_analytic': 'LQG-QKF (Analytic)'}
    
    # Plot 1: Estimation Error
    ax1 = axes[0, 0]
    for method in methods:
        ax1.plot(avg_results[method]['estimation_error'], 
                label=labels[method], color=colors[method], linewidth=2)
    ax1.set_title('State Estimation Error')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('RMSE')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Tracking Error
    ax2 = axes[0, 1]
    for method in methods:
        ax2.plot(avg_results[method]['tracking_error'], 
                label=labels[method], color=colors[method], linewidth=2)
    ax2.set_title('Path Tracking Error')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Tracking Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cost Function
    ax3 = axes[0, 2]
    for method in methods:
        ax3.plot(avg_results[method]['cost'], 
                label=labels[method], color=colors[method], linewidth=2)
    ax3.set_title('Control Cost')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Cost')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Estimation Variance
    ax4 = axes[1, 0]
    for method in methods:
        ax4.plot(avg_results[method]['variance'], 
                label=labels[method], color=colors[method], linewidth=2)
    ax4.set_title('Estimation Variance')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Trace of Covariance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Actual Trajectories vs Reference Path
    if 'reference_paths' in all_results and len(all_results['reference_paths']) > 0:
        ax5 = axes[1, 1]
        ref_path = all_results['reference_paths'][0]
        
        # Plot reference path
        ax5.plot(ref_path['x'], ref_path['y'], 'k--', linewidth=3, label='Reference Path', alpha=0.8)
        
        # Plot actual trajectories for each method
        for method in methods:
            if method in avg_results and 'trajectory_x' in avg_results[method]:
                ax5.plot(avg_results[method]['trajectory_x'], 
                        avg_results[method]['trajectory_y'], 
                        color=colors[method], linewidth=2, alpha=0.8,
                        label=f'{labels[method]} Actual')
        
        # Add start and end markers for reference
        ax5.plot(ref_path['x'][0], ref_path['y'][0], 'go', markersize=8, label='Start')
        ax5.plot(ref_path['x'][-1], ref_path['y'][-1], 'ro', markersize=8, label='End')
        
        ax5.set_title('Actual Trajectories vs Reference')
        ax5.set_xlabel('X Position')
        ax5.set_ylabel('Y Position')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # Focus on reference path and QKF trajectories only for axis limits
        x_coords = list(ref_path['x'])
        y_coords = list(ref_path['y'])
        
        # Add only successful QKF trajectories to determine bounds
        for method in ['lqg_qkf_numeric', 'lqg_qkf_analytic']:
            if method in avg_results and 'trajectory_x' in avg_results[method]:
                # Check if method diverged (indicated by very large tracking errors)
                track_err = avg_results[method]['tracking_error']
                if np.mean(track_err[-100:]) < 1e5:  # Only include non-diverged methods
                    traj_x = np.array(avg_results[method]['trajectory_x'])
                    traj_y = np.array(avg_results[method]['trajectory_y'])
                    valid_mask = (np.abs(traj_x) < 1e3) & (np.abs(traj_y) < 1e3)  # Filter diverged points
                    if np.any(valid_mask):  # Only add if there are valid points
                        x_coords.extend(traj_x[valid_mask])
                        y_coords.extend(traj_y[valid_mask])
                        print(f"  Including {method} trajectory in axis calculation")
        
        # Set axis limits based on reference + QKF paths with small margin
        if len(x_coords) > 0 and len(y_coords) > 0:
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            x_margin = (x_max - x_min) * 0.1 if x_max != x_min else 1.0
            y_margin = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
            ax5.set_xlim(x_min - x_margin, x_max + x_margin)
            ax5.set_ylim(y_min - y_margin, y_max + y_margin)
        ax5.set_aspect('equal', adjustable='box')
    
    # Plot 6: Performance Summary (Bar Chart)
    ax6 = axes[1, 2]
    final_track_errors = [avg_results[method]['tracking_error'][-100:].mean() for method in methods]
    bars = ax6.bar([labels[method] for method in methods], final_track_errors, 
                   color=[colors[method] for method in methods], alpha=0.7)
    ax6.set_title('Final Tracking Performance')
    ax6.set_ylabel('Average Tracking Error (last 100 steps)')
    ax6.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, final_track_errors):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(final_track_errors)*0.02, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(perf_dir + '/path_tracking_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create separate detailed path plot with all trajectories
    if 'reference_paths' in all_results and len(all_results['reference_paths']) > 0:
        plt.figure(figsize=(12, 8))
        ref_path = all_results['reference_paths'][0]
        
        # Plot reference path
        plt.plot(ref_path['x'], ref_path['y'], 'k--', linewidth=4, label='Reference Path', alpha=0.9)
        
        # Plot actual trajectories for each method
        for method in methods:
            if method in avg_results and 'trajectory_x' in avg_results[method]:
                plt.plot(avg_results[method]['trajectory_x'], 
                        avg_results[method]['trajectory_y'], 
                        color=colors[method], linewidth=2.5, alpha=0.8,
                        label=f'{labels[method]} Actual Path')
        
        # Add start and end markers
        plt.plot(ref_path['x'][0], ref_path['y'][0], 'go', markersize=12, label='Start', zorder=10)
        plt.plot(ref_path['x'][-1], ref_path['y'][-1], 'ro', markersize=12, label='End', zorder=10)
        
        plt.title('Path Tracking Comparison: Actual vs Reference Trajectories', fontsize=14)
        plt.xlabel('X Position', fontsize=12)
        plt.ylabel('Y Position', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Focus on reference path and QKF trajectories only for axis limits
        x_coords = list(ref_path['x'])
        y_coords = list(ref_path['y'])
        
        # Add only successful QKF trajectories to determine bounds
        for method in ['lqg_qkf_numeric', 'lqg_qkf_analytic']:
            if method in avg_results and 'trajectory_x' in avg_results[method]:
                # Check if method diverged (indicated by very large tracking errors)
                track_err = avg_results[method]['tracking_error']
                if np.mean(track_err[-100:]) < 1e5:  # Only include non-diverged methods
                    traj_x = np.array(avg_results[method]['trajectory_x'])
                    traj_y = np.array(avg_results[method]['trajectory_y'])
                    valid_mask = (np.abs(traj_x) < 1e3) & (np.abs(traj_y) < 1e3)  # Filter diverged points
                    if np.any(valid_mask):  # Only add if there are valid points
                        x_coords.extend(traj_x[valid_mask])
                        y_coords.extend(traj_y[valid_mask])
        
        # Set axis limits based on reference + QKF paths with reasonable margin
        if len(x_coords) > 0 and len(y_coords) > 0:
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            x_margin = (x_max - x_min) * 0.15 if x_max != x_min else 1.0
            y_margin = (y_max - y_min) * 0.15 if y_max != y_min else 1.0
            plt.xlim(x_min - x_margin, x_max + x_margin)
            plt.ylim(y_min - y_margin, y_max + y_margin)
        plt.gca().set_aspect('equal', adjustable='box')
        
        plt.savefig(perf_dir + '/path_tracking_paths.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Print summary
    print("\n=== Path Tracking Summary ===")
    for method in methods:
        final_track_error = avg_results[method]['tracking_error'][-100:].mean()
        final_est_error = avg_results[method]['estimation_error'][-100:].mean()
        
        # Check if method diverged (indicated by very large errors)
        if final_track_error > 1e5 or final_est_error > 1e5:
            print(f"{labels[method]}: DIVERGED")
        else:
            print(f"{labels[method]}: Tracking Error = {final_track_error:.4f}, Estimation Error = {final_est_error:.4f}")

def test(H=1000, trials=20, plot=True, noise_scale=1e-1, m_scale=1e2, Q_scale=1.0, R_scale=1.0, rand_seed=None):
    
    # Check if pkl files already exist
    pkl_dir = 'pkl/'
    os.makedirs(pkl_dir, exist_ok=True)
    
    ekf_file = pkl_dir + f'ekf_results-mscale={int(m_scale)}.pkl'
    qkf_file = pkl_dir + f'qkf_results-mscale={int(m_scale)}.pkl'
    qkf_analytic_file = pkl_dir + f'qkf_analytic_results-mscale={int(m_scale)}.pkl'
    ukf_file = pkl_dir + f'ukf_results-mscale={int(m_scale)}.pkl'
    
    # Check if all required files exist
    if (os.path.exists(ekf_file) and os.path.exists(qkf_file) and 
        os.path.exists(qkf_analytic_file) and os.path.exists(ukf_file)):
        print(f"Found existing pkl files for m_scale={int(m_scale)}. Skipping simulation.")
        print(f"Loading existing results from: {ekf_file}, {qkf_file}, {qkf_analytic_file}, {ukf_file}")
        
        # Load existing results
        with open(ekf_file, 'rb') as f:
            err_list_ekf_all, var_list_ekf_all, cost_list_ekf_all = pkl.load(f)
        with open(qkf_file, 'rb') as f:
            err_list_qkf_num_all, var_list_qkf_num_all, cost_list_qkf_num_all = pkl.load(f)
        with open(qkf_analytic_file, 'rb') as f:
            err_list_qkf_analytic_all, var_list_qkf_analytic_all, cost_list_qkf_analytic_all = pkl.load(f)
        with open(ukf_file, 'rb') as f:
            err_list_ukf_all, var_list_ukf_all, cost_list_ukf_all = pkl.load(f)
        
        # Convert to numpy arrays if they aren't already
        err_list_ekf_all = np.array(err_list_ekf_all)
        var_list_ekf_all = np.array(var_list_ekf_all)
        cost_list_ekf_all = np.array(cost_list_ekf_all)
        err_list_qkf_num_all = np.array(err_list_qkf_num_all)
        var_list_qkf_num_all = np.array(var_list_qkf_num_all)
        cost_list_qkf_num_all = np.array(cost_list_qkf_num_all)
        err_list_qkf_analytic_all = np.array(err_list_qkf_analytic_all)
        var_list_qkf_analytic_all = np.array(var_list_qkf_analytic_all)
        cost_list_qkf_analytic_all = np.array(cost_list_qkf_analytic_all)
        err_list_ukf_all = np.array(err_list_ukf_all)
        var_list_ukf_all = np.array(var_list_ukf_all)
        cost_list_ukf_all = np.array(cost_list_ukf_all)
        
        # Skip to averaging and plotting
        skip_simulation = True
    else:
        print(f"Running simulation for m_scale={int(m_scale)}...")
        skip_simulation = False
        
        err_list_ekf_all = []
        var_list_ekf_all = []
        cost_list_ekf_all = []

        err_list_qkf_num_all = []
        var_list_qkf_num_all = []
        cost_list_qkf_num_all = []

        err_list_qkf_analytic_all = []
        var_list_qkf_analytic_all = []
        cost_list_qkf_analytic_all = []

        err_list_ukf_all = []
        var_list_ukf_all = []
        cost_list_ukf_all = []

        for i in tqdm(range(trials)):
            seed_i = rand_seed + i if rand_seed is not None else None
            lqe_qkf_results, ekf_results, qkf_num_results, qkf_analytic_results, ukf_results = one_trial(
                H=H, noise_scale=noise_scale, m_scale=m_scale,
                Q_scale=Q_scale, R_scale=R_scale, rand_seed=seed_i
            )
            
            # lqe_qkf_results, ekf_results, qkf_results = one_trial(H=H, noise_scale=noise_scale, m_scale=m_scale, Q_scale=Q_scale, R_scale=R_scale, rand_seed=rand_seed)
            
            err_list_ekf_all.append(ekf_results[0])
            var_list_ekf_all.append(ekf_results[1])
            cost_list_ekf_all.append(ekf_results[2])
            
            err_list_qkf_num_all.append(qkf_num_results[0])
            var_list_qkf_num_all.append(qkf_num_results[1])
            cost_list_qkf_num_all.append(qkf_num_results[2])

            err_list_qkf_analytic_all.append(qkf_analytic_results[0])
            var_list_qkf_analytic_all.append(qkf_analytic_results[1])
            cost_list_qkf_analytic_all.append(qkf_analytic_results[2])

            err_list_ukf_all.append(ukf_results[0])
            var_list_ukf_all.append(ukf_results[1])
            cost_list_ukf_all.append(ukf_results[2])


    # average results
    err_list_ekf_avg = np.mean(np.array(err_list_ekf_all), axis=0)
    var_list_ekf_avg = np.mean(np.array(var_list_ekf_all), axis=0)
    cost_list_ekf_avg = np.mean(np.array(cost_list_ekf_all), axis=0)

    err_list_qkf_num_avg = np.mean(np.array(err_list_qkf_num_all), axis=0)
    var_list_qkf_num_avg = np.mean(np.array(var_list_qkf_num_all), axis=0)
    cost_list_qkf_num_avg = np.mean(np.array(cost_list_qkf_num_all), axis=0)

    err_list_qkf_analytic_avg = np.mean(np.array(err_list_qkf_analytic_all), axis=0)
    var_list_qkf_analytic_avg = np.mean(np.array(var_list_qkf_analytic_all), axis=0)
    cost_list_qkf_analytic_avg = np.mean(np.array(cost_list_qkf_analytic_all), axis=0)

    err_list_ukf_avg = np.mean(np.array(err_list_ukf_all), axis=0)
    var_list_ukf_avg = np.mean(np.array(var_list_ukf_all), axis=0)
    cost_list_ukf_avg = np.mean(np.array(cost_list_ukf_all), axis=0)

    # Only save pkl files if simulation was actually run
    if not skip_simulation:
        pkl_dir = 'pkl/'
        os.makedirs(pkl_dir, exist_ok=True)
        with open(pkl_dir + f'ekf_results-mscale={int(m_scale)}.pkl', 'wb') as f:
            pkl.dump((np.array(err_list_ekf_all), np.array(var_list_ekf_all), np.array(cost_list_ekf_all)), f)
        with open(pkl_dir + f'qkf_results-mscale={int(m_scale)}.pkl', 'wb') as f:
            pkl.dump((np.array(err_list_qkf_num_all), np.array(var_list_qkf_num_all), np.array(cost_list_qkf_num_all)), f)
        with open(pkl_dir + f'qkf_analytic_results-mscale={int(m_scale)}.pkl', 'wb') as f:
            pkl.dump((np.array(err_list_qkf_analytic_all), np.array(var_list_qkf_analytic_all), np.array(cost_list_qkf_analytic_all)), f)
        with open(pkl_dir + f'ukf_results-mscale={int(m_scale)}.pkl', 'wb') as f:
            pkl.dump((np.array(err_list_ukf_all), np.array(var_list_ukf_all), np.array(cost_list_ukf_all)), f)

    
    if plot:  
        # plot estimation peformance comparison
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        ax[0].set_title('Estimate error')
        ax[0].set_xlabel('Time step')
        ax[0].set_ylabel('Estimate error')
        ax[1].set_title('Estimate variance')
        ax[1].set_xlabel('Time step')   
        ax[1].set_ylabel('Estimate variance')
        
        ax[0].plot(err_list_ekf_avg, label='EKF error', color='blue')
        ax[0].plot(err_list_ukf_avg, label='UKF error', color='green')
        ax[0].plot(err_list_qkf_num_avg, label='QKF error', color='orange') 
        ax[1].plot(var_list_ekf_avg, label='EKF variance', color='blue')
        ax[1].plot(var_list_ukf_avg, label='UKF variance', color='green')
        ax[1].plot(var_list_qkf_num_avg, label='QKF variance', color='orange')

        ax[0].legend()
        ax[1].legend()
        ax[0].grid(True)
        ax[1].grid(True)
        plt.tight_layout()
        plt.savefig(perf_dir + '/estimation_performance.png')
        plt.close()

        # plot cost performance comparison
        plt.figure(figsize=(10, 6))
        plt.title('Cost performance comparison')
        plt.xlabel('Time step')
        plt.ylabel('Cost')
        plt.plot(cost_list_ekf_avg, label='LQR+EKF cost', color='blue')
        plt.plot(cost_list_ukf_avg, label='LQR+UKF cost', color='green')
        plt.plot(cost_list_qkf_num_avg, label='LQR+QKF cost', color='orange')
        plt.legend()
        plt.grid()
        plt.savefig(perf_dir + '/cost_performance.png')
        plt.close()
        
        # plot convergence comparison with improved detection
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))  # Increased figure size
        fig.suptitle('Convergence Analysis', fontsize=16, y=0.98)  # Moved title up and increased font size
        
        # Use improved convergence detection
        convergence_ekf = []
        convergence_qkf_num = []
        convergence_ukf = []
        
        tolerance = np.mean(cost_list_ekf_avg[:100]) * 0.01  # 1% of initial cost as tolerance
        
        for cnt in range(trials):
            # For each trial, detect convergence using the improved method
            conv_ekf, _ = detect_convergence(cost_list_ekf_all[cnt], tolerance=tolerance)
            conv_qkf_num, _ = detect_convergence(cost_list_qkf_num_all[cnt], tolerance=tolerance)
            conv_ukf, _ = detect_convergence(cost_list_ukf_all[cnt], tolerance=tolerance)
            
            convergence_ekf.append(conv_ekf if conv_ekf is not None else H)
            convergence_qkf_num.append(conv_qkf_num if conv_qkf_num is not None else H)
            convergence_ukf.append(conv_ukf if conv_ukf is not None else H)
        
        # Subplot 1: Convergence times
        ax1 = axes[0, 0]
        ax1.set_title('Time to Convergence', pad=25, fontsize=14)
        
        # Plot convergence times
        trials_range = range(trials)
        ax1.plot(trials_range, convergence_ekf, label='LQR+EKF', marker='o', alpha=0.7, color='blue')
        ax1.plot(trials_range, convergence_ukf, label='LQR+UKF', marker='d', alpha=0.7, color='green')
        ax1.plot(trials_range, convergence_qkf_num, label='LQR+QKF', marker='s', alpha=0.7, color='orange')
        ax1.set_xlabel('Trial', fontsize=12)
        ax1.set_ylabel('Convergence Time (steps)', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Convergence statistics
        ax2 = axes[0, 1]
        ax2.set_title('Convergence Statistics', pad=25, fontsize=14)
        methods = ['LQR+EKF', 'LQR+UKF', 'LQR+QKF']
        avg_conv_times = [np.mean(convergence_ekf), np.mean(convergence_ukf), np.mean(convergence_qkf_num)]
        std_conv_times = [np.std(convergence_ekf), np.std(convergence_ukf), np.std(convergence_qkf_num)]
        
        bars = ax2.bar(methods, avg_conv_times, yerr=std_conv_times, capsize=5, alpha=0.7)
        ax2.set_ylabel('Average Convergence Time', fontsize=12)
        ax2.tick_params(axis='x', rotation=45, labelsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars with better positioning
        for bar, avg_time in zip(bars, avg_conv_times):
            ax2.text(bar.get_x() + bar.get_width()/2 + bar.get_width()*0.3, bar.get_height() + max(std_conv_times) * 0.15, 
                    f'{avg_time:.0f}', ha='center', va='bottom', fontsize=10)
        
        # Subplot 3: Convergence rate (percentage converged vs time)
        ax3 = axes[1, 0]
        ax3.set_title('Convergence Rate Over Time', pad=25, fontsize=14)
        time_steps = np.arange(0, H, 10)
        
        ekf_conv_rate = []
        qkf_num_conv_rate = []
        ukf_conv_rate = []
        
        for t in time_steps:
            ekf_conv_rate.append(np.sum(np.array(convergence_ekf) <= t) / trials * 100)
            qkf_num_conv_rate.append(np.sum(np.array(convergence_qkf_num) <= t) / trials * 100)
            ukf_conv_rate.append(np.sum(np.array(convergence_ukf) <= t) / trials * 100)
        
        ax3.plot(time_steps, ekf_conv_rate, label='LQR+EKF', linewidth=2, color='blue')
        ax3.plot(time_steps, ukf_conv_rate, label='LQR+UKF', linewidth=2, color='green')
        ax3.plot(time_steps, qkf_num_conv_rate, label='LQR+QKF', linewidth=2, color='orange')
        ax3.set_xlabel('Time Steps', fontsize=12)
        ax3.set_ylabel('Convergence Rate (%)', fontsize=12)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Subplot 4: Final convergence status
        ax4 = axes[1, 1]
        ax4.set_title('Final Convergence Status', pad=25, fontsize=14)
        conv_counts = [
            np.sum(np.array(convergence_ekf) < H),
            np.sum(np.array(convergence_ukf) < H),
            # np.sum(np.array(convergence_qkf_analytic) < H),
            np.sum(np.array(convergence_qkf_num) < H),
        ]
        conv_percentages = [count/trials*100 for count in conv_counts]
        
        bars = ax4.bar(methods, conv_percentages, alpha=0.7, color=['blue', 'green', 'orange'])
        ax4.set_ylabel('Convergence Rate (%)', fontsize=12)
        ax4.tick_params(axis='x', rotation=45, labelsize=10)
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3)
        
        # Add percentage labels with better positioning
        for bar, pct in zip(bars, conv_percentages):
            ax4.text(bar.get_x() + bar.get_width()/2 + bar.get_width()*0.3, bar.get_height() + 3, 
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # Adjust layout to prevent overlap
        plt.subplots_adjust(top=0.92, bottom=0.12, left=0.1, right=0.95, hspace=0.35, wspace=0.3)
        plt.savefig(perf_dir + '/convergence_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print summary statistics
        print(f"\n=== Convergence Analysis Summary ===")
        print(f"Tolerance used: {tolerance:.2e}")
        print(f"LQR+EKF - Avg convergence time: {np.mean(convergence_ekf):.1f} ± {np.std(convergence_ekf):.1f}")
        print(f"LQR+UKF - Avg convergence time: {np.mean(convergence_ukf):.1f} ± {np.std(convergence_ukf):.1f}")
        print(f"LQR+QKF - Avg convergence time: {np.mean(convergence_qkf_num):.1f} ± {np.std(convergence_qkf_num):.1f}")
        # print(f"Analytic QKF - Avg convergence time: {np.mean(convergence_qkf_analytic):.1f} ± {np.std(convergence_qkf_analytic):.1f}")
        print(f"Convergence rates: LQR+EKF {conv_percentages[0]:.1f}%, LQR+UKF {conv_percentages[1]:.1f}%, LQR+QKF {conv_percentages[2]:.1f}%")

    # return cost_list_ekf_avg, cost_list_qkf_num_avg, cost_list_qkf_analytic_avg, cost_list_ukf_avg
    return cost_list_ekf_avg, cost_list_qkf_num_avg, cost_list_ukf_avg

def nonlinearity_test(H=1000, trials=20):
    m_scales = [0, 1, 1e1, 1e2, 1e3, 1e4]
    rand_seed = 100  # use the same base for all m_scales
    for i, m_scale in enumerate(m_scales):
        print(f"Testing with m_scale={m_scale}")
        # cost_list_ekf_avg, cost_list_qkf_num_avg, cost_list_qkf_analytic_avg, cost_list_ukf_avg = test(H=H, trials=trials, plot=False, m_scale=m_scale, rand_seed=rand_seed)
        cost_list_ekf_avg, cost_list_qkf_num_avg, cost_list_ukf_avg = test(H=H, trials=trials, plot=False, m_scale=m_scale, rand_seed=rand_seed)





if __name__ == "__main__":
    os.makedirs(perf_dir, exist_ok=True)
    
    # Run path tracking test
    print("=" * 60)
    print("RUNNING PATH TRACKING TEST - LQG-EKF vs LQG-UKF vs LQG-QKF")
    print("=" * 60)
    
    # Test hierarchical control approach
    print("Testing HIERARCHICAL control approach...")
    
    # Run single trial with hierarchical control
    hierarchical_results = one_trial_hierarchical_path_tracking(
        H=1000, local_horizon=50, noise_scale=1e-1, m_scale=1e-1, 
        Q_scale=1, R_scale=1, path_type='straight', rand_seed=42
    )
    
    # Plot hierarchical results
    methods = ['lqg_ekf', 'lqg_ukf', 'lqg_qkf_numeric', 'lqg_qkf_analytic']
    avg_results = {}
    
    for method in methods:
        avg_results[method] = {
            'estimation_error': np.array(hierarchical_results[method][0]),
            'variance': np.array(hierarchical_results[method][1]),
            'cost': np.array(hierarchical_results[method][2]),
            'tracking_error': np.array(hierarchical_results[method][3]),
            'trajectory_x': np.array(hierarchical_results[method][4]),
            'trajectory_y': np.array(hierarchical_results[method][5])
        }
    
    # Create compatible all_results structure for plotting
    all_results = {
        'reference_paths': [hierarchical_results['reference_path']]
    }
    for method in methods:
        all_results[method] = [hierarchical_results[method]]
    
    plot_path_tracking_results(avg_results, all_results, methods, 1000)
    
    print("\nPath tracking test completed!")
    print(f"Results saved in the '{perf_dir}' directory.")
    print("- path_tracking_performance.png: Detailed comparison")
    print("- path_tracking_paths.png: Reference trajectory visualization")
        



