import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Literal
from tqdm import tqdm
from stateDynamics import StateDynamics, sensor, Vec, invVec
from scipy.linalg import sqrtm
import pickle as pkl
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set publication-ready plotting style
plt.style.use('default')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman']
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.titlesize'] = 14
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['grid.linewidth'] = 0.8
mpl.rcParams['grid.alpha'] = 0.3

# Publication-ready color palette
PUBLICATION_COLORS = {
    'ekf': '#1f77b4',      # Blue
    'ukf': '#2ca02c',      # Green  
    'qkf_numeric': '#ff7f0e',      # Orange
    'qkf_analytic': '#d62728',  # Red
    'grid': '#e0e0e0',
    'text': '#2f2f2f'
}

small_value = 1e-6  # Small value to prevent numerical issues
test_dir = 'prelim_test/'
pkl_dir = test_dir + 'pkl/'
os.makedirs(pkl_dir, exist_ok=True)

perf_dir = test_dir + 'perf/'
os.makedirs(perf_dir, exist_ok=True)

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
    # Compute the LQG gain
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
                 filter_type: Literal['qkf', 'ekf', 'kf', 'ukf', 'pf'] = 'qkf',
                 lqr_type: Literal['orig', 'aug_analytic', 'aug_numeric', 'None'] = 'orig',
                 n_particles: int = 1000):
        
        self.filter_type = filter_type
        self.lqr_type = lqr_type
        # dynamics setting
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
        self.x_hat = np.zeros((self.n, 1)) # estimated state vector
        self.x_hat_prev = self.x_hat.copy()
        self.z_hat = np.zeros((self.n + self.n**2, 1)) # estimated augmented state vector
        self.x_goal = np.zeros((self.n, 1)) # goal state vector
        
        # lqr
        self.x_goal = goal_state
        self.Q = Q.astype(np.float64)
        self.R = R.astype(np.float64)
        self.P_lqr = Q.copy()[:self.n, :self.n] # cost-to-go matrix for LQG
        
        # lqe
        self.P_est = np.eye(self.n) * small_value  # estimation error covariance matrix 
        
        # particle filter (when filter_type == 'pf')
        self.n_particles = n_particles
        if self.filter_type == 'pf':
            chol_P = np.linalg.cholesky(self.P_est + np.eye(self.n) * 1e-10)
            self.particles = self.x_hat.flatten() + (chol_P @ np.random.randn(self.n, n_particles)).T  # (n_particles, n)
            self.weights = np.ones(n_particles) / n_particles
        
        # convergence tracking
        self.convergence_history = {
            'cost': [],
            'estimation_error': [],
            'control_effort': [],
            'is_converged': False,
            'convergence_step': None,
            'convergence_metrics': {}
        } 
    
    # iLQG related
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
        iter = 0
        diff_u = 1e10  
        diff_cost = 1e10
        epsilon_u = 1e-6  # convergence threshold for control input change
        epsilon_cost = 1e-8  # convergence threshold for cost change
        prev_cost = 1e10
        
        while iter < max_iter:
            iter += 1
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
                
            u_nominal = u_new
            x_nominal = x_new
            prev_cost = current_cost
            
        self.F.set_u(u_new)
        return
    
    def update_lqr_analytic(self, goal_state):
        # LQG update only with augmented state
        I_p = np.eye(self.p)  # shape (p, p)
        I_p2 = np.eye(self.p ** 2)  # shape (p^2, p^2)
        I_n = np.eye(self.n) # shape (n, n)
        B = self.F.get_B()  # shape (n, p)
        A = self.F.get_A()  # shape (n, n)
        
        
        # to be checked
        ####################################################
        # x_actual = self.F.get_x()  # shape (n, 1), current state vector
        x_pred = A @ self.x_hat_prev + B @ self.F.get_u()  # shape (n, 1), predicted state vector
        x = x_pred - goal_state # shape (n, 1)
        
        x_estimated = self.x_hat
        x_tilde = x_estimated - x_pred # shape (n, 1)
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
        
        S = np.zeros((self.p, self.p))  # shape (p, p)
        for i in range(self.p):
            e_i = np.zeros((self.p, 1)) # shape (p, 1)
            e_i[i] = 1
            term1 = (M @ np.kron(e_i, I_p)) # shape (n^2, p)
            term2 = term1.T @ q @ e_i.T  # shape (p, p)
            S += term2  # accumulate over p columns

        # Z = np.kron(A, B) @ np.kron(x, I_p)   + np.kron(B, A) @ np.kron(I_p, x) # shape (n^2, p)
        Z = np.kron(B, A @ x) + np.kron(A @ x, B) + np.kron(B, A @ x_tilde) + np.kron(A @ x_tilde, B)
        u_new = -np.linalg.inv(S + 2 * self.R) @ Z.T @ q # shape (p, 1)
        self.F.set_u(u_new)

    def update_lqr_orig(self, goal_state, ):
        # LQG update only with original state, no augmented state
        # P_lqr = scipy.linalg.solve_discrete_are(self.A, self.B, self.Q[:self.n, :self.n], self.R)  # P is the fixed-point
        P_lqr = finite_horizon_lqr(self.A, self.B, self.Q[:self.n, :self.n], self.R, N=1, Qf=self.P_lqr)
        self.P_lqr = P_lqr.copy() # update cost-to-go matrix
        feedback_gain = -np.linalg.pinv(self.R + self.B.T @ P_lqr @ self.B) @ self.B.T @ P_lqr @ self.A
        u_new = feedback_gain @ (goal_state - self.x_hat)  # control input
        self.F.set_u(u_new)
        return
        
    def update_lqr(self):
        if self.lqr_type == 'None':
            # No LQG update, no control input
            # self.F.set_u(np.ones((self.p, 1)))
            self.F.set_u(np.random.randn(self.p, 0))  # small random noise
            return 
        else:
            goal_state = self.x_goal
            if self.filter_type in ('ekf', 'kf', 'ukf', 'pf'):
                self.update_lqr_orig(goal_state)
            elif self.filter_type == 'qkf':
                if self.lqr_type == 'aug_numeric':
                    # self.update_lqr_newton(goal_state, infinite_horizon=False)
                    self.update_ilqr(goal_state, alpha=1)
                if self.lqr_type == 'aug_analytic':
                    self.update_lqr_analytic(goal_state)
                    # self.update_ilqr(goal_state, alpha=1)
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
        self.x_hat_prev = self.x_hat.copy()
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
        self.x_hat_prev = self.x_hat.copy()
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
        self.x_hat_prev = self.x_hat.copy()
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
            sqrt_P = np.linalg.cholesky((n + lambda_) * self.P_est)
        except np.linalg.LinAlgError:
            # If not positive definite, use eigendecomposition
            eigenvals, eigenvecs = np.linalg.eigh(self.P_est)
            eigenvals = np.maximum(eigenvals, 1e-8)  # Ensure positive
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
        
        # Kalman gain
        K = C_tilde @ np.linalg.pinv(S)
        
        # Measurement residual (consistent with EKF)
        y = self.sensor.measure(self.F.get_x())
        delta_y = y - y_predicted
        
        # Update the state estimate
        self.x_hat_prev = self.x_hat.copy()
        self.x_hat = x_predicted + K @ delta_y
        
        # Update the covariance estimate
        self.P_est = sigma_0 - K @ S @ K.T
        return K
    
    def _resample_systematic(self, particles, weights):
        """Systematic resampling for particle filter."""
        n_particles = len(weights)
        cumsum = np.cumsum(weights)
        u0 = np.random.rand() / n_particles
        indices = np.zeros(n_particles, dtype=int)
        j = 0
        for i in range(n_particles):
            u = u0 + i / n_particles
            while u > cumsum[j]:
                j += 1
            indices[i] = j
        return particles[indices], np.ones(n_particles) / n_particles
    
    def update_lqe_pf(self):
        """Particle filter: predict, weight by measurement likelihood, resample."""
        n_particles = self.n_particles
        A, B, W = self.F.A, self.F.B, self.F.W
        u = self.F.get_u()
        # Predict
        self.particles = (A @ self.particles.T).T + (B @ u).flatten()
        chol_W = np.linalg.cholesky(W + np.eye(self.n) * 1e-10)
        self.particles += (chol_W @ np.random.randn(self.n, n_particles)).T
        # Measurement
        y = self.sensor.measure(self.F.get_x())  # (m, 1)
        y = y.flatten()
        V = self.sensor.V
        # Log-likelihood for each particle (Gaussian: -0.5 * (y - h_i)' inv(V) (y - h_i))
        log_like = np.zeros(n_particles)
        for i in range(n_particles):
            h_i = self.sensor.measure_pred(self.particles[i].reshape(-1, 1)).flatten()
            diff = y - h_i
            log_like[i] = -0.5 * diff @ np.linalg.solve(V, diff)
        # Weight update (log-domain for stability)
        log_w = np.log(self.weights + 1e-300) + log_like
        log_w -= log_w.max()
        self.weights = np.exp(log_w)
        self.weights /= self.weights.sum()
        # Resample
        self.particles, self.weights = self._resample_systematic(self.particles, self.weights)
        # Mean and covariance
        self.x_hat_prev = self.x_hat.copy()
        self.x_hat = np.mean(self.particles, axis=0).reshape(-1, 1)
        diff = self.particles - self.x_hat.flatten()
        self.P_est = (diff.T @ diff) / n_particles + np.eye(self.n) * small_value
        return None  # PF has no Kalman gain
    
    def update_lqe(self):
        if self.filter_type == 'qkf':
            K = self.update_lqe_qkf()
        elif self.filter_type == 'ekf':
            K = self.update_lqe_ekf()
        elif self.filter_type == 'kf':
            K = self.update_lqe_kf()
        elif self.filter_type == 'ukf':
            K = self.update_lqe_ukf()
        elif self.filter_type == 'pf':
            self.update_lqe_pf()
        else:
            raise ValueError("Invalid filter type. Choose 'qkf', 'ekf', 'kf', 'ukf', or 'pf'.")
        t = self.F.t
        # print(f'  t={t:4d}', f'‖K_{self.filter_type}‖₂=', np.linalg.norm(K),) if t % 100 == 0 else None
        return

    
    def forward_state(self):
        self.F.forward()
    
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
        time_list = []  # Track step-wise time consumption
        
        for step in tqdm(range(1, self.H + 1, 1), desc=f"Running {self.filter_type}-{self.lqr_type}"):
            step_start_time = time.time()  # Start timing this step
            
            self.update_lqe()
            if self.lqr_type != 'None':
                self.update_lqr()
                self.forward_state()
            
            # record error
            estimate_error = np.linalg.norm(self.F.get_x() - self.x_hat).item() 
            rmse_list.append(estimate_error)
            
            # record variance
            if self.filter_type == 'qkf':
                var = np.trace(self.Pz_est[:self.n, :self.n])
            elif self.filter_type in ('ekf', 'kf', 'ukf', 'pf'):
                var = np.trace(self.P_est)
            else:
                var = 0.0  # Default case
            var_list.append(var)
            
            # record cost   
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
            
            # record step time
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            time_list.append(step_time)
            
            # Check for early convergence (optional - can save computation)
            if step > 200 and self.check_system_convergence():
                if step % 100 == 0:  # Print occasionally
                    print(f"  {self.filter_type}-{self.lqr_type}: Converged at step {step}")
                # Could break here for early stopping, but continue for full simulation
            
        cost_to_go_list = []
        for i in range(len(cost_list)):
            cost_to_go = np.sum(cost_list[i:])
            cost_to_go_list.append(cost_to_go)  
            
        return rmse_list, var_list, cost_to_go_list, time_list
        
            

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
    err_list_qkf, var_list_qkf, _, time_list_qkf = lqe_qkf.run_sim()
    
    lqg_ekf = LQG(n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, H=H, filter_type='ekf', lqr_type='orig', goal_state=goal_state)
    err_list_ekf, var_list_ekf, cost_list_ekf, time_list_ekf = lqg_ekf.run_sim()
    
    lqg_qkf_aug_num = LQG(n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, H=H, filter_type='qkf', lqr_type='aug_numeric', goal_state=goal_state)
    err_list_aug_num, var_list_aug_num, cost_list_aug_num, time_list_aug_num = lqg_qkf_aug_num.run_sim()
    
    lqg_qkf_aug_analytic = LQG(n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, H=H, filter_type='qkf', lqr_type='aug_analytic', goal_state=goal_state)
    err_list_aug_analytic, var_list_aug_analytic, cost_list_aug_analytic, time_list_aug_analytic = lqg_qkf_aug_analytic.run_sim()
    
    lqg_ukf = LQG(n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, H=H, filter_type='ukf', lqr_type='orig', goal_state=goal_state)
    err_list_ukf, var_list_ukf, cost_list_ukf, time_list_ukf = lqg_ukf.run_sim()

    return [err_list_qkf, var_list_qkf, time_list_qkf], [err_list_ekf, var_list_ekf, cost_list_ekf, time_list_ekf], [err_list_aug_num, var_list_aug_num, cost_list_aug_num, time_list_aug_num], [err_list_aug_analytic, var_list_aug_analytic, cost_list_aug_analytic, time_list_aug_analytic], [err_list_ukf, var_list_ukf, cost_list_ukf, time_list_ukf]

def test(H=1000, trials=20, plot=True, noise_scale=1e-1, m_scale=1e2, Q_scale=1.0, R_scale=1.0, rand_seed=None):
    # Check if pkl files already exist
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
            loaded_data = pkl.load(f)
            if len(loaded_data) == 3:  # Old format without time data
                err_list_ekf_all, var_list_ekf_all, cost_list_ekf_all = loaded_data
                time_list_ekf_all = [np.zeros(H) for _ in range(trials)]  # Default zeros for time
            else:  # New format with time data
                err_list_ekf_all, var_list_ekf_all, cost_list_ekf_all, time_list_ekf_all = loaded_data
        with open(qkf_file, 'rb') as f:
            loaded_data = pkl.load(f)
            if len(loaded_data) == 3:  # Old format without time data
                err_list_qkf_num_all, var_list_qkf_num_all, cost_list_qkf_num_all = loaded_data
                time_list_qkf_num_all = [np.zeros(H) for _ in range(trials)]  # Default zeros for time
            else:  # New format with time data
                err_list_qkf_num_all, var_list_qkf_num_all, cost_list_qkf_num_all, time_list_qkf_num_all = loaded_data
        with open(qkf_analytic_file, 'rb') as f:
            loaded_data = pkl.load(f)
            if len(loaded_data) == 3:  # Old format without time data
                err_list_qkf_analytic_all, var_list_qkf_analytic_all, cost_list_qkf_analytic_all = loaded_data
                time_list_qkf_analytic_all = [np.zeros(H) for _ in range(trials)]  # Default zeros for time
            else:  # New format with time data
                err_list_qkf_analytic_all, var_list_qkf_analytic_all, cost_list_qkf_analytic_all, time_list_qkf_analytic_all = loaded_data
        with open(ukf_file, 'rb') as f:
            loaded_data = pkl.load(f)
            if len(loaded_data) == 3:  # Old format without time data
                err_list_ukf_all, var_list_ukf_all, cost_list_ukf_all = loaded_data
                time_list_ukf_all = [np.zeros(H) for _ in range(trials)]  # Default zeros for time
            else:  # New format with time data
                err_list_ukf_all, var_list_ukf_all, cost_list_ukf_all, time_list_ukf_all = loaded_data
        
        # Convert to numpy arrays if they aren't already
        err_list_ekf_all = np.array(err_list_ekf_all)
        var_list_ekf_all = np.array(var_list_ekf_all)
        cost_list_ekf_all = np.array(cost_list_ekf_all)
        time_list_ekf_all = np.array(time_list_ekf_all)
        err_list_qkf_num_all = np.array(err_list_qkf_num_all)
        var_list_qkf_num_all = np.array(var_list_qkf_num_all)
        cost_list_qkf_num_all = np.array(cost_list_qkf_num_all)
        time_list_qkf_num_all = np.array(time_list_qkf_num_all)
        err_list_qkf_analytic_all = np.array(err_list_qkf_analytic_all)
        var_list_qkf_analytic_all = np.array(var_list_qkf_analytic_all)
        cost_list_qkf_analytic_all = np.array(cost_list_qkf_analytic_all)
        time_list_qkf_analytic_all = np.array(time_list_qkf_analytic_all)
        err_list_ukf_all = np.array(err_list_ukf_all)
        var_list_ukf_all = np.array(var_list_ukf_all)
        cost_list_ukf_all = np.array(cost_list_ukf_all)
        time_list_ukf_all = np.array(time_list_ukf_all)
        
        # Skip to averaging and plotting
        skip_simulation = True
    else:
        print(f"Running simulation for m_scale={int(m_scale)}...")
        skip_simulation = False
        
        err_list_ekf_all = []
        var_list_ekf_all = []
        cost_list_ekf_all = []
        time_list_ekf_all = []

        err_list_qkf_num_all = []
        var_list_qkf_num_all = []
        cost_list_qkf_num_all = []
        time_list_qkf_num_all = []

        err_list_qkf_analytic_all = []
        var_list_qkf_analytic_all = []
        cost_list_qkf_analytic_all = []
        time_list_qkf_analytic_all = []

        err_list_ukf_all = []
        var_list_ukf_all = []
        cost_list_ukf_all = []
        time_list_ukf_all = []

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
            time_list_ekf_all.append(ekf_results[3])
            
            err_list_qkf_num_all.append(qkf_num_results[0])
            var_list_qkf_num_all.append(qkf_num_results[1])
            cost_list_qkf_num_all.append(qkf_num_results[2])
            time_list_qkf_num_all.append(qkf_num_results[3])

            err_list_qkf_analytic_all.append(qkf_analytic_results[0])
            var_list_qkf_analytic_all.append(qkf_analytic_results[1])
            cost_list_qkf_analytic_all.append(qkf_analytic_results[2])
            time_list_qkf_analytic_all.append(qkf_analytic_results[3])

            err_list_ukf_all.append(ukf_results[0])
            var_list_ukf_all.append(ukf_results[1])
            cost_list_ukf_all.append(ukf_results[2])
            time_list_ukf_all.append(ukf_results[3])


    # average results
    err_list_ekf_avg = np.mean(np.array(err_list_ekf_all), axis=0)
    var_list_ekf_avg = np.mean(np.array(var_list_ekf_all), axis=0)
    cost_list_ekf_avg = np.mean(np.array(cost_list_ekf_all), axis=0)
    time_list_ekf_avg = np.mean(np.array(time_list_ekf_all), axis=0)

    err_list_qkf_num_avg = np.mean(np.array(err_list_qkf_num_all), axis=0)
    var_list_qkf_num_avg = np.mean(np.array(var_list_qkf_num_all), axis=0)
    cost_list_qkf_num_avg = np.mean(np.array(cost_list_qkf_num_all), axis=0)
    time_list_qkf_num_avg = np.mean(np.array(time_list_qkf_num_all), axis=0)

    err_list_qkf_analytic_avg = np.mean(np.array(err_list_qkf_analytic_all), axis=0)
    var_list_qkf_analytic_avg = np.mean(np.array(var_list_qkf_analytic_all), axis=0)
    cost_list_qkf_analytic_avg = np.mean(np.array(cost_list_qkf_analytic_all), axis=0)
    time_list_qkf_analytic_avg = np.mean(np.array(time_list_qkf_analytic_all), axis=0)

    err_list_ukf_avg = np.mean(np.array(err_list_ukf_all), axis=0)
    var_list_ukf_avg = np.mean(np.array(var_list_ukf_all), axis=0)
    cost_list_ukf_avg = np.mean(np.array(cost_list_ukf_all), axis=0)
    time_list_ukf_avg = np.mean(np.array(time_list_ukf_all), axis=0)

    # Only save pkl files if simulation was actually run
    if not skip_simulation:
        os.makedirs(pkl_dir, exist_ok=True)
        with open(pkl_dir + f'ekf_results-mscale={int(m_scale)}.pkl', 'wb') as f:
            pkl.dump((np.array(err_list_ekf_all), np.array(var_list_ekf_all), np.array(cost_list_ekf_all), np.array(time_list_ekf_all)), f)
        with open(pkl_dir + f'qkf_results-mscale={int(m_scale)}.pkl', 'wb') as f:
            pkl.dump((np.array(err_list_qkf_num_all), np.array(var_list_qkf_num_all), np.array(cost_list_qkf_num_all), np.array(time_list_qkf_num_all)), f)
        with open(pkl_dir + f'qkf_analytic_results-mscale={int(m_scale)}.pkl', 'wb') as f:
            pkl.dump((np.array(err_list_qkf_analytic_all), np.array(var_list_qkf_analytic_all), np.array(cost_list_qkf_analytic_all), np.array(time_list_qkf_analytic_all)), f)
        with open(pkl_dir + f'ukf_results-mscale={int(m_scale)}.pkl', 'wb') as f:
            pkl.dump((np.array(err_list_ukf_all), np.array(var_list_ukf_all), np.array(cost_list_ukf_all), np.array(time_list_ukf_all)), f)

    
    if plot:  
        # Plot estimation performance comparison with publication-ready styling
        fig, ax = plt.subplots(2, 1, figsize=(8, 12))
        fig.suptitle('Estimation Performance Comparison', fontsize=24, y=0.98, fontweight='bold')
        
        # Error subplot
        ax[0].set_title(f'Estimation Error, Nonlinearity Scale = {int(m_scale)}', fontsize=20, fontweight='bold', pad=15)
        ax[0].set_xlabel('Time Step', fontsize=20, fontweight='bold')
        ax[0].set_ylabel('Mean Squared Error', fontsize=20, fontweight='bold')
        ax[0].plot(err_list_ekf_avg, label='EKF', color=PUBLICATION_COLORS['ekf'], linewidth=2.5)
        ax[0].plot(err_list_ukf_avg, label='UKF', color=PUBLICATION_COLORS['ukf'], linewidth=2.5)
        ax[0].plot(err_list_qkf_num_avg, label='QKF(numeric)', color=PUBLICATION_COLORS['qkf_numeric'], linewidth=2.5)
        ax[0].plot(err_list_qkf_analytic_avg, label='QKF(analytic)', color=PUBLICATION_COLORS['qkf_analytic'], linewidth=2.5)
        ax[0].legend(frameon=True, fancybox=True, shadow=True, loc='upper right')
        ax[0].grid(True, alpha=0.3, linestyle='--')
        ax[0].set_yscale('log')
        ax[0].legend(
            frameon=True,
            fancybox=True,
            shadow=True,
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            fontsize=14
        )
        # Variance subplot
        ax[1].set_title('Estimation Variance', fontsize=20, fontweight='bold', pad=15)
        ax[1].set_xlabel('Time Step', fontsize=20, fontweight='bold')
        ax[1].set_ylabel('Estimation Variance', fontsize=20, fontweight='bold')
        ax[1].plot(var_list_ekf_avg, label='EKF', color=PUBLICATION_COLORS['ekf'], linewidth=2.5)
        ax[1].plot(var_list_qkf_num_avg, label='QKF(numeric)', color=PUBLICATION_COLORS['qkf_numeric'], linewidth=2.5)
        ax[1].plot(var_list_qkf_analytic_avg, label='QKF(analytic)', color=PUBLICATION_COLORS['qkf_analytic'], linewidth=2.5)
        ukf_line = ax[1].plot(var_list_ukf_avg, label='UKF', color=PUBLICATION_COLORS['ukf'], linewidth=2.5)[0]
        handles, labels = ax[1].get_legend_handles_labels()
        # Reorder legend to original: EKF, UKF, QKF(numeric), QKF(analytic)
        legend_order = [labels.index('EKF'), labels.index('UKF'), labels.index('QKF(numeric)'), labels.index('QKF(analytic)')]
        ax[1].legend([handles[i] for i in legend_order], [labels[i] for i in legend_order], frameon=True, fancybox=True, shadow=True, loc='upper right')
        ax[1].grid(True, alpha=0.3, linestyle='--')
        ax[1].set_yscale('log')
        ax[1].legend(
            frameon=True,
            fancybox=True,
            shadow=True,
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            fontsize=14
        )
        plt.tight_layout()
        plt.savefig(perf_dir + 'estimation_performance.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        # Bar chart: average time consumption per time step for each controller
        avg_times = [np.mean(time_list_ekf_avg), np.mean(time_list_ukf_avg), np.mean(time_list_qkf_num_avg), np.mean(time_list_qkf_analytic_avg)]
        controllers = ['LQG+EKF', 'LQG+UKF', 'iLQG+QKF(numeric)', 'LQG+QKF(analytic)']
        colors = [PUBLICATION_COLORS['ekf'], PUBLICATION_COLORS['ukf'], PUBLICATION_COLORS['qkf_numeric'], PUBLICATION_COLORS['qkf_analytic']]

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(controllers, avg_times, color=colors, edgecolor='black')
        ax.set_title(f'Average Computational Time per Step, Nonlinearity Scale = {int(m_scale)}', fontsize=20, fontweight='bold', pad=20)
        ax.set_ylabel('Average Time per Step (seconds)', fontsize=20, fontweight='bold')
        ax.set_xlabel('Controller', fontsize=20, fontweight='bold')
        ax.tick_params(axis='x', labelsize=12, rotation=15)
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')

        # Annotate bars with values
        for bar, value in zip(bars, avg_times):
            ax.text(bar.get_x() + bar.get_width()/2, value, f'{value:.2e}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(perf_dir + f'avg_timing_bar-mscale={int(m_scale)}.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        # Plot cost performance comparison with publication-ready styling
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(f'Cost Performance Comparison, Nonlinearity Scale = {int(m_scale)}', fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel('Time Step', fontsize=20, fontweight='bold')
        ax.set_ylabel('Cost-to-go (log scale)', fontsize=20, fontweight='bold')
        
        # Plot with enhanced styling
        ax.plot(cost_list_ekf_avg, label='LQG+EKF', color=PUBLICATION_COLORS['ekf'], linewidth=2.5, linestyle='-')
        ax.plot(cost_list_ukf_avg, label='LQG+UKF', color=PUBLICATION_COLORS['ukf'], linewidth=2.5, linestyle='--')
        ax.plot(cost_list_qkf_num_avg, label='iLQG+QKF(numeric)', color=PUBLICATION_COLORS['qkf_numeric'], linewidth=2.5, linestyle='-.')
        ax.plot(cost_list_qkf_analytic_avg, label='LQG+QKF(analytic)', color=PUBLICATION_COLORS['qkf_analytic'], linewidth=2.5, linestyle=':')
        
        # ax.legend(frameon=True, fancybox=True, shadow=True, loc='upper right', fontsize=11)
        ax.legend(
            frameon=True,
            fancybox=True,
            shadow=True,
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            fontsize=14
        )
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_yscale('log')
        
        # Enhance tick labels
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        plt.tight_layout()
        plt.savefig(perf_dir + f'cost_perf-mscale={int(m_scale)}.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        # Plot timing performance comparison with publication-ready styling
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(f'Computational Time per Step, Nonlinearity Scale = {int(m_scale)}', fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel('Time Step', fontsize=20, fontweight='bold')
        ax.set_ylabel('Time per Step (seconds)', fontsize=20, fontweight='bold')
        
        # Plot with enhanced styling
        ax.plot(time_list_ekf_avg, label='LQG+EKF', color=PUBLICATION_COLORS['ekf'], linewidth=2.5, linestyle='-')
        ax.plot(time_list_ukf_avg, label='LQG+UKF', color=PUBLICATION_COLORS['ukf'], linewidth=2.5, linestyle='--')
        ax.plot(time_list_qkf_num_avg, label='iLQG+QKF(numeric)', color=PUBLICATION_COLORS['qkf_numeric'], linewidth=2.5, linestyle='-.')
        ax.plot(time_list_qkf_analytic_avg, label='LQG+QKF(analytic)', color=PUBLICATION_COLORS['qkf_analytic'], linewidth=2.5, linestyle=':')
            
        ax.legend(frameon=True, fancybox=True, shadow=True, loc='upper right', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_yscale('log')
        
        # Enhance tick labels
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        plt.tight_layout()
        plt.savefig(perf_dir + f'timing_perf-mscale={int(m_scale)}.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Plot convergence comparison with publication-ready styling - only right two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        # fig.suptitle('Convergence Analysis', fontsize=20, y=0.98, fontweight='bold')
        
        # Use improved convergence detection
        convergence_ekf = []
        convergence_qkf_num = []
        convergence_ukf = []
        convergence_qkf_analytic = []
        
        tolerance = np.mean(cost_list_ekf_avg[:100]) * 0.01  # 1% of initial cost as tolerance
        
        # # Debug: Check array shapes
        # print(f"cost_list_ekf_all shape: {cost_list_ekf_all.shape}")
        # print(f"cost_list_qkf_analytic_all shape: {cost_list_qkf_analytic_all.shape}")
        # print(f"trials: {trials}")
        
        for cnt in range(trials):
            # For each trial, detect convergence using the improved method
            conv_ekf, _ = detect_convergence(cost_list_ekf_all[cnt], tolerance=tolerance)
            conv_qkf_num, _ = detect_convergence(cost_list_qkf_num_all[cnt], tolerance=tolerance)
            conv_ukf, _ = detect_convergence(cost_list_ukf_all[cnt], tolerance=tolerance)
            conv_qkf_analytic, _ = detect_convergence(cost_list_qkf_analytic_all[cnt], tolerance=tolerance)
            # print(conv_qkf_analytic)
            convergence_ekf.append(conv_ekf if conv_ekf is not None else H)
            convergence_qkf_num.append(conv_qkf_num if conv_qkf_num is not None else H)
            convergence_ukf.append(conv_ukf if conv_ukf is not None else H)
            convergence_qkf_analytic.append(conv_qkf_analytic if conv_qkf_analytic is not None else H)
        
        # Subplot 1: Average Convergence statistics
        ax1 = axes[0]
        ax1.set_title(f'Average Convergence Time, Nonlinearity Scale = {int(m_scale)}', pad=20, fontsize=24, fontweight='bold')
        methods = ['LQG+EKF', 'LQG+UKF', 'iLQG+QKF\n(numeric)', 'LQG+QKF\n(analytic)']
        print(methods)
        avg_conv_times = [np.mean(convergence_ekf), np.mean(convergence_ukf), np.mean(convergence_qkf_num), np.mean(convergence_qkf_analytic)]
        std_conv_times = [np.std(convergence_ekf), np.std(convergence_ukf), np.std(convergence_qkf_num), np.std(convergence_qkf_analytic)]
        
        # Enhanced bar plot with better colors and styling
        colors = [PUBLICATION_COLORS['ekf'], PUBLICATION_COLORS['ukf'], PUBLICATION_COLORS['qkf_numeric'], PUBLICATION_COLORS['qkf_analytic']]
        bars = ax1.bar(methods, avg_conv_times, yerr=std_conv_times, capsize=8, alpha=0.8, 
                       color=colors, edgecolor='black', linewidth=1)
        ax1.set_ylabel('Average Convergence Time (steps)', fontsize=20, fontweight='bold')
        ax1.tick_params(axis='x', rotation=0, labelsize=16, labelcolor=PUBLICATION_COLORS['text'])
        ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add value labels on bars with better positioning and styling
        for bar, avg_time, std_time in zip(bars, avg_conv_times, std_conv_times):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_time + max(std_conv_times) * 0.1, 
                    f'{avg_time:.0f}±{std_time:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Subplot 2: Final convergence status
        ax2 = axes[1]
        # Subplot 2: Final convergence status
        ax2 = axes[1]
        ax2.set_title(f'Convergence Rate, Nonlinearity Scale = {int(m_scale)}', pad=20, fontsize=24, fontweight='bold')
        conv_counts = [
            np.sum(np.array(convergence_ekf) < H),
            np.sum(np.array(convergence_ukf) < H),
            np.sum(np.array(convergence_qkf_num) < H),
            np.sum(np.array(convergence_qkf_analytic) < H),
        ]
        conv_percentages = [count/trials*100 for count in conv_counts]
        
        # Enhanced bar plot with better styling
        colors = [PUBLICATION_COLORS['ekf'], PUBLICATION_COLORS['ukf'], PUBLICATION_COLORS['qkf_numeric'], PUBLICATION_COLORS['qkf_analytic']]
        bars = ax2.bar(methods, conv_percentages, alpha=0.8, color=colors, edgecolor='black', linewidth=1)
        ax2.set_ylabel('Convergence Rate (%)', fontsize=20, fontweight='bold')
        ax2.tick_params(axis='x', rotation=0, labelsize=18, labelcolor=PUBLICATION_COLORS['text'])
        ax2.set_ylim(0, 110)  # Increase y-limit to prevent overlap
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add percentage labels with better positioning and styling
        for bar, pct in zip(bars, conv_percentages):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax2.tick_params(axis='both', which='major', labelsize=16)
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.savefig(perf_dir + 'convergence_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Print summary statistics
        print(f"\n=== Convergence Analysis Summary ===")
        print(f"Tolerance used: {tolerance:.2e}")
        print(f"LQG+EKF - Avg convergence time: {np.mean(convergence_ekf):.1f} ± {np.std(convergence_ekf):.1f}")
        print(f"LQG+UKF - Avg convergence time: {np.mean(convergence_ukf):.1f} ± {np.std(convergence_ukf):.1f}")
        print(f"iLQG+QKF(numeric) - Avg convergence time: {np.mean(convergence_qkf_num):.1f} ± {np.std(convergence_qkf_num):.1f}")
        print(f"LQG+QKF(analytic) - Avg convergence time: {np.mean(convergence_qkf_analytic):.1f} ± {np.std(convergence_qkf_analytic):.1f}")
        # print(f"Analytic QKF - Avg convergence time: {np.mean(convergence_qkf_analytic):.1f} ± {np.std(convergence_qkf_analytic):.1f}")
        print(f"Convergence rates: LQG+EKF {conv_percentages[0]:.1f}%, LQG+UKF {conv_percentages[1]:.1f}%, iLQG+QKF(numeric) {conv_percentages[2]:.1f}%, LQG+QKF(analytic) {conv_percentages[3]:.1f}%")
        
        # Print timing summary statistics
        print(f"\n=== Timing Performance Summary ===")
        print(f"LQG+EKF - Avg time per step: {np.mean(time_list_ekf_avg):.4f} ± {np.std(time_list_ekf_avg):.4f} seconds")
        print(f"LQG+UKF - Avg time per step: {np.mean(time_list_ukf_avg):.4f} ± {np.std(time_list_ukf_avg):.4f} seconds")
        print(f"iLQG+QKF(numeric) - Avg time per step: {np.mean(time_list_qkf_num_avg):.4f} ± {np.std(time_list_qkf_num_avg):.4f} seconds")
        print(f"LQG+QKF(analytic) - Avg time per step: {np.mean(time_list_qkf_analytic_avg):.4f} ± {np.std(time_list_qkf_analytic_avg):.4f} seconds")
        print(f"Total simulation time: LQG+EKF {np.sum(time_list_ekf_avg):.2f}s, LQG+UKF {np.sum(time_list_ukf_avg):.2f}s, iLQG+QKF(numeric) {np.sum(time_list_qkf_num_avg):.2f}s, LQG+QKF(analytic) {np.sum(time_list_qkf_analytic_avg):.2f}s")

    # return cost_list_ekf_avg, cost_list_qkf_num_avg, cost_list_qkf_analytic_avg, cost_list_ukf_avg
    return cost_list_ekf_avg, cost_list_ukf_avg, cost_list_qkf_num_avg, cost_list_qkf_analytic_avg

def nonlinearity_test(H=1000, trials=20):
    m_scales = [0, 1, 1e1, 1e2, 1e3, 1e4]
    rand_seed = 100  # use the same base for all m_scales
    
    # Store results for plotting
    all_results = {
        'm_scales': m_scales,
        'ekf_costs': [],
        'ukf_costs': [],
        'qkf_numeric_costs': [],
        'qkf_analytic_costs': []
    }
    
    for i, m_scale in enumerate(m_scales):
        print(f"Testing with m_scale={m_scale}")
        cost_list_ekf_avg, cost_list_ukf_avg, cost_list_qkf_num_avg, cost_list_qkf_analytic_avg = test(H=H, trials=trials, plot=False, m_scale=m_scale, rand_seed=rand_seed)
        
        # Store average costs over ALL time steps for comparison
        all_results['ekf_costs'].append(np.mean(cost_list_ekf_avg))  # Average over all time steps
        all_results['ukf_costs'].append(np.mean(cost_list_ukf_avg))
        all_results['qkf_numeric_costs'].append(np.mean(cost_list_qkf_num_avg))
        all_results['qkf_analytic_costs'].append(np.mean(cost_list_qkf_analytic_avg))
    
    # Plot nonlinearity analysis results
    plot_nonlinearity_analysis(all_results)
    
    return all_results


def plot_nonlinearity_analysis(results):
    """
    Create publication-ready plots for nonlinearity analysis.
    
    Args:
        results: Dictionary containing m_scales and cost data for each method
    """
    # Create figure with publication-ready styling
    fig, ax = plt.subplots(figsize=(8, 6))
    # Subplot 2: Cost vs Nonlinearity (log scale for better visualization)
    ax.set_title('Average Cost Performance vs. Nonlinearity', fontsize=20, fontweight='bold', pad=15)
    ax.set_xlabel('Nonlinearity Scale (m)', fontsize=20, fontweight='bold')
    ax.set_ylabel('Average Cost', fontsize=20, fontweight='bold')
    
    
    # Plot with enhanced styling
    ax.loglog(results['m_scales'], results['ekf_costs'],  
               label='LQG+EKF', color=PUBLICATION_COLORS['ekf'], 
               marker='o', markersize=8, linewidth=3, markeredgecolor='white', markeredgewidth=1)
    ax.loglog(results['m_scales'], results['ukf_costs'], 
               label='LQG+UKF', color=PUBLICATION_COLORS['ukf'], 
               marker='s', markersize=8, linewidth=3, markeredgecolor='white', markeredgewidth=1)
    ax.loglog(results['m_scales'], results['qkf_numeric_costs'], 
               label='iLQG+QKF(numeric)', color=PUBLICATION_COLORS['qkf_numeric'], 
               marker='^', markersize=8, linewidth=3, markeredgecolor='white', markeredgewidth=1)
    ax.loglog(results['m_scales'], results['qkf_analytic_costs'], 
               label='LQG+QKF(analytic)', color=PUBLICATION_COLORS['qkf_analytic'], 
               marker='d', markersize=8, linewidth=3, markeredgecolor='white', markeredgewidth=1)
    
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(perf_dir + 'nonlinearity_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n=== Nonlinearity Analysis Summary ===")
    print(f"Nonlinearity scales tested: {results['m_scales']}")
    print(f"EKF average costs: {[f'{cost:.2e}' for cost in results['ekf_costs']]}")
    print(f"UKF average costs: {[f'{cost:.2e}' for cost in results['ukf_costs']]}")
    print(f"QKF(numeric) average costs: {[f'{cost:.2e}' for cost in results['qkf_numeric_costs']]}")
    print(f"QKF(analytic) average costs: {[f'{cost:.2e}' for cost in results['qkf_analytic_costs']]}")


if __name__ == "__main__":
    test(H=1000, trials=100, plot=True, noise_scale=1e-1, m_scale=1e2, Q_scale=1.0, R_scale=1.0)
    nonlinearity_test(H=1000, trials=100)
        



