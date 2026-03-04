import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pickle as pkl
from datetime import datetime
import random
import time
from collections import Counter

PROJ_DIR = 'D:/AC/UCLA/ECE/UCLA_LEMUR/nonlinear_LQG/LQG_QKF/CDC/'
os.chdir(PROJ_DIR)

from LQG_QKF import LQG, generate_stable_system_parameters, validate_stable_parameters, generate_random_symmetric_matrix, generate_goal_state
from stateDynamics import StateDynamics, sensor, Vec, invVec
from typing import Literal
from tqdm import tqdm
import numpy as np
import random
import itertools
import time
import os

import sys

# Handle command-line arguments (optional for programmatic use)
if len(sys.argv) > 1:
    info_gain_method = sys.argv[1]
    if info_gain_method not in ['heuristic', 'baseline', 'van_trees']:
        raise ValueError(f"Invalid information gain method: {info_gain_method}")
    
    if info_gain_method == 'van_trees':
        assert len(sys.argv) == 3, "Van Trees method requires a metric"
        metric = sys.argv[2]
        if metric not in ['T', 'D', 'E', 'A']:
            raise ValueError(f"Invalid metric: {metric}")
    else:
        metric = 'T'  # Default metric (not used for heuristic/baseline)
else:
    # Default values when run programmatically
    info_gain_method = 'baseline'
    metric = 'T'

os.chdir(os.path.dirname(os.path.abspath(__file__)))
test_dir = f'sensor_sim-{info_gain_method}/'
pkl_dir = test_dir + 'pkl/'
os.makedirs(pkl_dir, exist_ok=True)
perf_dir = test_dir + 'perf/'
os.makedirs(perf_dir, exist_ok=True)

# build sensor selection simulator
random_seed = 42
small_value = 1e-6
class SensorSelectionSimulator(LQG):
    def __init__(self, n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, H = 50, m_scale=1e2,
                 filter_type: Literal['qkf', 'ekf', 'kf', 'ukf', 'pf'] = 'qkf',
                 lqr_type: Literal['orig', 'aug_analytic', 'aug_numeric', 'None'] = 'orig',
                 max_sensors=None,
                 update_interval=10,
                 info_gain_method: Literal['heuristic', 'baseline', 'van_trees'] = 'baseline',
                 metric: Literal['T', 'D', 'E', 'A'] = 'T',
                 n_particles: int = 1000):
        
        self.filter_type = filter_type
        self.lqr_type = lqr_type
        self.info_gain_method = info_gain_method
        self.metric = metric
        # dynamics setting
        self.F = StateDynamics(n1, n2, p , W, A_E, A_S, B_S)
        n = n1 + n2 # state size
        
        # sensor settings
        self.m_scale = m_scale
        M = M * m_scale
        self.sensor = sensor(C, M, V)
        self.V = self.sensor.get_V()
        
        # state settings
        self.A = np.zeros((n,n))
        self.A[:n1, :n1] = A_E
        self.A[n1:, n1:] = A_S
        self.B = np.zeros((n,p))
        self.B[n1:, :p] = B_S
        self.n1 = n1 
        self.n2 = n2
        self.n = n
        self.p = self.F.get_input_size() # control input size
        self.W = self.F.get_W()
        self.update_interval = update_interval
        
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
        self.z_hat = np.zeros((self.n + self.n**2, 1)) # estimated augmented state vector
        self.x_goal = np.zeros((self.n, 1)) # goal state vector
        
        # lqr
        self.x_goal = np.zeros((self.n, 1))  # Initialize with zeros, will be updated by get_next_sensor_selection()
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
        
        # sensor selection
        self.max_sensors = max_sensors if max_sensors is not None else self.sensor.m
        self.active_sensors = list(range(min(self.sensor.m, self.max_sensors)))  # Initialize with max allowed sensors
        
        # performance tracking
        self.performance_history = {
            'cost': [],
            'estimation_error': [],
            'control_effort': [],
            'sensor_selections': [],
            'information_gains': [],
            'covariance_traces': [],
            'time_consumption': []
        } 
        
    def get_next_sensor_selection(self):
        """
        Compute next sensor selection using greedy algorithm.
        Selects a subset of sensors that maximizes information gain.
        Returns the selected sensor indices and goal state.
        """
        # Get current state estimate
        x_current = self.x_hat
        
        # Generate all possible sensor subsets (binary combinations)
        sensor_subsets = self._generate_sensor_subsets()
        
        best_subset = None
        best_value = -np.inf
        info_gains = []
        
        # Greedy selection: choose sensor subset that maximizes information gain
        # NOTE: For heuristic method, all subsets of the same size have identical gain,
        # so this will select the lexicographically first subset (e.g., [0,1,2] for size 3)
        for subset in sensor_subsets:
            # Calculate information gain for this sensor subset
            info_gain = self._calculate_information_gain_subset(subset, method=self.info_gain_method, metric=self.metric)
            info_gains.append(info_gain)
            
            if info_gain > best_value:
                best_value = info_gain
                best_subset = subset
        
        # Record sensor selection performance
        self.performance_history['sensor_selections'].append(best_subset)
        self.performance_history['information_gains'].append(info_gains)
        
        # Update active sensors
        self.active_sensors = best_subset
        
        # Generate goal state based on selected sensors
        goal_state = self._generate_goal_from_sensors(best_subset, x_current)
        
        return goal_state
    
    def _generate_sensor_subsets(self):
        """
        Generate all possible sensor subsets (binary combinations).
        Each subset represents which sensors are active (ON) or inactive (OFF).
        Respects the max_sensors constraint.
        """
        import itertools
        
        # Get total number of sensors and max allowed
        num_sensors = self.sensor.m  # Number of available sensors
        max_allowed = min(num_sensors, self.max_sensors)  # Respect max_sensors constraint
        
        # Generate all possible subsets (including empty set and up to max_allowed)
        subsets = []
        for r in range(max_allowed + 1):  # 0 to max_allowed sensors
            for subset in itertools.combinations(range(num_sensors), r):
                subsets.append(list(subset))
        
        return subsets
    
    def _compute_covariance_for_sensors(self, sensor_subset):
        """
        Compute the covariance matrix after using a specific sensor subset.
        Similar to RAPID's cov_matrix method, but adapted for nonlinear filters.
        
        Args:
            sensor_subset: List of sensor indices to use
            
        Returns:
            Updated covariance matrix (P_est for EKF/KF/UKF, Pz_est[:n,:n] for QKF)
        """
        if len(sensor_subset) == 0:
            # No sensors: return current covariance
            if self.filter_type == 'qkf':
                return self.Pz_est[:self.n, :self.n].copy()
            else:
                return self.P_est.copy()
        
        # Get measurement matrices for selected sensors only
        C_subset = self.sensor.C[sensor_subset, :]
        M_subset = self.sensor.M[sensor_subset, :, :]
        V_subset = self.sensor.V[np.ix_(sensor_subset, sensor_subset)]
        
        if self.filter_type == 'qkf':
            # QKF: Use augmented state covariance
            # Get predicted augmented state covariance
            Phi_tilde = self.F.get_A_tilde()
            Sigma_tilde = self.F.get_Sigma_tilde()
            Pz_pred = Phi_tilde @ self.Pz_est @ Phi_tilde.T + Sigma_tilde
            
            # Build augmented measurement matrix for selected sensors
            measB_tilde_subset = self._build_aug_measB_subset(C_subset, M_subset)
            
            # Innovation covariance
            M_innov = measB_tilde_subset @ Pz_pred @ measB_tilde_subset.T + V_subset
            
            # Kalman gain
            K = Pz_pred @ measB_tilde_subset.T @ np.linalg.pinv(M_innov)
            
            # Updated covariance (posterior)
            Pz_post = Pz_pred - K @ M_innov @ K.T
            
            # Return only the state part (first n x n block)
            return Pz_post[:self.n, :self.n]
            
        elif self.filter_type == 'ekf':
            # EKF: Use Jacobian linearization
            # Get predicted state and covariance
            mu = self.F.B @ self.F.u
            Phi = self.F.A
            Sigma = self.F.W
            X_pred = mu + Phi @ self.x_hat
            P_pred = Phi @ self.P_est @ Phi.T + Sigma
            
            # Compute Jacobian for selected sensors
            g_subset = self._compute_jacobian_subset(X_pred, C_subset, M_subset)
            
            # Innovation covariance
            M_innov = g_subset @ P_pred @ g_subset.T + V_subset
            
            # Kalman gain
            K = P_pred @ g_subset.T @ np.linalg.pinv(M_innov)
            
            # Updated covariance (posterior)
            P_post = P_pred - K @ M_innov @ K.T
            
            return P_post
            
        elif self.filter_type == 'kf':
            # KF: Standard linear Kalman filter
            # Get predicted covariance
            P_pred = self.A @ self.P_est @ self.A.T + self.W
            
            # Innovation covariance
            M_innov = C_subset @ P_pred @ C_subset.T + V_subset
            
            # Kalman gain
            K = P_pred @ C_subset.T @ np.linalg.pinv(M_innov)
            
            # Updated covariance (posterior)
            P_post = P_pred - K @ M_innov @ K.T
            
            return P_post
            
        elif self.filter_type == 'ukf':
            # UKF: Full sigma-point update for the sensor subset (consistent with update_lqe_ukf)
            n = self.n
            alpha = 1e-3
            beta = 2
            kappa = 0
            lambda_ = alpha**2 * (n + kappa) - n

            # Sigma points from current (x_hat, P_est)
            sigma_points = np.zeros((2 * n + 1, n))
            sigma_points[0] = self.x_hat.flatten()
            try:
                sqrt_P = np.linalg.cholesky((n + lambda_) * self.P_est)
            except np.linalg.LinAlgError:
                eigenvals, eigenvecs = np.linalg.eigh(self.P_est)
                eigenvals = np.maximum(eigenvals, 1e-8)
                sqrt_P = eigenvecs @ np.diag(np.sqrt(eigenvals))
                sqrt_P = np.sqrt(n + lambda_) * sqrt_P
            for i in range(n):
                sigma_points[i + 1] = self.x_hat.flatten() + sqrt_P[i]
                sigma_points[n + i + 1] = self.x_hat.flatten() - sqrt_P[i]

            # Predict sigma points through state dynamics
            sigma_points_pred = np.zeros_like(sigma_points)
            for i in range(2 * n + 1):
                x_pred = self.F.A @ sigma_points[i].reshape(-1, 1) + self.F.B @ self.F.u
                sigma_points_pred[i] = x_pred.flatten()

            weights_mean = np.full(2 * n + 1, 1 / (2 * (n + lambda_)))
            weights_mean[0] = lambda_ / (n + lambda_)
            weights_cov = np.full(2 * n + 1, 1 / (2 * (n + lambda_)))
            weights_cov[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)

            x_predicted = np.sum(weights_mean[:, np.newaxis] * sigma_points_pred, axis=0).reshape(-1, 1)
            P_pred = self.F.W.copy()
            for i in range(2 * n + 1):
                diff = sigma_points_pred[i] - x_predicted.flatten()
                P_pred += weights_cov[i] * np.outer(diff, diff)

            # Propagate sigma points through subset measurement model
            m_subset = C_subset.shape[0]
            sigma_points_meas = np.zeros((2 * n + 1, m_subset))
            for i in range(2 * n + 1):
                sigma_points_meas[i] = self._measure_pred_subset(sigma_points_pred[i], C_subset, M_subset)

            y_predicted = np.sum(weights_mean[:, np.newaxis] * sigma_points_meas, axis=0)
            S = V_subset.copy()
            for i in range(2 * n + 1):
                diff = sigma_points_meas[i] - y_predicted
                S += weights_cov[i] * np.outer(diff, diff)

            C_tilde = np.zeros((n, m_subset))
            for i in range(2 * n + 1):
                diff_state = sigma_points_pred[i] - x_predicted.flatten()
                diff_meas = sigma_points_meas[i] - y_predicted
                C_tilde += weights_cov[i] * np.outer(diff_state, diff_meas)

            K = C_tilde @ np.linalg.pinv(S)
            P_post = P_pred - K @ S @ K.T
            return P_post
        
        elif self.filter_type == 'pf':
            # Particle filter: one-step predict + subset measurement update, return sample covariance
            n_particles_pf = 300
            mu = self.F.B @ self.F.u
            Phi = self.F.A
            Sigma = self.F.W
            x_pred = (mu + Phi @ self.x_hat).flatten()
            P_pred = Phi @ self.P_est @ Phi.T + Sigma
            chol_P = np.linalg.cholesky(P_pred + np.eye(self.n) * 1e-10)
            particles = x_pred + (chol_P @ np.random.randn(self.n, n_particles_pf)).T  # (N, n)
            y_dummy = self._measure_pred_subset(x_pred, C_subset, M_subset)  # (m_subset,)
            V_subset_inv = np.linalg.inv(V_subset + np.eye(V_subset.shape[0]) * 1e-10)
            log_w = np.zeros(n_particles_pf)
            for i in range(n_particles_pf):
                h_i = self._measure_pred_subset(particles[i], C_subset, M_subset)
                diff = y_dummy - h_i
                log_w[i] = -0.5 * diff @ V_subset_inv @ diff
            log_w -= log_w.max()
            weights = np.exp(log_w)
            weights /= weights.sum()
            # Systematic resampling
            cumsum = np.cumsum(weights)
            u0 = np.random.rand() / n_particles_pf
            indices = np.zeros(n_particles_pf, dtype=int)
            j = 0
            for i in range(n_particles_pf):
                u = u0 + i / n_particles_pf
                while j < n_particles_pf - 1 and u > cumsum[j]:
                    j += 1
                indices[i] = min(j, n_particles_pf - 1)
            particles = particles[indices]
            x_mean = particles.mean(axis=0)
            diff = particles - x_mean
            P_post = (diff.T @ diff) / n_particles_pf + np.eye(self.n) * small_value
            return P_post
        
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")
    
    def _build_aug_measB_subset(self, C_subset, M_subset):
        """
        Build augmented measurement matrix B_tilde for selected sensors.
        Similar to sensor.get_aug_measB() but for a subset.
        """
        m_subset, n = C_subset.shape
        
        # Build the right-hand block: each row i is vec(M[i].T)
        right_term = np.zeros((m_subset, n**2))
        for i in range(m_subset):
            right_term[i] = Vec(M_subset[i].T).squeeze()
        
        # Horizontal concatenation
        B_tilde = np.hstack((C_subset, right_term))  # shape (m_subset, n+n^2)
        return B_tilde
    
    def _measure_pred_subset(self, x, C_subset, M_subset):
        """
        Predicted measurement (no noise) for state x using only the sensor subset.
        y = C_subset @ x + [x' M_subset[i] x for each i]. Same as sensor.measure_pred but for subset.
        x: (n,) or (n, 1). Returns (m_subset,) for use in UKF sigma-point arrays.
        """
        x = np.asarray(x).reshape(-1, 1)
        term1 = (C_subset @ x).flatten()  # (m_subset,)
        term2 = np.zeros(C_subset.shape[0])
        for i in range(C_subset.shape[0]):
            term2[i] = (x.T @ M_subset[i] @ x).item()
        return term1 + term2

    def _compute_jacobian_subset(self, x, C_subset, M_subset):
        """
        Compute Jacobian g(x) for selected sensors.
        Similar to sensor.g() but for a subset.
        """
        m_subset, n = C_subset.shape
        term1 = C_subset  # Linear part
        term2 = np.zeros((m_subset, n))
        for i in range(m_subset):
            e = np.zeros((m_subset, 1))
            e[i] = 1
            term2 += e @ x.T @ M_subset[i]
        return term1 + 2 * term2  # shape (m_subset, n)
    
    def _calculate_information_gain_subset(self, sensor_subset, 
                                           method: Literal['heuristic', 'baseline', 'van_trees'] = 'baseline',
                                           metric: Literal['T', 'D', 'E', 'A'] = 'T'):
        """
        Calculate information gain for a given sensor subset.
        
        Args:
            sensor_subset: List of sensor indices
            x_current: Current state estimate (for compatibility, not used in all methods)
            method: Method to use for computing information gain
                - 'heuristic': Logarithmic approximation based on number of sensors
                - 'baseline': Rigorous approach using actual covariance reduction
                - 'van_trees': Van Trees information bound (placeholder)
        """
        if method == 'heuristic':
            # Heuristic approach: Logarithmic scaling based on number of sensors
            # NOTE: This method does NOT distinguish between different sensors - it only
            # considers the COUNT of sensors. Therefore, all subsets of the same size
            # will have identical information gain. The selection algorithm will pick the
            # lexicographically first subset encountered (e.g., [0,1,2] for size 3),
            # which is essentially arbitrary.
            # Get current uncertainty (trace of covariance matrix)
            if self.filter_type == 'qkf':
                current_uncertainty = np.trace(self.Pz_est[:self.n, :self.n])
            else:
                current_uncertainty = np.trace(self.P_est)
            
            # Calculate information gain based on sensor subset
            num_active_sensors = len(sensor_subset)
            total_sensors = self.sensor.m
            
            # Information gain increases with number of active sensors
            # but with diminishing returns (logarithmic scaling)
            if num_active_sensors == 0:
                info_gain = 0.0
            else:
                # Logarithmic scaling to model diminishing returns
                # WARNING: This ignores which specific sensors are selected!
                info_gain = current_uncertainty * np.log(1 + num_active_sensors) / np.log(1 + total_sensors)
            
            return info_gain
            
        elif method == 'baseline':
            # Baseline approach: Rigorous covariance reduction (similar to RAPID)
            # Get current uncertainty (trace of covariance matrix before update)
            if self.filter_type == 'qkf':
                # For QKF, we need to predict first, then compute what it would be after update
                Phi_tilde = self.F.get_A_tilde()
                Sigma_tilde = self.F.get_Sigma_tilde()
                Pz_pred = Phi_tilde @ self.Pz_est @ Phi_tilde.T + Sigma_tilde
                current_uncertainty = np.trace(Pz_pred[:self.n, :self.n])
            else:
                # For EKF/UKF/PF, predict covariance; KF uses linear prediction
                if self.filter_type in ('ekf', 'ukf', 'pf'):
                    mu = self.F.B @ self.F.u
                    Phi = self.F.A
                    Sigma = self.F.W
                    P_pred = Phi @ self.P_est @ Phi.T + Sigma
                else:  # KF
                    P_pred = self.A @ self.P_est @ self.A.T + self.W
                current_uncertainty = np.trace(P_pred)
            
            # Compute covariance after using this sensor subset
            if len(sensor_subset) == 0:
                # No sensors: no information gain
                return 0.0
            
            P_post = self._compute_covariance_for_sensors(sensor_subset)
            post_uncertainty = np.trace(P_post)
            
            # Information gain = reduction in uncertainty
            info_gain = current_uncertainty - post_uncertainty
            
            # Ensure non-negative (should always be, but numerical errors might cause issues)
            info_gain = max(0.0, info_gain)
            
            return info_gain
            
        elif method == 'van_trees':
            # 1. Initialize Information Matrix with the Prior (Inverse of current uncertainty)
            # F_S corresponds to the inverse of the Van Trees' bound B_S
            if self.filter_type == 'qkf':
                # For QKF, use the augmented state covariance
                P_prior = self.Pz_est[:self.n, :self.n].copy()
            else:
                P_prior = self.P_est.copy()
            
            # Prior Information Matrix (I_x in the paper)
            F_S = np.linalg.pinv(P_prior) # initialize information matrix with inv(covariance matrix) as prior knowledge
            I_prior = F_S.copy()

            if len(sensor_subset) == 0:
                return 0.0

            # 2. Add contributions from each sensor in the subset (Theorem 2)
            # Formula: F_S = I_x + sum( (1/sigma^2) * (Xi*P*Xi.T + zi*zi.T) )
            for i in sensor_subset:
                # Extract parameters for sensor i
                # X_i, z_i are known features/parameters from the observation model
                Xi = self.sensor.M[i] # Quadratic part
                zi = self.sensor.C[i:i+1, :].T # Linear part (as column vector)
                sigma2_i = self.sensor.V[i, i] # Noise variance
                
                # P is the covariance of the prior distribution
                P = P_prior 
                
                # Calculate the information contribution (rank-1 update in the paper's context)
                # Note: Xi*P*Xi.T + zi*zi.T represents the total information from quadratic observation
                inf_contribution = (Xi @ P @ Xi.T + zi @ zi.T) / sigma2_i
                F_S += inf_contribution

            # 3. Calculate Utility based on the specified metric (Theorems 3-6)
            # 'metric' should be passed into the class or handled via sys.argv as in your script
            if metric == 'T': # T-Optimality (Trace)
                return np.trace(F_S) - np.trace(I_prior)
            
            elif metric == 'D': # D-Optimality (Log-determinant)
                # np.linalg.slogdet computes the log determinant of a matrix
                return np.linalg.slogdet(F_S)[1] - np.linalg.slogdet(I_prior)[1]
            
            elif metric == 'A': # A-Optimality (Average MSE)
                # A-opt utility: Tr(inv(I_prior)) - Tr(inv(F_S))
                return np.trace(P_prior) - np.trace(np.linalg.pinv(F_S))
            
            elif metric == 'E': # E-Optimality (Worst-case)
                # minimize the the largest eigenvalue of B_S, which is the inverse of the covariance matrix
                # thus, the largest eigenvalue of B_S is the smallest eigenvalue of the covariance matrix
                # np.linalg.eigvalsh returns the eigenvalues of a symmetric matrix in ascending order
                return np.linalg.eigvalsh(F_S)[0] - np.linalg.eigvalsh(I_prior)[0]
            
            else:
                raise ValueError(f"Invalid metric for Van Trees method: {metric}")
            
        else:
            raise ValueError(f"Unknown information gain method: {method}")
    
    def _generate_goal_from_sensors(self, sensor_subset, x_current):
        """
        Generate goal state based on selected sensor subset.
        """
        goal_state = x_current.copy()
        
        # Simple goal: move towards zero with intensity based on number of active sensors
        num_active_sensors = len(sensor_subset)
        total_sensors = self.sensor.m
        
        if num_active_sensors == 0:
            # No sensors active: maintain current state
            goal_state = x_current.copy()
        else:
            # More active sensors = more aggressive control towards zero
            control_strength = num_active_sensors / total_sensors
            goal_state = (1 - control_strength) * x_current + control_strength * np.zeros((self.n, 1))
        
        # Add some small random perturbation to avoid getting stuck
        noise_scale = 0.1
        goal_state += np.random.normal(0, noise_scale, goal_state.shape)
        
        return goal_state
    
    def get_active_measurement_model(self):
        """
        Get measurement model using only active sensors.
        """
        if not self.active_sensors:
            # No sensors active - return zero measurement
            return np.zeros((0, 1)), np.zeros((0, 0))
        
        # Get measurement matrices for active sensors only
        C_active = self.sensor.C[self.active_sensors, :]
        M_active = self.sensor.M[self.active_sensors, :, :]
        V_active = self.sensor.V[np.ix_(self.active_sensors, self.active_sensors)]
        
        return C_active, M_active, V_active
    
    def run_sim(self):
        """
        Run the sensor selection simulation with comprehensive performance tracking.
        """
        # Only print if verbose (not during comparison runs)
        verbose = len(sys.argv) > 1
        if verbose:
            print(f"Starting sensor selection simulation with {self.filter_type.upper()} filter and {self.lqr_type} LQR...")
        
        # Use tqdm only if verbose, otherwise use simple range
        if verbose:
            step_range = tqdm(range(1, self.H + 1, 1), desc=f"Running simulation with filter:[{self.filter_type}], controller:[{self.lqr_type}], m_scale:[{self.m_scale}]")
        else:
            step_range = range(1, self.H + 1, 1)
        
        for step in step_range:
            step_start_time = time.time()
            
            # Update state estimation
            self.update_lqe()
            
            # Update sensor selection periodically
            if step % self.update_interval == 0:
                self.x_goal = self.get_next_sensor_selection()
            
            # Update control and forward dynamics
            if self.lqr_type != 'None':
                self.update_lqr()
                self.forward_state()
            
            # Record comprehensive performance metrics
            self._record_performance_metrics(step)
            
            # Record time consumption for this step
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            self.performance_history['time_consumption'].append(step_time)
            
        # Calculate cost-to-go
        cost_to_go_list = self._calculate_cost_to_go()
        
        # Calculate average metrics
        avg_cost = np.mean(self.performance_history['cost'])
        avg_error = np.mean(self.performance_history['estimation_error'])
        
        # Only print if verbose
        verbose = len(sys.argv) > 1
        if verbose:
            print(f"Simulation completed. Average cost: {avg_cost:.4f}")
            print(f"Average estimation error: {avg_error:.4f}")
        
        return (self.performance_history['estimation_error'], 
                self.performance_history['covariance_traces'], 
                cost_to_go_list,
                self.performance_history['time_consumption'])
    
    def _record_performance_metrics(self, step):
        """Record essential performance metrics for each time step."""
        # State information
        true_state = self.F.get_x()
        estimated_state = self.x_hat
        control_input = self.F.get_u()
        
        # Estimation error
        estimate_error = np.linalg.norm(true_state - estimated_state).item()
        
        # Covariance trace
        if self.filter_type == 'qkf':
            cov_trace = np.trace(self.Pz_est[:self.n, :self.n])
        else:
            cov_trace = np.trace(self.P_est)
        
        # Control cost
        dx = estimated_state - self.x_goal
        cost = dx.T @ self.Q[:self.n, :self.n] @ dx + control_input.T @ self.R @ control_input
        cost_value = cost.item()
        
        # Control effort
        control_effort = np.linalg.norm(control_input).item()
        
        # Record essential metrics only
        self.performance_history['estimation_error'].append(estimate_error)
        self.performance_history['covariance_traces'].append(cov_trace)
        self.performance_history['cost'].append(cost_value)
        self.performance_history['control_effort'].append(control_effort)
    
    def _calculate_cost_to_go(self):
        """Calculate cost-to-go for each time step."""
        cost_to_go_list = []
        for i in range(len(self.performance_history['cost'])):
            cost_to_go = np.sum(self.performance_history['cost'][i:])
            cost_to_go_list.append(cost_to_go)
        return cost_to_go_list
    
    
    def plot_performance(self, save_plots=True, plot_dir=None):
        """
        Create simple performance plots similar to LQG_QKF.py style.
        """
        if plot_dir is None:
            plot_dir = perf_dir
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        # Create simple 1x3 subplot layout for essential metrics only
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f'Sensor Selection Performance - {self.filter_type.upper()}', fontsize=14)
        
        # Time vector
        time_steps = np.arange(1, len(self.performance_history['cost']) + 1)
        
        # 1. Cost over time
        axes[0].plot(time_steps, self.performance_history['cost'], 'b-', linewidth=2)
        axes[0].set_title('Control Cost')
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Cost')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Estimation error over time
        axes[1].plot(time_steps, self.performance_history['estimation_error'], 'r-', linewidth=2)
        axes[1].set_title('Estimation Error')
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('||x_true - x_est||')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Covariance trace over time
        axes[2].plot(time_steps, self.performance_history['covariance_traces'], 'g-', linewidth=2)
        axes[2].set_title('Covariance Trace')
        axes[2].set_xlabel('Time Step')
        axes[2].set_ylabel('tr(P)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{plot_dir}/sensor_selection_perf_{self.filter_type}_{self.lqr_type}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Performance plots saved to: {filename}")
        
        plt.show()
        return fig
    
    def save_results(self, trial_idx=None, save_dir=pkl_dir):
        """
        Save simulation results to pickle file.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        if trial_idx is None:
            filename = f"sensor_selection_results_{self.filter_type}_{self.lqr_type}_mscale={self.m_scale}.pkl"
        else:
            filename = f"sensor_selection_results_{self.filter_type}_{self.lqr_type}_mscale={self.m_scale}-trial_{trial_idx}.pkl"
        
        filepath = os.path.join(save_dir, filename)
        
        # Prepare data for saving
        results = {
            'performance_history': self.performance_history,
            'simulation_params': {
                'filter_type': self.filter_type,
                'lqr_type': self.lqr_type,
                'H': self.H,
                'n1': self.n1,
                'n2': self.n2,
                'n': self.n,
                'p': self.p
            }
        }
        
        with open(filepath, 'wb') as f:
            pkl.dump(results, f)
        
        print(f"Results saved to: {filepath}")
        return filepath


def run_sensor_scheduling_sim(H=1000, update_interval=10, noise_scale=1e-1, m_scale=1e0, Q_scale=1.0, R_scale=1.0, num_sensors=2, max_sensors=None, rand_seed=random_seed, plot=True, trial_idx=None, save_dir=pkl_dir, info_gain_method='baseline', metric='T', 
                              A_E=None, A_S=None, B_S=None, C=None, M=None, W=None, V=None, Q=None, R=None):
    """
    Run sensor scheduling simulation.
    
    If A_E, A_S, B_S, C, M, W, V, Q, R are provided, they will be used instead of generating new ones.
    This ensures identical parameters across different method comparisons.
    """
    n1 = 2
    n2 = 2
    n = n1 + n2 # state size
    p = 3
    m = num_sensors  # number of sensors
    
    # Use provided parameters or generate new ones
    if A_E is None or A_S is None or B_S is None or C is None or M is None or W is None or V is None:
        if rand_seed is not None:
            np.random.seed(rand_seed)
        
        # Use stable parameter generation
        A_E, A_S, B_S, C, M, W, V = generate_stable_system_parameters(
            n1, n2, p, m, noise_scale, m_scale
        )
        
        # Validate stability
        if not validate_stable_parameters(A_E, A_S):
            print("Warning: Generated unstable parameters, but continuing...")
    
    if Q is None or R is None:
        if rand_seed is not None:
            np.random.seed(rand_seed)
        
        # Q, R must be symmetric positive definite matrices
        Q = generate_random_symmetric_matrix(n+n**2, scale=Q_scale)
        R = generate_random_symmetric_matrix(p, scale=R_scale)
    
    # goal_state = generate_goal_state(np.zeros((n1, 1)), n2) # goal state vector
    # lqg_kf_sys = LQG(n, p, W, A, B, C, M, V, Q, R, H=1000, filter_type='kf')
    # err_list_kf = lqg_kf_sys.run_sim()
    # plt.plot(err_list_kf, label=f'kf measure error')

    # Run simulations with different filter types
    simulators = {}
    
    # Suppress individual filter printouts when running comparison (they're verbose)
    verbose = len(sys.argv) > 1  # Only verbose when run from command line
    
    if verbose:
        print("Running EKF simulation...")
    lqg_ekf = SensorSelectionSimulator(n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, H=H, m_scale=m_scale, filter_type='ekf', lqr_type='orig', max_sensors=max_sensors, update_interval=update_interval, info_gain_method=info_gain_method, metric=metric)
    err_list_ekf, var_list_ekf, cost_list_ekf, time_list_ekf = lqg_ekf.run_sim()
    simulators['ekf'] = lqg_ekf
    
    if verbose:
        print("Running UKF simulation...")
    lqg_ukf = SensorSelectionSimulator(n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, H=H, m_scale=m_scale, filter_type='ukf', lqr_type='orig', max_sensors=max_sensors, update_interval=update_interval, info_gain_method=info_gain_method, metric=metric)
    err_list_ukf, var_list_ukf, cost_list_ukf, time_list_ukf = lqg_ukf.run_sim()
    simulators['ukf'] = lqg_ukf
    
    if verbose:
        print("Running QKF with augmented numeric LQR...")
    lqg_qkf_aug_num = SensorSelectionSimulator(n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, H=H, m_scale=m_scale, filter_type='qkf', lqr_type='aug_numeric', max_sensors=max_sensors, update_interval=update_interval, info_gain_method=info_gain_method, metric=metric)
    err_list_aug_num, var_list_aug_num, cost_list_aug_num, time_list_aug_num = lqg_qkf_aug_num.run_sim()
    simulators['qkf_aug_num'] = lqg_qkf_aug_num
    
    if verbose:
        print("Running QKF with augmented analytic LQR...")
    lqg_qkf_aug_analytic = SensorSelectionSimulator(n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, H=H, m_scale=m_scale, filter_type='qkf', lqr_type='aug_analytic', max_sensors=max_sensors, update_interval=update_interval, info_gain_method=info_gain_method, metric=metric)
    err_list_aug_analytic, var_list_aug_analytic, cost_list_aug_analytic, time_list_aug_analytic = lqg_qkf_aug_analytic.run_sim()
    simulators['qkf_aug_analytic'] = lqg_qkf_aug_analytic
    
    if verbose:
        print("Running PF simulation...")
    lqg_pf = SensorSelectionSimulator(n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, H=H, m_scale=m_scale, filter_type='pf', lqr_type='orig', max_sensors=max_sensors, update_interval=update_interval, info_gain_method=info_gain_method, metric=metric, n_particles=1000)
    err_list_pf, var_list_pf, cost_list_pf, time_list_pf = lqg_pf.run_sim()
    simulators['pf'] = lqg_pf
    
    # Save results for each simulator
    if verbose:
        print("\nSaving simulation results...")
        for name, simulator in simulators.items():
            print(f"Saving {name} results...")
            # simulator.save_results(trial_idx=trial_idx)
    
    # Return performance histories for consistency with pickle loading
    ekf_perf_history = simulators['ekf'].performance_history
    ukf_perf_history = simulators['ukf'].performance_history  
    qkf_num_perf_history = simulators['qkf_aug_num'].performance_history
    qkf_analytic_perf_history = simulators['qkf_aug_analytic'].performance_history
    pf_perf_history = simulators['pf'].performance_history
    
    # Create comparison plots only if requested
    if plot:
        all_results = [(err_list_ekf, var_list_ekf, cost_list_ekf, time_list_ekf), (err_list_ukf, var_list_ukf, cost_list_ukf, time_list_ukf), (err_list_aug_num, var_list_aug_num, cost_list_aug_num, time_list_aug_num), (err_list_aug_analytic, var_list_aug_analytic, cost_list_aug_analytic, time_list_aug_analytic), (err_list_pf, var_list_pf, cost_list_pf, time_list_pf)]
        plot_comparison(all_results, save_plots=True, update_interval=update_interval)
    
    return ekf_perf_history, ukf_perf_history, qkf_num_perf_history, qkf_analytic_perf_history, pf_perf_history

system_names = ['ekf', 'ukf', 'qkf_aug_num', 'qkf_aug_analytic', 'pf']
sensor_update_interval = 10

def get_cost_to_go(cost_list):
    cost_to_go = []
    for j in range(len(cost_list)):
        cost_to_go.append(np.sum(cost_list[j:]))
    return cost_to_go

plot_labels = {
    'ekf': 'LQG+EKF',
    'ukf': 'LQG+UKF',
    'qkf_aug_num': 'iLQG+QKF (Numeric)',
    'qkf_aug_analytic': 'LQG+QKF (Analytic)',
    'pf': 'LQG+PF'
}

axis_labels = {
    'ekf': 'LQG+EKF',
    'ukf': 'LQG+UKF',
    'qkf_aug_num': 'iLQG+QKF\n(Numeric)',
    'qkf_aug_analytic': 'LQG+QKF\n(Analytic)',
    'pf': 'LQG+PF'
}

def plot_comparison(all_results, save_plots=True, plot_dir=perf_dir, update_interval=10, m_scale=1e0, 
                   system_names=None, plot_labels=None, axis_labels=None):
    """
    Create publication-ready comparison plots across different filter types.
    
    Args:
        all_results: List of tuples (error, trace, cost_to_go, time) for each method
        system_names: Optional list of system names (defaults to ['ekf', 'ukf', 'qkf_aug_num', 'qkf_aug_analytic'])
        plot_labels: Optional dict mapping system names to plot labels
        axis_labels: Optional dict mapping system names to axis labels
    """
    # Use defaults if not provided (match length of all_results for backward compatibility)
    n_results = len(all_results)
    if system_names is None:
        system_names = (['ekf', 'ukf', 'qkf_aug_num', 'qkf_aug_analytic', 'pf'])[:n_results]
    if plot_labels is None:
        plot_labels = {
            'ekf': 'LQG+EKF',
            'ukf': 'LQG+UKF',
            'qkf_aug_num': 'iLQG+QKF (Numeric)',
            'qkf_aug_analytic': 'LQG+QKF (Analytic)',
            'pf': 'LQG+PF'
        }
    if axis_labels is None:
        axis_labels = {
            'ekf': 'LQG+EKF',
            'ukf': 'LQG+UKF',
            'qkf_aug_num': 'iLQG+QKF\n(Numeric)',
            'qkf_aug_analytic': 'LQG+QKF\n(Analytic)',
            'pf': 'LQG+PF'
        }
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    # Set publication-ready style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.linewidth': 2.5,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.spines.bottom': True,
        'axes.spines.left': True,
        'axes.edgecolor': 'black',
        'grid.alpha': 0.3,
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': True,
        'legend.fontsize': 10,
        'figure.dpi': 300
    })
    
    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Sensor Selection Performance Comparison (Nonlinearity Scale = ' + f'{m_scale}' + ')', 
                 fontsize=24, fontweight='bold', y=0.98)
    
    # Professional color palette and styles (expandable for more methods)
    base_colors = ['#2E86C1', '#28B463', '#F39C12', '#E74C3C', '#9B59B6', '#1ABC9C']  # Blue, Green, Orange, Red, Purple, Teal
    colors = [base_colors[i % len(base_colors)] for i in range(len(all_results))]
    linestyles = ['-', '--', '-.', ':', '-', '--']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    alphas = [0.9, 0.8, 0.8, 0.8, 0.8, 0.8]
    
    # 1. Cost-to-go comparison (top left)
    for i, result in enumerate(all_results):
        cost_list = result[2]
        time_steps = np.arange(1, len(result[2]) + 1)
        # Calculate cost-to-go for this simulator
        cost_to_go = []
        for j in range(len(cost_list)):
            cost_to_go.append(np.sum(cost_list[j:]))
        name = system_names[i]
        axes[0, 0].plot(time_steps, cost_to_go, 
                    color=colors[i % len(colors)], 
                    linestyle=linestyles[i % len(linestyles)],
                    label=plot_labels[name], linewidth=2.5, alpha=alphas[i])
    
    axes[0, 0].set_title('Cost-to-Go Comparison', fontsize=22, fontweight='bold', pad=10)
    axes[0, 0].set_xlabel('Time Step', fontsize=20, fontweight='bold')
    axes[0, 0].set_ylabel('Cost-to-Go (log scale)', fontsize=20, fontweight='bold')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend(loc='lower left', framealpha=0.9, fontsize=14)
    axes[0, 0].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    axes[0, 0].tick_params(axis='both', which='major', labelsize=16)
    # Add subplot index
    axes[0, 0].text(-0.06, 1.05, 'A', transform=axes[0, 0].transAxes, fontsize=28, fontweight='bold', 
                    va='bottom', ha='left')
    
    # 2. Staged cost-to-go comparison (top right)
    for i, result in enumerate(all_results):
        cost_list = result[2]
        time_steps = np.arange(1, len(result[2]) + 1)
        
        # Calculate staged cost-to-go for each goal state phase
        staged_costs = []
        
        # Group costs by sensor selection phases
        phase_costs = []
        staged_costs = []
        for j, cost in enumerate(cost_list):
            phase_costs.append(cost)
            
            # Check if we're at a sensor selection update point
            if (j + 1) % update_interval == 0:
                # Calculate cost accumulated during this phase only
                staged_costs.append(phase_costs)
                phase_costs = []  # Reset for next phase
        
        # Handle remaining costs if simulation doesn't end at sensor update
        if phase_costs:
            phase_cost = np.sum(phase_costs)
            staged_costs.append(phase_cost)
        
        # Plot staged costs - use phase indices (0, 1, 2, ...)
        staged_cost_to_go = []
        # print(len(staged_costs))
        for staged_costs in staged_costs:
            staged_cost_to_go.append(get_cost_to_go(staged_costs))
        staged_cost_to_go = np.concatenate(staged_cost_to_go)
        name = system_names[i]
        phase_indices = np.arange(len(staged_cost_to_go))
        
        axes[0, 1].plot(phase_indices, staged_cost_to_go, 
                    color=colors[i % len(colors)], 
                    linestyle=linestyles[i % len(linestyles)],
                    label=plot_labels[name], linewidth=2.5, marker=markers[i], 
                    markersize=4, alpha=alphas[i], markevery=5)
    
    axes[0, 1].set_title('Staged Cost-to-Go (Goal State Phases)', fontsize=22, fontweight='bold', pad=10)
    axes[0, 1].set_xlabel('Phase Index', fontsize=20, fontweight='bold')
    axes[0, 1].set_ylabel('Phase Cost-to-Go (log scale)', fontsize=20, fontweight='bold')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend(loc='lower left', framealpha=0.9, fontsize=14)
    axes[0, 1].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    axes[0, 1].tick_params(axis='both', which='major', labelsize=16)
    # Add subplot index
    axes[0, 1].text(-0.06, 1.05, 'B', transform=axes[0, 1].transAxes, fontsize=28, fontweight='bold', 
                    va='bottom', ha='left')
    
    # 3. Average estimation error comparison (bottom left)
    avg_errors = []
    for i, result in enumerate(all_results):
        avg_error = np.mean(result[0])  # Average estimation error
        avg_errors.append(avg_error)
    
    if len(avg_errors) == 0:
        print("Warning: No results to plot for estimation error comparison")
        return fig
    
    plot_system_names = [axis_labels[name] for name in system_names]
    bars_error = axes[1, 0].bar(plot_system_names, avg_errors, 
                               color=colors[:len(system_names)], 
                               edgecolor='black', linewidth=1.2, alpha=0.8)
    axes[1, 0].set_title('Average Estimation Error', fontsize=22, fontweight='bold', pad=10)
    # axes[1, 0].set_xlabel('Filter Type', fontsize=20, fontweight='bold')
    axes[1, 0].set_ylabel('Average $||x_{true} - x_{est}||$', fontsize=20, fontweight='bold')
    
    # Set y-axis limit to prevent overlap with top border
    max_error = max(avg_errors) if len(avg_errors) > 0 else 1.0
    axes[1, 0].set_ylim(0, max_error * 1.15)
    
    axes[1, 0].grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
    axes[1, 0].tick_params(axis='both', which='major', labelsize=16)
    # Rotate x-axis labels for better readability
    axes[1, 0].set_xticklabels(plot_system_names, rotation=45)
    # Adjust label alignment after setting labels
    for label in axes[1, 0].get_xticklabels():
        label.set_ha('right')
    # Add subplot index
    axes[1, 0].text(-0.06, 1.05, 'C', transform=axes[1, 0].transAxes, fontsize=28, fontweight='bold', 
                    va='bottom', ha='left')
    
    # Add value labels on bars
    for bar, avg_error in zip(bars_error, avg_errors):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + max_error*0.03,
                f'{avg_error:.4f}', ha='center', va='bottom', 
                fontsize=14, fontweight='bold')
    
    # 4. Average time consumption comparison (bottom right)
    avg_times = []
    for i, result in enumerate(all_results):
        time_data = result[3]  # Time data is now consistently at index 3
        avg_time = np.mean(time_data)
        avg_times.append(avg_time)
    axis_label_names = [axis_labels[name] for name in system_names]
    bars = axes[1, 1].bar(axis_label_names, avg_times, 
                         color=colors[:len(system_names)], 
                         edgecolor='black', linewidth=1.2, alpha=0.8)
    axes[1, 1].set_title('Average Time Consumption', fontsize=22, fontweight='bold', pad=10)
    # axes[1, 1].set_xlabel('Filter Type', fontsize=20, fontweight='bold')
    axes[1, 1].set_ylabel('Average Time per Step (seconds)', fontsize=20, fontweight='bold')
    
    # Set y-axis limit to prevent overlap with top border
    max_time = max(avg_times)
    axes[1, 1].set_ylim(0, max_time * 1.15)
    
    # Format y-axis ticks in scientific notation (e.g., 8e-03)
    def sci_formatter(x, pos):
        if abs(x) < 1e-10:  # Handle zero and very small numbers
            return '0'
        # Convert to scientific notation
        exp = int(np.floor(np.log10(abs(x))))
        coeff = x / (10 ** exp)
        # Round coefficient to integer if close
        if abs(coeff - round(coeff)) < 0.01:
            coeff_str = f'{int(round(coeff))}'
        else:
            coeff_str = f'{coeff:.1f}'
        # Format as "coefficient e exponent" (e.g., 8e-03)
        if exp < 0:
            return f'{coeff_str}e{exp:02d}'
        elif exp > 0:
            return f'{coeff_str}e+{exp:02d}'
        else:
            return coeff_str
    
    axes[1, 1].yaxis.set_major_formatter(FuncFormatter(sci_formatter))
    
    axes[1, 1].grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
    axes[1, 1].tick_params(axis='both', which='major', labelsize=16)
    # Rotate x-axis labels for better readability
    axes[1, 1].set_xticklabels(axis_label_names, rotation=45)
    # Adjust label alignment after setting labels
    for label in axes[1, 1].get_xticklabels():
        label.set_ha('right')
    # Add subplot index
    axes[1, 1].text(-0.06, 1.05, 'D', transform=axes[1, 1].transAxes, fontsize=28, fontweight='bold', 
                    va='bottom', ha='left')
    
    # Add value labels on bars (in scientific notation)
    for bar, avg_time in zip(bars, avg_times):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + max_time*0.03,
                f'{avg_time:.2e}', ha='center', va='bottom', 
                fontsize=14, fontweight='bold')
    
    # Enhanced prominent black borders for all subplots
    for i in range(2):
        for j in range(2):
            for spine in axes[i, j].spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(2.5)
                spine.set_visible(True)
    
    # Enhance overall figure appearance
    plt.tight_layout(pad=1.5)
    fig.patch.set_facecolor('white')
    
    if save_plots:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{plot_dir}/sensor_selection_comparison_mscale={m_scale:.0e}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Plots saved to: {filename}")
    
    # plt.show()
    return fig

import random
def run_comprehensive_test(n_trials=1, H=1000, update_interval=10, num_sensors=6, max_sensors=3, plot=True, m_scale=1e0):
    if not os.path.exists(pkl_dir + f"sensor_selection_comprehensive_results_mscale={m_scale:.0e}_trials={n_trials}.pkl"):
        all_ekf_results = []
        all_ukf_results = []
        all_qkf_num_results = []
        all_qkf_analytic_results = []
        all_pf_results = []
        
        for idx in range(n_trials):
            if not os.path.exists(pkl_dir + f"sensor_selection_results_ekf_orig_mscale={m_scale:.0e}-trial_{idx}.pkl"):
                # print(pkl_dir + f"sensor_selection_results_ekf_orig_mscale={m_scale}-trial_{idx}.pkl")

                seed_i = random.randint(0, 1000000)
                print(f"Running trial [{idx+1}/{n_trials}]")
                ekf_result, ukf_result, qkf_num_result, qkf_analytic_result, pf_result = run_sensor_scheduling_sim(H=H, num_sensors=num_sensors, max_sensors=max_sensors, rand_seed=seed_i, plot=False, update_interval=update_interval, m_scale=m_scale, trial_idx=idx)
            else:
                ekf_result = pkl.load(open(pkl_dir + f"sensor_selection_results_ekf_orig_mscale={m_scale:.0e}-trial_{idx}.pkl", 'rb'))['performance_history']
                ukf_result = pkl.load(open(pkl_dir + f"sensor_selection_results_ukf_orig_mscale={m_scale:.0e}-trial_{idx}.pkl", 'rb'))['performance_history']
                qkf_num_result = pkl.load(open(pkl_dir + f"sensor_selection_results_qkf_aug_numeric_mscale={m_scale:.0e}-trial_{idx}.pkl", 'rb'))['performance_history']
                qkf_analytic_result = pkl.load(open(pkl_dir + f"sensor_selection_results_qkf_aug_analytic_mscale={m_scale:.0e}-trial_{idx}.pkl", 'rb'))['performance_history']
                pf_path = pkl_dir + f"sensor_selection_results_pf_orig_mscale={m_scale:.0e}-trial_{idx}.pkl"
                pf_result = pkl.load(open(pf_path, 'rb'))['performance_history'] if os.path.exists(pf_path) else None

            # append results
            all_ekf_results.append(ekf_result)
            all_ukf_results.append(ukf_result)
            all_qkf_num_results.append(qkf_num_result)
            all_qkf_analytic_results.append(qkf_analytic_result)
            if pf_result is not None:
                all_pf_results.append(pf_result)
        
        # Extract specific metrics for averaging
        # Get cost data for averaging
        ekf_costs = [result['cost'] for result in all_ekf_results]
        ukf_costs = [result['cost'] for result in all_ukf_results]
        qkf_num_costs = [result['cost'] for result in all_qkf_num_results]
        qkf_analytic_costs = [result['cost'] for result in all_qkf_analytic_results]
        
        # Get estimation error data for averaging
        ekf_errors = [result['estimation_error'] for result in all_ekf_results]
        ukf_errors = [result['estimation_error'] for result in all_ukf_results]
        qkf_num_errors = [result['estimation_error'] for result in all_qkf_num_results]
        qkf_analytic_errors = [result['estimation_error'] for result in all_qkf_analytic_results]
        
        # Get covariance traces for averaging
        ekf_traces = [result['covariance_traces'] for result in all_ekf_results]
        ukf_traces = [result['covariance_traces'] for result in all_ukf_results]
        qkf_num_traces = [result['covariance_traces'] for result in all_qkf_num_results]
        qkf_analytic_traces = [result['covariance_traces'] for result in all_qkf_analytic_results]
        
        # get time consumption for averaging
        ekf_times = [result['time_consumption'] for result in all_ekf_results]
        ukf_times = [result['time_consumption'] for result in all_ukf_results]
        qkf_num_times = [result['time_consumption'] for result in all_qkf_num_results]
        qkf_analytic_times = [result['time_consumption'] for result in all_qkf_analytic_results]
        
        # Average the metrics
        avg_ekf_cost = np.mean(ekf_costs, axis=0)
        avg_ukf_cost = np.mean(ukf_costs, axis=0)
        avg_qkf_num_cost = np.mean(qkf_num_costs, axis=0)
        avg_qkf_analytic_cost = np.mean(qkf_analytic_costs, axis=0)
        
        avg_ekf_error = np.mean(ekf_errors, axis=0)
        avg_ukf_error = np.mean(ukf_errors, axis=0)
        avg_qkf_num_error = np.mean(qkf_num_errors, axis=0)
        avg_qkf_analytic_error = np.mean(qkf_analytic_errors, axis=0)
        
        avg_ekf_trace = np.mean(ekf_traces, axis=0)
        avg_ukf_trace = np.mean(ukf_traces, axis=0)
        avg_qkf_num_trace = np.mean(qkf_num_traces, axis=0)
        avg_qkf_analytic_trace = np.mean(qkf_analytic_traces, axis=0)
        
        avg_ekf_time = np.mean(ekf_times, axis=0)
        avg_ukf_time = np.mean(ukf_times, axis=0)
        avg_qkf_num_time = np.mean(qkf_num_times, axis=0)
        avg_qkf_analytic_time = np.mean(qkf_analytic_times, axis=0)
        
        # Calculate cost-to-go for each averaged cost
        def get_cost_to_go_from_cost(cost_list):
            cost_to_go = []
            for j in range(len(cost_list)):
                cost_to_go.append(np.sum(cost_list[j:]))
            return cost_to_go
        
        avg_ekf_cost_to_go = get_cost_to_go_from_cost(avg_ekf_cost)
        avg_ukf_cost_to_go = get_cost_to_go_from_cost(avg_ukf_cost)
        avg_qkf_num_cost_to_go = get_cost_to_go_from_cost(avg_qkf_num_cost)
        avg_qkf_analytic_cost_to_go = get_cost_to_go_from_cost(avg_qkf_analytic_cost)
        
        # Create tuples in the expected format: (error, trace, cost_to_go, time)
        avg_all_ekf_results = (avg_ekf_error, avg_ekf_trace, avg_ekf_cost_to_go, avg_ekf_time)
        avg_all_ukf_results = (avg_ukf_error, avg_ukf_trace, avg_ukf_cost_to_go, avg_ukf_time)
        avg_all_qkf_num_results = (avg_qkf_num_error, avg_qkf_num_trace, avg_qkf_num_cost_to_go, avg_qkf_num_time)
        avg_all_qkf_analytic_results = (avg_qkf_analytic_error, avg_qkf_analytic_trace, avg_qkf_analytic_cost_to_go, avg_qkf_analytic_time)

        avg_all_results = [avg_all_ekf_results, avg_all_ukf_results, avg_all_qkf_num_results, avg_all_qkf_analytic_results]
        if len(all_pf_results) == n_trials:
            pf_costs = [r['cost'] for r in all_pf_results]
            pf_errors = [r['estimation_error'] for r in all_pf_results]
            pf_traces = [r['covariance_traces'] for r in all_pf_results]
            pf_times = [r['time_consumption'] for r in all_pf_results]
            avg_pf_cost_to_go = get_cost_to_go_from_cost(np.mean(pf_costs, axis=0))
            avg_all_pf_results = (np.mean(pf_errors, axis=0), np.mean(pf_traces, axis=0), avg_pf_cost_to_go, np.mean(pf_times, axis=0))
            avg_all_results.append(avg_all_pf_results)
        pkl.dump(avg_all_results, open(pkl_dir + f"sensor_selection_comprehensive_results_mscale={m_scale:.0e}_trials={n_trials}.pkl", 'wb'))
    else:
        print(f"Loading existing comprehensive results for m_scale={m_scale:.0e}, trials={n_trials}...")
        avg_all_results = pkl.load(open(pkl_dir + f"sensor_selection_comprehensive_results_mscale={m_scale:.0e}_trials={n_trials}.pkl", 'rb'))

    if plot:
        plot_comparison(avg_all_results, save_plots=True, update_interval=update_interval, m_scale=m_scale)
    
    
    return avg_all_results

def run_method_comparison(n_trials=1, H=1000, update_interval=10, noise_scale=1e-1, m_scale=1e0, Q_scale=1.0, R_scale=1.0, num_sensors=6, max_sensors=3, plot=True, rand_seed=None, sensor_variance=0.75, reset=False):
    """
    Compare different information gain methods with identical random parameters.
    
    Methods compared:
    - heuristic
    - baseline
    - van_trees-T (Trace optimality)
    - van_trees-D (Determinant optimality)
    - van_trees-E (Eigenvalue optimality)
    - van_trees-A (Average MSE optimality)
    
    Args:
        n_trials: Number of trials to run
        H: Simulation horizon
        update_interval: Sensor selection update interval
        noise_scale: Noise scale for parameter generation
        m_scale: Nonlinearity scale
        Q_scale: Q matrix scale
        R_scale: R matrix scale
        num_sensors: Total number of sensors
        max_sensors: Maximum sensors to select
        plot: Whether to plot results
        rand_seed: Random seed (None for random each trial)
        sensor_variance: Controls how different sensor qualities are (0.0 = identical, higher = more variance)
                        Noise levels will range from (1.0 - sensor_variance) to (1.0 + sensor_variance)
                        Default 0.75 gives range [0.25x, 1.75x] base noise
        reset: If True, rerun simulation even if pickle file exists. If False, load from pickle if available.
    """
    methods = [
        # ('heuristic', 'T'),  # metric not used for heuristic - DISABLED
        ('baseline', 'T'),   # metric not used for baseline
        ('van_trees', 'T'),
        ('van_trees', 'D'),
        ('van_trees', 'E'),
        ('van_trees', 'A'),
    ]
    
    # Storage for all results
    all_results = {f'{method}_{metric}': [] for method, metric in methods}
    
    # Create comparison directory
    comp_dir = 'sensor_sim-comparison/'
    comp_pkl_dir = comp_dir + 'pkl/'
    comp_perf_dir = comp_dir + 'perf/'
    os.makedirs(comp_pkl_dir, exist_ok=True)
    os.makedirs(comp_perf_dir, exist_ok=True)
    
    # Check if results already exist
    results_file = comp_pkl_dir + f'method_comparison_results_mscale={m_scale:.0e}_trials={n_trials}.pkl'
    skip_simulation = os.path.exists(results_file) and not reset
    
    if skip_simulation:
        print("=" * 80)
        print("LOADING EXISTING RESULTS")
        print("=" * 80)
        print(f"Found existing results file: {results_file}")
        print("Skipping simulation and loading from pickle...")
        print("(Set reset=True to force rerun)")
        print("=" * 80)
        
        # Load existing results
        comparison_results = pkl.load(open(results_file, 'rb'))
        print(f"✓ Loaded results for {len(comparison_results)} methods")
        
        # Skip simulation and sensor analysis, go directly to summary and plotting
        all_results = {}  # Empty since we're skipping analysis
        sensor_selection_analysis = {}  # Initialize to empty dict
    else:
        print("=" * 80)
        print("METHOD COMPARISON: Information Gain Approaches")
        print("=" * 80)
        print(f"Methods: {', '.join([f'{m}_{met}' for m, met in methods])}")
        print(f"Trials: {n_trials} | Horizon: {H} | Update Interval: {update_interval} | Sensors: {num_sensors}/{max_sensors} | m_scale: {m_scale} | Sensor Variance: {sensor_variance:.2f}")
        print("=" * 80)
        
        for trial_idx in range(n_trials):
            # Generate parameters ONCE per trial (ensures identical parameters for all methods)
            if rand_seed is not None:
                trial_seed = rand_seed + trial_idx
            else:
                trial_seed = random.randint(0, 1000000)
            
            print(f"\n{'─' * 80}")
            print(f"TRIAL {trial_idx + 1}/{n_trials} | Seed: {trial_seed}")
            print(f"{'─' * 80}")
            
            # Set random seed and generate parameters
            np.random.seed(trial_seed)
            random.seed(trial_seed)
            
            n1 = 2
            n2 = 2
            n = n1 + n2
            p = 3
            m = num_sensors
            
            # Generate parameters ONCE
            print("Generating system parameters (shared across all methods)...")
            A_E, A_S, B_S, C, M, W, V = generate_stable_system_parameters(
                n1, n2, p, m, noise_scale, m_scale
            )
            
            # RECOMMENDATION 2: Create sensors with intentionally different qualities
            # Modify V (measurement noise covariance) to have varying noise levels
            # This makes sensor selection more meaningful - some sensors are better than others
            print(f"  Creating sensors with different quality levels (variance={sensor_variance:.2f})...")
            V_modified = V.copy()
            # Create a range of noise levels: some sensors are precise (low noise), some are noisy (high noise)
            # sensor_variance controls the spread: 0.0 = identical sensors, higher = more variance
            noise_min = max(0.1, 1.0 - sensor_variance)  # Ensure minimum is positive
            noise_max = 1.0 + sensor_variance
            noise_levels = np.linspace(noise_min, noise_max, m)  # Evenly spaced across range
            np.random.shuffle(noise_levels)  # Randomize which sensors get which quality
            for i in range(m):
                V_modified[i, i] = V[i, i] * noise_levels[i]
            # Update off-diagonal elements proportionally (maintain correlation structure)
            for i in range(m):
                for j in range(i+1, m):
                    if V[i, j] != 0:
                        V_modified[i, j] = V[i, j] * np.sqrt(noise_levels[i] * noise_levels[j])
                        V_modified[j, i] = V_modified[i, j]
            V = V_modified
            noise_levels_str = ', '.join([f'{x:.3f}' for x in noise_levels[:5]])
            print(f"  ✓ Sensor noise levels: [{noise_levels_str}...] (range: {noise_levels.min():.3f}x - {noise_levels.max():.3f}x, variance={sensor_variance:.2f})")
            
            if not validate_stable_parameters(A_E, A_S):
                print("  ⚠ Warning: Generated unstable parameters, but continuing...")
            
            Q = generate_random_symmetric_matrix(n+n**2, scale=Q_scale)
            R = generate_random_symmetric_matrix(p, scale=R_scale)
            print("  ✓ Parameters generated successfully")
            
            # Run each method with the SAME parameters
            method_results_summary = {}
            for method_idx, (method, metric) in enumerate(methods, 1):
                method_name = f'{method}_{metric}'
                method_display = f"{method.upper()}" + (f"-{metric}" if method == 'van_trees' else "")
                print(f"\n  [{method_idx}/{len(methods)}] Running {method_display:20s}...", end=' ', flush=True)
                
                # Reset random seed for reproducibility within each method run
                # (but use the same initial parameters)
                np.random.seed(trial_seed)
                random.seed(trial_seed)
                
                try:
                    # Run simulation with this method using PRE-GENERATED parameters
                    # This ensures all methods use identical system parameters
                    ekf_result, ukf_result, qkf_num_result, qkf_analytic_result, pf_result = run_sensor_scheduling_sim(
                        H=H,
                        update_interval=update_interval,
                        noise_scale=noise_scale,
                        m_scale=m_scale,
                        Q_scale=Q_scale,
                        R_scale=R_scale,
                        num_sensors=num_sensors,
                        max_sensors=max_sensors,
                        rand_seed=None,  # Don't generate new params
                        plot=False,
                        trial_idx=None,
                        save_dir=comp_pkl_dir,
                        info_gain_method=method,
                        metric=metric,
                        # Pass pre-generated parameters to ensure identical systems
                        A_E=A_E, A_S=A_S, B_S=B_S, C=C, M=M, W=W, V=V, Q=Q, R=R
                    )

                    # Store results (ekf_result, etc. are performance_history dictionaries)
                    all_results[method_name].append({
                        'ekf': ekf_result,
                        'ukf': ukf_result,
                        'qkf_num': qkf_num_result,
                        'qkf_analytic': qkf_analytic_result,
                        'pf': pf_result,
                        'seed': trial_seed
                    })
                    
                    # Print summary for this method
                    avg_error = np.mean(qkf_num_result['estimation_error'])
                    avg_cost = np.mean(qkf_num_result['cost'])
                    method_results_summary[method_name] = {'error': avg_error, 'cost': avg_cost}
                    print(f"✓ | Error: {avg_error:.4f} | Cost: {avg_cost:.4f}")
                    
                except Exception as e:
                    print(f"✗ ERROR: {e}")
                    continue
            
            # Print trial summary with sensor selection comparison
            if method_results_summary:
                print(f"\n  Trial {trial_idx + 1} Summary:")
                for method_name, summary in method_results_summary.items():
                    print(f"    {method_name:20s}: Error={summary['error']:.4f}, Cost={summary['cost']:.4f}")
                
                # Compare sensor selections across methods for this trial
                print(f"\n  Sensor Selection Comparison (Trial {trial_idx + 1}, first 5 selections):")
                selection_comparison = {}
                for method_name in method_results_summary.keys():
                    # Get sensor selections for this method and trial
                    if len(all_results[method_name]) > 0:
                        trial_result = all_results[method_name][-1]  # Last result is current trial
                        selections = trial_result['qkf_num'].get('sensor_selections', [])
                        if len(selections) > 0:
                            selection_comparison[method_name] = selections[:5]  # First 5 selections
                            method_display = method_name.replace('_', '-').upper()
                            print(f"    {method_display:<20s}: {selections[:5]}")
                
                # Check if selections are actually different
                if len(selection_comparison) > 0:
                    all_selections_list = list(selection_comparison.values())
                    # Convert each selection (list of lists) to tuple of tuples for hashing
                    selection_tuples = [tuple(tuple(inner_sel) for inner_sel in sel) for sel in all_selections_list]
                    if len(set(selection_tuples)) == 1:
                        print(f"    ⚠ WARNING: All methods selected identical sensors!")
                    else:
                        # Count unique selections
                        unique_selections = len(set(selection_tuples))
                        print(f"    → {unique_selections}/{len(all_selections_list)} methods have unique selections")
        
        # Aggregate results and create comparison plots
        print(f"\n{'=' * 80}")
        print("AGGREGATING RESULTS AND CREATING COMPARISON PLOTS")
        print(f"{'=' * 80}")
        
        # Prepare data for plotting (similar to plot_comparison format)
        comparison_results = {}
        for method_name, trial_results in all_results.items():
            if len(trial_results) == 0:
                continue
            
            # Average across trials
            avg_ekf_error = np.mean([r['ekf']['estimation_error'] for r in trial_results], axis=0)
            avg_ukf_error = np.mean([r['ukf']['estimation_error'] for r in trial_results], axis=0)
            avg_qkf_num_error = np.mean([r['qkf_num']['estimation_error'] for r in trial_results], axis=0)
            avg_qkf_analytic_error = np.mean([r['qkf_analytic']['estimation_error'] for r in trial_results], axis=0)
            
            avg_ekf_trace = np.mean([r['ekf']['covariance_traces'] for r in trial_results], axis=0)
            avg_ukf_trace = np.mean([r['ukf']['covariance_traces'] for r in trial_results], axis=0)
            avg_qkf_num_trace = np.mean([r['qkf_num']['covariance_traces'] for r in trial_results], axis=0)
            avg_qkf_analytic_trace = np.mean([r['qkf_analytic']['covariance_traces'] for r in trial_results], axis=0)
            
            avg_ekf_cost = np.mean([r['ekf']['cost'] for r in trial_results], axis=0)
            avg_ukf_cost = np.mean([r['ukf']['cost'] for r in trial_results], axis=0)
            avg_qkf_num_cost = np.mean([r['qkf_num']['cost'] for r in trial_results], axis=0)
            avg_qkf_analytic_cost = np.mean([r['qkf_analytic']['cost'] for r in trial_results], axis=0)
            
            avg_ekf_time = np.mean([r['ekf']['time_consumption'] for r in trial_results], axis=0)
            avg_ukf_time = np.mean([r['ukf']['time_consumption'] for r in trial_results], axis=0)
            avg_qkf_num_time = np.mean([r['qkf_num']['time_consumption'] for r in trial_results], axis=0)
            avg_qkf_analytic_time = np.mean([r['qkf_analytic']['time_consumption'] for r in trial_results], axis=0)
            avg_pf_error = np.mean([r['pf']['estimation_error'] for r in trial_results], axis=0)
            avg_pf_trace = np.mean([r['pf']['covariance_traces'] for r in trial_results], axis=0)
            avg_pf_cost = np.mean([r['pf']['cost'] for r in trial_results], axis=0)
            avg_pf_time = np.mean([r['pf']['time_consumption'] for r in trial_results], axis=0)
            
            # Calculate cost-to-go
            def get_cost_to_go(cost_list):
                return [np.sum(cost_list[j:]) for j in range(len(cost_list))]
            
            comparison_results[method_name] = {
                'ekf': (avg_ekf_error, avg_ekf_trace, get_cost_to_go(avg_ekf_cost), avg_ekf_time),
                'ukf': (avg_ukf_error, avg_ukf_trace, get_cost_to_go(avg_ukf_cost), avg_ukf_time),
                'qkf_num': (avg_qkf_num_error, avg_qkf_num_trace, get_cost_to_go(avg_qkf_num_cost), avg_qkf_num_time),
                'qkf_analytic': (avg_qkf_analytic_error, avg_qkf_analytic_trace, get_cost_to_go(avg_qkf_analytic_cost), avg_qkf_analytic_time),
                'pf': (avg_pf_error, avg_pf_trace, get_cost_to_go(avg_pf_cost), avg_pf_time),
            }
        
        # Save comparison results
        pkl.dump(comparison_results, open(results_file, 'wb'))
        print(f"✓ Results saved to: {results_file}")
    
    # RECOMMENDATION 3: Analyze sensor selection differences in detail
    # Only run analysis if we actually ran simulations (not loading from pickle)
    if not skip_simulation:
        print("\n" + "=" * 80)
        print("SENSOR SELECTION ANALYSIS")
        print("=" * 80)
        sensor_selection_analysis = {}
        
        # Analyze selections across all trials
        for method_name, trial_results in all_results.items():
            if len(trial_results) == 0:
                continue
            
            # Aggregate selections across all trials
            all_selections = []
            for trial_result in trial_results:
                selections = trial_result['qkf_num'].get('sensor_selections', [])
                all_selections.extend(selections)
            
            if all_selections:
                # Count frequency of each sensor being selected
                sensor_counts = {}
                selection_sizes = []
                unique_selections = set()
                
                for sel in all_selections:
                    sel_tuple = tuple(sorted(sel))  # Normalize for comparison
                    unique_selections.add(sel_tuple)
                    selection_sizes.append(len(sel))
                    for sensor_idx in sel:
                        sensor_counts[sensor_idx] = sensor_counts.get(sensor_idx, 0) + 1
                
                # Find most common selection pattern
                selection_patterns = Counter([tuple(sorted(sel)) for sel in all_selections])
                most_common_pattern = selection_patterns.most_common(1)[0] if selection_patterns else None
                
                sensor_selection_analysis[method_name] = {
                    'sensor_counts': sensor_counts,
                    'most_selected_sensors': sorted(sensor_counts.items(), key=lambda x: x[1], reverse=True)[:5],
                    'avg_selection_size': np.mean(selection_sizes) if selection_sizes else 0,
                    'unique_patterns': len(unique_selections),
                    'most_common_pattern': most_common_pattern,
                    'sample_selections': all_selections[:min(10, len(all_selections))]
                }
    
    # Print detailed analysis
    if sensor_selection_analysis:
        print("\nSensor Selection Patterns:")
        print(f"{'Method':<25s} {'Top Sensors':<30s} {'Avg Size':<10s} {'Unique Patterns':<15s}")
        print("─" * 80)
        
        for method_name, analysis in sensor_selection_analysis.items():
            top_sensors_str = ", ".join([f"s{i}({count})" for i, count in analysis['most_selected_sensors'][:5]])
            method_display = method_name.replace('_', '-').upper()
            print(f"{method_display:<25s} {top_sensors_str:<30s} {analysis['avg_selection_size']:<10.2f} {analysis['unique_patterns']:<15d}")
        
        print("\n" + "─" * 80)
        print("Most Common Selection Pattern per Method:")
        for method_name, analysis in sensor_selection_analysis.items():
            if analysis['most_common_pattern']:
                pattern, count = analysis['most_common_pattern']
                method_display = method_name.replace('_', '-').upper()
                print(f"  {method_display:<25s}: {list(pattern)} (selected {count} times)")
        
        # Compare selections between methods
        print("\n" + "─" * 80)
        print("Selection Overlap Analysis:")
        method_names_list = list(sensor_selection_analysis.keys())
        for i, method1 in enumerate(method_names_list):
            for method2 in method_names_list[i+1:]:
                counts1 = set(sensor_selection_analysis[method1]['most_selected_sensors'][:3])
                counts2 = set(sensor_selection_analysis[method2]['most_selected_sensors'][:3])
                overlap = len(counts1.intersection(counts2))
                method1_display = method1.replace('_', '-').upper()
                method2_display = method2.replace('_', '-').upper()
                print(f"  {method1_display} vs {method2_display}: {overlap}/3 top sensors overlap")
        
        print("─" * 80)
    
    # Print summary statistics
    print(f"\n{'─' * 80}")
    print("FINAL SUMMARY (QKF Augmented Numeric):")
    print(f"{'─' * 80}")
    print(f"{'Method':<25s} {'Avg Error':<15s} {'Avg Cost':<15s} {'Error Range':<15s}")
    print(f"{'─' * 80}")
    
    errors_list = []
    for method_name in comparison_results.keys():
        error, trace, cost_to_go, time_data = comparison_results[method_name]['qkf_num']
        avg_error = np.mean(error)
        avg_cost = np.mean([np.sum(cost_to_go[i:]) for i in range(len(cost_to_go))]) / len(cost_to_go) if len(cost_to_go) > 0 else 0
        error_range = f"{np.min(error):.4f}-{np.max(error):.4f}"
        method_display = method_name.replace('_', '-').upper()
        errors_list.append(avg_error)
        print(f"{method_display:<25s} {avg_error:<15.4f} {avg_cost:<15.4f} {error_range:<15s}")
    
    # Calculate and print difference statistics
    if len(errors_list) > 1:
        min_error = min(errors_list)
        max_error = max(errors_list)
        error_range = max_error - min_error
        error_range_pct = (error_range / min_error) * 100 if min_error > 0 else 0
        print(f"{'─' * 80}")
        print(f"Range: {error_range:.6f} ({error_range_pct:.2f}% of minimum)")
        print(f"Best: {min_error:.6f} | Worst: {max_error:.6f}")
        
        # Analysis of why differences might be small
        print(f"\n{'─' * 80}")
        print("ANALYSIS: Why errors might be similar:")
        print(f"{'─' * 80}")
        print(f"1. Sensor selection frequency: Every {update_interval} steps (only {H//update_interval} selections in {H} steps)")
        print(f"   → Impact is limited: sensors are fixed for {update_interval} steps between selections")
        print(f"2. QKF filter robustness: The QKF filter may be robust enough that suboptimal")
        print(f"   sensor selections don't significantly degrade performance")
        print(f"3. All methods might select similar sensors: Even with different information gain")
        print(f"   calculations, methods may converge to similar optimal sensor sets")
        print(f"4. System well-conditioning: If the measurement model is well-conditioned, most")
        print(f"   sensor combinations provide reasonable estimation quality")
        print(f"5. Sensor variance ({sensor_variance:.2f}): Current variance may not create enough")
        print(f"   difference between sensors to make selection matter significantly")
        print(f"\n  Recommendations to increase differences:")
        print(f"   - Increase sensor_variance (e.g., 1.5-2.0) to create larger quality gaps")
        print(f"   - Reduce update_interval (e.g., 1-5) for more frequent selection")
        print(f"   - Check sensor_selections output above to verify methods select different sensors")
        print(f"   - Consider using a more challenging system (higher noise, worse conditioning)")
        print(f"{'─' * 80}")
    print(f"{'─' * 80}")
    
    # Create comparison plots (focus on QKF augmented numeric for now)
    if plot:
        print("\nGenerating comparison plots...")
        plot_method_comparison(comparison_results, save_plots=True, plot_dir=comp_perf_dir, update_interval=update_interval, m_scale=m_scale)
        print(f"✓ Plots saved to: {comp_perf_dir}")
    
    print(f"\n{'=' * 80}")
    print("COMPARISON COMPLETE")
    print(f"{'=' * 80}\n")
    
    return comparison_results


def plot_method_comparison(comparison_results, save_plots=True, plot_dir='sensor_sim-comparison/perf/', update_interval=10, m_scale=1e0):
    """
    Plot comparison of different information gain methods.
    """
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    # Use QKF augmented numeric results
    filter_key = 'qkf_num'
    
    # Extract data for each method
    method_names = list(comparison_results.keys())
    all_results = [comparison_results[m][filter_key] for m in method_names]
    
    # Create labels - use shorter names for x-axis to avoid overlap
    plot_labels = {
        'heuristic_T': 'Heuristic',
        'baseline_T': 'Baseline',
        'van_trees_T': 'Van Trees (T)',
        'van_trees_D': 'Van Trees (D)',
        'van_trees_E': 'Van Trees (E)',
        'van_trees_A': 'Van Trees (A)',
    }
    
    # Create axis_labels - use shorter names for x-axis
    axis_labels_dict = {
        'heuristic_T': 'Heuristic',
        'baseline_T': 'Baseline',
        'van_trees_T': 'VT-T',
        'van_trees_D': 'VT-D',
        'van_trees_E': 'VT-E',
        'van_trees_A': 'VT-A',
    }
    
    # Use existing plot_comparison function but with method names
    plot_comparison(all_results, save_plots=save_plots, plot_dir=plot_dir, 
                   update_interval=update_interval, m_scale=m_scale,
                   system_names=method_names, plot_labels=plot_labels, axis_labels=axis_labels_dict)


def run_nonlinearity_test(nonlinearity_factors=[1e-2, 1, 1e2], n_trials=5, H=1000, update_interval=100, num_sensors=6, max_sensors=3, plot=True):
    for m_scale in nonlinearity_factors:
        run_comprehensive_test(n_trials=n_trials, H=H, update_interval=update_interval, num_sensors=num_sensors, max_sensors=max_sensors, plot=plot, m_scale=m_scale)

if __name__ == "__main__":
    # Check if running from command line with arguments
    if len(sys.argv) > 1:
        # Original behavior: run with specified method from command line
        # Run comprehensive test with multiple trials
        run_comprehensive_test(n_trials=100, H=1000, update_interval=100, num_sensors=10, max_sensors=5, plot=True, m_scale=1e2)
    else:
        # Programmatic use: run method comparison
        # RECOMMENDATION 1: Reduced update_interval from 100 to 10 for more frequent selection
        # This allows sensor selection to have more impact on performance
        # sensor_variance controls how different sensor qualities are (higher = more variance)
        run_method_comparison(n_trials=50, H=1000, update_interval=100, 
                              num_sensors=10, max_sensors=5, m_scale=1e2, 
                              sensor_variance=1,
                              reset=True,
                              plot=True)
    
    # # Run nonlinearity test
    # nonlinearity_factors = [0, 1, 1e1, 1e2]
    # nonlinearity_factors = [1e2]
    # run_nonlinearity_test(nonlinearity_factors=nonlinearity_factors, n_trials=100, H=1000, update_interval=100, num_sensors=10, max_sensors=5, plot=True)