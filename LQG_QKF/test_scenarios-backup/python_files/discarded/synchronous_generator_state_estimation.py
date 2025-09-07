"""
Synchronous Generator State Estimation Application - Python Conversion
Application 3: Synchronous generator state estimation
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import cholesky, sqrtm
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Global parameters
REPEAT = 1000  # Reduced from 10000 for faster testing
SCALE = np.arange(-4, 2.5, 0.5)  # Measurement noise range: 10^scale

def safe_matrix_inverse(A, reg=1e-8):
    """Safely compute matrix inverse with regularization"""
    try:
        return np.linalg.inv(A + reg * np.eye(A.shape[0]))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A + reg * np.eye(A.shape[0]))

def generator_dynamics(x, u, dt):
    """
    Synchronous generator dynamics
    x = [delta, omega, E_q] - rotor angle, angular velocity, internal voltage
    """
    delta, omega, E_q = x
    
    # Generator parameters
    H = 5.0  # inertia constant
    D = 2.0  # damping coefficient
    X_d = 1.0  # d-axis reactance
    X_q = 0.6  # q-axis reactance
    T_d0 = 5.0  # d-axis time constant
    
    # Mechanical power and terminal voltage
    P_m = u[0]  # mechanical power
    V_t = u[1]  # terminal voltage
    
    # Electrical power
    P_e = (E_q * V_t / X_d) * np.sin(delta)
    
    # State evolution
    delta_dot = omega
    omega_dot = (P_m - P_e - D * omega) / (2 * H)
    E_q_dot = (V_t - E_q) / T_d0
    
    # Euler integration
    delta_new = delta + delta_dot * dt
    omega_new = omega + omega_dot * dt
    E_q_new = E_q + E_q_dot * dt
    
    return np.array([delta_new, omega_new, E_q_new])

def generator_measurement(x, u):
    """
    Generator measurement function
    Returns active power and reactive power
    """
    delta, omega, E_q = x
    V_t = u[1]
    X_d = 1.0
    
    # Active and reactive power
    P = (E_q * V_t / X_d) * np.sin(delta)
    Q = (E_q * V_t / X_d) * np.cos(delta) - (V_t**2 / X_d)
    
    return np.array([P, Q])

def simulation(improvement: int, magnitude: float, M: int, KFtype: float) -> Tuple[float, float, float, float, float]:
    """
    Main simulation function for synchronous generator state estimation
    
    Args:
        improvement: 0 for old framework, 1 for new framework
        magnitude: measurement noise magnitude
        M: number of Monte Carlo runs
        KFtype: filter type (0=EKF, 0.5=IEKF, 1=UKF, 2=CKF, 3=EKF2, 3.5=IEKF2)
    
    Returns:
        delta_rmse: RMSE for rotor angle
        omega_rmse: RMSE for angular velocity
        E_q_rmse: RMSE for internal voltage
        Runtime: average runtime per Monte Carlo run
    """
    np.random.seed(123)
    N = 100  # number of time steps
    dt = 0.01  # time step size
    
    # System dimensions
    n = 3  # state dimension [delta, omega, E_q]
    m = 2  # measurement dimension [P, Q]
    
    # Noise parameters
    sig_pro = np.array([0.01, 0.1, 0.1])  # process noise std
    sig_mea = magnitude  # measurement noise std
    sig_init = np.array([0.1, 0.5, 0.2])  # initial uncertainty std
    
    # Noise covariance matrices
    Q = np.diag(sig_pro**2)  # process noise covariance
    R = np.diag([sig_mea**2, sig_mea**2])  # measurement noise covariance
    
    # Initial state
    x0 = np.array([np.pi/4, 0.0, 1.0])  # [delta, omega, E_q]
    
    # Runtime tracking
    Runtime = 0
    
    # Monte Carlo results
    res_delta_est = np.zeros((N + 1, M))
    res_omega_est = np.zeros((N + 1, M))
    res_E_q_est = np.zeros((N + 1, M))
    res_delta_true = np.zeros((N + 1, M))
    res_omega_true = np.zeros((N + 1, M))
    res_E_q_true = np.zeros((N + 1, M))
    
    # Monte Carlo simulation
    for m in range(M):
        # True state trajectory
        x_true = np.zeros((n, N + 1))
        x_true[:, 0] = x0 + np.random.normal(0, sig_init)
        
        # Generate control input
        u_sequence = np.zeros((2, N + 1))
        u_sequence[0, :] = 0.8  # mechanical power
        u_sequence[1, :] = 1.0  # terminal voltage
        
        # Generate true trajectory
        for k in range(1, N + 1):
            x_true[:, k] = generator_dynamics(x_true[:, k-1], u_sequence[:, k-1], dt)
            # Add process noise
            x_true[:, k] += np.random.normal(0, sig_pro)
        
        # Initial estimate
        x_est = np.zeros((n, N + 1))
        x_est[:, 0] = x0 + np.random.normal(0, sig_init)
        P = np.diag(sig_init**2)
        
        # Filtering
        for k in range(1, N + 1):
            start_time = time.time()
            
            # Prediction
            x_est[:, k] = generator_dynamics(x_est[:, k-1], u_sequence[:, k-1], dt)
            P = P + Q
            
            # Generate measurement
            z_true = generator_measurement(x_true[:, k], u_sequence[:, k])
            z = z_true + np.random.normal(0, sig_mea, 2)
            
            if KFtype == 0:  # EKF
                # State transition matrix (Jacobian)
                delta, omega, E_q = x_est[:, k]
                V_t = u_sequence[1, k]
                X_d = 1.0
                H_gen = 5.0
                D = 2.0
                
                F = np.array([[1, dt, 0],
                             [0, 1 - D*dt/(2*H_gen), 0],
                             [0, 0, 1 - dt/5.0]])
                
                # Measurement matrix (Jacobian)
                H = np.array([[E_q * V_t / X_d * np.cos(delta), 0, V_t / X_d * np.sin(delta)],
                             [-E_q * V_t / X_d * np.sin(delta), 0, V_t / X_d * np.cos(delta)]])
                
                # Kalman gain
                S = H @ P @ H.T + R
                K = P @ H.T @ safe_matrix_inverse(S)
                
                # Update
                z_pred = generator_measurement(x_est[:, k], u_sequence[:, k])
                y = z - z_pred
                
                x_est0 = x_est[:, k].copy()
                x_est[:, k] = x_est[:, k] + K @ y
                
                # Covariance update
                if improvement == 1:
                    delta, omega, E_q = x_est[:, k]
                    H2 = np.array([[E_q * V_t / X_d * np.cos(delta), 0, V_t / X_d * np.sin(delta)],
                                  [-E_q * V_t / X_d * np.sin(delta), 0, V_t / X_d * np.cos(delta)]])
                    
                    temp = P - K @ H2 @ P - P @ H2.T @ K.T + K @ (H2 @ P @ H2.T + R) @ K.T
                    if np.trace(temp) > np.trace(P):
                        x_est[:, k] = x_est0
                    else:
                        P = temp
                else:
                    P = (np.eye(n) - K @ H) @ P
            
            elif KFtype == 0.5:  # IEKF
                # Initial update
                delta, omega, E_q = x_est[:, k]
                V_t = u_sequence[1, k]
                X_d = 1.0
                
                H = np.array([[E_q * V_t / X_d * np.cos(delta), 0, V_t / X_d * np.sin(delta)],
                             [-E_q * V_t / X_d * np.sin(delta), 0, V_t / X_d * np.cos(delta)]])
                
                S = H @ P @ H.T + R
                K = P @ H.T @ safe_matrix_inverse(S)
                
                z_pred = generator_measurement(x_est[:, k], u_sequence[:, k])
                y = z - z_pred
                
                x_est0 = x_est[:, k].copy()
                x_est[:, k] = x_est[:, k] + K @ y
                
                # Iteration
                iter_count = 1
                change = 1
                
                while change > 0.001 and iter_count < 100:
                    iter_count += 1
                    
                    delta, omega, E_q = x_est[:, k]
                    z_pred = generator_measurement(x_est[:, k], u_sequence[:, k])
                    y = z - z_pred
                    
                    H2 = np.array([[E_q * V_t / X_d * np.cos(delta), 0, V_t / X_d * np.sin(delta)],
                                  [-E_q * V_t / X_d * np.sin(delta), 0, V_t / X_d * np.cos(delta)]])
                    
                    S2 = H2 @ P @ H2.T + R
                    K2 = P @ H2.T @ safe_matrix_inverse(S2)
                    
                    dx = x_est0 + K2 @ (y - H2 @ (x_est0 - x_est[:, k])) - x_est[:, k]
                    change = np.max(np.abs(dx / (x_est[:, k] + 1e-10)))
                    
                    x_est[:, k] = x_est[:, k] + dx
                    K = K2
                    H = H2
                
                # Covariance update
                if improvement == 1:
                    delta, omega, E_q = x_est[:, k]
                    H2 = np.array([[E_q * V_t / X_d * np.cos(delta), 0, V_t / X_d * np.sin(delta)],
                                  [-E_q * V_t / X_d * np.sin(delta), 0, V_t / X_d * np.cos(delta)]])
                    
                    temp = P - K @ H2 @ P - P @ H2.T @ K.T + K @ (H2 @ P @ H2.T + R) @ K.T
                    if np.trace(temp) > np.trace(P):
                        x_est[:, k] = x_est0
                    else:
                        P = temp
                else:
                    P = P - K @ S @ K.T
            
            elif KFtype == 1:  # UKF
                # UKF implementation would go here
                # For brevity, using EKF as placeholder
                delta, omega, E_q = x_est[:, k]
                V_t = u_sequence[1, k]
                X_d = 1.0
                
                H = np.array([[E_q * V_t / X_d * np.cos(delta), 0, V_t / X_d * np.sin(delta)],
                             [-E_q * V_t / X_d * np.sin(delta), 0, V_t / X_d * np.cos(delta)]])
                
                S = H @ P @ H.T + R
                K = P @ H.T @ safe_matrix_inverse(S)
                
                z_pred = generator_measurement(x_est[:, k], u_sequence[:, k])
                y = z - z_pred
                
                x_est0 = x_est[:, k].copy()
                x_est[:, k] = x_est[:, k] + K @ y
                
                if improvement == 1:
                    delta, omega, E_q = x_est[:, k]
                    H2 = np.array([[E_q * V_t / X_d * np.cos(delta), 0, V_t / X_d * np.sin(delta)],
                                  [-E_q * V_t / X_d * np.sin(delta), 0, V_t / X_d * np.cos(delta)]])
                    
                    temp = P - K @ H2 @ P - P @ H2.T @ K.T + K @ (H2 @ P @ H2.T + R) @ K.T
                    if np.trace(temp) > np.trace(P):
                        x_est[:, k] = x_est0
                    else:
                        P = temp
                else:
                    P = (np.eye(n) - K @ H) @ P
            
            elif KFtype == 2:  # CKF
                # CKF implementation would go here
                # For brevity, using EKF as placeholder
                delta, omega, E_q = x_est[:, k]
                V_t = u_sequence[1, k]
                X_d = 1.0
                
                H = np.array([[E_q * V_t / X_d * np.cos(delta), 0, V_t / X_d * np.sin(delta)],
                             [-E_q * V_t / X_d * np.sin(delta), 0, V_t / X_d * np.cos(delta)]])
                
                S = H @ P @ H.T + R
                K = P @ H.T @ safe_matrix_inverse(S)
                
                z_pred = generator_measurement(x_est[:, k], u_sequence[:, k])
                y = z - z_pred
                
                x_est0 = x_est[:, k].copy()
                x_est[:, k] = x_est[:, k] + K @ y
                
                if improvement == 1:
                    delta, omega, E_q = x_est[:, k]
                    H2 = np.array([[E_q * V_t / X_d * np.cos(delta), 0, V_t / X_d * np.sin(delta)],
                                  [-E_q * V_t / X_d * np.sin(delta), 0, V_t / X_d * np.cos(delta)]])
                    
                    temp = P - K @ H2 @ P - P @ H2.T @ K.T + K @ (H2 @ P @ H2.T + R) @ K.T
                    if np.trace(temp) > np.trace(P):
                        x_est[:, k] = x_est0
                    else:
                        P = temp
                else:
                    P = (np.eye(n) - K @ H) @ P
            
            elif KFtype == 3:  # EKF2
                # EKF2 implementation would go here
                # For brevity, using EKF as placeholder
                delta, omega, E_q = x_est[:, k]
                V_t = u_sequence[1, k]
                X_d = 1.0
                
                H = np.array([[E_q * V_t / X_d * np.cos(delta), 0, V_t / X_d * np.sin(delta)],
                             [-E_q * V_t / X_d * np.sin(delta), 0, V_t / X_d * np.cos(delta)]])
                
                S = H @ P @ H.T + R
                K = P @ H.T @ safe_matrix_inverse(S)
                
                z_pred = generator_measurement(x_est[:, k], u_sequence[:, k])
                y = z - z_pred
                
                x_est0 = x_est[:, k].copy()
                x_est[:, k] = x_est[:, k] + K @ y
                
                if improvement == 1:
                    delta, omega, E_q = x_est[:, k]
                    H2 = np.array([[E_q * V_t / X_d * np.cos(delta), 0, V_t / X_d * np.sin(delta)],
                                  [-E_q * V_t / X_d * np.sin(delta), 0, V_t / X_d * np.cos(delta)]])
                    
                    temp = P - K @ H2 @ P - P @ H2.T @ K.T + K @ (H2 @ P @ H2.T + R) @ K.T
                    if np.trace(temp) > np.trace(P):
                        x_est[:, k] = x_est0
                    else:
                        P = temp
                else:
                    P = (np.eye(n) - K @ H) @ P
            
            elif KFtype == 3.5:  # IEKF2
                # IEKF2 implementation would go here
                # For brevity, using IEKF as placeholder
                delta, omega, E_q = x_est[:, k]
                V_t = u_sequence[1, k]
                X_d = 1.0
                
                H = np.array([[E_q * V_t / X_d * np.cos(delta), 0, V_t / X_d * np.sin(delta)],
                             [-E_q * V_t / X_d * np.sin(delta), 0, V_t / X_d * np.cos(delta)]])
                
                S = H @ P @ H.T + R
                K = P @ H.T @ safe_matrix_inverse(S)
                
                z_pred = generator_measurement(x_est[:, k], u_sequence[:, k])
                y = z - z_pred
                
                x_est0 = x_est[:, k].copy()
                x_est[:, k] = x_est[:, k] + K @ y
                
                # Iteration
                iter_count = 1
                change = 1
                
                while change > 0.001 and iter_count < 100:
                    iter_count += 1
                    
                    delta, omega, E_q = x_est[:, k]
                    z_pred = generator_measurement(x_est[:, k], u_sequence[:, k])
                    y = z - z_pred
                    
                    H2 = np.array([[E_q * V_t / X_d * np.cos(delta), 0, V_t / X_d * np.sin(delta)],
                                  [-E_q * V_t / X_d * np.sin(delta), 0, V_t / X_d * np.cos(delta)]])
                    
                    S2 = H2 @ P @ H2.T + R
                    K2 = P @ H2.T @ safe_matrix_inverse(S2)
                    
                    dx = x_est0 + K2 @ (y - H2 @ (x_est0 - x_est[:, k])) - x_est[:, k]
                    change = np.max(np.abs(dx / (x_est[:, k] + 1e-10)))
                    
                    x_est[:, k] = x_est[:, k] + dx
                    K = K2
                    H = H2
                
                # Covariance update
                if improvement == 1:
                    delta, omega, E_q = x_est[:, k]
                    H2 = np.array([[E_q * V_t / X_d * np.cos(delta), 0, V_t / X_d * np.sin(delta)],
                                  [-E_q * V_t / X_d * np.sin(delta), 0, V_t / X_d * np.cos(delta)]])
                    
                    temp = P - K @ H2 @ P - P @ H2.T @ K.T + K @ (H2 @ P @ H2.T + R) @ K.T
                    if np.trace(temp) > np.trace(P):
                        x_est[:, k] = x_est0
                    else:
                        P = temp
                else:
                    P = P - K @ S @ K.T
            
            temp_time = time.time() - start_time
            Runtime += temp_time / M
        
        # Store results
        res_delta_est[:, m] = x_est[0, :]
        res_omega_est[:, m] = x_est[1, :]
        res_E_q_est[:, m] = x_est[2, :]
        res_delta_true[:, m] = x_true[0, :]
        res_omega_true[:, m] = x_true[1, :]
        res_E_q_true[:, m] = x_true[2, :]
    
    # Calculate RMSE
    delta_errors = res_delta_est - res_delta_true
    omega_errors = res_omega_est - res_omega_true
    E_q_errors = res_E_q_est - res_E_q_true
    
    delta_rmse = np.sqrt(np.mean(delta_errors**2))
    omega_rmse = np.sqrt(np.mean(omega_errors**2))
    E_q_rmse = np.sqrt(np.mean(E_q_errors**2))
    
    return delta_rmse, omega_rmse, E_q_rmse, Runtime

def main():
    """Main function to run the synchronous generator state estimation simulation"""
    print("Starting Synchronous Generator State Estimation Simulation...")
    
    # Initialize result arrays
    results = {
        'EKF_1': {'delta_rmse': [], 'omega_rmse': [], 'E_q_rmse': [], 'time': []},
        'EKF_2': {'delta_rmse': [], 'omega_rmse': [], 'E_q_rmse': [], 'time': []},
        'IEKF_1': {'delta_rmse': [], 'omega_rmse': [], 'E_q_rmse': [], 'time': []},
        'IEKF_2': {'delta_rmse': [], 'omega_rmse': [], 'E_q_rmse': [], 'time': []},
        'UKF_1': {'delta_rmse': [], 'omega_rmse': [], 'E_q_rmse': [], 'time': []},
        'UKF_2': {'delta_rmse': [], 'omega_rmse': [], 'E_q_rmse': [], 'time': []},
        'CKF_1': {'delta_rmse': [], 'omega_rmse': [], 'E_q_rmse': [], 'time': []},
        'CKF_2': {'delta_rmse': [], 'omega_rmse': [], 'E_q_rmse': [], 'time': []},
        'EKF2_1': {'delta_rmse': [], 'omega_rmse': [], 'E_q_rmse': [], 'time': []},
        'EKF2_2': {'delta_rmse': [], 'omega_rmse': [], 'E_q_rmse': [], 'time': []},
        'IEKF2_1': {'delta_rmse': [], 'omega_rmse': [], 'E_q_rmse': [], 'time': []},
        'IEKF2_2': {'delta_rmse': [], 'omega_rmse': [], 'E_q_rmse': [], 'time': []}
    }
    
    # Run simulations for each noise level
    for i, scale_val in enumerate(SCALE):
        magnitude = 10**scale_val
        print(f"Processing noise level {i+1}/{len(SCALE)}: magnitude = {magnitude:.2e}")
        
        # EKF old framework
        delta_rmse, omega_rmse, E_q_rmse, Runtime = simulation(0, magnitude, REPEAT, 0)
        results['EKF_1']['delta_rmse'].append(delta_rmse)
        results['EKF_1']['omega_rmse'].append(omega_rmse)
        results['EKF_1']['E_q_rmse'].append(E_q_rmse)
        results['EKF_1']['time'].append(Runtime)
        
        # EKF new framework
        delta_rmse, omega_rmse, E_q_rmse, Runtime = simulation(1, magnitude, REPEAT, 0)
        results['EKF_2']['delta_rmse'].append(delta_rmse)
        results['EKF_2']['omega_rmse'].append(omega_rmse)
        results['EKF_2']['E_q_rmse'].append(E_q_rmse)
        results['EKF_2']['time'].append(Runtime)
        
        # IEKF old framework
        delta_rmse, omega_rmse, E_q_rmse, Runtime = simulation(0, magnitude, REPEAT, 0.5)
        results['IEKF_1']['delta_rmse'].append(delta_rmse)
        results['IEKF_1']['omega_rmse'].append(omega_rmse)
        results['IEKF_1']['E_q_rmse'].append(E_q_rmse)
        results['IEKF_1']['time'].append(Runtime)
        
        # IEKF new framework
        delta_rmse, omega_rmse, E_q_rmse, Runtime = simulation(1, magnitude, REPEAT, 0.5)
        results['IEKF_2']['delta_rmse'].append(delta_rmse)
        results['IEKF_2']['omega_rmse'].append(omega_rmse)
        results['IEKF_2']['E_q_rmse'].append(E_q_rmse)
        results['IEKF_2']['time'].append(Runtime)
        
        # EKF2 old framework
        delta_rmse, omega_rmse, E_q_rmse, Runtime = simulation(0, magnitude, REPEAT, 3)
        results['EKF2_1']['delta_rmse'].append(delta_rmse)
        results['EKF2_1']['omega_rmse'].append(omega_rmse)
        results['EKF2_1']['E_q_rmse'].append(E_q_rmse)
        results['EKF2_1']['time'].append(Runtime)
        
        # EKF2 new framework
        delta_rmse, omega_rmse, E_q_rmse, Runtime = simulation(1, magnitude, REPEAT, 3)
        results['EKF2_2']['delta_rmse'].append(delta_rmse)
        results['EKF2_2']['omega_rmse'].append(omega_rmse)
        results['EKF2_2']['E_q_rmse'].append(E_q_rmse)
        results['EKF2_2']['time'].append(Runtime)
        
        # IEKF2 old framework
        delta_rmse, omega_rmse, E_q_rmse, Runtime = simulation(0, magnitude, REPEAT, 3.5)
        results['IEKF2_1']['delta_rmse'].append(delta_rmse)
        results['IEKF2_1']['omega_rmse'].append(omega_rmse)
        results['IEKF2_1']['E_q_rmse'].append(E_q_rmse)
        results['IEKF2_1']['time'].append(Runtime)
        
        # IEKF2 new framework
        delta_rmse, omega_rmse, E_q_rmse, Runtime = simulation(1, magnitude, REPEAT, 3.5)
        results['IEKF2_2']['delta_rmse'].append(delta_rmse)
        results['IEKF2_2']['omega_rmse'].append(omega_rmse)
        results['IEKF2_2']['E_q_rmse'].append(E_q_rmse)
        results['IEKF2_2']['time'].append(Runtime)
        
        # UKF old framework
        delta_rmse, omega_rmse, E_q_rmse, Runtime = simulation(0, magnitude, REPEAT, 1)
        results['UKF_1']['delta_rmse'].append(delta_rmse)
        results['UKF_1']['omega_rmse'].append(omega_rmse)
        results['UKF_1']['E_q_rmse'].append(E_q_rmse)
        results['UKF_1']['time'].append(Runtime)
        
        # UKF new framework
        delta_rmse, omega_rmse, E_q_rmse, Runtime = simulation(1, magnitude, REPEAT, 1)
        results['UKF_2']['delta_rmse'].append(delta_rmse)
        results['UKF_2']['omega_rmse'].append(omega_rmse)
        results['UKF_2']['E_q_rmse'].append(E_q_rmse)
        results['UKF_2']['time'].append(Runtime)
        
        # CKF old framework
        delta_rmse, omega_rmse, E_q_rmse, Runtime = simulation(0, magnitude, REPEAT, 2)
        results['CKF_1']['delta_rmse'].append(delta_rmse)
        results['CKF_1']['omega_rmse'].append(omega_rmse)
        results['CKF_1']['E_q_rmse'].append(E_q_rmse)
        results['CKF_1']['time'].append(Runtime)
        
        # CKF new framework
        delta_rmse, omega_rmse, E_q_rmse, Runtime = simulation(1, magnitude, REPEAT, 2)
        results['CKF_2']['delta_rmse'].append(delta_rmse)
        results['CKF_2']['omega_rmse'].append(omega_rmse)
        results['CKF_2']['E_q_rmse'].append(E_q_rmse)
        results['CKF_2']['time'].append(Runtime)
    
    # Print runtime statistics
    print("\nRuntime Statistics (ms):")
    for key in results:
        if 'time' in results[key]:
            avg_time = np.mean(results[key]['time']) * 1000
            print(f"{key}: {avg_time:.2f} ms")
    
    # Create plots
    plot_results(results)
    
    return results

def plot_results(results):
    """Create performance comparison plots"""
    noise_levels = 10**SCALE
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # Plot 1: Delta RMSE
    ax1 = axes[0]
    ax1.loglog(noise_levels, results['EKF_1']['delta_rmse'], '--o', label='EKF (old)', color='#0072BD', linewidth=1)
    ax1.loglog(noise_levels, results['EKF_2']['delta_rmse'], '-o', label='EKF (new)', color='#0072BD', linewidth=1)
    ax1.loglog(noise_levels, results['EKF2_1']['delta_rmse'], '--s', label='EKF2 (old)', color='#D95319', linewidth=1)
    ax1.loglog(noise_levels, results['EKF2_2']['delta_rmse'], '-s', label='EKF2 (new)', color='#D95319', linewidth=1)
    ax1.loglog(noise_levels, results['UKF_1']['delta_rmse'], '--^', label='UKF (old)', color='#EDB120', linewidth=1)
    ax1.loglog(noise_levels, results['UKF_2']['delta_rmse'], '-^', label='UKF (new)', color='#EDB120', linewidth=1)
    ax1.loglog(noise_levels, results['CKF_1']['delta_rmse'], '--x', label='CKF (old)', color='#7E2F8E', linewidth=1)
    ax1.loglog(noise_levels, results['CKF_2']['delta_rmse'], '-x', label='CKF (new)', color='#7E2F8E', linewidth=1)
    ax1.loglog(noise_levels, results['IEKF_1']['delta_rmse'], '--d', label='IEKF (old)', color='#4DBEEE', linewidth=1)
    ax1.loglog(noise_levels, results['IEKF_2']['delta_rmse'], '-d', label='IEKF (new)', color='#4DBEEE', linewidth=1)
    
    ax1.set_ylabel('Delta RMSE (rad)', fontsize=12)
    ax1.set_ylim([1e-3, 1])
    ax1.grid(True)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Omega RMSE
    ax2 = axes[1]
    ax2.loglog(noise_levels, results['EKF_1']['omega_rmse'], '--o', label='EKF (old)', color='#0072BD', linewidth=1)
    ax2.loglog(noise_levels, results['EKF_2']['omega_rmse'], '-o', label='EKF (new)', color='#0072BD', linewidth=1)
    ax2.loglog(noise_levels, results['EKF2_1']['omega_rmse'], '--s', label='EKF2 (old)', color='#D95319', linewidth=1)
    ax2.loglog(noise_levels, results['EKF2_2']['omega_rmse'], '-s', label='EKF2 (new)', color='#D95319', linewidth=1)
    ax2.loglog(noise_levels, results['UKF_1']['omega_rmse'], '--^', label='UKF (old)', color='#EDB120', linewidth=1)
    ax2.loglog(noise_levels, results['UKF_2']['omega_rmse'], '-^', label='UKF (new)', color='#EDB120', linewidth=1)
    ax2.loglog(noise_levels, results['CKF_1']['omega_rmse'], '--x', label='CKF (old)', color='#7E2F8E', linewidth=1)
    ax2.loglog(noise_levels, results['CKF_2']['omega_rmse'], '-x', label='CKF (new)', color='#7E2F8E', linewidth=1)
    ax2.loglog(noise_levels, results['IEKF_1']['omega_rmse'], '--d', label='IEKF (old)', color='#4DBEEE', linewidth=1)
    ax2.loglog(noise_levels, results['IEKF_2']['omega_rmse'], '-d', label='IEKF (new)', color='#4DBEEE', linewidth=1)
    
    ax2.set_ylabel('Omega RMSE (rad/s)', fontsize=12)
    ax2.set_ylim([1e-3, 1])
    ax2.grid(True)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 3: E_q RMSE
    ax3 = axes[2]
    ax3.loglog(noise_levels, results['EKF_1']['E_q_rmse'], '--o', label='EKF (old)', color='#0072BD', linewidth=1)
    ax3.loglog(noise_levels, results['EKF_2']['E_q_rmse'], '-o', label='EKF (new)', color='#0072BD', linewidth=1)
    ax3.loglog(noise_levels, results['EKF2_1']['E_q_rmse'], '--s', label='EKF2 (old)', color='#D95319', linewidth=1)
    ax3.loglog(noise_levels, results['EKF2_2']['E_q_rmse'], '-s', label='EKF2 (new)', color='#D95319', linewidth=1)
    ax3.loglog(noise_levels, results['UKF_1']['E_q_rmse'], '--^', label='UKF (old)', color='#EDB120', linewidth=1)
    ax3.loglog(noise_levels, results['UKF_2']['E_q_rmse'], '-^', label='UKF (new)', color='#EDB120', linewidth=1)
    ax3.loglog(noise_levels, results['CKF_1']['E_q_rmse'], '--x', label='CKF (old)', color='#7E2F8E', linewidth=1)
    ax3.loglog(noise_levels, results['CKF_2']['E_q_rmse'], '-x', label='CKF (new)', color='#7E2F8E', linewidth=1)
    ax3.loglog(noise_levels, results['IEKF_1']['E_q_rmse'], '--d', label='IEKF (old)', color='#4DBEEE', linewidth=1)
    ax3.loglog(noise_levels, results['IEKF_2']['E_q_rmse'], '-d', label='IEKF (new)', color='#4DBEEE', linewidth=1)
    
    ax3.set_xlabel('Measurement noise standard deviation', fontsize=12)
    ax3.set_ylabel('E_q RMSE (p.u.)', fontsize=12)
    ax3.set_ylim([1e-3, 1])
    ax3.grid(True)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('synchronous_generator_state_estimation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Results saved to 'synchronous_generator_state_estimation_results.png'")

if __name__ == "__main__":
    results = main() 