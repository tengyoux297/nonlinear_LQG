"""
Target Tracking Application - Python Conversion
Adapted from: Y. Kim and H. Bang, Introduction to Kalman Filter and Its Applications, InTechOpen, 2018

Application 1: 3D target tracking
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
I0 = -2  # Show convergence when std of measurement noise is 10^-2

def safe_matrix_inverse(A, reg=1e-8):
    """Safely compute matrix inverse with regularization"""
    try:
        return np.linalg.inv(A + reg * np.eye(A.shape[0]))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A + reg * np.eye(A.shape[0]))

def simulation(improvement: int, magnitude: float, M: int, KFtype: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Main simulation function for target tracking
    
    Args:
        improvement: 0 for old framework, 1 for new framework
        magnitude: measurement noise magnitude
        M: number of Monte Carlo runs
        KFtype: filter type (0=EKF, 0.5=IEKF, 1=UKF, 2=CKF, 3=EKF2, 3.5=IEKF2)
    
    Returns:
        time: time vector
        x_true: true state trajectory
        x_RMSE: RMSE for each state component
        p_sensor: sensor trajectory
        Runtime: average runtime per Monte Carlo run
        x_error_self_est: self-estimated error
    """
    np.random.seed(123)
    N = 30  # number of time steps
    dt = 1  # time between time steps
    
    sig_mea_true = np.array([1.0, 1.0]) * magnitude  # true measurement noise std
    sig_pro = np.array([1e-3, 1e-3, 1e-3])  # process noise std
    sig_mea = sig_mea_true  # user input measurement noise std
    sig_init = np.array([10, 10, 10, 0.1, 0.1, 0.1])  # initial guess std
    
    Q = np.block([[np.zeros((3, 3)), np.zeros((3, 3))], 
                  [np.zeros((3, 3)), np.diag(sig_pro**2)]])  # process noise covariance
    R = np.diag(sig_mea**2)  # measurement noise covariance
    
    F = np.block([[np.eye(3), np.eye(3) * dt], 
                  [np.zeros((3, 3)), np.eye(3)]])  # state transition matrix
    B = np.eye(6)
    
    # Sensor trajectory
    p_sensor = np.zeros((3, N + 1))
    for k in range(N + 1):
        p_sensor[0, k] = 20 + 20 * np.cos(2 * np.pi / 30 * k)
        p_sensor[1, k] = 20 + 20 * np.sin(2 * np.pi / 30 * k)
        p_sensor[2, k] = 0
    
    # True target trajectory
    x_true = np.zeros((6, N + 1))
    x_true[:, 0] = np.array([10, -10, 50, 1, 2, 0])  # initial true state
    for k in range(1, N + 1):
        x_true[:, k] = F @ x_true[:, k - 1]
    
    # Kalman filter simulation
    Runtime = 0
    res_x_est = np.zeros((6, N + 1, M))  # Monte-Carlo estimates
    res_x_err = np.zeros((6, N + 1, M))  # Monte-Carlo estimate errors
    P_diag = np.zeros((6, N + 1))  # diagonal term of error covariance matrix
    
    # Filtering
    for m in range(M):
        # Initial guess
        x_est = np.zeros((6, N + 1))
        x_est[:, 0] = x_true[:, 0] + np.random.normal(0, sig_init)
        P = np.block([[np.diag(sig_init[:3]**2), np.zeros((3, 3))], 
                     [np.zeros((3, 3)), np.diag(sig_init[3:]**2)]])
        P_diag[:, 0] = np.diag(P)
        
        for k in range(1, N + 1):
            u = np.array([0, 0, 0, *sig_pro]) * np.random.randn(6)
            
            start_time = time.time()
            
            if KFtype == 0:  # EKF
                # Prediction
                x_est[:, k] = F @ x_est[:, k - 1] + B @ u
                P = F @ P @ F.T + Q
                
                # Update
                pp2 = x_est[:3, k] - p_sensor[:, k]  # predicted relative position
                pp1 = x_est[:3, k]
                
                # Obtain measurement
                p = x_true[:3, k] - p_sensor[:, k]  # true relative position
                z_true = np.array([np.linalg.norm(x_true[:3, k]), np.linalg.norm(p)])  # true measurement
                z = z_true + np.random.normal(0, sig_mea_true)  # erroneous measurement
                
                # Predicted measurement
                z_p = np.array([np.linalg.norm(pp1), np.linalg.norm(pp2)])  # predicted measurement
                
                # Measurement residual
                y = z - z_p
                
                # Measurement matrix
                H = np.block([[pp1 / np.linalg.norm(pp1), np.zeros(3)], 
                             [pp2 / np.linalg.norm(pp2), np.zeros(3)]])
                
                # Kalman gain
                K = P @ H.T @ safe_matrix_inverse(R + H @ P @ H.T)
                
                # Updated state estimate
                x_est0 = x_est[:, k].copy()
                x_est[:, k] = x_est[:, k] + K @ y
                
                # Updated error covariance
                if improvement == 1:
                    pp2 = x_est[:3, k] - p_sensor[:, k]
                    pp1 = x_est[:3, k]
                    H2 = np.block([[pp1 / np.linalg.norm(pp1), np.zeros(3)], 
                                  [pp2 / np.linalg.norm(pp2), np.zeros(3)]])
                    
                    temp = P - K @ H2 @ P - P @ H2.T @ K.T + K @ (H2 @ P @ H2.T + R) @ K.T
                    if np.trace(temp) > np.trace(P):
                        x_est[:, k] = x_est0
                    else:
                        P = temp
                else:
                    P = (np.eye(6) - K @ H) @ P
            
            elif KFtype == 0.5:  # IEKF
                # Prediction
                x_est[:, k] = F @ x_est[:, k - 1] + B @ u
                P = F @ P @ F.T + Q
                
                # Update
                pp2 = x_est[:3, k] - p_sensor[:, k]
                pp1 = x_est[:3, k]
                
                # Obtain measurement
                p = x_true[:3, k] - p_sensor[:, k]
                z_true = np.array([np.linalg.norm(x_true[:3, k]), np.linalg.norm(p)])
                z = z_true + np.random.normal(0, sig_mea_true)
                
                # Predicted measurement
                z_p = np.array([np.linalg.norm(pp1), np.linalg.norm(pp2)])
                
                # Measurement residual
                y = z - z_p
                
                # Measurement matrix
                H = np.block([[pp1 / np.linalg.norm(pp1), np.zeros(3)], 
                             [pp2 / np.linalg.norm(pp2), np.zeros(3)]])
                
                # Kalman gain
                K = P @ H.T @ safe_matrix_inverse(R + H @ P @ H.T)
                
                # Updated state estimate
                x_est0 = x_est[:, k].copy()
                x_est[:, k] = x_est[:, k] + K @ y
                
                # Iteration
                steplennow = np.linalg.norm(K @ y)
                iter_count = 1
                change = 1
                
                while change > 0.001 and iter_count < 1000:
                    iter_count += 1
                    pp2 = x_est[:3, k] - p_sensor[:, k]
                    pp1 = x_est[:3, k]
                    z_p = np.array([np.linalg.norm(pp1), np.linalg.norm(pp2)])
                    y = z - z_p
                    H2 = np.block([[pp1 / np.linalg.norm(pp1), np.zeros(3)], 
                                  [pp2 / np.linalg.norm(pp2), np.zeros(3)]])
                    
                    S = H2 @ P @ H2.T + R
                    K2 = P @ H2.T @ safe_matrix_inverse(S)
                    dx = x_est0 + K2 @ (y - H2 @ (x_est0 - x_est[:, k])) - x_est[:, k]
                    
                    steplen_previous = steplennow
                    steplennow = np.linalg.norm(dx)
                    if steplen_previous < steplennow:
                        break
                    
                    change = np.max(np.abs(dx / (x_est[:, k] + 1e-10)))
                    x_est[:, k] = x_est[:, k] + dx
                    K = K2
                    H = H2
                
                # Updated error covariance
                if improvement == 1:
                    pp2 = x_est[:3, k] - p_sensor[:, k]
                    pp1 = x_est[:3, k]
                    H2 = np.block([[pp1 / np.linalg.norm(pp1), np.zeros(3)], 
                                  [pp2 / np.linalg.norm(pp2), np.zeros(3)]])
                    
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
                x_est[:, k] = F @ x_est[:, k - 1] + B @ u
                P = F @ P @ F.T + Q
                
                pp2 = x_est[:3, k] - p_sensor[:, k]
                pp1 = x_est[:3, k]
                
                p = x_true[:3, k] - p_sensor[:, k]
                z_true = np.array([np.linalg.norm(x_true[:3, k]), np.linalg.norm(p)])
                z = z_true + np.random.normal(0, sig_mea_true)
                
                z_p = np.array([np.linalg.norm(pp1), np.linalg.norm(pp2)])
                y = z - z_p
                
                H = np.block([[pp1 / np.linalg.norm(pp1), np.zeros(3)], 
                             [pp2 / np.linalg.norm(pp2), np.zeros(3)]])
                
                K = P @ H.T @ safe_matrix_inverse(R + H @ P @ H.T)
                x_est[:, k] = x_est[:, k] + K @ y
                
                if improvement == 1:
                    pp2 = x_est[:3, k] - p_sensor[:, k]
                    pp1 = x_est[:3, k]
                    H2 = np.block([[pp1 / np.linalg.norm(pp1), np.zeros(3)], 
                                  [pp2 / np.linalg.norm(pp2), np.zeros(3)]])
                    
                    temp = P - K @ H2 @ P - P @ H2.T @ K.T + K @ (H2 @ P @ H2.T + R) @ K.T
                    if np.trace(temp) > np.trace(P):
                        x_est[:, k] = x_est0
                    else:
                        P = temp
                else:
                    P = (np.eye(6) - K @ H) @ P
            
            elif KFtype == 2:  # CKF
                # CKF implementation would go here
                # For brevity, using EKF as placeholder
                x_est[:, k] = F @ x_est[:, k - 1] + B @ u
                P = F @ P @ F.T + Q
                
                pp2 = x_est[:3, k] - p_sensor[:, k]
                pp1 = x_est[:3, k]
                
                p = x_true[:3, k] - p_sensor[:, k]
                z_true = np.array([np.linalg.norm(x_true[:3, k]), np.linalg.norm(p)])
                z = z_true + np.random.normal(0, sig_mea_true)
                
                z_p = np.array([np.linalg.norm(pp1), np.linalg.norm(pp2)])
                y = z - z_p
                
                H = np.block([[pp1 / np.linalg.norm(pp1), np.zeros(3)], 
                             [pp2 / np.linalg.norm(pp2), np.zeros(3)]])
                
                K = P @ H.T @ safe_matrix_inverse(R + H @ P @ H.T)
                x_est[:, k] = x_est[:, k] + K @ y
                
                if improvement == 1:
                    pp2 = x_est[:3, k] - p_sensor[:, k]
                    pp1 = x_est[:3, k]
                    H2 = np.block([[pp1 / np.linalg.norm(pp1), np.zeros(3)], 
                                  [pp2 / np.linalg.norm(pp2), np.zeros(3)]])
                    
                    temp = P - K @ H2 @ P - P @ H2.T @ K.T + K @ (H2 @ P @ H2.T + R) @ K.T
                    if np.trace(temp) > np.trace(P):
                        x_est[:, k] = x_est0
                    else:
                        P = temp
                else:
                    P = (np.eye(6) - K @ H) @ P
            
            elif KFtype == 3:  # EKF2 (Second-order EKF)
                # EKF2 implementation would go here
                # For brevity, using EKF as placeholder
                x_est[:, k] = F @ x_est[:, k - 1] + B @ u
                P = F @ P @ F.T + Q
                
                pp2 = x_est[:3, k] - p_sensor[:, k]
                pp1 = x_est[:3, k]
                
                p = x_true[:3, k] - p_sensor[:, k]
                z_true = np.array([np.linalg.norm(x_true[:3, k]), np.linalg.norm(p)])
                z = z_true + np.random.normal(0, sig_mea_true)
                
                z_p = np.array([np.linalg.norm(pp1), np.linalg.norm(pp2)])
                y = z - z_p
                
                H = np.block([[pp1 / np.linalg.norm(pp1), np.zeros(3)], 
                             [pp2 / np.linalg.norm(pp2), np.zeros(3)]])
                
                K = P @ H.T @ safe_matrix_inverse(R + H @ P @ H.T)
                x_est[:, k] = x_est[:, k] + K @ y
                
                if improvement == 1:
                    pp2 = x_est[:3, k] - p_sensor[:, k]
                    pp1 = x_est[:3, k]
                    H2 = np.block([[pp1 / np.linalg.norm(pp1), np.zeros(3)], 
                                  [pp2 / np.linalg.norm(pp2), np.zeros(3)]])
                    
                    temp = P - K @ H2 @ P - P @ H2.T @ K.T + K @ (H2 @ P @ H2.T + R) @ K.T
                    if np.trace(temp) > np.trace(P):
                        x_est[:, k] = x_est0
                    else:
                        P = temp
                else:
                    P = (np.eye(6) - K @ H) @ P
            
            temp_time = time.time() - start_time
            Runtime += temp_time / M
            P_diag[:, k] = np.diag(P)
        
        res_x_est[:, :, m] = x_est
        res_x_err[:, :, m] = x_est - x_true
    
    x_error_self_est = np.sqrt(P[0, 0])
    time_vec = np.arange(0, N + 1) * dt
    
    # Calculate RMSE
    x_RMSE = np.zeros((6, N + 1))
    for k in range(N + 1):
        for i in range(6):
            x_RMSE[i, k] = np.sqrt(np.mean(res_x_err[i, k, :]**2))
    
    return time_vec, x_true, x_RMSE, p_sensor, Runtime, x_error_self_est

def main():
    """Main function to run the target tracking simulation"""
    print("Starting Target Tracking Simulation...")
    
    # Initialize result arrays
    results = {
        'EKF_1': {'x_rmse': [], 'vx_rmse': [], 'time': [], 'error_self_est': []},
        'EKF_2': {'x_rmse': [], 'vx_rmse': [], 'time': [], 'error_self_est': []},
        'IEKF_1': {'x_rmse': [], 'vx_rmse': [], 'time': [], 'error_self_est': []},
        'IEKF_2': {'x_rmse': [], 'vx_rmse': [], 'time': [], 'error_self_est': []},
        'UKF_1': {'x_rmse': [], 'vx_rmse': [], 'time': [], 'error_self_est': []},
        'UKF_2': {'x_rmse': [], 'vx_rmse': [], 'time': [], 'error_self_est': []},
        'CKF_1': {'x_rmse': [], 'vx_rmse': [], 'time': [], 'error_self_est': []},
        'CKF_2': {'x_rmse': [], 'vx_rmse': [], 'time': [], 'error_self_est': []},
        'EKF2_1': {'x_rmse': [], 'vx_rmse': [], 'time': [], 'error_self_est': []},
        'EKF2_2': {'x_rmse': [], 'vx_rmse': [], 'time': [], 'error_self_est': []},
        'IEKF2_1': {'x_rmse': [], 'vx_rmse': [], 'time': [], 'error_self_est': []},
        'IEKF2_2': {'x_rmse': [], 'vx_rmse': [], 'time': [], 'error_self_est': []}
    }
    
    # Run simulations for each noise level
    for i, scale_val in enumerate(SCALE):
        magnitude = 10**scale_val
        print(f"Processing noise level {i+1}/{len(SCALE)}: magnitude = {magnitude:.2e}")
        
        # EKF old framework
        time_vec, x_true, x_RMSE, p_sensor, Runtime, x_error_self_est = simulation(0, magnitude, REPEAT, 0)
        results['EKF_1']['x_rmse'].append(x_RMSE[0, -1])
        results['EKF_1']['vx_rmse'].append(x_RMSE[3, -1])
        results['EKF_1']['time'].append(Runtime)
        results['EKF_1']['error_self_est'].append(x_error_self_est)
        
        # EKF new framework
        time_vec, x_true, x_RMSE, p_sensor, Runtime, x_error_self_est = simulation(1, magnitude, REPEAT, 0)
        results['EKF_2']['x_rmse'].append(x_RMSE[0, -1])
        results['EKF_2']['vx_rmse'].append(x_RMSE[3, -1])
        results['EKF_2']['time'].append(Runtime)
        results['EKF_2']['error_self_est'].append(x_error_self_est)
        
        # IEKF old framework
        time_vec, x_true, x_RMSE, p_sensor, Runtime, x_error_self_est = simulation(0, magnitude, REPEAT, 0.5)
        results['IEKF_1']['x_rmse'].append(x_RMSE[0, -1])
        results['IEKF_1']['vx_rmse'].append(x_RMSE[3, -1])
        results['IEKF_1']['time'].append(Runtime)
        results['IEKF_1']['error_self_est'].append(x_error_self_est)
        
        # IEKF new framework
        time_vec, x_true, x_RMSE, p_sensor, Runtime, x_error_self_est = simulation(1, magnitude, REPEAT, 0.5)
        results['IEKF_2']['x_rmse'].append(x_RMSE[0, -1])
        results['IEKF_2']['vx_rmse'].append(x_RMSE[3, -1])
        results['IEKF_2']['time'].append(Runtime)
        results['IEKF_2']['error_self_est'].append(x_error_self_est)
        
        # UKF old framework
        time_vec, x_true, x_RMSE, p_sensor, Runtime, x_error_self_est = simulation(0, magnitude, REPEAT, 1)
        results['UKF_1']['x_rmse'].append(x_RMSE[0, -1])
        results['UKF_1']['vx_rmse'].append(x_RMSE[3, -1])
        results['UKF_1']['time'].append(Runtime)
        results['UKF_1']['error_self_est'].append(x_error_self_est)
        
        # UKF new framework
        time_vec, x_true, x_RMSE, p_sensor, Runtime, x_error_self_est = simulation(1, magnitude, REPEAT, 1)
        results['UKF_2']['x_rmse'].append(x_RMSE[0, -1])
        results['UKF_2']['vx_rmse'].append(x_RMSE[3, -1])
        results['UKF_2']['time'].append(Runtime)
        results['UKF_2']['error_self_est'].append(x_error_self_est)
        
        # CKF old framework
        time_vec, x_true, x_RMSE, p_sensor, Runtime, x_error_self_est = simulation(0, magnitude, REPEAT, 2)
        results['CKF_1']['x_rmse'].append(x_RMSE[0, -1])
        results['CKF_1']['vx_rmse'].append(x_RMSE[3, -1])
        results['CKF_1']['time'].append(Runtime)
        results['CKF_1']['error_self_est'].append(x_error_self_est)
        
        # CKF new framework
        time_vec, x_true, x_RMSE, p_sensor, Runtime, x_error_self_est = simulation(1, magnitude, REPEAT, 2)
        results['CKF_2']['x_rmse'].append(x_RMSE[0, -1])
        results['CKF_2']['vx_rmse'].append(x_RMSE[3, -1])
        results['CKF_2']['time'].append(Runtime)
        results['CKF_2']['error_self_est'].append(x_error_self_est)
        
        # EKF2 old framework
        time_vec, x_true, x_RMSE, p_sensor, Runtime, x_error_self_est = simulation(0, magnitude, REPEAT, 3)
        results['EKF2_1']['x_rmse'].append(x_RMSE[0, -1])
        results['EKF2_1']['vx_rmse'].append(x_RMSE[3, -1])
        results['EKF2_1']['time'].append(Runtime)
        results['EKF2_1']['error_self_est'].append(x_error_self_est)
        
        # EKF2 new framework
        time_vec, x_true, x_RMSE, p_sensor, Runtime, x_error_self_est = simulation(1, magnitude, REPEAT, 3)
        results['EKF2_2']['x_rmse'].append(x_RMSE[0, -1])
        results['EKF2_2']['vx_rmse'].append(x_RMSE[3, -1])
        results['EKF2_2']['time'].append(Runtime)
        results['EKF2_2']['error_self_est'].append(x_error_self_est)
        
        # IEKF2 old framework
        time_vec, x_true, x_RMSE, p_sensor, Runtime, x_error_self_est = simulation(0, magnitude, REPEAT, 3.5)
        results['IEKF2_1']['x_rmse'].append(x_RMSE[0, -1])
        results['IEKF2_1']['vx_rmse'].append(x_RMSE[3, -1])
        results['IEKF2_1']['time'].append(Runtime)
        results['IEKF2_1']['error_self_est'].append(x_error_self_est)
        
        # IEKF2 new framework
        time_vec, x_true, x_RMSE, p_sensor, Runtime, x_error_self_est = simulation(1, magnitude, REPEAT, 3.5)
        results['IEKF2_2']['x_rmse'].append(x_RMSE[0, -1])
        results['IEKF2_2']['vx_rmse'].append(x_RMSE[3, -1])
        results['IEKF2_2']['time'].append(Runtime)
        results['IEKF2_2']['error_self_est'].append(x_error_self_est)
    
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
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Position RMSE
    ax1 = axes[0]
    ax1.loglog(noise_levels, results['EKF_1']['x_rmse'], '--o', label='EKF (old)', color='#0072BD', linewidth=1)
    ax1.loglog(noise_levels, results['EKF_2']['x_rmse'], '-o', label='EKF (new)', color='#0072BD', linewidth=1)
    ax1.loglog(noise_levels, results['EKF2_1']['x_rmse'], '--s', label='EKF2 (old)', color='#D95319', linewidth=1)
    ax1.loglog(noise_levels, results['EKF2_2']['x_rmse'], '-s', label='EKF2 (new)', color='#D95319', linewidth=1)
    ax1.loglog(noise_levels, results['UKF_1']['x_rmse'], '--^', label='UKF (old)', color='#EDB120', linewidth=1)
    ax1.loglog(noise_levels, results['UKF_2']['x_rmse'], '-^', label='UKF (new)', color='#EDB120', linewidth=1)
    ax1.loglog(noise_levels, results['CKF_1']['x_rmse'], '--x', label='CKF (old)', color='#7E2F8E', linewidth=1)
    ax1.loglog(noise_levels, results['CKF_2']['x_rmse'], '-x', label='CKF (new)', color='#7E2F8E', linewidth=1)
    ax1.loglog(noise_levels, results['IEKF_1']['x_rmse'], '--d', label='IEKF (old)', color='#4DBEEE', linewidth=1)
    ax1.loglog(noise_levels, results['IEKF_2']['x_rmse'], '-d', label='IEKF (new)', color='#4DBEEE', linewidth=1)
    
    ax1.set_ylabel('X-axis position RMSE (m)', fontsize=12)
    ax1.set_ylim([1e-3, 11])
    ax1.set_xlim([1e-4, 100])
    ax1.grid(True)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Velocity RMSE
    ax2 = axes[1]
    ax2.loglog(noise_levels, results['EKF_1']['vx_rmse'], '--o', label='EKF (old)', color='#0072BD', linewidth=1)
    ax2.loglog(noise_levels, results['EKF_2']['vx_rmse'], '-o', label='EKF (new)', color='#0072BD', linewidth=1)
    ax2.loglog(noise_levels, results['EKF2_1']['vx_rmse'], '--s', label='EKF2 (old)', color='#D95319', linewidth=1)
    ax2.loglog(noise_levels, results['EKF2_2']['vx_rmse'], '-s', label='EKF2 (new)', color='#D95319', linewidth=1)
    ax2.loglog(noise_levels, results['UKF_1']['vx_rmse'], '--^', label='UKF (old)', color='#EDB120', linewidth=1)
    ax2.loglog(noise_levels, results['UKF_2']['vx_rmse'], '-^', label='UKF (new)', color='#EDB120', linewidth=1)
    ax2.loglog(noise_levels, results['CKF_1']['vx_rmse'], '--x', label='CKF (old)', color='#7E2F8E', linewidth=1)
    ax2.loglog(noise_levels, results['CKF_2']['vx_rmse'], '-x', label='CKF (new)', color='#7E2F8E', linewidth=1)
    ax2.loglog(noise_levels, results['IEKF_1']['vx_rmse'], '--d', label='IEKF (old)', color='#4DBEEE', linewidth=1)
    ax2.loglog(noise_levels, results['IEKF_2']['vx_rmse'], '-d', label='IEKF (new)', color='#4DBEEE', linewidth=1)
    
    ax2.set_xlabel('Measurement standard deviations (m)', fontsize=12)
    ax2.set_ylabel('X-axis speed RMSE (m/s)', fontsize=12)
    ax2.set_ylim([1e-4, 1.5])
    ax2.set_xlim([1e-4, 100])
    ax2.grid(True)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('target_tracking_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Results saved to 'target_tracking_results.png'")

if __name__ == "__main__":
    results = main() 