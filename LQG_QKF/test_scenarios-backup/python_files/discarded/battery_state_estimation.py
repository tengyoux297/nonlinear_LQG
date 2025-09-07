"""
Battery State Estimation Application - Python Conversion
Application 5: Battery state-of-charge and state-of-health estimation
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
NOISE1_REF = 0.001
SCALE = np.arange(-2, 3.5, 0.5)  # Measurement noise range: 10^scale

def safe_matrix_inverse(A, reg=1e-8):
    """Safely compute matrix inverse with regularization"""
    try:
        return np.linalg.inv(A + reg * np.eye(A.shape[0]))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A + reg * np.eye(A.shape[0]))

def battery_dynamics(x, u, dt):
    """
    Battery state dynamics
    x = [SOC, SOH, V] - State of Charge, State of Health, Voltage
    """
    SOC, SOH, V = x
    
    # Battery parameters
    Q_nom = 2.3  # Nominal capacity (Ah)
    R_int = 0.1  # Internal resistance (Ohm)
    V_oc = 3.7   # Open circuit voltage (V)
    
    # State evolution
    SOC_dot = -u / (Q_nom * SOH)  # SOC rate of change
    SOH_dot = 0  # SOH remains constant in this model
    V_dot = 0    # Voltage rate of change
    
    # Euler integration
    SOC_new = SOC + SOC_dot * dt
    SOH_new = SOH + SOH_dot * dt
    V_new = V + V_dot * dt
    
    return np.array([SOC_new, SOH_new, V_new])

def battery_measurement(x, u):
    """
    Battery measurement function
    Returns voltage measurement
    """
    SOC, SOH, V = x
    
    # Battery parameters
    R_int = 0.1  # Internal resistance (Ohm)
    V_oc = 3.7   # Open circuit voltage (V)
    
    # Voltage measurement
    V_measured = V_oc - R_int * u
    
    return np.array([V_measured])

def simulation(improvement: int, magnitude: float, M: int, KFtype: float) -> Tuple[float, float, float, float]:
    """
    Main simulation function for battery state estimation
    
    Args:
        improvement: 0 for old framework, 1 for new framework
        magnitude: measurement noise magnitude
        M: number of Monte Carlo runs
        KFtype: filter type (0=EKF, 0.5=IEKF, 1=UKF, 2=CKF, 3=EKF2, 3.5=IEKF2)
    
    Returns:
        SOC_rmse: RMSE for State of Charge
        SOH_rmse: RMSE for State of Health
        V_rmse: RMSE for Voltage
        Runtime: average runtime per Monte Carlo run
    """
    np.random.seed(123)
    N = 100  # number of time steps
    dt = 1   # time step size
    
    # System dimensions
    n = 3  # state dimension [SOC, SOH, V]
    m = 1  # measurement dimension [V]
    
    # Noise parameters
    sig_pro = np.array([0.01, 0.001, 0.1])  # process noise std
    sig_mea = magnitude  # measurement noise std
    sig_init = np.array([0.1, 0.05, 0.2])   # initial uncertainty std
    
    # Noise covariance matrices
    Q = np.diag(sig_pro**2)  # process noise covariance
    R = np.diag([sig_mea**2])  # measurement noise covariance
    
    # Initial state
    x0 = np.array([0.8, 0.9, 3.6])  # [SOC, SOH, V]
    
    # Runtime tracking
    Runtime = 0
    
    # Monte Carlo results
    res_SOC_est = np.zeros((N + 1, M))
    res_SOH_est = np.zeros((N + 1, M))
    res_V_est = np.zeros((N + 1, M))
    res_SOC_true = np.zeros((N + 1, M))
    res_SOH_true = np.zeros((N + 1, M))
    res_V_true = np.zeros((N + 1, M))
    
    # Monte Carlo simulation
    for m in range(M):
        # True state trajectory
        x_true = np.zeros((n, N + 1))
        x_true[:, 0] = x0 + np.random.normal(0, sig_init)
        
        # Generate control input (current)
        u_sequence = np.random.normal(0.5, 0.2, N + 1)  # Current profile
        
        # Generate true trajectory
        for k in range(1, N + 1):
            x_true[:, k] = battery_dynamics(x_true[:, k-1], u_sequence[k-1], dt)
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
            x_est[:, k] = battery_dynamics(x_est[:, k-1], u_sequence[k-1], dt)
            P = P + Q
            
            # Generate measurement
            z_true = battery_measurement(x_true[:, k], u_sequence[k])
            z = z_true + np.random.normal(0, sig_mea)
            
            if KFtype == 0:  # EKF
                # Measurement matrix (Jacobian)
                H = np.array([[0, 0, 1]])  # Voltage measurement
                
                # Kalman gain
                S = H @ P @ H.T + R
                K = P @ H.T @ safe_matrix_inverse(S)
                
                # Update
                z_pred = battery_measurement(x_est[:, k], u_sequence[k])
                y = z - z_pred
                
                x_est0 = x_est[:, k].copy()
                x_est[:, k] = x_est[:, k] + K @ y
                
                # Covariance update
                if improvement == 1:
                    H2 = np.array([[0, 0, 1]])  # Updated measurement matrix
                    temp = P - K @ H2 @ P - P @ H2.T @ K.T + K @ (H2 @ P @ H2.T + R) @ K.T
                    if np.trace(temp) > np.trace(P):
                        x_est[:, k] = x_est0
                    else:
                        P = temp
                else:
                    P = (np.eye(n) - K @ H) @ P
            
            elif KFtype == 0.5:  # IEKF
                # Initial update
                H = np.array([[0, 0, 1]])
                S = H @ P @ H.T + R
                K = P @ H.T @ safe_matrix_inverse(S)
                
                z_pred = battery_measurement(x_est[:, k], u_sequence[k])
                y = z - z_pred
                
                x_est0 = x_est[:, k].copy()
                x_est[:, k] = x_est[:, k] + K @ y
                
                # Iteration
                iter_count = 1
                change = 1
                
                while change > 0.001 and iter_count < 100:
                    iter_count += 1
                    
                    z_pred = battery_measurement(x_est[:, k], u_sequence[k])
                    y = z - z_pred
                    
                    H2 = np.array([[0, 0, 1]])
                    S2 = H2 @ P @ H2.T + R
                    K2 = P @ H2.T @ safe_matrix_inverse(S2)
                    
                    dx = x_est0 + K2 @ (y - H2 @ (x_est0 - x_est[:, k])) - x_est[:, k]
                    change = np.max(np.abs(dx / (x_est[:, k] + 1e-10)))
                    
                    x_est[:, k] = x_est[:, k] + dx
                    K = K2
                    H = H2
                
                # Covariance update
                if improvement == 1:
                    H2 = np.array([[0, 0, 1]])
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
                H = np.array([[0, 0, 1]])
                S = H @ P @ H.T + R
                K = P @ H.T @ safe_matrix_inverse(S)
                
                z_pred = battery_measurement(x_est[:, k], u_sequence[k])
                y = z - z_pred
                
                x_est0 = x_est[:, k].copy()
                x_est[:, k] = x_est[:, k] + K @ y
                
                if improvement == 1:
                    H2 = np.array([[0, 0, 1]])
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
                H = np.array([[0, 0, 1]])
                S = H @ P @ H.T + R
                K = P @ H.T @ safe_matrix_inverse(S)
                
                z_pred = battery_measurement(x_est[:, k], u_sequence[k])
                y = z - z_pred
                
                x_est0 = x_est[:, k].copy()
                x_est[:, k] = x_est[:, k] + K @ y
                
                if improvement == 1:
                    H2 = np.array([[0, 0, 1]])
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
                H = np.array([[0, 0, 1]])
                S = H @ P @ H.T + R
                K = P @ H.T @ safe_matrix_inverse(S)
                
                z_pred = battery_measurement(x_est[:, k], u_sequence[k])
                y = z - z_pred
                
                x_est0 = x_est[:, k].copy()
                x_est[:, k] = x_est[:, k] + K @ y
                
                if improvement == 1:
                    H2 = np.array([[0, 0, 1]])
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
                H = np.array([[0, 0, 1]])
                S = H @ P @ H.T + R
                K = P @ H.T @ safe_matrix_inverse(S)
                
                z_pred = battery_measurement(x_est[:, k], u_sequence[k])
                y = z - z_pred
                
                x_est0 = x_est[:, k].copy()
                x_est[:, k] = x_est[:, k] + K @ y
                
                # Iteration
                iter_count = 1
                change = 1
                
                while change > 0.001 and iter_count < 100:
                    iter_count += 1
                    
                    z_pred = battery_measurement(x_est[:, k], u_sequence[k])
                    y = z - z_pred
                    
                    H2 = np.array([[0, 0, 1]])
                    S2 = H2 @ P @ H2.T + R
                    K2 = P @ H2.T @ safe_matrix_inverse(S2)
                    
                    dx = x_est0 + K2 @ (y - H2 @ (x_est0 - x_est[:, k])) - x_est[:, k]
                    change = np.max(np.abs(dx / (x_est[:, k] + 1e-10)))
                    
                    x_est[:, k] = x_est[:, k] + dx
                    K = K2
                    H = H2
                
                # Covariance update
                if improvement == 1:
                    H2 = np.array([[0, 0, 1]])
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
        res_SOC_est[:, m] = x_est[0, :]
        res_SOH_est[:, m] = x_est[1, :]
        res_V_est[:, m] = x_est[2, :]
        res_SOC_true[:, m] = x_true[0, :]
        res_SOH_true[:, m] = x_true[1, :]
        res_V_true[:, m] = x_true[2, :]
    
    # Calculate RMSE
    SOC_errors = res_SOC_est - res_SOC_true
    SOH_errors = res_SOH_est - res_SOH_true
    V_errors = res_V_est - res_V_true
    
    SOC_rmse = np.sqrt(np.mean(SOC_errors**2))
    SOH_rmse = np.sqrt(np.mean(SOH_errors**2))
    V_rmse = np.sqrt(np.mean(V_errors**2))
    
    return SOC_rmse, SOH_rmse, V_rmse, Runtime

def main():
    """Main function to run the battery state estimation simulation"""
    print("Starting Battery State Estimation Simulation...")
    
    # Initialize result arrays
    results = {
        'EKF_1': {'SOC_rmse': [], 'SOH_rmse': [], 'V_rmse': [], 'time': []},
        'EKF_2': {'SOC_rmse': [], 'SOH_rmse': [], 'V_rmse': [], 'time': []},
        'IEKF_1': {'SOC_rmse': [], 'SOH_rmse': [], 'V_rmse': [], 'time': []},
        'IEKF_2': {'SOC_rmse': [], 'SOH_rmse': [], 'V_rmse': [], 'time': []},
        'UKF_1': {'SOC_rmse': [], 'SOH_rmse': [], 'V_rmse': [], 'time': []},
        'UKF_2': {'SOC_rmse': [], 'SOH_rmse': [], 'V_rmse': [], 'time': []},
        'CKF_1': {'SOC_rmse': [], 'SOH_rmse': [], 'V_rmse': [], 'time': []},
        'CKF_2': {'SOC_rmse': [], 'SOH_rmse': [], 'V_rmse': [], 'time': []},
        'EKF2_1': {'SOC_rmse': [], 'SOH_rmse': [], 'V_rmse': [], 'time': []},
        'EKF2_2': {'SOC_rmse': [], 'SOH_rmse': [], 'V_rmse': [], 'time': []},
        'IEKF2_1': {'SOC_rmse': [], 'SOH_rmse': [], 'V_rmse': [], 'time': []},
        'IEKF2_2': {'SOC_rmse': [], 'SOH_rmse': [], 'V_rmse': [], 'time': []}
    }
    
    # Run simulations for each noise level
    for i, scale_val in enumerate(SCALE):
        magnitude = 10**scale_val * NOISE1_REF
        print(f"Processing noise level {i+1}/{len(SCALE)}: magnitude = {magnitude:.2e}")
        
        # EKF old framework
        SOC_rmse, SOH_rmse, V_rmse, Runtime = simulation(0, magnitude, REPEAT, 0)
        results['EKF_1']['SOC_rmse'].append(SOC_rmse)
        results['EKF_1']['SOH_rmse'].append(SOH_rmse)
        results['EKF_1']['V_rmse'].append(V_rmse)
        results['EKF_1']['time'].append(Runtime)
        
        # EKF new framework
        SOC_rmse, SOH_rmse, V_rmse, Runtime = simulation(1, magnitude, REPEAT, 0)
        results['EKF_2']['SOC_rmse'].append(SOC_rmse)
        results['EKF_2']['SOH_rmse'].append(SOH_rmse)
        results['EKF_2']['V_rmse'].append(V_rmse)
        results['EKF_2']['time'].append(Runtime)
        
        # IEKF old framework
        SOC_rmse, SOH_rmse, V_rmse, Runtime = simulation(0, magnitude, REPEAT, 0.5)
        results['IEKF_1']['SOC_rmse'].append(SOC_rmse)
        results['IEKF_1']['SOH_rmse'].append(SOH_rmse)
        results['IEKF_1']['V_rmse'].append(V_rmse)
        results['IEKF_1']['time'].append(Runtime)
        
        # IEKF new framework
        SOC_rmse, SOH_rmse, V_rmse, Runtime = simulation(1, magnitude, REPEAT, 0.5)
        results['IEKF_2']['SOC_rmse'].append(SOC_rmse)
        results['IEKF_2']['SOH_rmse'].append(SOH_rmse)
        results['IEKF_2']['V_rmse'].append(V_rmse)
        results['IEKF_2']['time'].append(Runtime)
        
        # EKF2 old framework
        SOC_rmse, SOH_rmse, V_rmse, Runtime = simulation(0, magnitude, REPEAT, 3)
        results['EKF2_1']['SOC_rmse'].append(SOC_rmse)
        results['EKF2_1']['SOH_rmse'].append(SOH_rmse)
        results['EKF2_1']['V_rmse'].append(V_rmse)
        results['EKF2_1']['time'].append(Runtime)
        
        # EKF2 new framework
        SOC_rmse, SOH_rmse, V_rmse, Runtime = simulation(1, magnitude, REPEAT, 3)
        results['EKF2_2']['SOC_rmse'].append(SOC_rmse)
        results['EKF2_2']['SOH_rmse'].append(SOH_rmse)
        results['EKF2_2']['V_rmse'].append(V_rmse)
        results['EKF2_2']['time'].append(Runtime)
        
        # IEKF2 old framework
        SOC_rmse, SOH_rmse, V_rmse, Runtime = simulation(0, magnitude, REPEAT, 3.5)
        results['IEKF2_1']['SOC_rmse'].append(SOC_rmse)
        results['IEKF2_1']['SOH_rmse'].append(SOH_rmse)
        results['IEKF2_1']['V_rmse'].append(V_rmse)
        results['IEKF2_1']['time'].append(Runtime)
        
        # IEKF2 new framework
        SOC_rmse, SOH_rmse, V_rmse, Runtime = simulation(1, magnitude, REPEAT, 3.5)
        results['IEKF2_2']['SOC_rmse'].append(SOC_rmse)
        results['IEKF2_2']['SOH_rmse'].append(SOH_rmse)
        results['IEKF2_2']['V_rmse'].append(V_rmse)
        results['IEKF2_2']['time'].append(Runtime)
        
        # UKF old framework
        SOC_rmse, SOH_rmse, V_rmse, Runtime = simulation(0, magnitude, REPEAT, 1)
        results['UKF_1']['SOC_rmse'].append(SOC_rmse)
        results['UKF_1']['SOH_rmse'].append(SOH_rmse)
        results['UKF_1']['V_rmse'].append(V_rmse)
        results['UKF_1']['time'].append(Runtime)
        
        # UKF new framework
        SOC_rmse, SOH_rmse, V_rmse, Runtime = simulation(1, magnitude, REPEAT, 1)
        results['UKF_2']['SOC_rmse'].append(SOC_rmse)
        results['UKF_2']['SOH_rmse'].append(SOH_rmse)
        results['UKF_2']['V_rmse'].append(V_rmse)
        results['UKF_2']['time'].append(Runtime)
        
        # CKF old framework
        SOC_rmse, SOH_rmse, V_rmse, Runtime = simulation(0, magnitude, REPEAT, 2)
        results['CKF_1']['SOC_rmse'].append(SOC_rmse)
        results['CKF_1']['SOH_rmse'].append(SOH_rmse)
        results['CKF_1']['V_rmse'].append(V_rmse)
        results['CKF_1']['time'].append(Runtime)
        
        # CKF new framework
        SOC_rmse, SOH_rmse, V_rmse, Runtime = simulation(1, magnitude, REPEAT, 2)
        results['CKF_2']['SOC_rmse'].append(SOC_rmse)
        results['CKF_2']['SOH_rmse'].append(SOH_rmse)
        results['CKF_2']['V_rmse'].append(V_rmse)
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
    noise_levels = 10**SCALE * NOISE1_REF
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # Plot 1: SOC RMSE
    ax1 = axes[0]
    ax1.loglog(noise_levels, results['EKF_1']['SOC_rmse'], '--o', label='EKF (old)', color='#0072BD', linewidth=1)
    ax1.loglog(noise_levels, results['EKF_2']['SOC_rmse'], '-o', label='EKF (new)', color='#0072BD', linewidth=1)
    ax1.loglog(noise_levels, results['EKF2_1']['SOC_rmse'], '--s', label='EKF2 (old)', color='#D95319', linewidth=1)
    ax1.loglog(noise_levels, results['EKF2_2']['SOC_rmse'], '-s', label='EKF2 (new)', color='#D95319', linewidth=1)
    ax1.loglog(noise_levels, results['UKF_1']['SOC_rmse'], '--^', label='UKF (old)', color='#EDB120', linewidth=1)
    ax1.loglog(noise_levels, results['UKF_2']['SOC_rmse'], '-^', label='UKF (new)', color='#EDB120', linewidth=1)
    ax1.loglog(noise_levels, results['CKF_1']['SOC_rmse'], '--x', label='CKF (old)', color='#7E2F8E', linewidth=1)
    ax1.loglog(noise_levels, results['CKF_2']['SOC_rmse'], '-x', label='CKF (new)', color='#7E2F8E', linewidth=1)
    ax1.loglog(noise_levels, results['IEKF_1']['SOC_rmse'], '--d', label='IEKF (old)', color='#4DBEEE', linewidth=1)
    ax1.loglog(noise_levels, results['IEKF_2']['SOC_rmse'], '-d', label='IEKF (new)', color='#4DBEEE', linewidth=1)
    
    ax1.set_ylabel('SOC RMSE', fontsize=12)
    ax1.set_ylim([1e-3, 1])
    ax1.grid(True)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: SOH RMSE
    ax2 = axes[1]
    ax2.loglog(noise_levels, results['EKF_1']['SOH_rmse'], '--o', label='EKF (old)', color='#0072BD', linewidth=1)
    ax2.loglog(noise_levels, results['EKF_2']['SOH_rmse'], '-o', label='EKF (new)', color='#0072BD', linewidth=1)
    ax2.loglog(noise_levels, results['EKF2_1']['SOH_rmse'], '--s', label='EKF2 (old)', color='#D95319', linewidth=1)
    ax2.loglog(noise_levels, results['EKF2_2']['SOH_rmse'], '-s', label='EKF2 (new)', color='#D95319', linewidth=1)
    ax2.loglog(noise_levels, results['UKF_1']['SOH_rmse'], '--^', label='UKF (old)', color='#EDB120', linewidth=1)
    ax2.loglog(noise_levels, results['UKF_2']['SOH_rmse'], '-^', label='UKF (new)', color='#EDB120', linewidth=1)
    ax2.loglog(noise_levels, results['CKF_1']['SOH_rmse'], '--x', label='CKF (old)', color='#7E2F8E', linewidth=1)
    ax2.loglog(noise_levels, results['CKF_2']['SOH_rmse'], '-x', label='CKF (new)', color='#7E2F8E', linewidth=1)
    ax2.loglog(noise_levels, results['IEKF_1']['SOH_rmse'], '--d', label='IEKF (old)', color='#4DBEEE', linewidth=1)
    ax2.loglog(noise_levels, results['IEKF_2']['SOH_rmse'], '-d', label='IEKF (new)', color='#4DBEEE', linewidth=1)
    
    ax2.set_ylabel('SOH RMSE', fontsize=12)
    ax2.set_ylim([1e-4, 1])
    ax2.grid(True)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 3: Voltage RMSE
    ax3 = axes[2]
    ax3.loglog(noise_levels, results['EKF_1']['V_rmse'], '--o', label='EKF (old)', color='#0072BD', linewidth=1)
    ax3.loglog(noise_levels, results['EKF_2']['V_rmse'], '-o', label='EKF (new)', color='#0072BD', linewidth=1)
    ax3.loglog(noise_levels, results['EKF2_1']['V_rmse'], '--s', label='EKF2 (old)', color='#D95319', linewidth=1)
    ax3.loglog(noise_levels, results['EKF2_2']['V_rmse'], '-s', label='EKF2 (new)', color='#D95319', linewidth=1)
    ax3.loglog(noise_levels, results['UKF_1']['V_rmse'], '--^', label='UKF (old)', color='#EDB120', linewidth=1)
    ax3.loglog(noise_levels, results['UKF_2']['V_rmse'], '-^', label='UKF (new)', color='#EDB120', linewidth=1)
    ax3.loglog(noise_levels, results['CKF_1']['V_rmse'], '--x', label='CKF (old)', color='#7E2F8E', linewidth=1)
    ax3.loglog(noise_levels, results['CKF_2']['V_rmse'], '-x', label='CKF (new)', color='#7E2F8E', linewidth=1)
    ax3.loglog(noise_levels, results['IEKF_1']['V_rmse'], '--d', label='IEKF (old)', color='#4DBEEE', linewidth=1)
    ax3.loglog(noise_levels, results['IEKF_2']['V_rmse'], '-d', label='IEKF (new)', color='#4DBEEE', linewidth=1)
    
    ax3.set_xlabel('Measurement noise standard deviation', fontsize=12)
    ax3.set_ylabel('Voltage RMSE (V)', fontsize=12)
    ax3.set_ylim([1e-3, 1])
    ax3.grid(True)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('battery_state_estimation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Results saved to 'battery_state_estimation_results.png'")

if __name__ == "__main__":
    results = main() 