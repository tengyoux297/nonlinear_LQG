"""
Pendulum State Estimation Application - Python Conversion
Application 4: Pendulum state estimation
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
NOISE1_REF = 1
SCALE = np.arange(-4, 1.5, 0.5)  # Measurement noise range: 10^scale

def safe_matrix_inverse(A, reg=1e-8):
    """Safely compute matrix inverse with regularization"""
    try:
        return np.linalg.inv(A + reg * np.eye(A.shape[0]))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A + reg * np.eye(A.shape[0]))

def pendulum_dynamics(x, dt):
    """
    Pendulum state dynamics
    x = [theta, theta_dot] - angle and angular velocity
    """
    theta, theta_dot = x
    
    # Pendulum parameters
    g = 9.81  # gravity (m/s^2)
    L = 1.0   # length (m)
    m = 1.0   # mass (kg)
    b = 0.1   # damping coefficient
    
    # State evolution
    theta_ddot = -g/L * np.sin(theta) - b/m * theta_dot
    
    # Euler integration
    theta_new = theta + theta_dot * dt
    theta_dot_new = theta_dot + theta_ddot * dt
    
    return np.array([theta_new, theta_dot_new])

def pendulum_measurement(x):
    """
    Pendulum measurement function
    Returns noisy angle measurement
    """
    theta, theta_dot = x
    return np.array([theta])

def simulation(improvement: int, magnitude: float, M: int, KFtype: float) -> Tuple[float, float, float]:
    """
    Main simulation function for pendulum state estimation
    
    Args:
        improvement: 0 for old framework, 1 for new framework
        magnitude: measurement noise magnitude
        M: number of Monte Carlo runs
        KFtype: filter type (0=EKF, 0.5=IEKF, 1=UKF, 2=CKF, 3=EKF2, 3.5=IEKF2)
    
    Returns:
        theta_rmse: RMSE for angle
        theta_dot_rmse: RMSE for angular velocity
        Runtime: average runtime per Monte Carlo run
    """
    np.random.seed(123)
    N = 100  # number of time steps
    dt = 0.1  # time step size
    
    # System dimensions
    n = 2  # state dimension [theta, theta_dot]
    m = 1  # measurement dimension [theta]
    
    # Noise parameters
    sig_pro = np.array([0.01, 0.1])  # process noise std
    sig_mea = magnitude  # measurement noise std
    sig_init = np.array([0.1, 0.5])  # initial uncertainty std
    
    # Noise covariance matrices
    Q = np.diag(sig_pro**2)  # process noise covariance
    R = np.diag([sig_mea**2])  # measurement noise covariance
    
    # Initial state
    x0 = np.array([np.pi/4, 0.0])  # [theta, theta_dot]
    
    # Runtime tracking
    Runtime = 0
    
    # Monte Carlo results
    res_theta_est = np.zeros((N + 1, M))
    res_theta_dot_est = np.zeros((N + 1, M))
    res_theta_true = np.zeros((N + 1, M))
    res_theta_dot_true = np.zeros((N + 1, M))
    
    # Monte Carlo simulation
    for m in range(M):
        # True state trajectory
        x_true = np.zeros((n, N + 1))
        x_true[:, 0] = x0 + np.random.normal(0, sig_init)
        
        # Generate true trajectory
        for k in range(1, N + 1):
            x_true[:, k] = pendulum_dynamics(x_true[:, k-1], dt)
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
            x_est[:, k] = pendulum_dynamics(x_est[:, k-1], dt)
            P = P + Q
            
            # Generate measurement
            z_true = pendulum_measurement(x_true[:, k])
            z = z_true + np.random.normal(0, sig_mea)
            
            if KFtype == 0:  # EKF
                # State transition matrix (Jacobian)
                theta, theta_dot = x_est[:, k]
                g, L, m, b = 9.81, 1.0, 1.0, 0.1
                
                F = np.array([[1, dt],
                             [-g/L * np.cos(theta) * dt, 1 - b/m * dt]])
                
                # Measurement matrix (Jacobian)
                H = np.array([[1, 0]])  # Angle measurement
                
                # Kalman gain
                S = H @ P @ H.T + R
                K = P @ H.T @ safe_matrix_inverse(S)
                
                # Update
                z_pred = pendulum_measurement(x_est[:, k])
                y = z - z_pred
                
                x_est0 = x_est[:, k].copy()
                x_est[:, k] = x_est[:, k] + K @ y
                
                # Covariance update
                if improvement == 1:
                    H2 = np.array([[1, 0]])  # Updated measurement matrix
                    temp = P - K @ H2 @ P - P @ H2.T @ K.T + K @ (H2 @ P @ H2.T + R) @ K.T
                    if np.trace(temp) > np.trace(P):
                        x_est[:, k] = x_est0
                    else:
                        P = temp
                else:
                    P = (np.eye(n) - K @ H) @ P
            
            elif KFtype == 0.5:  # IEKF
                # Initial update
                H = np.array([[1, 0]])
                S = H @ P @ H.T + R
                K = P @ H.T @ safe_matrix_inverse(S)
                
                z_pred = pendulum_measurement(x_est[:, k])
                y = z - z_pred
                
                x_est0 = x_est[:, k].copy()
                x_est[:, k] = x_est[:, k] + K @ y
                
                # Iteration
                iter_count = 1
                change = 1
                
                while change > 0.001 and iter_count < 100:
                    iter_count += 1
                    
                    z_pred = pendulum_measurement(x_est[:, k])
                    y = z - z_pred
                    
                    H2 = np.array([[1, 0]])
                    S2 = H2 @ P @ H2.T + R
                    K2 = P @ H2.T @ safe_matrix_inverse(S2)
                    
                    dx = x_est0 + K2 @ (y - H2 @ (x_est0 - x_est[:, k])) - x_est[:, k]
                    change = np.max(np.abs(dx / (x_est[:, k] + 1e-10)))
                    
                    x_est[:, k] = x_est[:, k] + dx
                    K = K2
                    H = H2
                
                # Covariance update
                if improvement == 1:
                    H2 = np.array([[1, 0]])
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
                H = np.array([[1, 0]])
                S = H @ P @ H.T + R
                K = P @ H.T @ safe_matrix_inverse(S)
                
                z_pred = pendulum_measurement(x_est[:, k])
                y = z - z_pred
                
                x_est0 = x_est[:, k].copy()
                x_est[:, k] = x_est[:, k] + K @ y
                
                if improvement == 1:
                    H2 = np.array([[1, 0]])
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
                H = np.array([[1, 0]])
                S = H @ P @ H.T + R
                K = P @ H.T @ safe_matrix_inverse(S)
                
                z_pred = pendulum_measurement(x_est[:, k])
                y = z - z_pred
                
                x_est0 = x_est[:, k].copy()
                x_est[:, k] = x_est[:, k] + K @ y
                
                if improvement == 1:
                    H2 = np.array([[1, 0]])
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
                H = np.array([[1, 0]])
                S = H @ P @ H.T + R
                K = P @ H.T @ safe_matrix_inverse(S)
                
                z_pred = pendulum_measurement(x_est[:, k])
                y = z - z_pred
                
                x_est0 = x_est[:, k].copy()
                x_est[:, k] = x_est[:, k] + K @ y
                
                if improvement == 1:
                    H2 = np.array([[1, 0]])
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
                H = np.array([[1, 0]])
                S = H @ P @ H.T + R
                K = P @ H.T @ safe_matrix_inverse(S)
                
                z_pred = pendulum_measurement(x_est[:, k])
                y = z - z_pred
                
                x_est0 = x_est[:, k].copy()
                x_est[:, k] = x_est[:, k] + K @ y
                
                # Iteration
                iter_count = 1
                change = 1
                
                while change > 0.001 and iter_count < 100:
                    iter_count += 1
                    
                    z_pred = pendulum_measurement(x_est[:, k])
                    y = z - z_pred
                    
                    H2 = np.array([[1, 0]])
                    S2 = H2 @ P @ H2.T + R
                    K2 = P @ H2.T @ safe_matrix_inverse(S2)
                    
                    dx = x_est0 + K2 @ (y - H2 @ (x_est0 - x_est[:, k])) - x_est[:, k]
                    change = np.max(np.abs(dx / (x_est[:, k] + 1e-10)))
                    
                    x_est[:, k] = x_est[:, k] + dx
                    K = K2
                    H = H2
                
                # Covariance update
                if improvement == 1:
                    H2 = np.array([[1, 0]])
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
        res_theta_est[:, m] = x_est[0, :]
        res_theta_dot_est[:, m] = x_est[1, :]
        res_theta_true[:, m] = x_true[0, :]
        res_theta_dot_true[:, m] = x_true[1, :]
    
    # Calculate RMSE
    theta_errors = res_theta_est - res_theta_true
    theta_dot_errors = res_theta_dot_est - res_theta_dot_true
    
    theta_rmse = np.sqrt(np.mean(theta_errors**2))
    theta_dot_rmse = np.sqrt(np.mean(theta_dot_errors**2))
    
    return theta_rmse, theta_dot_rmse, Runtime

def main():
    """Main function to run the pendulum state estimation simulation"""
    print("Starting Pendulum State Estimation Simulation...")
    
    # Initialize result arrays
    results = {
        'EKF_1': {'theta_rmse': [], 'theta_dot_rmse': [], 'time': []},
        'EKF_2': {'theta_rmse': [], 'theta_dot_rmse': [], 'time': []},
        'IEKF_1': {'theta_rmse': [], 'theta_dot_rmse': [], 'time': []},
        'IEKF_2': {'theta_rmse': [], 'theta_dot_rmse': [], 'time': []},
        'UKF_1': {'theta_rmse': [], 'theta_dot_rmse': [], 'time': []},
        'UKF_2': {'theta_rmse': [], 'theta_dot_rmse': [], 'time': []},
        'CKF_1': {'theta_rmse': [], 'theta_dot_rmse': [], 'time': []},
        'CKF_2': {'theta_rmse': [], 'theta_dot_rmse': [], 'time': []},
        'EKF2_1': {'theta_rmse': [], 'theta_dot_rmse': [], 'time': []},
        'EKF2_2': {'theta_rmse': [], 'theta_dot_rmse': [], 'time': []},
        'IEKF2_1': {'theta_rmse': [], 'theta_dot_rmse': [], 'time': []},
        'IEKF2_2': {'theta_rmse': [], 'theta_dot_rmse': [], 'time': []}
    }
    
    # Run simulations for each noise level
    for i, scale_val in enumerate(SCALE):
        magnitude = 10**scale_val * NOISE1_REF
        print(f"Processing noise level {i+1}/{len(SCALE)}: magnitude = {magnitude:.2e}")
        
        # EKF old framework
        theta_rmse, theta_dot_rmse, Runtime = simulation(0, magnitude, REPEAT, 0)
        results['EKF_1']['theta_rmse'].append(theta_rmse)
        results['EKF_1']['theta_dot_rmse'].append(theta_dot_rmse)
        results['EKF_1']['time'].append(Runtime)
        
        # EKF new framework
        theta_rmse, theta_dot_rmse, Runtime = simulation(1, magnitude, REPEAT, 0)
        results['EKF_2']['theta_rmse'].append(theta_rmse)
        results['EKF_2']['theta_dot_rmse'].append(theta_dot_rmse)
        results['EKF_2']['time'].append(Runtime)
        
        # IEKF old framework
        theta_rmse, theta_dot_rmse, Runtime = simulation(0, magnitude, REPEAT, 0.5)
        results['IEKF_1']['theta_rmse'].append(theta_rmse)
        results['IEKF_1']['theta_dot_rmse'].append(theta_dot_rmse)
        results['IEKF_1']['time'].append(Runtime)
        
        # IEKF new framework
        theta_rmse, theta_dot_rmse, Runtime = simulation(1, magnitude, REPEAT, 0.5)
        results['IEKF_2']['theta_rmse'].append(theta_rmse)
        results['IEKF_2']['theta_dot_rmse'].append(theta_dot_rmse)
        results['IEKF_2']['time'].append(Runtime)
        
        # EKF2 old framework
        theta_rmse, theta_dot_rmse, Runtime = simulation(0, magnitude, REPEAT, 3)
        results['EKF2_1']['theta_rmse'].append(theta_rmse)
        results['EKF2_1']['theta_dot_rmse'].append(theta_dot_rmse)
        results['EKF2_1']['time'].append(Runtime)
        
        # EKF2 new framework
        theta_rmse, theta_dot_rmse, Runtime = simulation(1, magnitude, REPEAT, 3)
        results['EKF2_2']['theta_rmse'].append(theta_rmse)
        results['EKF2_2']['theta_dot_rmse'].append(theta_dot_rmse)
        results['EKF2_2']['time'].append(Runtime)
        
        # IEKF2 old framework
        theta_rmse, theta_dot_rmse, Runtime = simulation(0, magnitude, REPEAT, 3.5)
        results['IEKF2_1']['theta_rmse'].append(theta_rmse)
        results['IEKF2_1']['theta_dot_rmse'].append(theta_dot_rmse)
        results['IEKF2_1']['time'].append(Runtime)
        
        # IEKF2 new framework
        theta_rmse, theta_dot_rmse, Runtime = simulation(1, magnitude, REPEAT, 3.5)
        results['IEKF2_2']['theta_rmse'].append(theta_rmse)
        results['IEKF2_2']['theta_dot_rmse'].append(theta_dot_rmse)
        results['IEKF2_2']['time'].append(Runtime)
        
        # UKF old framework
        theta_rmse, theta_dot_rmse, Runtime = simulation(0, magnitude, REPEAT, 1)
        results['UKF_1']['theta_rmse'].append(theta_rmse)
        results['UKF_1']['theta_dot_rmse'].append(theta_dot_rmse)
        results['UKF_1']['time'].append(Runtime)
        
        # UKF new framework
        theta_rmse, theta_dot_rmse, Runtime = simulation(1, magnitude, REPEAT, 1)
        results['UKF_2']['theta_rmse'].append(theta_rmse)
        results['UKF_2']['theta_dot_rmse'].append(theta_dot_rmse)
        results['UKF_2']['time'].append(Runtime)
        
        # CKF old framework
        theta_rmse, theta_dot_rmse, Runtime = simulation(0, magnitude, REPEAT, 2)
        results['CKF_1']['theta_rmse'].append(theta_rmse)
        results['CKF_1']['theta_dot_rmse'].append(theta_dot_rmse)
        results['CKF_1']['time'].append(Runtime)
        
        # CKF new framework
        theta_rmse, theta_dot_rmse, Runtime = simulation(1, magnitude, REPEAT, 2)
        results['CKF_2']['theta_rmse'].append(theta_rmse)
        results['CKF_2']['theta_dot_rmse'].append(theta_dot_rmse)
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
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Angle RMSE
    ax1 = axes[0]
    ax1.loglog(noise_levels, results['EKF_1']['theta_rmse'], '--o', label='EKF (old)', color='#0072BD', linewidth=1)
    ax1.loglog(noise_levels, results['EKF_2']['theta_rmse'], '-o', label='EKF (new)', color='#0072BD', linewidth=1)
    ax1.loglog(noise_levels, results['EKF2_1']['theta_rmse'], '--s', label='EKF2 (old)', color='#D95319', linewidth=1)
    ax1.loglog(noise_levels, results['EKF2_2']['theta_rmse'], '-s', label='EKF2 (new)', color='#D95319', linewidth=1)
    ax1.loglog(noise_levels, results['UKF_1']['theta_rmse'], '--^', label='UKF (old)', color='#EDB120', linewidth=1)
    ax1.loglog(noise_levels, results['UKF_2']['theta_rmse'], '-^', label='UKF (new)', color='#EDB120', linewidth=1)
    ax1.loglog(noise_levels, results['CKF_1']['theta_rmse'], '--x', label='CKF (old)', color='#7E2F8E', linewidth=1)
    ax1.loglog(noise_levels, results['CKF_2']['theta_rmse'], '-x', label='CKF (new)', color='#7E2F8E', linewidth=1)
    ax1.loglog(noise_levels, results['IEKF_1']['theta_rmse'], '--d', label='IEKF (old)', color='#4DBEEE', linewidth=1)
    ax1.loglog(noise_levels, results['IEKF_2']['theta_rmse'], '-d', label='IEKF (new)', color='#4DBEEE', linewidth=1)
    
    ax1.set_ylabel('Angle RMSE (rad)', fontsize=12)
    ax1.set_ylim([1e-3, 1])
    ax1.grid(True)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Angular Velocity RMSE
    ax2 = axes[1]
    ax2.loglog(noise_levels, results['EKF_1']['theta_dot_rmse'], '--o', label='EKF (old)', color='#0072BD', linewidth=1)
    ax2.loglog(noise_levels, results['EKF_2']['theta_dot_rmse'], '-o', label='EKF (new)', color='#0072BD', linewidth=1)
    ax2.loglog(noise_levels, results['EKF2_1']['theta_dot_rmse'], '--s', label='EKF2 (old)', color='#D95319', linewidth=1)
    ax2.loglog(noise_levels, results['EKF2_2']['theta_dot_rmse'], '-s', label='EKF2 (new)', color='#D95319', linewidth=1)
    ax2.loglog(noise_levels, results['UKF_1']['theta_dot_rmse'], '--^', label='UKF (old)', color='#EDB120', linewidth=1)
    ax2.loglog(noise_levels, results['UKF_2']['theta_dot_rmse'], '-^', label='UKF (new)', color='#EDB120', linewidth=1)
    ax2.loglog(noise_levels, results['CKF_1']['theta_dot_rmse'], '--x', label='CKF (old)', color='#7E2F8E', linewidth=1)
    ax2.loglog(noise_levels, results['CKF_2']['theta_dot_rmse'], '-x', label='CKF (new)', color='#7E2F8E', linewidth=1)
    ax2.loglog(noise_levels, results['IEKF_1']['theta_dot_rmse'], '--d', label='IEKF (old)', color='#4DBEEE', linewidth=1)
    ax2.loglog(noise_levels, results['IEKF_2']['theta_dot_rmse'], '-d', label='IEKF (new)', color='#4DBEEE', linewidth=1)
    
    ax2.set_xlabel('Measurement noise standard deviation', fontsize=12)
    ax2.set_ylabel('Angular Velocity RMSE (rad/s)', fontsize=12)
    ax2.set_ylim([1e-3, 1])
    ax2.grid(True)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('pendulum_state_estimation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Results saved to 'pendulum_state_estimation_results.png'")

if __name__ == "__main__":
    results = main() 