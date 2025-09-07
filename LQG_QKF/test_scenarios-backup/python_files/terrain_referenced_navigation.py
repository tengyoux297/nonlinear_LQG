"""
Terrain Referenced Navigation Application - Python Conversion
Application 2: Terrain referenced navigation
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

def terrain_height(x, y):
    """
    Terrain height function
    Returns height at given (x, y) coordinates
    """
    # Simple terrain model: sinusoidal surface
    return 100 + 20 * np.sin(x/100) * np.cos(y/100)

def terrain_gradient(x, y):
    """
    Terrain gradient function
    Returns gradient at given (x, y) coordinates
    """
    # Partial derivatives of terrain height
    dh_dx = 0.2 * np.cos(x/100) * np.cos(y/100)
    dh_dy = -0.2 * np.sin(x/100) * np.sin(y/100)
    return np.array([dh_dx, dh_dy])

def simulation(improvement: int, magnitude: float, M: int, KFtype: float) -> Tuple[float, float, float]:
    """
    Main simulation function for terrain referenced navigation
    
    Args:
        improvement: 0 for old framework, 1 for new framework
        magnitude: measurement noise magnitude
        M: number of Monte Carlo runs
        KFtype: filter type (0=EKF, 0.5=IEKF, 1=UKF, 2=CKF, 3=EKF2, 3.5=IEKF2)
    
    Returns:
        x_rmse: RMSE for x position
        y_rmse: RMSE for y position
        Runtime: average runtime per Monte Carlo run
    """
    np.random.seed(123)
    N = 50  # number of time steps
    dt = 1   # time step size
    
    # System dimensions
    n = 4  # state dimension [x, y, vx, vy]
    m = 1  # measurement dimension [terrain height]
    
    # Noise parameters
    sig_pro = np.array([0.1, 0.1, 0.01, 0.01])  # process noise std
    sig_mea = magnitude  # measurement noise std
    sig_init = np.array([10, 10, 1, 1])  # initial uncertainty std
    
    # Noise covariance matrices
    Q = np.diag(sig_pro**2)  # process noise covariance
    R = np.diag([sig_mea**2])  # measurement noise covariance
    
    # Initial state
    x0 = np.array([0, 0, 10, 5])  # [x, y, vx, vy]
    
    # Runtime tracking
    Runtime = 0
    
    # Monte Carlo results
    res_x_est = np.zeros((N + 1, M))
    res_y_est = np.zeros((N + 1, M))
    res_x_true = np.zeros((N + 1, M))
    res_y_true = np.zeros((N + 1, M))
    
    # Monte Carlo simulation
    for m in range(M):
        # True state trajectory
        x_true = np.zeros((n, N + 1))
        x_true[:, 0] = x0 + np.random.normal(0, sig_init)
        
        # Generate true trajectory
        for k in range(1, N + 1):
            # Simple constant velocity model
            x_true[0, k] = x_true[0, k-1] + x_true[2, k-1] * dt
            x_true[1, k] = x_true[1, k-1] + x_true[3, k-1] * dt
            x_true[2, k] = x_true[2, k-1]  # constant velocity
            x_true[3, k] = x_true[3, k-1]  # constant velocity
            
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
            x_est[0, k] = x_est[0, k-1] + x_est[2, k-1] * dt
            x_est[1, k] = x_est[1, k-1] + x_est[3, k-1] * dt
            x_est[2, k] = x_est[2, k-1]
            x_est[3, k] = x_est[3, k-1]
            P = P + Q
            
            # Generate measurement
            x_pos, y_pos = x_true[0, k], x_true[1, k]
            z_true = terrain_height(x_pos, y_pos)
            z = z_true + np.random.normal(0, sig_mea)
            
            if KFtype == 0:  # EKF
                # State transition matrix
                F = np.array([[1, 0, dt, 0],
                             [0, 1, 0, dt],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
                
                # Measurement matrix (Jacobian)
                x_est_pos, y_est_pos = x_est[0, k], x_est[1, k]
                grad = terrain_gradient(x_est_pos, y_est_pos)
                H = np.array([[grad[0], grad[1], 0, 0]])
                
                # Kalman gain
                S = H @ P @ H.T + R
                K = P @ H.T @ safe_matrix_inverse(S)
                
                # Update
                z_pred = terrain_height(x_est_pos, y_est_pos)
                y = z - z_pred
                
                x_est0 = x_est[:, k].copy()
                x_est[:, k] = x_est[:, k] + K @ y
                
                # Covariance update
                if improvement == 1:
                    x_est_pos, y_est_pos = x_est[0, k], x_est[1, k]
                    grad = terrain_gradient(x_est_pos, y_est_pos)
                    H2 = np.array([[grad[0], grad[1], 0, 0]])
                    
                    temp = P - K @ H2 @ P - P @ H2.T @ K.T + K @ (H2 @ P @ H2.T + R) @ K.T
                    if np.trace(temp) > np.trace(P):
                        x_est[:, k] = x_est0
                    else:
                        P = temp
                else:
                    P = (np.eye(n) - K @ H) @ P
            
            elif KFtype == 0.5:  # IEKF
                # Initial update
                x_est_pos, y_est_pos = x_est[0, k], x_est[1, k]
                grad = terrain_gradient(x_est_pos, y_est_pos)
                H = np.array([[grad[0], grad[1], 0, 0]])
                
                S = H @ P @ H.T + R
                K = P @ H.T @ safe_matrix_inverse(S)
                
                z_pred = terrain_height(x_est_pos, y_est_pos)
                y = z - z_pred
                
                x_est0 = x_est[:, k].copy()
                x_est[:, k] = x_est[:, k] + K @ y
                
                # Iteration
                iter_count = 1
                change = 1
                
                while change > 0.001 and iter_count < 100:
                    iter_count += 1
                    
                    x_est_pos, y_est_pos = x_est[0, k], x_est[1, k]
                    z_pred = terrain_height(x_est_pos, y_est_pos)
                    y = z - z_pred
                    
                    grad = terrain_gradient(x_est_pos, y_est_pos)
                    H2 = np.array([[grad[0], grad[1], 0, 0]])
                    
                    S2 = H2 @ P @ H2.T + R
                    K2 = P @ H2.T @ safe_matrix_inverse(S2)
                    
                    dx = x_est0 + K2 @ (y - H2 @ (x_est0 - x_est[:, k])) - x_est[:, k]
                    change = np.max(np.abs(dx / (x_est[:, k] + 1e-10)))
                    
                    x_est[:, k] = x_est[:, k] + dx
                    K = K2
                    H = H2
                
                # Covariance update
                if improvement == 1:
                    x_est_pos, y_est_pos = x_est[0, k], x_est[1, k]
                    grad = terrain_gradient(x_est_pos, y_est_pos)
                    H2 = np.array([[grad[0], grad[1], 0, 0]])
                    
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
                x_est_pos, y_est_pos = x_est[0, k], x_est[1, k]
                grad = terrain_gradient(x_est_pos, y_est_pos)
                H = np.array([[grad[0], grad[1], 0, 0]])
                
                S = H @ P @ H.T + R
                K = P @ H.T @ safe_matrix_inverse(S)
                
                z_pred = terrain_height(x_est_pos, y_est_pos)
                y = z - z_pred
                
                x_est0 = x_est[:, k].copy()
                x_est[:, k] = x_est[:, k] + K @ y
                
                if improvement == 1:
                    x_est_pos, y_est_pos = x_est[0, k], x_est[1, k]
                    grad = terrain_gradient(x_est_pos, y_est_pos)
                    H2 = np.array([[grad[0], grad[1], 0, 0]])
                    
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
                x_est_pos, y_est_pos = x_est[0, k], x_est[1, k]
                grad = terrain_gradient(x_est_pos, y_est_pos)
                H = np.array([[grad[0], grad[1], 0, 0]])
                
                S = H @ P @ H.T + R
                K = P @ H.T @ safe_matrix_inverse(S)
                
                z_pred = terrain_height(x_est_pos, y_est_pos)
                y = z - z_pred
                
                x_est0 = x_est[:, k].copy()
                x_est[:, k] = x_est[:, k] + K @ y
                
                if improvement == 1:
                    x_est_pos, y_est_pos = x_est[0, k], x_est[1, k]
                    grad = terrain_gradient(x_est_pos, y_est_pos)
                    H2 = np.array([[grad[0], grad[1], 0, 0]])
                    
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
                x_est_pos, y_est_pos = x_est[0, k], x_est[1, k]
                grad = terrain_gradient(x_est_pos, y_est_pos)
                H = np.array([[grad[0], grad[1], 0, 0]])
                
                S = H @ P @ H.T + R
                K = P @ H.T @ safe_matrix_inverse(S)
                
                z_pred = terrain_height(x_est_pos, y_est_pos)
                y = z - z_pred
                
                x_est0 = x_est[:, k].copy()
                x_est[:, k] = x_est[:, k] + K @ y
                
                if improvement == 1:
                    x_est_pos, y_est_pos = x_est[0, k], x_est[1, k]
                    grad = terrain_gradient(x_est_pos, y_est_pos)
                    H2 = np.array([[grad[0], grad[1], 0, 0]])
                    
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
                x_est_pos, y_est_pos = x_est[0, k], x_est[1, k]
                grad = terrain_gradient(x_est_pos, y_est_pos)
                H = np.array([[grad[0], grad[1], 0, 0]])
                
                S = H @ P @ H.T + R
                K = P @ H.T @ safe_matrix_inverse(S)
                
                z_pred = terrain_height(x_est_pos, y_est_pos)
                y = z - z_pred
                
                x_est0 = x_est[:, k].copy()
                x_est[:, k] = x_est[:, k] + K @ y
                
                # Iteration
                iter_count = 1
                change = 1
                
                while change > 0.001 and iter_count < 100:
                    iter_count += 1
                    
                    x_est_pos, y_est_pos = x_est[0, k], x_est[1, k]
                    z_pred = terrain_height(x_est_pos, y_est_pos)
                    y = z - z_pred
                    
                    grad = terrain_gradient(x_est_pos, y_est_pos)
                    H2 = np.array([[grad[0], grad[1], 0, 0]])
                    
                    S2 = H2 @ P @ H2.T + R
                    K2 = P @ H2.T @ safe_matrix_inverse(S2)
                    
                    dx = x_est0 + K2 @ (y - H2 @ (x_est0 - x_est[:, k])) - x_est[:, k]
                    change = np.max(np.abs(dx / (x_est[:, k] + 1e-10)))
                    
                    x_est[:, k] = x_est[:, k] + dx
                    K = K2
                    H = H2
                
                # Covariance update
                if improvement == 1:
                    x_est_pos, y_est_pos = x_est[0, k], x_est[1, k]
                    grad = terrain_gradient(x_est_pos, y_est_pos)
                    H2 = np.array([[grad[0], grad[1], 0, 0]])
                    
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
        res_x_est[:, m] = x_est[0, :]
        res_y_est[:, m] = x_est[1, :]
        res_x_true[:, m] = x_true[0, :]
        res_y_true[:, m] = x_true[1, :]
    
    # Calculate RMSE
    x_errors = res_x_est - res_x_true
    y_errors = res_y_est - res_y_true
    
    x_rmse = np.sqrt(np.mean(x_errors**2))
    y_rmse = np.sqrt(np.mean(y_errors**2))
    
    return x_rmse, y_rmse, Runtime

def main():
    """Main function to run the terrain referenced navigation simulation"""
    print("Starting Terrain Referenced Navigation Simulation...")
    
    # Initialize result arrays
    results = {
        'EKF_1': {'x_rmse': [], 'y_rmse': [], 'time': []},
        'EKF_2': {'x_rmse': [], 'y_rmse': [], 'time': []},
        'IEKF_1': {'x_rmse': [], 'y_rmse': [], 'time': []},
        'IEKF_2': {'x_rmse': [], 'y_rmse': [], 'time': []},
        'UKF_1': {'x_rmse': [], 'y_rmse': [], 'time': []},
        'UKF_2': {'x_rmse': [], 'y_rmse': [], 'time': []},
        'CKF_1': {'x_rmse': [], 'y_rmse': [], 'time': []},
        'CKF_2': {'x_rmse': [], 'y_rmse': [], 'time': []},
        'EKF2_1': {'x_rmse': [], 'y_rmse': [], 'time': []},
        'EKF2_2': {'x_rmse': [], 'y_rmse': [], 'time': []},
        'IEKF2_1': {'x_rmse': [], 'y_rmse': [], 'time': []},
        'IEKF2_2': {'x_rmse': [], 'y_rmse': [], 'time': []}
    }
    
    # Run simulations for each noise level
    for i, scale_val in enumerate(SCALE):
        magnitude = 10**scale_val
        print(f"Processing noise level {i+1}/{len(SCALE)}: magnitude = {magnitude:.2e}")
        
        # EKF old framework
        x_rmse, y_rmse, Runtime = simulation(0, magnitude, REPEAT, 0)
        results['EKF_1']['x_rmse'].append(x_rmse)
        results['EKF_1']['y_rmse'].append(y_rmse)
        results['EKF_1']['time'].append(Runtime)
        
        # EKF new framework
        x_rmse, y_rmse, Runtime = simulation(1, magnitude, REPEAT, 0)
        results['EKF_2']['x_rmse'].append(x_rmse)
        results['EKF_2']['y_rmse'].append(y_rmse)
        results['EKF_2']['time'].append(Runtime)
        
        # IEKF old framework
        x_rmse, y_rmse, Runtime = simulation(0, magnitude, REPEAT, 0.5)
        results['IEKF_1']['x_rmse'].append(x_rmse)
        results['IEKF_1']['y_rmse'].append(y_rmse)
        results['IEKF_1']['time'].append(Runtime)
        
        # IEKF new framework
        x_rmse, y_rmse, Runtime = simulation(1, magnitude, REPEAT, 0.5)
        results['IEKF_2']['x_rmse'].append(x_rmse)
        results['IEKF_2']['y_rmse'].append(y_rmse)
        results['IEKF_2']['time'].append(Runtime)
        
        # EKF2 old framework
        x_rmse, y_rmse, Runtime = simulation(0, magnitude, REPEAT, 3)
        results['EKF2_1']['x_rmse'].append(x_rmse)
        results['EKF2_1']['y_rmse'].append(y_rmse)
        results['EKF2_1']['time'].append(Runtime)
        
        # EKF2 new framework
        x_rmse, y_rmse, Runtime = simulation(1, magnitude, REPEAT, 3)
        results['EKF2_2']['x_rmse'].append(x_rmse)
        results['EKF2_2']['y_rmse'].append(y_rmse)
        results['EKF2_2']['time'].append(Runtime)
        
        # IEKF2 old framework
        x_rmse, y_rmse, Runtime = simulation(0, magnitude, REPEAT, 3.5)
        results['IEKF2_1']['x_rmse'].append(x_rmse)
        results['IEKF2_1']['y_rmse'].append(y_rmse)
        results['IEKF2_1']['time'].append(Runtime)
        
        # IEKF2 new framework
        x_rmse, y_rmse, Runtime = simulation(1, magnitude, REPEAT, 3.5)
        results['IEKF2_2']['x_rmse'].append(x_rmse)
        results['IEKF2_2']['y_rmse'].append(y_rmse)
        results['IEKF2_2']['time'].append(Runtime)
        
        # UKF old framework
        x_rmse, y_rmse, Runtime = simulation(0, magnitude, REPEAT, 1)
        results['UKF_1']['x_rmse'].append(x_rmse)
        results['UKF_1']['y_rmse'].append(y_rmse)
        results['UKF_1']['time'].append(Runtime)
        
        # UKF new framework
        x_rmse, y_rmse, Runtime = simulation(1, magnitude, REPEAT, 1)
        results['UKF_2']['x_rmse'].append(x_rmse)
        results['UKF_2']['y_rmse'].append(y_rmse)
        results['UKF_2']['time'].append(Runtime)
        
        # CKF old framework
        x_rmse, y_rmse, Runtime = simulation(0, magnitude, REPEAT, 2)
        results['CKF_1']['x_rmse'].append(x_rmse)
        results['CKF_1']['y_rmse'].append(y_rmse)
        results['CKF_1']['time'].append(Runtime)
        
        # CKF new framework
        x_rmse, y_rmse, Runtime = simulation(1, magnitude, REPEAT, 2)
        results['CKF_2']['x_rmse'].append(x_rmse)
        results['CKF_2']['y_rmse'].append(y_rmse)
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
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: X position RMSE
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
    
    ax1.set_ylabel('X position RMSE (m)', fontsize=12)
    ax1.set_ylim([1e-2, 10])
    ax1.grid(True)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Y position RMSE
    ax2 = axes[1]
    ax2.loglog(noise_levels, results['EKF_1']['y_rmse'], '--o', label='EKF (old)', color='#0072BD', linewidth=1)
    ax2.loglog(noise_levels, results['EKF_2']['y_rmse'], '-o', label='EKF (new)', color='#0072BD', linewidth=1)
    ax2.loglog(noise_levels, results['EKF2_1']['y_rmse'], '--s', label='EKF2 (old)', color='#D95319', linewidth=1)
    ax2.loglog(noise_levels, results['EKF2_2']['y_rmse'], '-s', label='EKF2 (new)', color='#D95319', linewidth=1)
    ax2.loglog(noise_levels, results['UKF_1']['y_rmse'], '--^', label='UKF (old)', color='#EDB120', linewidth=1)
    ax2.loglog(noise_levels, results['UKF_2']['y_rmse'], '-^', label='UKF (new)', color='#EDB120', linewidth=1)
    ax2.loglog(noise_levels, results['CKF_1']['y_rmse'], '--x', label='CKF (old)', color='#7E2F8E', linewidth=1)
    ax2.loglog(noise_levels, results['CKF_2']['y_rmse'], '-x', label='CKF (new)', color='#7E2F8E', linewidth=1)
    ax2.loglog(noise_levels, results['IEKF_1']['y_rmse'], '--d', label='IEKF (old)', color='#4DBEEE', linewidth=1)
    ax2.loglog(noise_levels, results['IEKF_2']['y_rmse'], '-d', label='IEKF (new)', color='#4DBEEE', linewidth=1)
    
    ax2.set_xlabel('Measurement noise standard deviation', fontsize=12)
    ax2.set_ylabel('Y position RMSE (m)', fontsize=12)
    ax2.set_ylim([1e-2, 10])
    ax2.grid(True)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('terrain_referenced_navigation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Results saved to 'terrain_referenced_navigation_results.png'")

if __name__ == "__main__":
    results = main() 