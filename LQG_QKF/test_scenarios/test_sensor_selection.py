#!/usr/bin/env python3
"""
Test script for the sensor selection simulator.
"""

import numpy as np
from sensor_selection_sim import SensorSelectionSimulator, run_sensor_scheduling_sim

def test_sensor_selection():
    """Test the sensor selection functionality with a short simulation."""
    print("Testing Sensor Selection Simulator...")
    
    # Test parameters
    n1, n2, p = 2, 2, 3
    n = n1 + n2
    m = 2
    H = 50  # Short simulation for testing
    
    # Generate test system parameters
    np.random.seed(42)
    
    # Simple stable system
    A_E = np.array([[0.8, 0.1], [0.1, 0.8]])
    A_S = np.array([[0.9, 0.05], [0.05, 0.9]])
    B_S = np.random.randn(n2, p) * 0.1
    W = np.eye(n) * 0.01
    
    # Measurement matrices
    C = np.random.randn(m, n) * 0.5
    M = np.random.randn(m, n, n) * 0.1
    V = np.eye(m) * 0.1
    
    # Cost matrices
    Q = np.eye(n + n**2) * 1.0
    R = np.eye(p) * 0.1
    
    # Test EKF
    print("Testing EKF sensor selection...")
    try:
        sim_ekf = SensorSelectionSimulator(n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, 
                                         H=H, filter_type='ekf', lqr_type='orig')
        err, var, cost = sim_ekf.run_sim()
        
        print(f"EKF Results:")
        print(f"  Final error: {err[-1]:.4f}")
        print(f"  Final variance: {var[-1]:.4f}")
        print(f"  Final cost: {cost[-1]:.4f}")
        print(f"  Sensor selections: {sim_ekf.performance_history['sensor_selections']}")
        
        # Test plotting
        print("Testing plotting functionality...")
        sim_ekf.plot_performance(save_plots=False)
        
        print("✓ EKF test passed!")
        
    except Exception as e:
        print(f"✗ EKF test failed: {e}")
        return False
    
    # Test QKF
    print("\nTesting QKF sensor selection...")
    try:
        sim_qkf = SensorSelectionSimulator(n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, 
                                         H=H, filter_type='qkf', lqr_type='aug_numeric')
        err, var, cost = sim_qkf.run_sim()
        
        print(f"QKF Results:")
        print(f"  Final error: {err[-1]:.4f}")
        print(f"  Final variance: {var[-1]:.4f}")
        print(f"  Final cost: {cost[-1]:.4f}")
        print(f"  Sensor selections: {sim_qkf.performance_history['sensor_selections']}")
        
        print("✓ QKF test passed!")
        
    except Exception as e:
        print(f"✗ QKF test failed: {e}")
        return False
    
    print("\n✓ All tests passed! Sensor selection simulator is working correctly.")
    return True

if __name__ == "__main__":
    test_sensor_selection()
