"""
Application 4: Pendulum State Estimation with Quadratic Measurement Function
Python conversion of the MATLAB code with exact same functionality
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import cholesky, sqrtm
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Global parameters
REPEAT = 10000  # The number of times that the simulation is repeated at each measurement noise setup
NOISE1_REF = 1
SCALE = np.arange(-4, 1.5, 0.5)  # The range of measurement noise is 10^scale

def safe_matrix_inverse(A, reg=1e-8):
    """Safely compute matrix inverse with regularization"""
    try:
        return np.linalg.inv(A + reg * np.eye(A.shape[0]))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A + reg * np.eye(A.shape[0]))

def simulation(improvement1: int, noise1: float, repeat: int, KF_type: float) -> Tuple[float, float, float]:
    """
    Main simulation function for pendulum state estimation with quadratic measurement
    
    Args:
        improvement1: 0 for old framework, 1 for new framework
        noise1: measurement noise magnitude
        repeat: number of Monte Carlo runs
        KF_type: filter type (0=EKF, 0.5=IEKF, 1=UKF, 2=CKF, 3=EKF2, 3.5=IEKF2)
    
    Returns:
        sitadot_rmse: RMSE for angular velocity
        sita_rmse: RMSE for angular position
        Runtime: average runtime per Monte Carlo run
    """
    # Problem setup
    delta_t = 0.01
    len_pendulum = 1
    mass = 1
    g = 9.8
    Tlimit = 1
    sig_F = 1e-3
    Q = np.array([[(sig_F*delta_t/mass/len_pendulum**2)**2, 0], [0, 0]])
    
    sitadot_rmse = 0
    sita_rmse = 0
    np.random.seed(123)
    Runtime = 0
    
    for iii in range(repeat):
        states_true = np.array([[0], [np.pi/4]])  # [sitadot; sita]
        sig_init = np.array([np.pi/18, np.pi/18])
        Variance = np.array([[sig_init[0]**2, 0], [0, sig_init[1]**2]])
        states_est = states_true + np.random.normal(0, sig_init).reshape(2, 1)
        
        # Generate profile
        Measurementnoise = noise1**2
        t = np.arange(delta_t, Tlimit + delta_t, delta_t)
        v_true = np.zeros(len(t))
        v = v_true + (sig_F*delta_t/mass/len_pendulum**2) * np.random.randn(len(t))
        stepnum = len(t)
        
        # Generate true trajectory
        for i in range(stepnum):
            states = states_true[:, i].reshape(2, 1)
            states = states + np.array([[-g/len_pendulum*np.sin(states[1, 0]) + v_true[i]/mass/len_pendulum**2], 
                                       [states[0, 0]]]) * delta_t
            states_true = np.hstack([states_true, states])
        
        sitadot = states_true[0, :]
        sita = states_true[1, :]
        
        # Quadratic measurement function: y_k = mg cos(θ_k) sin(θ_k) + mlω_k² sin(θ_k)
        Measurements = (mass*g*np.cos(sita) + mass*len_pendulum*sitadot**2) * np.sin(sita)
        Measurements_noisy = Measurements + np.random.randn(stepnum + 1) * noise1
        
        # Kalman filtering
        start_time = time.time()
        
        for i in range(stepnum):
            if KF_type == 0:  # EKF
                states = states_est[:, i].reshape(2, 1)
                F0 = np.array([[1, -g/len_pendulum*np.cos(states[1, 0])*delta_t], 
                               [delta_t, 1]])
                states = states + np.array([[-g/len_pendulum*np.sin(states[1, 0]) + v[i]/mass/len_pendulum**2], 
                                           [states[0, 0]]]) * delta_t
                Variance = F0 @ Variance @ F0.T + Q
                
                # Quadratic measurement function
                residual = Measurements_noisy[i+1] - (mass*g*np.cos(states[1, 0]) + mass*len_pendulum*states[0, 0]**2) * np.sin(states[1, 0])
                
                # Jacobian of measurement function
                H = np.array([[2*mass*len_pendulum*states[0, 0]*np.sin(states[1, 0]), 
                              mass*g*np.cos(2*states[1, 0]) + mass*len_pendulum*states[0, 0]**2*np.cos(states[1, 0])]])
                
                S = H @ Variance @ H.T + Measurementnoise
                K = Variance @ H.T @ safe_matrix_inverse(S)
                states0 = states.copy()
                states = states0 + K * residual
                
                if improvement1 == 1:
                    H2 = np.array([[2*mass*len_pendulum*states[0, 0]*np.sin(states[1, 0]), 
                                   mass*g*np.cos(2*states[1, 0]) + mass*len_pendulum*states[0, 0]**2*np.cos(states[1, 0])]])
                    temp = Variance - K @ H2 @ Variance - Variance @ H2.T @ K.T + K @ (H2 @ Variance @ H2.T + Measurementnoise) @ K.T
                    if np.trace(temp) < np.trace(Variance):
                        Variance = temp
                    else:
                        states = states0
                else:
                    Variance = Variance - K @ H @ Variance
                
                states_est = np.hstack([states_est, states])
                
            elif KF_type == 0.5:  # IEKF
                states = states_est[:, i].reshape(2, 1)
                F0 = np.array([[1, -g/len_pendulum*np.cos(states[1, 0])*delta_t], 
                               [delta_t, 1]])
                states = states + np.array([[-g/len_pendulum*np.sin(states[1, 0]) + v[i]/mass/len_pendulum**2], 
                                           [states[0, 0]]]) * delta_t
                Variance = F0 @ Variance @ F0.T + Q
                
                residual = Measurements_noisy[i+1] - (mass*g*np.cos(states[1, 0]) + mass*len_pendulum*states[0, 0]**2) * np.sin(states[1, 0])
                H = np.array([[2*mass*len_pendulum*states[0, 0]*np.sin(states[1, 0]), 
                              mass*g*np.cos(2*states[1, 0]) + mass*len_pendulum*states[0, 0]**2*np.cos(states[1, 0])]])
                S = H @ Variance @ H.T + Measurementnoise
                K = Variance @ H.T @ safe_matrix_inverse(S)
                states0 = states.copy()
                states = states0 + K * residual
                
                steplennow = np.linalg.norm(K * residual)
                change = 1
                iter_count = 1
                
                while change > 0.001 and iter_count < 1000:
                    iter_count += 1
                    residual = Measurements_noisy[i+1] - (mass*g*np.cos(states[1, 0]) + mass*len_pendulum*states[0, 0]**2) * np.sin(states[1, 0])
                    H2 = np.array([[2*mass*len_pendulum*states[0, 0]*np.sin(states[1, 0]), 
                                   mass*g*np.cos(2*states[1, 0]) + mass*len_pendulum*states[0, 0]**2*np.cos(states[1, 0])]])
                    S2 = H2 @ Variance @ H2.T + Measurementnoise
                    K2 = Variance @ H2.T @ safe_matrix_inverse(S2)
                    dx = states0 + K2 * (residual - H2 @ (states0 - states)) - states
                    steplen_previous = steplennow
                    steplennow = np.linalg.norm(dx)
                    if steplen_previous < steplennow:
                        break
                    else:
                        change = np.max(np.abs(dx / (states + 1e-10)))
                        states = states + dx
                        K = K2
                        H = H2
                
                if improvement1 == 1:
                    H2 = np.array([[2*mass*len_pendulum*states[0, 0]*np.sin(states[1, 0]), 
                                   mass*g*np.cos(2*states[1, 0]) + mass*len_pendulum*states[0, 0]**2*np.cos(states[1, 0])]])
                    temp = Variance - K @ H2 @ Variance - Variance @ H2.T @ K.T + K @ (H2 @ Variance @ H2.T + Measurementnoise) @ K.T
                    if np.trace(temp) < np.trace(Variance):
                        Variance = temp
                    else:
                        states = states0
                else:
                    Variance = Variance - K @ H @ Variance
                
                states_est = np.hstack([states_est, states]) 

            elif KF_type == 1:  # UKF
                L = cholesky(Variance).T
                state = states_est[:, i].reshape(2, 1)
                n = 2
                lambda_param = (1e-6 - 1) * n
                alpha = 1e-3
                beta = 2
                mnum = 1
                states = np.zeros((n, n*2 + 1))
                states[:, 0] = state.flatten()
                
                for ii in range(n):
                    states[:, 1 + ii] = state.flatten() + np.sqrt(lambda_param + n) * L[:, ii]
                    states[:, n + ii + 1] = state.flatten() - np.sqrt(lambda_param + n) * L[:, ii]
                
                for ii in range(2*n + 1):
                    states[:, ii] = states[:, ii] + np.array([-g/len_pendulum*np.sin(states[1, ii]) + v[i]/mass/len_pendulum**2, 
                                                             states[0, ii]]) * delta_t
                    if ii == 0:
                        state = states[:, ii].reshape(2, 1) * lambda_param/(n + lambda_param)
                    else:
                        state = state + states[:, ii].reshape(2, 1)/(n + lambda_param)/2
                
                Variance = (states[:, 0].reshape(2, 1) - state) @ (states[:, 0].reshape(2, 1) - state).T * (lambda_param/(n + lambda_param) + 1 - alpha**2 + beta) + Q
                for ii in range(1, 2*n + 1):
                    Variance = Variance + (state - states[:, ii].reshape(2, 1)) @ (state - states[:, ii].reshape(2, 1)).T/(n + lambda_param)/2
                
                # Obtain measurement
                z = Measurements_noisy[i+1]
                
                # Predict Measurement From Propagated Sigma Points
                L = cholesky(Variance).T
                states[:, 0] = state.flatten()
                for ii in range(n):
                    states[:, 1 + ii] = state.flatten() + np.sqrt(lambda_param + n) * L[:, ii]
                    states[:, n + ii + 1] = state.flatten() - np.sqrt(lambda_param + n) * L[:, ii]
                
                measures = np.zeros((mnum, 2*n + 1))
                for ii in range(2*n + 1):
                    measures[:, ii] = (mass*g*np.cos(states[1, ii]) + mass*len_pendulum*states[0, ii]**2) * np.sin(states[1, ii])
                    if ii == 0:
                        m_exp = lambda_param/(n + lambda_param) * measures[:, ii]
                    else:
                        m_exp = m_exp + 1/(n + lambda_param)/2 * measures[:, ii]
                
                # Estimate Mean And Covariance of Predicted Measurement
                Py = (lambda_param/(n + lambda_param) + 1 - alpha**2 + beta) * (measures[:, 0] - m_exp) @ (measures[:, 0] - m_exp).T + Measurementnoise
                Pxy = (lambda_param/(n + lambda_param) + 1 - alpha**2 + beta) * (states[:, 0].reshape(2, 1) - state) @ (measures[:, 0] - m_exp).T
                for ii in range(1, 2*n + 1):
                    Py = Py + 1/(n + lambda_param)/2 * (measures[:, ii] - m_exp) @ (measures[:, ii] - m_exp).T
                    Pxy = Pxy + 1/(n + lambda_param)/2 * (states[:, ii].reshape(2, 1) - state) @ (measures[:, ii] - m_exp).T
                
                # Kalman gain
                K = Pxy @ safe_matrix_inverse(Py)
                
                # Update
                state0 = state.copy()
                dstate = K @ (z - m_exp)
                state = state + dstate
                
                if improvement1 == 1:
                    for ii in range(2*n + 1):
                        states[:, ii] = states[:, ii] + dstate.flatten()
                    
                    measures = np.zeros((mnum, 2*n + 1))
                    for ii in range(2*n + 1):
                        measures[:, ii] = (mass*g*np.cos(states[1, ii]) + mass*len_pendulum*states[0, ii]**2) * np.sin(states[1, ii])
                        if ii == 0:
                            m_exp = lambda_param/(n + lambda_param) * measures[:, ii]
                        else:
                            m_exp = m_exp + 1/(n + lambda_param)/2 * measures[:, ii]
                    
                    # Estimate Mean And Covariance of Predicted Measurement
                    Py = (lambda_param/(n + lambda_param) + 1 - alpha**2 + beta) * (measures[:, 0] - m_exp) @ (measures[:, 0] - m_exp).T + Measurementnoise
                    Pxy = (lambda_param/(n + lambda_param) + 1 - alpha**2 + beta) * (states[:, 0].reshape(2, 1) - state) @ (measures[:, 0] - m_exp).T
                    for ii in range(1, 2*n + 1):
                        Py = Py + 1/(n + lambda_param)/2 * (measures[:, ii] - m_exp) @ (measures[:, ii] - m_exp).T
                        Pxy = Pxy + 1/(n + lambda_param)/2 * (states[:, ii].reshape(2, 1) - state) @ (measures[:, ii] - m_exp).T
                    
                    temp = Variance + K @ Py @ K.T - Pxy @ K.T - K @ Pxy.T
                    if np.trace(temp) < np.trace(Variance):
                        Variance = temp
                    else:
                        state = state0
                else:
                    Variance = Variance - K @ Py @ K.T
                
                states_est = np.hstack([states_est, state])
                
            elif KF_type == 2:  # CKF
                L = cholesky(Variance).T
                state = states_est[:, i].reshape(2, 1)
                n = 2
                mnum = 1
                states = np.zeros((n, n*2))
                
                for ii in range(n):
                    states[:, ii] = state.flatten() + np.sqrt(n) * L[:, ii]
                    states[:, n + ii] = state.flatten() - np.sqrt(n) * L[:, ii]
                
                state = np.zeros((2, 1))
                for ii in range(2*n):
                    states[:, ii] = states[:, ii] + np.array([-g/len_pendulum*np.sin(states[1, ii]) + v[i]/mass/len_pendulum**2, 
                                                            states[0, ii]]) * delta_t
                    state = state + states[:, ii].reshape(2, 1)/n/2
                
                Variance = Q
                for ii in range(2*n):
                    Variance = Variance + (state - states[:, ii].reshape(2, 1)) @ (state - states[:, ii].reshape(2, 1)).T/n/2
                
                # Obtain measurement
                z = Measurements_noisy[i+1]
                
                # Predict Measurement From Propagated Sigma Points
                L = cholesky(Variance).T
                states = np.zeros((n, n*2))
                for ii in range(n):
                    states[:, ii] = state.flatten() + np.sqrt(n) * L[:, ii]
                    states[:, n + ii] = state.flatten() - np.sqrt(n) * L[:, ii]
                
                measures = np.zeros((mnum, 2*n))
                m_exp = 0
                for ii in range(2*n):
                    measures[:, ii] = (mass*g*np.cos(states[1, ii]) + mass*len_pendulum*states[0, ii]**2) * np.sin(states[1, ii])
                    m_exp = m_exp + 1/n/2 * measures[:, ii]
                
                # Estimate Mean And Covariance of Predicted Measurement
                Py = Measurementnoise
                Pxy = 0
                for ii in range(2*n):
                    Py = Py + 1/n/2 * (measures[:, ii] - m_exp) @ (measures[:, ii] - m_exp).T
                    Pxy = Pxy + 1/n/2 * (states[:, ii].reshape(2, 1) - state) @ (measures[:, ii] - m_exp).T
                
                # Kalman gain
                K = Pxy @ safe_matrix_inverse(Py)
                
                # Update
                state0 = state.copy()
                state = state + K @ (z - m_exp)
                
                if improvement1 == 1:
                    states = np.zeros((n, n*2))
                    for ii in range(n):
                        states[:, ii] = state.flatten() + np.sqrt(n) * L[:, ii]
                        states[:, n + ii] = state.flatten() - np.sqrt(n) * L[:, ii]
                    
                    measures = np.zeros((mnum, 2*n))
                    m_exp = 0
                    for ii in range(2*n):
                        measures[:, ii] = (mass*g*np.cos(states[1, ii]) + mass*len_pendulum*states[0, ii]**2) * np.sin(states[1, ii])
                        m_exp = m_exp + 1/n/2 * measures[:, ii]
                    
                    # Estimate Mean And Covariance of Predicted Measurement
                    Py = Measurementnoise
                    Pxy = 0
                    for ii in range(2*n):
                        Py = Py + 1/n/2 * (measures[:, ii] - m_exp) @ (measures[:, ii] - m_exp).T
                        Pxy = Pxy + 1/n/2 * (states[:, ii].reshape(2, 1) - state) @ (measures[:, ii] - m_exp).T
                    
                    temp = Variance + K @ Py @ K.T - Pxy @ K.T - K @ Pxy.T
                    if np.trace(temp) < np.trace(Variance):
                        Variance = temp
                    else:
                        state = state0
                else:
                    Variance = Variance - K @ Py @ K.T
                
                states_est = np.hstack([states_est, state]) 

            elif KF_type == 3:  # 2-EKF
                states = states_est[:, i].reshape(2, 1)
                n = len(states)
                mnum = 1
                F = np.array([[1, -g/len_pendulum*np.cos(states[1, 0])*delta_t], 
                              [delta_t, 1]])
                states = states + np.array([[-g/len_pendulum*np.sin(states[1, 0]) + v[i]/mass/len_pendulum**2], 
                                           [states[0, 0]]]) * delta_t
                
                Fxx = np.zeros((n, n, n))
                deltaP = np.zeros((n, n))
                Fxx[1, 1, 0] = g/len_pendulum*np.sin(states[1, 0])*delta_t
                
                for ii in range(n):
                    states[ii, 0] = states[ii, 0] + 0.5*np.sum(np.diag(Fxx[:, :, ii] @ Variance))
                    for jj in range(n):
                        deltaP[ii, jj] = np.sum(np.diag(Fxx[:, :, ii] @ Variance @ Fxx[:, :, jj] @ Variance))
                
                Variance = F @ Variance @ F.T + 0.5*deltaP + Q
                
                # Update
                Hxx = np.zeros((n, n, mnum))
                Hxx[:, :, 0] = np.array([[2*mass*len_pendulum*np.sin(states[1, 0]), 2*mass*len_pendulum*states[0, 0]*np.cos(states[1, 0])],
                                         [2*mass*len_pendulum*states[0, 0]*np.cos(states[1, 0]), -2*mass*g*np.sin(2*states[1, 0]) - mass*len_pendulum*states[0, 0]**2*np.sin(states[1, 0])]])
                
                H = np.array([[2*mass*len_pendulum*states[0, 0]*np.sin(states[1, 0]), 
                              mass*g*np.cos(2*states[1, 0]) + mass*len_pendulum*states[0, 0]**2*np.cos(states[1, 0])]])
                
                S = H @ Variance @ H.T + Measurementnoise
                for ii in range(mnum):
                    for jj in range(mnum):
                        S[ii, jj] = S[ii, jj] + 0.5*np.sum(np.diag(Hxx[:, :, ii] @ Variance @ Hxx[:, :, jj] @ Variance))
                
                K = Variance @ H.T @ safe_matrix_inverse(S)
                states0 = states.copy()
                residual = Measurements_noisy[i+1] - (mass*g*np.cos(states[1, 0]) + mass*len_pendulum*states[0, 0]**2) * np.sin(states[1, 0])
                
                for ii in range(mnum):
                    residual = residual - 0.5*np.sum(np.diag(Hxx[:, :, ii] @ Variance))
                
                states = states0 + K * residual
                
                if improvement1 == 1:
                    Hxx2 = np.zeros((n, n, mnum))
                    Hxx2[:, :, 0] = np.array([[2*mass*len_pendulum*np.sin(states[1, 0]), 2*mass*len_pendulum*states[0, 0]*np.cos(states[1, 0])],
                                              [2*mass*len_pendulum*states[0, 0]*np.cos(states[1, 0]), -2*mass*g*np.sin(2*states[1, 0]) - mass*len_pendulum*states[0, 0]**2*np.sin(states[1, 0])]])
                    
                    H2 = np.array([[2*mass*len_pendulum*states[0, 0]*np.sin(states[1, 0]), 
                                   mass*g*np.cos(2*states[1, 0]) + mass*len_pendulum*states[0, 0]**2*np.cos(states[1, 0])]])
                    
                    S2 = H2 @ Variance @ H2.T + Measurementnoise
                    for ii in range(mnum):
                        for jj in range(mnum):
                            S2[ii, jj] = S2[ii, jj] + 0.5*np.sum(np.diag(Hxx2[:, :, ii] @ Variance @ Hxx2[:, :, jj] @ Variance))
                    
                    temp = Variance + K @ S2 @ K.T - Variance @ H2.T @ K.T - K @ H2 @ Variance
                    if np.trace(temp) < np.trace(Variance):
                        Variance = temp
                    else:
                        states = states0
                else:
                    Variance = Variance - K @ S @ K.T
                
                states_est = np.hstack([states_est, states])
                
            else:  # 2-IEKF
                states = states_est[:, i].reshape(2, 1)
                n = len(states)
                mnum = 1
                F = np.array([[1, -g/len_pendulum*np.cos(states[1, 0])*delta_t], 
                              [delta_t, 1]])
                states = states + np.array([[-g/len_pendulum*np.sin(states[1, 0]) + v[i]/mass/len_pendulum**2], 
                                           [states[0, 0]]]) * delta_t
                
                Fxx = np.zeros((n, n, n))
                deltaP = np.zeros((n, n))
                Fxx[1, 1, 0] = g/len_pendulum*np.sin(states[1, 0])*delta_t
                
                for ii in range(n):
                    states[ii, 0] = states[ii, 0] + 0.5*np.sum(np.diag(Fxx[:, :, ii] @ Variance))
                    for jj in range(n):
                        deltaP[ii, jj] = np.sum(np.diag(Fxx[:, :, ii] @ Variance @ Fxx[:, :, jj] @ Variance))
                
                Variance = F @ Variance @ F.T + 0.5*deltaP + Q
                
                # Update
                Hxx = np.zeros((n, n, mnum))
                Hxx[:, :, 0] = np.array([[2*mass*len_pendulum*np.sin(states[1, 0]), 2*mass*len_pendulum*states[0, 0]*np.cos(states[1, 0])],
                                         [2*mass*len_pendulum*states[0, 0]*np.cos(states[1, 0]), -2*mass*g*np.sin(2*states[1, 0]) - mass*len_pendulum*states[0, 0]**2*np.sin(states[1, 0])]])
                
                H = np.array([[2*mass*len_pendulum*states[0, 0]*np.sin(states[1, 0]), 
                              mass*g*np.cos(2*states[1, 0]) + mass*len_pendulum*states[0, 0]**2*np.cos(states[1, 0])]])
                
                S = H @ Variance @ H.T + Measurementnoise
                for ii in range(mnum):
                    for jj in range(mnum):
                        S[ii, jj] = S[ii, jj] + 0.5*np.sum(np.diag(Hxx[:, :, ii] @ Variance @ Hxx[:, :, jj] @ Variance))
                
                K = Variance @ H.T @ safe_matrix_inverse(S)
                states0 = states.copy()
                residual = Measurements_noisy[i+1] - (mass*g*np.cos(states[1, 0]) + mass*len_pendulum*states[0, 0]**2) * np.sin(states[1, 0])
                
                for ii in range(mnum):
                    residual = residual - 0.5*np.sum(np.diag(Hxx[:, :, ii] @ Variance))
                
                states = states0 + K * residual
                steplennow = np.linalg.norm(K * residual)
                change = 1
                iter_count = 1
                
                while change > 0.001 and iter_count < 1000:
                    iter_count += 1
                    Hxx2 = np.zeros((n, n, mnum))
                    Hxx2[:, :, 0] = np.array([[2*mass*len_pendulum*np.sin(states[1, 0]), 2*mass*len_pendulum*states[0, 0]*np.cos(states[1, 0])],
                                              [2*mass*len_pendulum*states[0, 0]*np.cos(states[1, 0]), -2*mass*g*np.sin(2*states[1, 0]) - mass*len_pendulum*states[0, 0]**2*np.sin(states[1, 0])]])
                    
                    H2 = np.array([[2*mass*len_pendulum*states[0, 0]*np.sin(states[1, 0]), 
                                   mass*g*np.cos(2*states[1, 0]) + mass*len_pendulum*states[0, 0]**2*np.cos(states[1, 0])]])
                    
                    residual = Measurements_noisy[i+1] - (mass*g*np.cos(states[1, 0]) + mass*len_pendulum*states[0, 0]**2) * np.sin(states[1, 0])
                    
                    for ii in range(mnum):
                        residual = residual - 0.5*(states0 - states).T @ Hxx2[:, :, ii] @ (states0 - states)
                        residual = residual - 0.5*np.sum(np.diag(Hxx[:, :, ii] @ Variance))
                    
                    S2 = H2 @ Variance @ H2.T + Measurementnoise
                    for ii in range(mnum):
                        for jj in range(mnum):
                            S2[ii, jj] = S2[ii, jj] + 0.5*np.sum(np.diag(Hxx2[:, :, ii] @ Variance @ Hxx2[:, :, jj] @ Variance))
                    
                    K2 = Variance @ H2.T @ safe_matrix_inverse(S2)
                    dx = states0 + K2 * (residual - H2 @ (states0 - states)) - states
                    steplen_previous = steplennow
                    steplennow = np.linalg.norm(dx)
                    if steplen_previous < steplennow:
                        break
                    else:
                        change = np.max(np.abs(dx / (states + 1e-10)))
                        states = states + dx
                        K = K2
                        S = S2
                
                if improvement1 == 1:
                    Hxx2 = np.zeros((n, n, mnum))
                    Hxx2[:, :, 0] = np.array([[2*mass*len_pendulum*np.sin(states[1, 0]), 2*mass*len_pendulum*states[0, 0]*np.cos(states[1, 0])],
                                              [2*mass*len_pendulum*states[0, 0]*np.cos(states[1, 0]), -2*mass*g*np.sin(2*states[1, 0]) - mass*len_pendulum*states[0, 0]**2*np.sin(states[1, 0])]])
                    
                    H2 = np.array([[2*mass*len_pendulum*states[0, 0]*np.sin(states[1, 0]), 
                                   mass*g*np.cos(2*states[1, 0]) + mass*len_pendulum*states[0, 0]**2*np.cos(states[1, 0])]])
                    
                    S2 = H2 @ Variance @ H2.T + Measurementnoise
                    for ii in range(mnum):
                        for jj in range(mnum):
                            S2[ii, jj] = S2[ii, jj] + 0.5*np.sum(np.diag(Hxx2[:, :, ii] @ Variance @ Hxx2[:, :, jj] @ Variance))
                    
                    temp = Variance + K @ S2 @ K.T - Variance @ H2.T @ K.T - K @ H2 @ Variance
                    if np.trace(temp) < np.trace(Variance):
                        Variance = temp
                    else:
                        states = states0
                else:
                    Variance = Variance - K @ S @ K.T
                
                states_est = np.hstack([states_est, states])
        
        temp = time.time() - start_time
        Runtime = Runtime + temp/repeat
        
        sitadot_est = states_est[0, :]
        sita_est = states_est[1, :]
        sitadot_rmse = sitadot_rmse + (sitadot[-1] - sitadot_est[-1])**2/repeat
        sita_rmse = sita_rmse + (sita[-1] - sita_est[-1])**2/repeat
    
    sita_rmse = np.sqrt(sita_rmse)
    sitadot_rmse = np.sqrt(sitadot_rmse)
    
    return sitadot_rmse, sita_rmse, Runtime 

def main():
    """
    Main function that runs all simulations and creates plots
    """
    print("Starting Pendulum State Estimation with Quadratic Measurement Function")
    print("This simulation may take several minutes...")
    
    # Initialize arrays for plotting
    xplotEKF_1 = []
    yplotEKF_1 = []
    Time_EKF_1 = []
    xplotEKF2_1 = []
    yplotEKF2_1 = []
    Time_EKF2_1 = []
    xplotUKF_1 = []
    yplotUKF_1 = []
    Time_UKF_1 = []
    xplotCKF_1 = []
    yplotCKF_1 = []
    Time_CKF_1 = []
    xplotEKF_2 = []
    yplotEKF_2 = []
    Time_EKF_2 = []
    xplotEKF2_2 = []
    yplotEKF2_2 = []
    Time_EKF2_2 = []
    xplotUKF_2 = []
    yplotUKF_2 = []
    Time_UKF_2 = []
    xplotCKF_2 = []
    yplotCKF_2 = []
    Time_CKF_2 = []
    xplotIEKF_1 = []
    yplotIEKF_1 = []
    Time_IEKF_1 = []
    xplotIEKF_2 = []
    yplotIEKF_2 = []
    Time_IEKF_2 = []
    xplotIEKF2_1 = []
    yplotIEKF2_1 = []
    Time_IEKF2_1 = []
    xplotIEKF2_2 = []
    yplotIEKF2_2 = []
    Time_IEKF2_2 = []
    
    for i in SCALE:
        noise1 = NOISE1_REF * 10**i
        print(f"Processing noise level {i+1}/{len(SCALE)}: magnitude = {noise1:.2e}")
        
        # IEKF
        x_rmse, y_rmse, Runtime = simulation(0, noise1, REPEAT, 0.5)
        xplotIEKF_1.append(x_rmse)
        yplotIEKF_1.append(y_rmse)
        Time_IEKF_1.append(Runtime)
        
        x_rmse, y_rmse, Runtime = simulation(1, noise1, REPEAT, 0.5)
        xplotIEKF_2.append(x_rmse)
        yplotIEKF_2.append(y_rmse)
        Time_IEKF_2.append(Runtime)
        
        # EKF
        x_rmse, y_rmse, Runtime = simulation(0, noise1, REPEAT, 0)
        xplotEKF_1.append(x_rmse)
        yplotEKF_1.append(y_rmse)
        Time_EKF_1.append(Runtime)
        
        x_rmse, y_rmse, Runtime = simulation(1, noise1, REPEAT, 0)
        xplotEKF_2.append(x_rmse)
        yplotEKF_2.append(y_rmse)
        Time_EKF_2.append(Runtime)
        
        # EKF2
        x_rmse, y_rmse, Runtime = simulation(0, noise1, REPEAT, 3)
        xplotEKF2_1.append(x_rmse)
        yplotEKF2_1.append(y_rmse)
        Time_EKF2_1.append(Runtime)
        
        x_rmse, y_rmse, Runtime = simulation(1, noise1, REPEAT, 3)
        xplotEKF2_2.append(x_rmse)
        yplotEKF2_2.append(y_rmse)
        Time_EKF2_2.append(Runtime)
        
        # IEKF2
        x_rmse, y_rmse, Runtime = simulation(0, noise1, REPEAT, 3.5)
        xplotIEKF2_1.append(x_rmse)
        yplotIEKF2_1.append(y_rmse)
        Time_IEKF2_1.append(Runtime)
        
        x_rmse, y_rmse, Runtime = simulation(1, noise1, REPEAT, 3.5)
        xplotIEKF2_2.append(x_rmse)
        yplotIEKF2_2.append(y_rmse)
        Time_IEKF2_2.append(Runtime)
        
        # UKF
        x_rmse, y_rmse, Runtime = simulation(0, noise1, REPEAT, 1)
        xplotUKF_1.append(x_rmse)
        yplotUKF_1.append(y_rmse)
        Time_UKF_1.append(Runtime)
        
        x_rmse, y_rmse, Runtime = simulation(1, noise1, REPEAT, 1)
        xplotUKF_2.append(x_rmse)
        yplotUKF_2.append(y_rmse)
        Time_UKF_2.append(Runtime)
        
        # CKF
        x_rmse, y_rmse, Runtime = simulation(0, noise1, REPEAT, 2)
        xplotCKF_1.append(x_rmse)
        yplotCKF_1.append(y_rmse)
        Time_CKF_1.append(Runtime)
        
        x_rmse, y_rmse, Runtime = simulation(1, noise1, REPEAT, 2)
        xplotCKF_2.append(x_rmse)
        yplotCKF_2.append(y_rmse)
        Time_CKF_2.append(Runtime)
    
    # Create plots
    plot_results({
        'xplotEKF_1': xplotEKF_1, 'yplotEKF_1': yplotEKF_1, 'Time_EKF_1': Time_EKF_1,
        'xplotEKF2_1': xplotEKF2_1, 'yplotEKF2_1': yplotEKF2_1, 'Time_EKF2_1': Time_EKF2_1,
        'xplotUKF_1': xplotUKF_1, 'yplotUKF_1': yplotUKF_1, 'Time_UKF_1': Time_UKF_1,
        'xplotCKF_1': xplotCKF_1, 'yplotCKF_1': yplotCKF_1, 'Time_CKF_1': Time_CKF_1,
        'xplotEKF_2': xplotEKF_2, 'yplotEKF_2': yplotEKF_2, 'Time_EKF_2': Time_EKF_2,
        'xplotEKF2_2': xplotEKF2_2, 'yplotEKF2_2': yplotEKF2_2, 'Time_EKF2_2': Time_EKF2_2,
        'xplotUKF_2': xplotUKF_2, 'yplotUKF_2': yplotUKF_2, 'Time_UKF_2': Time_UKF_2,
        'xplotCKF_2': xplotCKF_2, 'yplotCKF_2': yplotCKF_2, 'Time_CKF_2': Time_CKF_2,
        'xplotIEKF_1': xplotIEKF_1, 'yplotIEKF_1': yplotIEKF_1, 'Time_IEKF_1': Time_IEKF_1,
        'xplotIEKF_2': xplotIEKF_2, 'yplotIEKF_2': yplotIEKF_2, 'Time_IEKF_2': Time_IEKF_2,
        'xplotIEKF2_1': xplotIEKF2_1, 'yplotIEKF2_1': yplotIEKF2_1, 'Time_IEKF2_1': Time_IEKF2_1,
        'xplotIEKF2_2': xplotIEKF2_2, 'yplotIEKF2_2': yplotIEKF2_2, 'Time_IEKF2_2': Time_IEKF2_2
    })
    
    return {
        'xplotEKF_1': xplotEKF_1, 'yplotEKF_1': yplotEKF_1, 'Time_EKF_1': Time_EKF_1,
        'xplotEKF2_1': xplotEKF2_1, 'yplotEKF2_1': yplotEKF2_1, 'Time_EKF2_1': Time_EKF2_1,
        'xplotUKF_1': xplotUKF_1, 'yplotUKF_1': yplotUKF_1, 'Time_UKF_1': Time_UKF_1,
        'xplotCKF_1': xplotCKF_1, 'yplotCKF_1': yplotCKF_1, 'Time_CKF_1': Time_CKF_1,
        'xplotEKF_2': xplotEKF_2, 'yplotEKF_2': yplotEKF_2, 'Time_EKF_2': Time_EKF_2,
        'xplotEKF2_2': xplotEKF2_2, 'yplotEKF2_2': yplotEKF2_2, 'Time_EKF2_2': Time_EKF2_2,
        'xplotUKF_2': xplotUKF_2, 'yplotUKF_2': yplotUKF_2, 'Time_UKF_2': Time_UKF_2,
        'xplotCKF_2': xplotCKF_2, 'yplotCKF_2': yplotCKF_2, 'Time_CKF_2': Time_CKF_2,
        'xplotIEKF_1': xplotIEKF_1, 'yplotIEKF_1': yplotIEKF_1, 'Time_IEKF_1': Time_IEKF_1,
        'xplotIEKF_2': xplotIEKF_2, 'yplotIEKF_2': yplotIEKF_2, 'Time_IEKF_2': Time_IEKF_2,
        'xplotIEKF2_1': xplotIEKF2_1, 'yplotIEKF2_1': yplotIEKF2_1, 'Time_IEKF2_1': Time_IEKF2_1,
        'xplotIEKF2_2': xplotIEKF2_2, 'yplotIEKF2_2': yplotIEKF2_2, 'Time_IEKF2_2': Time_IEKF2_2
    }

def plot_results(results):
    """
    Create plots for the simulation results
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Angular Velocity RMSE
    ax1.semilogy(SCALE, results['xplotEKF_1'], 'b-o', label='EKF (Old)', linewidth=2, markersize=6)
    ax1.semilogy(SCALE, results['xplotEKF_2'], 'b--s', label='EKF (New)', linewidth=2, markersize=6)
    ax1.semilogy(SCALE, results['xplotIEKF_1'], 'r-o', label='IEKF (Old)', linewidth=2, markersize=6)
    ax1.semilogy(SCALE, results['xplotIEKF_2'], 'r--s', label='IEKF (New)', linewidth=2, markersize=6)
    ax1.semilogy(SCALE, results['xplotUKF_1'], 'g-o', label='UKF (Old)', linewidth=2, markersize=6)
    ax1.semilogy(SCALE, results['xplotUKF_2'], 'g--s', label='UKF (New)', linewidth=2, markersize=6)
    ax1.semilogy(SCALE, results['xplotCKF_1'], 'm-o', label='CKF (Old)', linewidth=2, markersize=6)
    ax1.semilogy(SCALE, results['xplotCKF_2'], 'm--s', label='CKF (New)', linewidth=2, markersize=6)
    ax1.semilogy(SCALE, results['xplotEKF2_1'], 'c-o', label='EKF2 (Old)', linewidth=2, markersize=6)
    ax1.semilogy(SCALE, results['xplotEKF2_2'], 'c--s', label='EKF2 (New)', linewidth=2, markersize=6)
    ax1.semilogy(SCALE, results['xplotIEKF2_1'], 'y-o', label='IEKF2 (Old)', linewidth=2, markersize=6)
    ax1.semilogy(SCALE, results['xplotIEKF2_2'], 'y--s', label='IEKF2 (New)', linewidth=2, markersize=6)
    ax1.set_xlabel('Measurement noise standard deviation (log scale)', fontsize=12)
    ax1.set_ylabel('Angular Velocity RMSE', fontsize=12)
    ax1.set_title('Pendulum State Estimation - Angular Velocity RMSE', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Angular Position RMSE
    ax2.semilogy(SCALE, results['yplotEKF_1'], 'b-o', label='EKF (Old)', linewidth=2, markersize=6)
    ax2.semilogy(SCALE, results['yplotEKF_2'], 'b--s', label='EKF (New)', linewidth=2, markersize=6)
    ax2.semilogy(SCALE, results['yplotIEKF_1'], 'r-o', label='IEKF (Old)', linewidth=2, markersize=6)
    ax2.semilogy(SCALE, results['yplotIEKF_2'], 'r--s', label='IEKF (New)', linewidth=2, markersize=6)
    ax2.semilogy(SCALE, results['yplotUKF_1'], 'g-o', label='UKF (Old)', linewidth=2, markersize=6)
    ax2.semilogy(SCALE, results['yplotUKF_2'], 'g--s', label='UKF (New)', linewidth=2, markersize=6)
    ax2.semilogy(SCALE, results['yplotCKF_1'], 'm-o', label='CKF (Old)', linewidth=2, markersize=6)
    ax2.semilogy(SCALE, results['yplotCKF_2'], 'm--s', label='CKF (New)', linewidth=2, markersize=6)
    ax2.semilogy(SCALE, results['yplotEKF2_1'], 'c-o', label='EKF2 (Old)', linewidth=2, markersize=6)
    ax2.semilogy(SCALE, results['yplotEKF2_2'], 'c--s', label='EKF2 (New)', linewidth=2, markersize=6)
    ax2.semilogy(SCALE, results['yplotIEKF2_1'], 'y-o', label='IEKF2 (Old)', linewidth=2, markersize=6)
    ax2.semilogy(SCALE, results['yplotIEKF2_2'], 'y--s', label='IEKF2 (New)', linewidth=2, markersize=6)
    ax2.set_xlabel('Measurement noise standard deviation (log scale)', fontsize=12)
    ax2.set_ylabel('Angular Position RMSE', fontsize=12)
    ax2.set_title('Pendulum State Estimation - Angular Position RMSE', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Runtime comparison
    ax3.plot(SCALE, results['Time_EKF_1'], 'b-o', label='EKF (Old)', linewidth=2, markersize=6)
    ax3.plot(SCALE, results['Time_EKF_2'], 'b--s', label='EKF (New)', linewidth=2, markersize=6)
    ax3.plot(SCALE, results['Time_IEKF_1'], 'r-o', label='IEKF (Old)', linewidth=2, markersize=6)
    ax3.plot(SCALE, results['Time_IEKF_2'], 'r--s', label='IEKF (New)', linewidth=2, markersize=6)
    ax3.plot(SCALE, results['Time_UKF_1'], 'g-o', label='UKF (Old)', linewidth=2, markersize=6)
    ax3.plot(SCALE, results['Time_UKF_2'], 'g--s', label='UKF (New)', linewidth=2, markersize=6)
    ax3.plot(SCALE, results['Time_CKF_1'], 'm-o', label='CKF (Old)', linewidth=2, markersize=6)
    ax3.plot(SCALE, results['Time_CKF_2'], 'm--s', label='CKF (New)', linewidth=2, markersize=6)
    ax3.plot(SCALE, results['Time_EKF2_1'], 'c-o', label='EKF2 (Old)', linewidth=2, markersize=6)
    ax3.plot(SCALE, results['Time_EKF2_2'], 'c--s', label='EKF2 (New)', linewidth=2, markersize=6)
    ax3.plot(SCALE, results['Time_IEKF2_1'], 'y-o', label='IEKF2 (Old)', linewidth=2, markersize=6)
    ax3.plot(SCALE, results['Time_IEKF2_2'], 'y--s', label='IEKF2 (New)', linewidth=2, markersize=6)
    ax3.set_xlabel('Measurement noise standard deviation', fontsize=12)
    ax3.set_ylabel('Average Runtime (seconds)', fontsize=12)
    ax3.set_title('Pendulum State Estimation - Runtime Comparison', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance summary
    noise_levels = [10**s for s in SCALE]
    ax4.bar(range(len(noise_levels)), [results['xplotIEKF_2'][i] for i in range(len(noise_levels))], 
             label='IEKF (New)', alpha=0.7)
    ax4.set_xlabel('Noise Level Index', fontsize=12)
    ax4.set_ylabel('Angular Velocity RMSE', fontsize=12)
    ax4.set_title('Performance Summary - IEKF (New)', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pendulum_state_estimation_quadratic_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Simulation completed! Results saved to 'pendulum_state_estimation_quadratic_results.png'")

if __name__ == "__main__":
    results = main() 