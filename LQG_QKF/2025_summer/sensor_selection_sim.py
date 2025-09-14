import os
import matplotlib.pyplot as plt
import pickle as pkl
from datetime import datetime
import random
import itertools
import time

# Import specific modules instead of wildcard imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '2025_summer'))
from LQG_QKF import LQG, generate_stable_system_parameters, validate_stable_parameters, generate_random_symmetric_matrix, generate_goal_state
from stateDynamics import StateDynamics, sensor, Vec, invVec
from typing import Literal
from tqdm import tqdm
import numpy as np
import random
import itertools
import time
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
test_dir = 'sensor_sim/'
pkl_dir = test_dir + 'pkl/'
os.makedirs(pkl_dir, exist_ok=True)
perf_dir = test_dir + 'perf/'
os.makedirs(perf_dir, exist_ok=True)

# build sensor selection simulator
random_seed = 42
small_value = 1e-6
class SensorSelectionSimulator(LQG):
    def __init__(self, n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, H = 50, m_scale=1e2,
                 filter_type: Literal['qkf', 'ekf', 'kf', 'ukf'] = 'qkf',
                 lqr_type: Literal['orig', 'aug_analytic', 'aug_numeric', 'None'] = 'orig',
                 max_sensors=None,
                 update_interval=10):
        
        self.filter_type = filter_type
        self.lqr_type = lqr_type
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
        for subset in sensor_subsets:
            # Calculate information gain for this sensor subset
            info_gain = self._calculate_information_gain_subset(subset, x_current)
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
    
    def _calculate_information_gain_subset(self, sensor_subset, x_current):
        """
        Calculate information gain for a given sensor subset.
        Uses trace of covariance matrix as uncertainty measure.
        """
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
            info_gain = current_uncertainty * np.log(1 + num_active_sensors) / np.log(1 + total_sensors)
        
        # Add some randomness to avoid always selecting the same subset
        # info_gain += np.random.normal(0, 0.05 * info_gain)
        
        return info_gain
    
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
        print(f"Starting sensor selection simulation with {self.filter_type} filter and {self.lqr_type} LQR...")
        
        for step in tqdm(range(1, self.H + 1, 1), desc=f"Running simulation with filter:[{self.filter_type}], controller:[{self.lqr_type}], m_scale:[{self.m_scale}]"):
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
    
    
    def plot_performance(self, save_plots=True, plot_dir=perf_dir):
        """
        Create simple performance plots similar to LQG_QKF.py style.
        """
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


def run_sensor_scheduling_sim(H=1000, update_interval=10, noise_scale=1e-1, m_scale=1e0, Q_scale=1.0, R_scale=1.0, num_sensors=2, max_sensors=None, rand_seed=random_seed, plot=True, trial_idx=None, save_dir=pkl_dir):
    n1 = 2
    n2 = 2
    n = n1 + n2 # state size
    p = 3
    m = num_sensors  # number of sensors
    
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
    
    # goal_state = generate_goal_state(np.zeros((n1, 1)), n2) # goal state vector
    # lqg_kf_sys = LQG(n, p, W, A, B, C, M, V, Q, R, H=1000, filter_type='kf')
    # err_list_kf = lqg_kf_sys.run_sim()
    # plt.plot(err_list_kf, label=f'kf measure error')

    # Run simulations with different filter types
    simulators = {}
    
    print("Running EKF simulation...")
    lqg_ekf = SensorSelectionSimulator(n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, H=H, m_scale=m_scale, filter_type='ekf', lqr_type='orig', max_sensors=max_sensors, update_interval=update_interval)
    err_list_ekf, var_list_ekf, cost_list_ekf, time_list_ekf = lqg_ekf.run_sim()
    simulators['ekf'] = lqg_ekf
    
    print("Running UKF simulation...")
    lqg_ukf = SensorSelectionSimulator(n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, H=H, m_scale=m_scale, filter_type='ukf', lqr_type='orig', max_sensors=max_sensors, update_interval=update_interval)
    err_list_ukf, var_list_ukf, cost_list_ukf, time_list_ukf = lqg_ukf.run_sim()
    simulators['ukf'] = lqg_ukf
    
    print("Running QKF with augmented numeric LQR...")
    lqg_qkf_aug_num = SensorSelectionSimulator(n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, H=H, m_scale=m_scale, filter_type='qkf', lqr_type='aug_numeric', max_sensors=max_sensors, update_interval=update_interval)
    err_list_aug_num, var_list_aug_num, cost_list_aug_num, time_list_aug_num = lqg_qkf_aug_num.run_sim()
    simulators['qkf_aug_num'] = lqg_qkf_aug_num
    
    print("Running QKF with augmented analytic LQR...")
    lqg_qkf_aug_analytic = SensorSelectionSimulator(n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, H=H, m_scale=m_scale, filter_type='qkf', lqr_type='aug_analytic', max_sensors=max_sensors, update_interval=update_interval)
    err_list_aug_analytic, var_list_aug_analytic, cost_list_aug_analytic, time_list_aug_analytic = lqg_qkf_aug_analytic.run_sim()
    simulators['qkf_aug_analytic'] = lqg_qkf_aug_analytic
    
    # Save results for each simulator
    print("\nSaving simulation results...")
    for name, simulator in simulators.items():
        print(f"Saving {name} results...")
        # simulator.save_results(trial_idx=trial_idx)
    
    # Return performance histories for consistency with pickle loading
    ekf_perf_history = simulators['ekf'].performance_history
    ukf_perf_history = simulators['ukf'].performance_history  
    qkf_num_perf_history = simulators['qkf_aug_num'].performance_history
    qkf_analytic_perf_history = simulators['qkf_aug_analytic'].performance_history
    
    # Create comparison plots only if requested
    if plot:
        all_results = [(err_list_ekf, var_list_ekf, cost_list_ekf, time_list_ekf), (err_list_ukf, var_list_ukf, cost_list_ukf, time_list_ukf), (err_list_aug_num, var_list_aug_num, cost_list_aug_num, time_list_aug_num), (err_list_aug_analytic, var_list_aug_analytic, cost_list_aug_analytic, time_list_aug_analytic)]
        plot_comparison(all_results, save_plots=True, update_interval=update_interval)
    
    return ekf_perf_history, ukf_perf_history, qkf_num_perf_history, qkf_analytic_perf_history

system_names = ['ekf', 'ukf', 'qkf_aug_num', 'qkf_aug_analytic']
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
    'qkf_aug_analytic': 'LQG+QKF (Analytic)'
}

axis_labels = {
    'ekf': 'LQG+EKF',
    'ukf': 'LQG+UKF',
    'qkf_aug_num': 'iLQG+QKF\n(Numeric)',
    'qkf_aug_analytic': 'LQG+QKF\n(Analytic)'
}

def plot_comparison(all_results, save_plots=True, plot_dir=perf_dir, update_interval=10, m_scale=1e0):
    """
    Create publication-ready comparison plots across different filter types.
    """
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
    
    # Professional color palette and styles
    colors = ['#2E86C1', '#28B463', '#F39C12', '#E74C3C']  # Blue, Green, Orange, Red
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D']
    alphas = [0.9, 0.8, 0.8, 0.8]
    
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
    
    axes[0, 0].set_title('Cost-to-Go Comparison', fontsize=18, fontweight='bold', pad=10)
    axes[0, 0].set_xlabel('Time Step', fontsize=16, fontweight='bold')
    axes[0, 0].set_ylabel('Cost-to-Go (log scale)', fontsize=16, fontweight='bold')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend(loc='lower left', framealpha=0.9, fontsize=12)
    axes[0, 0].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    axes[0, 0].tick_params(axis='both', which='major', labelsize=12)
    
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
    
    axes[0, 1].set_title('Staged Cost-to-Go (Goal State Phases)', fontsize=18, fontweight='bold', pad=10)
    axes[0, 1].set_xlabel('Phase Index', fontsize=16, fontweight='bold')
    axes[0, 1].set_ylabel('Phase Cost-to-Go (log scale)', fontsize=16, fontweight='bold')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend(loc='lower left', framealpha=0.9, fontsize=12)
    axes[0, 1].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    axes[0, 1].tick_params(axis='both', which='major', labelsize=12)
    
    # 3. Average estimation error comparison (bottom left)
    avg_errors = []
    for i, result in enumerate(all_results):
        avg_error = np.mean(result[0])  # Average estimation error
        avg_errors.append(avg_error)
    plot_system_names = [axis_labels[name] for name in system_names]
    bars_error = axes[1, 0].bar(plot_system_names, avg_errors, 
                               color=colors[:len(system_names)], 
                               edgecolor='black', linewidth=1.2, alpha=0.8)
    axes[1, 0].set_title('Average Estimation Error', fontsize=18, fontweight='bold', pad=10)
    # axes[1, 0].set_xlabel('Filter Type', fontsize=16, fontweight='bold')
    axes[1, 0].set_ylabel('Average $||x_{true} - x_{est}||$', fontsize=16, fontweight='bold')
    
    # Set y-axis limit to prevent overlap with top border
    max_error = max(avg_errors)
    axes[1, 0].set_ylim(0, max_error * 1.15)
    
    axes[1, 0].grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
    axes[1, 0].tick_params(axis='both', which='major', labelsize=14)
    axes[1, 0].tick_params(axis='x', rotation=0)
    
    # Add value labels on bars
    for bar, avg_error in zip(bars_error, avg_errors):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + max_error*0.03,
                f'{avg_error:.4f}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
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
    axes[1, 1].set_title('Average Time Consumption', fontsize=18, fontweight='bold', pad=10)
    # axes[1, 1].set_xlabel('Filter Type', fontsize=16, fontweight='bold')
    axes[1, 1].set_ylabel('Average Time (seconds)', fontsize=16, fontweight='bold')
    
    # Set y-axis limit to prevent overlap with top border
    max_time = max(avg_times)
    axes[1, 1].set_ylim(0, max_time * 1.15)
    
    axes[1, 1].grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
    axes[1, 1].tick_params(axis='both', which='major', labelsize=14)
    axes[1, 1].tick_params(axis='x', rotation=0)
    
    # Add value labels on bars
    for bar, avg_time in zip(bars, avg_times):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + max_time*0.03,
                f'{avg_time:.6f}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
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
    if not os.path.exists(pkl_dir + f"sensor_selection_comprehensive_results_mscale={m_scale}_trials={n_trials}.pkl"):
        all_ekf_results = []
        all_ukf_results = []
        all_qkf_num_results = []
        all_qkf_analytic_results = []
        
        for idx in range(n_trials):
            if not os.path.exists(pkl_dir + f"sensor_selection_results_ekf_orig_mscale={m_scale}-trial_{idx}.pkl"):
                # print(pkl_dir + f"sensor_selection_results_ekf_orig_mscale={m_scale}-trial_{idx}.pkl")

                seed_i = random.randint(0, 1000000)
                print(f"Running trial [{idx+1}/{n_trials}]")
                ekf_result, ukf_result, qkf_num_result, qkf_analytic_result = run_sensor_scheduling_sim(H=H, num_sensors=num_sensors, max_sensors=max_sensors, rand_seed=seed_i, plot=False, update_interval=update_interval, m_scale=m_scale, trial_idx=idx)
            else:
                ekf_result = pkl.load(open(pkl_dir + f"sensor_selection_results_ekf_orig_mscale={m_scale}-trial_{idx}.pkl", 'rb'))['performance_history']
                ukf_result = pkl.load(open(pkl_dir + f"sensor_selection_results_ukf_orig_mscale={m_scale}-trial_{idx}.pkl", 'rb'))['performance_history']
                qkf_num_result = pkl.load(open(pkl_dir + f"sensor_selection_results_qkf_aug_numeric_mscale={m_scale}-trial_{idx}.pkl", 'rb'))['performance_history']
                qkf_analytic_result = pkl.load(open(pkl_dir + f"sensor_selection_results_qkf_aug_analytic_mscale={m_scale}-trial_{idx}.pkl", 'rb'))['performance_history']

            # append results
            all_ekf_results.append(ekf_result)
            all_ukf_results.append(ukf_result)
            all_qkf_num_results.append(qkf_num_result)
            all_qkf_analytic_results.append(qkf_analytic_result)
        
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
        pkl.dump(avg_all_results, open(pkl_dir + f"sensor_selection_comprehensive_results_mscale={m_scale}_trials={n_trials}.pkl", 'wb'))
    else:
        print(f"Loading existing comprehensive results for m_scale={m_scale}, trials={n_trials}...")
        avg_all_results = pkl.load(open(pkl_dir + f"sensor_selection_comprehensive_results_mscale={m_scale}_trials={n_trials}.pkl", 'rb'))
    
    if plot:
        plot_comparison(avg_all_results, save_plots=True, update_interval=update_interval, m_scale=m_scale)
    
    
    return avg_all_results

def run_nonlinearity_test(nonlinearity_factors=[1e-2, 1, 1e2], n_trials=5, H=1000, update_interval=100, num_sensors=6, max_sensors=3, plot=True):
    for m_scale in nonlinearity_factors:
        run_comprehensive_test(n_trials=n_trials, H=H, update_interval=update_interval, num_sensors=num_sensors, max_sensors=max_sensors, plot=plot, m_scale=m_scale)

if __name__ == "__main__":
    # Run single trial (original behavior)
    # run_sensor_scheduling_sim(H=1000, update_interval=100, num_sensors=10, max_sensors=5)
    
    # Run comprehensive test with multiple trials
    run_comprehensive_test(n_trials=1, H=1000, update_interval=100, num_sensors=10, max_sensors=5, plot=True, m_scale=1e0)
    
    # # Run nonlinearity test
    nonlinearity_factors = [0, 1, 1e1, 1e2]
    run_nonlinearity_test(nonlinearity_factors=nonlinearity_factors, n_trials=1, H=1000, update_interval=100, num_sensors=10, max_sensors=5, plot=True)