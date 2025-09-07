import os
import matplotlib.pyplot as plt
import pickle as pkl
from datetime import datetime

# from LQG_QKF.test_scenarios.LQG_QKF import *
# from LQG_QKF.test_scenarios.stateDynamics import *
from LQG_QKF import *
from stateDynamics import *
from typing import Literal
from tqdm import tqdm
import numpy as np

random_seed = 42
small_value = 1e-6
sensor_update_interval = 10

os.chdir(os.path.dirname(os.path.abspath(__file__)))
pkl_dir = 'pkl/'
os.makedirs(pkl_dir, exist_ok=True)
perf_dir = 'perf/'
os.makedirs(perf_dir, exist_ok=True)

# build sensor selection simulator
class SensorSelectionSimulator(LQG):
    def __init__(self, n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, H = 50, 
                 filter_type: Literal['qkf', 'ekf', 'kf', 'ukf'] = 'qkf',
                 lqr_type: Literal['orig', 'aug_analytic', 'aug_numeric', 'None'] = 'orig'):
        
        self.filter_type = filter_type
        self.lqr_type = lqr_type
        # dynamics setting
        self.F = StateDynamics(n1, n2, p , W, A_E, A_S, B_S)
        n = n1 + n2 # state size
        
        # sensor settings
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
        
        # performance tracking
        self.performance_history = {
            'cost': [],
            'estimation_error': [],
            'control_effort': [],
            'sensor_selections': [],
            'information_gains': [],
            'covariance_traces': []
        } 
        
    def get_next_sensor_selection(self):
        """
        Compute next sensor selection using greedy algorithm.
        Selects sensor configuration that maximizes information gain or minimizes uncertainty.
        Returns the goal state based on the selected sensor configuration.
        """
        # Get current state estimate
        x_current = self.x_hat
        
        # Define possible sensor configurations (simplified example)
        # In practice, this could be more sophisticated based on available sensors
        sensor_configs = self._generate_sensor_configurations()
        
        best_config = None
        best_value = -np.inf
        info_gains = []
        
        # Greedy selection: choose configuration that maximizes information gain
        for config in sensor_configs:
            # Calculate information gain for this configuration
            info_gain = self._calculate_information_gain(config, x_current)
            info_gains.append(info_gain)
            
            if info_gain > best_value:
                best_value = info_gain
                best_config = config
        
        # Record sensor selection performance
        self.performance_history['sensor_selections'].append(best_config['type'])
        self.performance_history['information_gains'].append(info_gains)
        
        # Generate goal state based on selected sensor configuration
        goal_state = self._generate_goal_from_config(best_config, x_current)
        
        return goal_state
    
    def _generate_sensor_configurations(self):
        """
        Generate possible sensor configurations.
        This is a simplified example - in practice, this would depend on
        the specific sensor setup and constraints.
        """
        configs = []
        
        # Configuration 1: Focus on earth state (first n1 components)
        config1 = {
            'type': 'earth_focus',
            'target_components': list(range(self.n1)),
            'weight': 1.0
        }
        configs.append(config1)
        
        # Configuration 2: Focus on sensor state (last n2 components)  
        config2 = {
            'type': 'sensor_focus',
            'target_components': list(range(self.n1, self.n)),
            'weight': 1.0
        }
        configs.append(config2)
        
        # Configuration 3: Balanced approach
        config3 = {
            'type': 'balanced',
            'target_components': list(range(self.n)),
            'weight': 0.5
        }
        configs.append(config3)
        
        return configs
    
    def _calculate_information_gain(self, config, x_current):
        """
        Calculate information gain for a given sensor configuration.
        Uses trace of covariance matrix as uncertainty measure.
        """
        # Get current uncertainty (trace of covariance matrix)
        if self.filter_type == 'qkf':
            current_uncertainty = np.trace(self.Pz_est[:self.n, :self.n])
        else:
            current_uncertainty = np.trace(self.P_est)
        
        # Simulate information gain based on configuration
        # Higher weight on target components leads to more information gain
        target_components = config['target_components']
        weight = config['weight']
        
        # Calculate information gain as reduction in uncertainty
        # This is a simplified model - in practice, you'd compute the actual
        # posterior covariance after incorporating the sensor measurement
        info_gain = weight * len(target_components) / self.n * current_uncertainty
        
        # Add some randomness to avoid always selecting the same configuration
        info_gain += np.random.normal(0, 0.1 * info_gain)
        
        return info_gain
    
    def _generate_goal_from_config(self, config, x_current):
        """
        Generate goal state based on selected sensor configuration.
        """
        goal_state = x_current.copy()
        
        if config['type'] == 'earth_focus':
            # Goal: move earth state towards zero, keep sensor state as is
            goal_state[:self.n1] = np.zeros((self.n1, 1))
            
        elif config['type'] == 'sensor_focus':
            # Goal: move sensor state towards zero, keep earth state as is
            goal_state[self.n1:] = np.zeros((self.n2, 1))
            
        elif config['type'] == 'balanced':
            # Goal: move entire state towards zero
            goal_state = np.zeros((self.n, 1))
            
        # Add some small random perturbation to avoid getting stuck
        noise_scale = 0.1
        goal_state += np.random.normal(0, noise_scale, goal_state.shape)
        
        return goal_state
    
    def run_sim(self):
        """
        Run the sensor selection simulation with comprehensive performance tracking.
        """
        print(f"Starting sensor selection simulation with {self.filter_type} filter and {self.lqr_type} LQR...")
        
        for step in tqdm(range(1, self.H + 1, 1), desc="Simulation Progress"):
            # Update state estimation
            self.update_lqe()
            
            # Update sensor selection periodically
            if step % sensor_update_interval == 0:
                self.x_goal = self.get_next_sensor_selection()
            
            # Update control and forward dynamics
            if self.lqr_type != 'None':
                self.update_lqr()
                self.forward_state()
            
            # Record comprehensive performance metrics
            self._record_performance_metrics(step)
            
        # Calculate cost-to-go
        cost_to_go_list = self._calculate_cost_to_go()
        
        # Calculate average metrics
        avg_cost = np.mean(self.performance_history['cost'])
        avg_error = np.mean(self.performance_history['estimation_error'])
        
        print(f"Simulation completed. Average cost: {avg_cost:.4f}")
        print(f"Average estimation error: {avg_error:.4f}")
        
        return (self.performance_history['estimation_error'], 
                self.performance_history['covariance_traces'], 
                cost_to_go_list)
    
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
    
    def save_results(self, filename=None, save_dir=pkl_dir):
        """
        Save simulation results to pickle file.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sensor_selection_results_{self.filter_type}_{self.lqr_type}_{timestamp}.pkl"
        
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


def run_sensor_scheduling_sim(H=1000, noise_scale=1e-1, m_scale=1e2, Q_scale=1.0, R_scale=1.0, rand_seed=random_seed):
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
    
    # goal_state = generate_goal_state(np.zeros((n1, 1)), n2) # goal state vector
    # lqg_kf_sys = LQG(n, p, W, A, B, C, M, V, Q, R, H=1000, filter_type='kf')
    # err_list_kf = lqg_kf_sys.run_sim()
    # plt.plot(err_list_kf, label=f'kf measure error')

    # Run simulations with different filter types
    simulators = {}
    
    print("Running EKF simulation...")
    lqg_ekf = SensorSelectionSimulator(n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, H=H, filter_type='ekf', lqr_type='orig')
    err_list_ekf, var_list_ekf, cost_list_ekf = lqg_ekf.run_sim()
    simulators['ekf'] = lqg_ekf
    
    print("Running QKF with augmented numeric LQR...")
    lqg_qkf_aug_num = SensorSelectionSimulator(n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, H=H, filter_type='qkf', lqr_type='aug_numeric')
    err_list_aug_num, var_list_aug_num, cost_list_aug_num = lqg_qkf_aug_num.run_sim()
    simulators['qkf_aug_num'] = lqg_qkf_aug_num
    
    print("Running QKF with augmented analytic LQR...")
    lqg_qkf_aug_analytic = SensorSelectionSimulator(n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, H=H, filter_type='qkf', lqr_type='aug_analytic')
    err_list_aug_analytic, var_list_aug_analytic, cost_list_aug_analytic = lqg_qkf_aug_analytic.run_sim()
    simulators['qkf_aug_analytic'] = lqg_qkf_aug_analytic
    
    print("Running UKF simulation...")
    lqg_ukf = SensorSelectionSimulator(n1, n2, p, W, A_E, A_S, B_S, C, M, V, Q, R, H=H, filter_type='ukf', lqr_type='orig')
    err_list_ukf, var_list_ukf, cost_list_ukf = lqg_ukf.run_sim()
    simulators['ukf'] = lqg_ukf
    
    # Save results for each simulator
    print("\nSaving simulation results...")
    for name, simulator in simulators.items():
        print(f"Saving {name} results...")
        simulator.save_results()
    
    # Create comparison plots only
    plot_comparison(simulators, save_plots=True)
    
    return [err_list_ekf, var_list_ekf, cost_list_ekf], [err_list_aug_num, var_list_aug_num, cost_list_aug_num], [err_list_aug_analytic, var_list_aug_analytic, cost_list_aug_analytic], [err_list_ukf, var_list_ukf, cost_list_ukf]


def plot_comparison(simulators, save_plots=True, plot_dir=perf_dir):
    """
    Create simple comparison plots across different filter types.
    """
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Sensor Selection Performance Comparison', fontsize=14)
    
    colors = ['blue', 'red', 'green', 'orange']
    linestyles = ['-', '--', '-.', ':']
    
    # 1. Cost-to-go comparison
    for i, (name, sim) in enumerate(simulators.items()):
        time_steps = np.arange(1, len(sim.performance_history['cost']) + 1)
        # Calculate cost-to-go for this simulator
        cost_to_go = []
        for j in range(len(sim.performance_history['cost'])):
            cost_to_go.append(np.sum(sim.performance_history['cost'][j:]))
        
        axes[0].plot(time_steps, cost_to_go, 
                    color=colors[i % len(colors)], 
                    linestyle=linestyles[i % len(linestyles)],
                    label=name, linewidth=2)
    
    axes[0].set_title('Cost-to-Go Comparison')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Cost-to-Go')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Estimation error comparison
    for i, (name, sim) in enumerate(simulators.items()):
        time_steps = np.arange(1, len(sim.performance_history['estimation_error']) + 1)
        axes[1].plot(time_steps, sim.performance_history['estimation_error'], 
                    color=colors[i % len(colors)], 
                    linestyle=linestyles[i % len(linestyles)],
                    label=name, linewidth=2)
    
    axes[1].set_title('Estimation Error Comparison')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('||x_true - x_est||')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{plot_dir}/sensor_selection_comparison_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Comparison plots saved to: {filename}")
    
    plt.show()
    return fig

if __name__ == "__main__":
    run_sensor_scheduling_sim()