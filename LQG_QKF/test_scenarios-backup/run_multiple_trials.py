import numpy as np
import os
from LQG_QKF import one_trial
import time

def run_multiple_trials(num_trials=10, **kwargs):
    """
    Run multiple trials with different random seeds and aggregate results.
    
    Args:
        num_trials: Number of trials to run
        **kwargs: Additional arguments to pass to one_trial()
    """
    print(f"Running {num_trials} trials...")
    
    # Default parameters
    default_params = {
        'H': 200,
        'process_noise_scale': 1e-3,
        'measurement_noise_scale': 1e-3,
        'nonlinearity_scale': 1e3,
        'Q_scale': 1e-2,
        'R_scale': 1e-2
    }
    
    # Update with any provided parameters
    default_params.update(kwargs)
    
    # Run trials with different random seeds
    for trial_num in range(num_trials):
        print(f"\n{'='*50}")
        print(f"TRIAL {trial_num + 1}/{num_trials}")
        print(f"{'='*50}")
        
        # Use trial number as random seed for reproducibility
        one_trial(rand_seed=trial_num, trial_num=trial_num, **default_params)
    
    print(f"\n{'='*50}")
    print(f"COMPLETED {num_trials} TRIALS")
    print(f"{'='*50}")

def run_parameter_sweep():
    """
    Run trials with different parameter combinations.
    """
    # Different noise levels
    noise_levels = [1e-4, 1e-3, 1e-2]
    
    # Different nonlinearity scales
    nonlinearity_scales = [1e1, 1e2, 1e3]
    
    trial_count = 0
    
    for noise_scale in noise_levels:
        for nonlinearity_scale in nonlinearity_scales:
            print(f"\n{'='*60}")
            print(f"PARAMETER SWEEP: noise={noise_scale}, nonlinearity={nonlinearity_scale}")
            print(f"{'='*60}")
            
            run_multiple_trials(
                num_trials=5,  # Fewer trials per parameter combination
                process_noise_scale=noise_scale,
                measurement_noise_scale=noise_scale,
                nonlinearity_scale=nonlinearity_scale,
                trial_num_offset=trial_count
            )
            
            trial_count += 5

if __name__ == "__main__":
    # Run multiple trials with default parameters
    run_multiple_trials(num_trials=10)
    
    # Uncomment to run parameter sweep
    # run_parameter_sweep()
