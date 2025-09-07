import numpy as np
import pickle
import os
import glob

def check_trial_lengths(pkl_dir, path_type='figure8'):
    """
    Check the actual lengths of trials for each controller.
    """
    pattern = os.path.join(pkl_dir, f"tracking_{path_type}-*.pkl")
    pkl_files = sorted(glob.glob(pattern))
    
    results_list = []
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                results = pickle.load(f)
                results_list.append(results)
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
    
    controllers = ['A', 'B', 'C', 'D']
    controller_names = {
        'A': 'QKF + Aug-LQR-Analytic',
        'B': 'QKF + Aug-iLQR-Numeric', 
        'C': 'EKF + Orig-LQR',
        'D': 'UKF + Orig-LQR'
    }
    
    print("TRIAL LENGTH ANALYSIS")
    print("="*60)
    
    for controller in controllers:
        print(f"\n{controller_names[controller]}:")
        
        lengths = []
        divergence_steps = []
        
        for i, results in enumerate(results_list):
            if controller in results:
                # Get the length of the time series (e.g., rmse)
                rmse_series = results[controller]['rmse']
                length = len(rmse_series)
                lengths.append(length)
                
                # Get divergence step
                div_step = results[controller]['divergence_step']
                is_diverged = results[controller]['is_diverged']
                divergence_steps.append(div_step if is_diverged else 201)
        
        print(f"  Trial lengths: {lengths}")
        print(f"  Min length: {min(lengths)}")
        print(f"  Max length: {max(lengths)}")
        print(f"  Mean length: {np.mean(lengths):.1f}")
        print(f"  Divergence steps: {divergence_steps}")
        print(f"  Min divergence step: {min(divergence_steps)}")
        
        # Show which trials are truncated
        min_length = min(lengths)
        truncated_trials = [i for i, length in enumerate(lengths) if length > min_length]
        if truncated_trials:
            print(f"  Trials truncated to {min_length}: {truncated_trials}")
        else:
            print(f"  No truncation needed (all trials same length)")

if __name__ == "__main__":
    pkl_dir = "pkl"
    check_trial_lengths(pkl_dir)
