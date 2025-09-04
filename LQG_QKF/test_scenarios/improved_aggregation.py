import numpy as np
import pickle
import os
import glob
import matplotlib.pyplot as plt

def aggregate_time_series_robust(results_list, metric_name, controller_keys=['A', 'B', 'C', 'D']):
    """
    Improved time series aggregation that properly handles diverged trials.
    """
    aggregated = {}
    
    for controller in controller_keys:
        time_series_list = []
        valid_trials = 0
        
        for results in results_list:
            if controller in results and metric_name in results[controller]:
                series = results[controller][metric_name]
                series = np.array(series, dtype=float)
                
                # Find the last valid (non-NaN) value
                valid_mask = ~np.isnan(series)
                if np.any(valid_mask):
                    last_valid_idx = np.where(valid_mask)[0][-1]
                    # Only use data up to the last valid point
                    truncated_series = series[:last_valid_idx + 1]
                    time_series_list.append(truncated_series)
                    valid_trials += 1
        
        if time_series_list:
            # Find the minimum length among valid portions
            min_length = min(len(series) for series in time_series_list)
            truncated_series = [series[:min_length] for series in time_series_list]
            
            # Stack all series and compute statistics
            stacked = np.array(truncated_series)
            
            aggregated[controller] = {
                'mean': np.mean(stacked, axis=0),
                'std': np.std(stacked, axis=0),
                'median': np.median(stacked, axis=0),
                'q25': np.percentile(stacked, 25, axis=0),
                'q75': np.percentile(stacked, 75, axis=0),
                'min': np.min(stacked, axis=0),
                'max': np.max(stacked, axis=0),
                'valid_trials': valid_trials,
                'length': min_length,
                'method': 'robust_truncation'
            }
        else:
            aggregated[controller] = None
    
    return aggregated

def aggregate_time_series_separate(results_list, metric_name, controller_keys=['A', 'B', 'C', 'D']):
    """
    Separate aggregation for successful vs diverged trials.
    """
    aggregated = {}
    
    for controller in controller_keys:
        successful_trials = []
        diverged_trials = []
        
        for results in results_list:
            if controller in results and metric_name in results[controller]:
                series = results[controller][metric_name]
                series = np.array(series, dtype=float)
                is_diverged = results[controller]['is_diverged']
                
                if is_diverged:
                    # For diverged trials, use data up to divergence point
                    valid_mask = ~np.isnan(series)
                    if np.any(valid_mask):
                        last_valid_idx = np.where(valid_mask)[0][-1]
                        diverged_trials.append(series[:last_valid_idx + 1])
                else:
                    # For successful trials, use all data
                    successful_trials.append(series)
        
        aggregated[controller] = {
            'successful_trials': successful_trials,
            'diverged_trials': diverged_trials,
            'successful_count': len(successful_trials),
            'diverged_count': len(diverged_trials),
            'total_count': len(successful_trials) + len(diverged_trials)
        }
        
        # Compute statistics for successful trials only
        if successful_trials:
            min_length = min(len(series) for series in successful_trials)
            truncated_successful = [series[:min_length] for series in successful_trials]
            stacked_successful = np.array(truncated_successful)
            
            aggregated[controller]['successful_stats'] = {
                'mean': np.mean(stacked_successful, axis=0),
                'std': np.std(stacked_successful, axis=0),
                'median': np.median(stacked_successful, axis=0),
                'length': min_length
            }
        
        # Compute statistics for diverged trials (up to divergence point)
        if diverged_trials:
            min_length = min(len(series) for series in diverged_trials)
            truncated_diverged = [series[:min_length] for series in diverged_trials]
            stacked_diverged = np.array(truncated_diverged)
            
            aggregated[controller]['diverged_stats'] = {
                'mean': np.mean(stacked_diverged, axis=0),
                'std': np.std(stacked_diverged, axis=0),
                'median': np.median(stacked_diverged, axis=0),
                'length': min_length
            }
    
    return aggregated

def plot_improved_aggregation(aggregated_data, metric_name, time_axis=None, save_path=None):
    """
    Plot improved aggregation results.
    """
    controller_labels = {
        'A': 'QKF + Aug-LQR-Analytic',
        'B': 'QKF + Aug-iLQR-Numeric', 
        'C': 'EKF + Orig-LQR',
        'D': 'UKF + Orig-LQR'
    }
    
    colors = {
        'A': '#1f77b4',  # Blue
        'B': '#ff7f0e',  # Orange
        'C': '#2ca02c',  # Green
        'D': '#d62728'   # Red
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Successful trials only
    for controller, data in aggregated_data.items():
        if data is not None and 'successful_stats' in data:
            stats = data['successful_stats']
            mean = stats['mean']
            std = stats['std']
            length = stats['length']
            
            if time_axis is None:
                time_axis = np.arange(length)
            
            ax1.plot(time_axis, mean, color=colors[controller], 
                    label=f"{controller_labels[controller]} (n={data['successful_count']})", 
                    linewidth=2)
            ax1.fill_between(time_axis, mean - std, mean + std, 
                           color=colors[controller], alpha=0.2)
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel(metric_name.replace('_', ' ').title())
    ax1.set_title(f'{metric_name.replace("_", " ").title()} - Successful Trials Only')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Comparison of successful vs diverged
    for controller, data in aggregated_data.items():
        if data is not None:
            if 'successful_stats' in data and 'diverged_stats' in data:
                # Plot successful trials
                succ_stats = data['successful_stats']
                succ_mean = succ_stats['mean']
                succ_length = succ_stats['length']
                succ_time = np.arange(succ_length)
                
                ax2.plot(succ_time, succ_mean, color=colors[controller], 
                        linestyle='-', linewidth=2, 
                        label=f"{controller_labels[controller]} (successful)")
                
                # Plot diverged trials
                div_stats = data['diverged_stats']
                div_mean = div_stats['mean']
                div_length = div_stats['length']
                div_time = np.arange(div_length)
                
                ax2.plot(div_time, div_mean, color=colors[controller], 
                        linestyle='--', linewidth=2, alpha=0.7,
                        label=f"{controller_labels[controller]} (diverged)")
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel(metric_name.replace('_', ' ').title())
    ax2.set_title(f'{metric_name.replace("_", " ").title()} - Successful vs Diverged')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    return fig, (ax1, ax2)

def analyze_improved_aggregation(pkl_dir, path_type='figure8'):
    """
    Run improved aggregation analysis.
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
    
    print(f"Loaded {len(results_list)} trials")
    
    # Run improved aggregation
    metrics = ['rmse', 'track_err', 'stage_cost']
    
    for metric in metrics:
        print(f"\nAnalyzing {metric} with improved aggregation...")
        
        # Separate successful vs diverged trials
        agg_data = aggregate_time_series_separate(results_list, metric)
        
        # Print summary
        for controller, data in agg_data.items():
            if data is not None:
                print(f"  Controller {controller}:")
                print(f"    Successful: {data['successful_count']}/{data['total_count']} trials")
                print(f"    Diverged: {data['diverged_count']}/{data['total_count']} trials")
        
        # Plot results
        fig, (ax1, ax2) = plot_improved_aggregation(agg_data, metric)
        save_path = os.path.join(pkl_dir, f"improved_aggregation_{metric}_{path_type}.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")

if __name__ == "__main__":
    pkl_dir = "pkl"
    analyze_improved_aggregation(pkl_dir)
