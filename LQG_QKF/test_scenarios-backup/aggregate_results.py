import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from scipy import stats
import glob

def load_all_trials(pkl_dir, path_type='figure8', max_trials=None):
    """
    Load all trial results from pickle files.
    
    Args:
        pkl_dir: Directory containing pickle files
        path_type: Type of path (e.g., 'figure8')
        max_trials: Maximum number of trials to load (None for all)
    
    Returns:
        list: List of results dictionaries
    """
    pattern = os.path.join(pkl_dir, f"tracking_{path_type}-*.pkl")
    pkl_files = sorted(glob.glob(pattern))
    
    if max_trials is not None:
        pkl_files = pkl_files[:max_trials]
    
    results_list = []
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                results = pickle.load(f)
                results_list.append(results)
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
    
    return results_list

def aggregate_metrics(results_list, metric_name, controller_keys=['A', 'B', 'C', 'D']):
    """
    Aggregate a specific metric across all trials for all controllers.
    
    Args:
        results_list: List of results dictionaries from multiple trials
        metric_name: Name of metric to aggregate (e.g., 'rmse', 'execution_time')
        controller_keys: List of controller keys to process
    
    Returns:
        dict: Aggregated statistics for each controller
    """
    aggregated = {}
    
    for controller in controller_keys:
        values = []
        for results in results_list:
            if controller in results and metric_name in results[controller]:
                value = results[controller][metric_name]
                if isinstance(value, (int, float)) and not np.isnan(value):
                    values.append(value)
        
        if values:
            values = np.array(values)
            aggregated[controller] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values),
                'q25': np.percentile(values, 25),
                'q75': np.percentile(values, 75),
                'count': len(values),
                'raw_values': values
            }
        else:
            aggregated[controller] = None
    
    return aggregated

def aggregate_time_series(results_list, metric_name, controller_keys=['A', 'B', 'C', 'D']):
    """
    Aggregate time series data across trials.
    
    Args:
        results_list: List of results dictionaries
        metric_name: Name of time series metric (e.g., 'rmse', 'track_err')
        controller_keys: List of controller keys
    
    Returns:
        dict: Aggregated time series statistics
    """
    aggregated = {}
    
    for controller in controller_keys:
        time_series_list = []
        valid_trials = 0
        
        for results in results_list:
            if controller in results and metric_name in results[controller]:
                series = results[controller][metric_name]
                # Convert to numpy array and filter out NaN values
                series = np.array(series, dtype=float)
                if len(series) > 0 and not np.all(np.isnan(series)):
                    time_series_list.append(series)
                    valid_trials += 1
        
        if time_series_list:
            # Find the minimum length to avoid issues with different trial lengths
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
                'length': min_length
            }
        else:
            aggregated[controller] = None
    
    return aggregated

def compute_confidence_interval(values, confidence=0.95):
    """Compute confidence interval for a set of values."""
    if len(values) < 2:
        return np.nan, np.nan
    
    mean = np.mean(values)
    std_err = stats.sem(values)
    h = std_err * stats.t.ppf((1 + confidence) / 2, len(values) - 1)
    
    return mean - h, mean + h

def create_summary_table(results_list, controller_keys=['A', 'B', 'C', 'D']):
    """
    Create a comprehensive summary table of all metrics.
    """
    metrics = ['execution_time', 'rmse', 'track_err', 'stage_cost', 'cost_to_go']
    
    summary = {}
    for controller in controller_keys:
        summary[controller] = {}
        for metric in metrics:
            if metric == 'execution_time':
                # Scalar metric
                agg = aggregate_metrics(results_list, metric, [controller])
                if agg[controller] is not None:
                    summary[controller][metric] = agg[controller]
            else:
                # Time series metric - get final values
                values = []
                for results in results_list:
                    if controller in results and metric in results[controller]:
                        series = results[controller][metric]
                        if isinstance(series, list) and len(series) > 0:
                            # Get the last non-NaN value
                            series = np.array(series, dtype=float)
                            valid_values = series[~np.isnan(series)]
                            if len(valid_values) > 0:
                                values.append(valid_values[-1])
                
                if values:
                    values = np.array(values)
                    summary[controller][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'median': np.median(values),
                        'count': len(values)
                    }
    
    return summary

def plot_aggregated_results(aggregated_data, metric_name, time_axis=None, 
                          controller_labels=None, save_path=None):
    """
    Plot aggregated time series results with confidence bands.
    """
    if controller_labels is None:
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
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for controller, data in aggregated_data.items():
        if data is not None:
            mean = data['mean']
            std = data['std']
            q25 = data['q25']
            q75 = data['q75']
            
            if time_axis is None:
                time_axis = np.arange(len(mean))
            
            # Plot mean line
            ax.plot(time_axis, mean, color=colors[controller], 
                   label=controller_labels[controller], linewidth=2)
            
            # Plot confidence band (25th-75th percentile)
            ax.fill_between(time_axis, q25, q75, 
                           color=colors[controller], alpha=0.2)
            
            # Plot std band
            ax.fill_between(time_axis, mean - std, mean + std, 
                           color=colors[controller], alpha=0.1)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel(metric_name.replace('_', ' ').title())
    ax.set_title(f'Aggregated {metric_name.replace("_", " ").title()} Across Trials')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    return fig, ax

# Example usage function
def analyze_multiple_trials(pkl_dir, path_type='figure8', max_trials=None):
    """
    Complete analysis of multiple trials.
    """
    print(f"Loading trials from {pkl_dir}...")
    results_list = load_all_trials(pkl_dir, path_type, max_trials)
    print(f"Loaded {len(results_list)} trials")
    
    if len(results_list) == 0:
        print("No trials found!")
        return
    
    # 1. Execution time analysis
    print("\n=== Execution Time Analysis ===")
    time_agg = aggregate_metrics(results_list, 'execution_time')
    for controller, data in time_agg.items():
        if data is not None:
            print(f"Controller {controller}: {data['mean']:.4f} ± {data['std']:.4f} seconds "
                  f"(n={data['count']})")
    
    # 2. Final performance metrics
    print("\n=== Final Performance Metrics ===")
    summary = create_summary_table(results_list)
    for controller, metrics in summary.items():
        print(f"\nController {controller}:")
        for metric, stats in metrics.items():
            if stats is not None:
                print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    # 3. Time series analysis
    print("\n=== Time Series Analysis ===")
    time_series_metrics = ['rmse', 'track_err', 'stage_cost']
    for metric in time_series_metrics:
        print(f"\nAnalyzing {metric}...")
        agg_data = aggregate_time_series(results_list, metric)
        
        # Plot results
        fig, ax = plot_aggregated_results(agg_data, metric)
        save_path = os.path.join(pkl_dir, f"aggregated_{metric}_{path_type}.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {save_path}")
    
    return results_list, summary

if __name__ == "__main__":
    # Example usage
    pkl_dir = "pkl"
    results_list, summary = analyze_multiple_trials(pkl_dir, max_trials=10)
