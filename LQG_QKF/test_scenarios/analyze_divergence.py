import numpy as np
import pickle
import os
import glob
import matplotlib.pyplot as plt

def analyze_divergence_patterns(pkl_dir, path_type='figure8'):
    """
    Analyze divergence patterns and their impact on execution times.
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
    
    # Analyze each controller
    controllers = ['A', 'B', 'C', 'D']
    controller_names = {
        'A': 'QKF + Aug-LQR-Analytic',
        'B': 'QKF + Aug-iLQR-Numeric', 
        'C': 'EKF + Orig-LQR',
        'D': 'UKF + Orig-LQR'
    }
    
    divergence_analysis = {}
    
    for controller in controllers:
        execution_times = []
        divergence_steps = []
        divergence_rates = []
        completed_trials = 0
        
        for i, results in enumerate(results_list):
            if controller in results:
                # Execution time
                exec_time = results[controller]['execution_time']
                execution_times.append(exec_time)
                
                # Divergence info
                is_diverged = results[controller]['is_diverged']
                divergence_step = results[controller]['divergence_step']
                
                if is_diverged:
                    divergence_steps.append(divergence_step)
                    divergence_rates.append(1.0)  # Diverged
                else:
                    divergence_steps.append(201)  # Completed full trial
                    divergence_rates.append(0.0)  # Not diverged
                    completed_trials += 1
        
        divergence_analysis[controller] = {
            'name': controller_names[controller],
            'execution_times': execution_times,
            'divergence_steps': divergence_steps,
            'divergence_rates': divergence_rates,
            'completed_trials': completed_trials,
            'total_trials': len(results_list),
            'divergence_percentage': (len(results_list) - completed_trials) / len(results_list) * 100,
            'mean_execution_time': np.mean(execution_times),
            'mean_divergence_step': np.mean(divergence_steps)
        }
    
    # Print analysis
    print("\n" + "="*80)
    print("DIVERGENCE ANALYSIS")
    print("="*80)
    
    for controller, data in divergence_analysis.items():
        print(f"\n{data['name']}:")
        print(f"  Divergence Rate: {data['divergence_percentage']:.1f}% ({data['total_trials'] - data['completed_trials']}/{data['total_trials']} trials)")
        print(f"  Mean Execution Time: {data['mean_execution_time']:.4f} seconds")
        print(f"  Mean Divergence Step: {data['mean_divergence_step']:.1f} (out of 201)")
        print(f"  Completed Trials: {data['completed_trials']}/{data['total_trials']}")
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Execution times vs divergence rates
    controllers_list = list(divergence_analysis.keys())
    exec_times = [divergence_analysis[c]['mean_execution_time'] for c in controllers_list]
    div_rates = [divergence_analysis[c]['divergence_percentage'] for c in controllers_list]
    names = [divergence_analysis[c]['name'] for c in controllers_list]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax1.bar(range(len(controllers_list)), exec_times, color=colors)
    ax1.set_xlabel('Controller')
    ax1.set_ylabel('Mean Execution Time (seconds)')
    ax1.set_title('Execution Time vs Divergence Rate')
    ax1.set_xticks(range(len(controllers_list)))
    ax1.set_xticklabels([f'{name.split()[0]}\n{name.split()[1]}' for name in names], rotation=45, ha='right')
    
    # Add divergence rate as text on bars
    for i, (bar, div_rate) in enumerate(zip(bars, div_rates)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{div_rate:.1f}%\ndiverged', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Divergence step distribution
    for i, controller in enumerate(controllers_list):
        div_steps = divergence_analysis[controller]['divergence_steps']
        ax2.hist(div_steps, bins=20, alpha=0.6, label=names[i].split()[0], color=colors[i])
    
    ax2.set_xlabel('Divergence Step')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Divergence Steps')
    ax2.legend()
    ax2.axvline(x=201, color='black', linestyle='--', alpha=0.7, label='Full Trial (201 steps)')
    
    # Plot 3: Execution time vs divergence step correlation
    for i, controller in enumerate(controllers_list):
        exec_times = divergence_analysis[controller]['execution_times']
        div_steps = divergence_analysis[controller]['divergence_steps']
        ax3.scatter(div_steps, exec_times, color=colors[i], label=names[i].split()[0], alpha=0.7)
    
    ax3.set_xlabel('Divergence Step')
    ax3.set_ylabel('Execution Time (seconds)')
    ax3.set_title('Execution Time vs Divergence Step')
    ax3.legend()
    ax3.axvline(x=201, color='black', linestyle='--', alpha=0.7, label='Full Trial')
    
    # Plot 4: Summary table
    ax4.axis('off')
    table_data = []
    for controller in controllers_list:
        data = divergence_analysis[controller]
        table_data.append([
            data['name'].split()[0],  # Short name
            f"{data['divergence_percentage']:.1f}%",
            f"{data['mean_execution_time']:.4f}s",
            f"{data['mean_divergence_step']:.0f}",
            f"{data['completed_trials']}/{data['total_trials']}"
        ])
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Controller', 'Divergence\nRate', 'Mean Time', 'Mean Step', 'Completed'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title('Summary Statistics')
    
    plt.tight_layout()
    plt.savefig(os.path.join(pkl_dir, 'divergence_analysis.png'), dpi=200, bbox_inches='tight')
    plt.show()
    
    return divergence_analysis

if __name__ == "__main__":
    pkl_dir = "pkl"
    analysis = analyze_divergence_patterns(pkl_dir)
