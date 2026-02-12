"""
Utility functions for Eureka MWE
"""
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any, List

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def create_output_dirs(base_dir: str):
    """Create necessary output directories"""
    
    dirs = [
        base_dir,
        os.path.join(base_dir, 'plots'),
        os.path.join(base_dir, 'metrics'),
        'rewards'
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    print(f"✓ Output directories created in: {base_dir}")


def plot_results(results: Dict[str, Any], output_dir: str):
    """Generate visualization plots from Eureka results"""
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70 + "\n")
    
    # Extract data
    baselines = results['baselines']
    best_per_iteration = results['best_per_iteration']
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Progression over iterations
    ax1 = axes[0]
    iterations = [r['iteration'] + 1 for r in best_per_iteration]
    performances = [r['performance'] for r in best_per_iteration]
    stds = [r.get('std', 0) for r in best_per_iteration]
    
    ax1.plot(iterations, performances, 'o-', linewidth=2, markersize=8, 
             label='Eureka Best', color='#2E86AB')
    ax1.fill_between(iterations, 
                     np.array(performances) - np.array(stds),
                     np.array(performances) + np.array(stds),
                     alpha=0.2, color='#2E86AB')
    
    # Add baseline lines
    for name, data in baselines.items():
        perf = data['performance']
        ax1.axhline(y=perf, linestyle='--', linewidth=2, 
                   label=f'{name.capitalize()} baseline', alpha=0.7)
    
    ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
    ax1.set_title('Eureka Reward Evolution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Performance comparison
    ax2 = axes[1]
    
    # Collect all performances
    labels = []
    values = []
    colors = []
    
    # Baselines
    for name, data in baselines.items():
        labels.append(f'{name.capitalize()}\nBaseline')
        values.append(data['performance'])
        colors.append('#A23B72')
    
    # Best Eureka rewards
    for i, r in enumerate(best_per_iteration):
        labels.append(f'Eureka\nIter {i+1}')
        values.append(r['performance'])
        colors.append('#2E86AB')
    
    bars = ax2.bar(range(len(labels)), values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
    ax2.set_title('Performance Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'plots', 'eureka_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_path}")
    
    plt.close()
    
    # Create detailed progression plot
    plot_detailed_progression(results, output_dir)


def plot_detailed_progression(results: Dict[str, Any], output_dir: str):
    """Create detailed plot showing all samples per iteration"""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract all results
    all_results = results.get('all_iterations', [])
    
    if not all_results:
        return
    
    # Group by iteration
    iterations = {}
    for r in all_results:
        iter_num = r['iteration']
        if iter_num not in iterations:
            iterations[iter_num] = []
        iterations[iter_num].append(r['performance'])
    
    # Plot box plots for each iteration
    positions = []
    data = []
    for iter_num in sorted(iterations.keys()):
        positions.append(iter_num + 1)
        data.append(iterations[iter_num])
    
    bp = ax.boxplot(data, positions=positions, widths=0.6,
                   patch_artist=True, showmeans=True)
    
    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('#2E86AB')
        patch.set_alpha(0.6)
    
    # Add baseline references
    baselines = results['baselines']
    for name, data in baselines.items():
        ax.axhline(y=data['performance'], linestyle='--', linewidth=2, 
                  label=f'{name.capitalize()} baseline', alpha=0.7)
    
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
    ax.set_title('Reward Distribution Across Iterations', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plot_path = os.path.join(output_dir, 'plots', 'reward_distribution.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Distribution plot saved to: {plot_path}")
    
    plt.close()


def save_results(results: Dict[str, Any], output_dir: str):
    """Save detailed results to JSON"""
    
    output_file = os.path.join(output_dir, 'detailed_results.json')
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Detailed results saved to: {output_file}")


def print_reward_analysis(reward_code: str):
    """Print analysis of a reward function"""
    
    print("\n" + "-"*60)
    print("REWARD FUNCTION ANALYSIS")
    print("-"*60)
    print(reward_code)
    print("-"*60 + "\n")


def compare_rewards(reward_codes: List[str], performances: List[float]):
    """Compare multiple reward functions"""
    
    print("\n" + "="*70)
    print("REWARD COMPARISON")
    print("="*70 + "\n")
    
    for i, (code, perf) in enumerate(zip(reward_codes, performances)):
        print(f"Reward {i+1} (Performance: {perf:.2f}):")
        print("-" * 60)
        print(code)
        print("-" * 60 + "\n")


def create_summary_table(results: Dict[str, Any], output_dir: str):
    """Create a summary table of results"""
    
    # Create markdown table
    table = "# Eureka MWE Results Summary\n\n"
    table += "## Configuration\n"
    table += f"- Environment: {results['config']['env_name']}\n"
    table += f"- Iterations: {results['config']['num_iterations']}\n"
    table += f"- Samples per iteration: {results['config']['num_samples']}\n"
    table += f"- Total timesteps per training: {results['config']['total_timesteps']}\n\n"
    
    table += "## Baseline Performance\n"
    table += "| Reward Type | Mean Reward | Std |\n"
    table += "|-------------|-------------|-----|\n"
    for name, data in results['baselines'].items():
        metrics = data['metrics']
        table += f"| {name.capitalize()} | {metrics['mean_reward']:.2f} | {metrics['std_reward']:.2f} |\n"
    
    table += "\n## Eureka Progression\n"
    table += "| Iteration | Best Reward | Std |\n"
    table += "|-----------|-------------|-----|\n"
    for best in results['best_per_iteration']:
        table += f"| {best['iteration'] + 1} | {best['performance']:.2f} | {best.get('std', 0):.2f} |\n"
    
    # Save table
    table_path = os.path.join(output_dir, 'summary.md')
    with open(table_path, 'w') as f:
        f.write(table)
    
    print(f"✓ Summary table saved to: {table_path}")


def log_iteration_details(iteration: int, results: List[Dict], output_dir: str):
    """Log detailed results for an iteration"""
    
    log_dir = os.path.join(output_dir, 'metrics')
    log_file = os.path.join(log_dir, f'iteration_{iteration}.json')
    
    with open(log_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Iteration {iteration} details logged to: {log_file}")


# Example usage
if __name__ == "__main__":
    # Test utility functions
    create_output_dirs('./test_output')
    print("✓ Utility functions test complete!")
