# main.py
"""Main experiment runner for Byzantine-robust federated optimization"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from federated_system import *

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 12)
plt.rcParams['font.size'] = 10

LR_VALUES_BASELINE = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
LR_VALUES_BYZ = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

OUTPUT_DIR = "outputs/"


def run_multiple_seeds(attack_type, aggregator_type, optimizer_type,
                      n_seeds=1, n_iterations=1500, lr=0.01, momentum=0.9, verbose=False, system=None):
    """Run experiment with multiple random seeds and return statistics"""
    all_gradient_norms = []
    all_objective_values = []
    
    for seed in range(n_seeds):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Seed {seed + 1}/{n_seeds}")
            
        if system is None:
            system = FederatedSystem(
                n_workers=20,
                n_byzantine=3,
                dimension=10,
                seed=seed
            )
        else:
            np.random.seed(seed)
        
        grad_norms, obj_values = system.run_experiment(
            attack_type=attack_type,
            aggregator_type=aggregator_type,
            optimizer_type=optimizer_type,
            n_iterations=n_iterations,
            lr=lr,
            momentum=momentum,
            verbose=verbose
        )
        
        all_gradient_norms.append(grad_norms)
        all_objective_values.append(obj_values)
    
    all_gradient_norms = np.array(all_gradient_norms)
    all_objective_values = np.array(all_objective_values)
    
    mean_grad_norms = np.mean(all_gradient_norms, axis=0)
    std_grad_norms = np.std(all_gradient_norms, axis=0)
    mean_obj_values = np.mean(all_objective_values, axis=0)
    std_obj_values = np.std(all_objective_values, axis=0)
    
    return {
        'mean_grad_norms': mean_grad_norms,
        'std_grad_norms': std_grad_norms,
        'mean_obj_values': mean_obj_values,
        'std_obj_values': std_obj_values,
        'all_grad_norms': all_gradient_norms,
        'all_obj_values': all_objective_values
    }


def tune_learning_rate(attack_type, aggregator_type, optimizer_type,
                       n_iterations=1000, n_seeds=1, verbose=False):
    """Tune learning rate by minimizing final mean gradient norm"""
    if optimizer_type in ['byz-nsgdm', 'baseline-decay']:
        lr_values = LR_VALUES_BYZ
    else:
        lr_values = LR_VALUES_BASELINE
    
    best_lr = None
    best_score = float('inf')
    scores = {}

    shared_system = FederatedSystem(
        n_workers=20,
        n_byzantine=3,
        dimension=10,
        seed=0
    )

    if verbose:
        print(f"\nTuning LR for setup: {attack_type.upper()} + {aggregator_type.upper()} + {optimizer_type.upper()}")
        print(f"Trying {len(lr_values)} learning rates for {n_iterations} iterations each...")

    for lr in lr_values:
        res = run_multiple_seeds(
            attack_type=attack_type,
            aggregator_type=aggregator_type,
            optimizer_type=optimizer_type,
            n_seeds=n_seeds,
            n_iterations=n_iterations,
            lr=lr,
            verbose=False,
            system=shared_system,
        )
        final_grad_norm = float(res['mean_grad_norms'][-1])
        if not np.isfinite(final_grad_norm):
            final_grad_norm = float('inf')
        scores[lr] = final_grad_norm
        if verbose:
            print(f"  LR={lr:g} -> final grad norm = {final_grad_norm:.3e}")
        if final_grad_norm < best_score:
            best_score = final_grad_norm
            best_lr = lr
        
        if final_grad_norm > 1e+10:
            if verbose:
                print(f"  Early stopping: metric > 1e+10, skipping remaining LRs")
            break

    if best_lr is None:
        best_lr = 0.01
        if verbose:
            print("All candidate LRs produced non-finite scores. Falling back to LR=0.01")
    else:
        if verbose:
            print(f"Best LR: {best_lr:g} (final grad norm {best_score:.3e})")

    return best_lr, scores


def plot_results(results_dict, save_path='exp11_comparison.png'):
    """Create 3x3 subplot grid comparing different configurations"""
    
    attacks = ['bf', 'mimic', 'alie']
    attack_names = ['BF', 'Mimic', 'ALIE']
    aggregators = ['rfa', 'krum', 'cm']
    aggregator_names = ['RFA', 'Krum', 'CM']
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 6))
    
    colors = {'baseline': '#1f77b4', 'baseline-decay': '#2ca02c', 'byz-nsgdm': '#ff7f0e'}
    labels = {'baseline': 'Baseline', 'baseline-decay': 'Baseline-Decay', 'byz-nsgdm': 'Byz-NSGDM'}
    
    for i, attack in enumerate(attacks):
        for j, aggregator in enumerate(aggregators):
            ax = axes[i, j]
            
            for optimizer in ['baseline', 'baseline-decay', 'byz-nsgdm']:
                key = f"{attack}_{aggregator}_{optimizer}"
                if key in results_dict:
                    data = results_dict[key]
                    mean_vals = data['mean_grad_norms']
                    std_vals = data['std_grad_norms']
                    best_lr = data['best_lr']
                    
                    x_indices = range(0, len(mean_vals), 20)
                    mean_vals_sampled = mean_vals[::20]
                    std_vals_sampled = std_vals[::20]
                    
                    ax.plot(x_indices, mean_vals_sampled, 
                           label=f"{labels[optimizer]} (lr={best_lr:.3g})", 
                           color=colors[optimizer], 
                           linewidth=1.5)
                    
                    ax.fill_between(x_indices,
                                   mean_vals_sampled - std_vals_sampled,
                                   mean_vals_sampled + std_vals_sampled,
                                   color=colors[optimizer],
                                   alpha=0.2)
            
            ax.set_title(f"{attack_names[i]} + {aggregator_names[j]}", fontsize=10)
            ax.set_yscale('log')
            ax.set_xlim(0, 3000)
            ax.set_ylim(1e-6, 1e2)
            ax.grid(True, alpha=0.3)
            for spine in ax.spines.values():
                spine.set_color('black')
                spine.set_linewidth(0.8)
            if i == 2:
                ax.set_xlabel("Iterations", fontsize=10)
            if j == 0:
                ax.set_ylabel("Gradient Norm", fontsize=10)
            ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
            ax.tick_params(axis='both', which='both', direction='out', length=4, labelsize=9)
            ax.xaxis.set_tick_params(which='both', bottom=True, top=False)
            ax.yaxis.set_tick_params(which='both', left=True, right=False)
    
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")
    
    return fig


def print_summary_statistics(results_dict):
    """Print summary statistics for all experiments"""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    attacks = ['bf', 'mimic', 'alie']
    attack_names = {'bf': 'Bit Flipping', 'mimic': 'Mimic', 'alie': 'ALIE'}
    aggregators = ['rfa', 'krum', 'cm']
    aggregator_names = {'rfa': 'RFA', 'krum': 'KRUM', 'cm': 'Coord. Median'}
    
    for attack in attacks:
        print(f"\n{attack_names[attack]} Attack:")
        print("-" * 40)
        
        for aggregator in aggregators:
            print(f"\n  {aggregator_names[aggregator]} Aggregator:")
            
            for optimizer in ['baseline', 'baseline-decay', 'byz-nsgdm']:
                key = f"{attack}_{aggregator}_{optimizer}"
                if key in results_dict:
                    data = results_dict[key]
                    final_grad_norm = data['mean_grad_norms'][-1]
                    final_obj = data['mean_obj_values'][-1]
                    
                    if len(data['mean_grad_norms']) > 100:
                        last_100 = data['mean_grad_norms'][-100:]
                        convergence_rate = (last_100[0] - last_100[-1]) / last_100[0]
                    else:
                        convergence_rate = 0
                    
                    if optimizer == 'baseline':
                        optimizer_name = 'Baseline'
                    elif optimizer == 'baseline-decay':
                        optimizer_name = 'Baseline-Decay'
                    else:
                        optimizer_name = 'Byz-NSGDM'
                    print(f"    {optimizer_name:15s}: Final grad norm = {final_grad_norm:.2e}, "
                          f"Final obj = {final_obj:.4f}, Conv. rate = {convergence_rate:.2%}")


def run_single_experiment(attack, aggregator, optimizer):
    """Run a single experiment (for parallel execution)"""
    best_lr, _ = tune_learning_rate(
        attack_type=attack,
        aggregator_type=aggregator,
        optimizer_type=optimizer,
        n_iterations=500,
        n_seeds=1,
        verbose=False,
    )
    
    results = run_multiple_seeds(
        attack_type=attack,
        aggregator_type=aggregator,
        optimizer_type=optimizer,
        n_seeds=3,
        n_iterations=3000,
        lr=best_lr,
        verbose=False,
    )
    
    results['best_lr'] = best_lr
    key = f"{attack}_{aggregator}_{optimizer}"
    return key, results


def main():
    """Main experiment runner"""
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    attacks = ['bf', 'mimic', 'alie']
    aggregators = ['rfa', 'krum', 'cm']
    optimizers = ['baseline', 'baseline-decay', 'byz-nsgdm']
    
    results_file = os.path.join(OUTPUT_DIR, 'exp11_results.pkl')
    if os.path.exists(results_file):
        print(f"Loading existing results from {results_file}")
        with open(results_file, 'rb') as f:
            results_dict = pickle.load(f)
        print(f"Loaded {len(results_dict)} experiment results")
    else:
        results_dict = {}
    
    experiments_to_run = []
    for attack in attacks:
        for aggregator in aggregators:
            for optimizer in optimizers:
                key = f"{attack}_{aggregator}_{optimizer}"
                if key not in results_dict or 'best_lr' not in results_dict[key]:
                    experiments_to_run.append((attack, aggregator, optimizer))
    
    if experiments_to_run:
        print(f"\nRunning {len(experiments_to_run)} experiments in parallel...")
        
        with ProcessPoolExecutor(max_workers=min(os.cpu_count(), len(experiments_to_run))) as executor:
            futures = {executor.submit(run_single_experiment, a, agg, opt): (a, agg, opt) 
                       for a, agg, opt in experiments_to_run}
            
            for future in as_completed(futures):
                attack, aggregator, optimizer = futures[future]
                try:
                    key, results = future.result()
                    results_dict[key] = results
                    print(f"Completed: {key} (best_lr={results['best_lr']:.3g})")
                    
                    with open(results_file, 'wb') as f:
                        pickle.dump(results_dict, f)
                except Exception as e:
                    print(f"Failed: {attack}_{aggregator}_{optimizer}: {e}")
    
    print_summary_statistics(results_dict)
    
    print("\nGenerating visualization...")
    fig = plot_results(results_dict, save_path=os.path.join(OUTPUT_DIR, 'exp11_comparison.png'))
    plt.show()
    
    return results_dict


if __name__ == "__main__":
    results = main()