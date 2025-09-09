"""
Honest QAOA Test - 15 assets, select 4
Run 20 experiments with REAL quantum performance
"""

import numpy as np
import json
import os
from datetime import datetime
import time
from honest_qaoa import HonestQAOA
import matplotlib.pyplot as plt

def generate_market_scenario(n_assets: int, scenario_type: str, seed: int):
    """Generate different market scenarios"""
    np.random.seed(seed)
    
    if scenario_type == 'bull':
        mean_return = 0.15
        volatility = 0.15
        correlation_base = 0.3
    elif scenario_type == 'bear':
        mean_return = 0.02
        volatility = 0.35
        correlation_base = 0.5
    elif scenario_type == 'volatile':
        mean_return = 0.08
        volatility = 0.40
        correlation_base = 0.4
    else:  # balanced
        mean_return = 0.10
        volatility = 0.20
        correlation_base = 0.35
    
    # Generate returns
    expected_returns = np.random.normal(mean_return, volatility/3, n_assets)
    expected_returns = np.clip(expected_returns, 0.01, 0.30)
    
    # Generate correlation matrix
    correlation = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            corr = np.random.uniform(-correlation_base, correlation_base * 1.5)
            correlation[i, j] = correlation[j, i] = corr
    
    # Generate covariance
    std_devs = np.random.uniform(volatility * 0.5, volatility * 1.5, n_assets)
    covariance = np.outer(std_devs, std_devs) * correlation
    
    # Ensure positive definite
    min_eig = np.min(np.linalg.eigvals(covariance))
    if min_eig < 0:
        covariance += np.eye(n_assets) * (-min_eig + 0.01)
    
    return expected_returns, covariance

def run_single_experiment(exp_id: int, n_assets: int = 15, budget: int = 4):
    """Run a single honest QAOA experiment"""
    
    print(f"\n{'='*60}")
    print(f"Experiment {exp_id}/20")
    print(f"{'='*60}")
    
    # Determine scenario
    scenarios = ['balanced', 'bull', 'bear', 'volatile']
    scenario = scenarios[exp_id % 4]
    
    # Risk factors
    risk_factors = [0.3, 0.5, 0.7]
    risk_factor = risk_factors[exp_id % 3]
    
    print(f"Scenario: {scenario}, Risk Factor: {risk_factor}")
    
    # Generate market data
    expected_returns, covariance = generate_market_scenario(n_assets, scenario, seed=1000 + exp_id)
    
    # Create and run HONEST QAOA
    qaoa = HonestQAOA(n_assets, budget, risk_factor)
    
    # Use reasonable parameters for honest performance
    qaoa.p = 6  # Moderate depth for better performance
    qaoa.shots = 8192  # Good shot count
    
    start_time = time.time()
    
    # Run optimization with limited iterations for speed
    results = qaoa.optimize(expected_returns, covariance, max_iterations=75)
    
    computation_time = time.time() - start_time
    
    # Prepare results in required format
    experiment_results = {
        'experiment_id': exp_id,
        'timestamp': datetime.now().isoformat(),
        'scenario': scenario,
        'risk_factor': risk_factor,
        'n_assets': n_assets,
        'budget': budget,
        
        # Key metrics
        'feasibility_rate': results['feasibility_rate'],
        'approximation_ratio': results['approximation_ratio'],
        'best_solution_probability': results['best_solution_probability'],
        'best_objective': results['best_objective'],
        
        # Top 10 solutions with probabilities
        'top_10_solutions': results['top_10_solutions'],
        
        # Additional metrics
        'n_unique_states': results['n_unique_states'],
        'n_feasible_states': results['n_feasible_states'],
        'circuit_depth': results['circuit_depth'],
        'circuit_executions': results['circuit_executions'],
        'total_shots': results['total_shots'],
        'computation_time': computation_time,
        
        # Classical comparison
        'classical_best': results['classical_best']
    }
    
    return experiment_results

def create_probability_distribution_plot(results: dict, exp_id: int, output_dir: str):
    """Create probability distribution plot for top 10 solutions"""
    
    if not results['top_10_solutions']:
        return
    
    # Extract data
    states = [sol['state'][-4:] for sol in results['top_10_solutions'][:10]]  # Show last 4 bits
    probs = [sol['probability'] * 100 for sol in results['top_10_solutions'][:10]]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(states)), probs, color='steelblue', alpha=0.8)
    
    # Highlight best solution
    bars[0].set_color('darkgreen')
    bars[0].set_alpha(1.0)
    
    plt.xlabel('Solution States (last 4 bits shown)', fontsize=12)
    plt.ylabel('Probability (%)', fontsize=12)
    plt.title(f'Experiment {exp_id}: Top 10 Solution Probabilities (Honest QAOA)', fontsize=14)
    plt.xticks(range(len(states)), states, rotation=45)
    
    # Add value labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{prob:.3f}%', ha='center', va='bottom', fontsize=9)
    
    # Add info text
    info_text = (f"Feasibility: {results['feasibility_rate']*100:.1f}%\n"
                f"Best Prob: {results['best_solution_probability']*100:.3f}%\n"
                f"AR: {results['approximation_ratio']:.3f}")
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            verticalalignment='top', fontsize=10)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, f'run_{exp_id:02d}_distribution.png')
    plt.savefig(plot_file, dpi=100)
    plt.close()
    
    print(f"  Saved distribution plot: {plot_file}")

def run_all_experiments():
    """Run all 20 experiments"""
    
    print("\n" + "="*80)
    print("HONEST QAOA TEST - 15 Assets, Select 4")
    print("Running 20 Experiments with REAL Quantum Performance")
    print("="*80)
    
    # Create output directory
    output_dir = '20_by_15_v3'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    all_results = []
    
    # Run 20 experiments
    for exp_id in range(1, 21):
        try:
            # Run experiment
            results = run_single_experiment(exp_id)
            
            # Save individual results
            result_file = os.path.join(output_dir, f'run_{exp_id:02d}_results.json')
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Create distribution plot
            create_probability_distribution_plot(results, exp_id, output_dir)
            
            all_results.append(results)
            
            # Print summary
            print(f"  Results: Feasibility={results['feasibility_rate']*100:.1f}%, "
                  f"Best Prob={results['best_solution_probability']*100:.3f}%, "
                  f"AR={results['approximation_ratio']:.3f}")
            
        except Exception as e:
            print(f"  Error in experiment {exp_id}: {str(e)}")
            continue
    
    # Calculate summary statistics
    if all_results:
        summary = calculate_summary_statistics(all_results)
        
        # Save summary
        summary_file = os.path.join(output_dir, 'summary_statistics.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print_final_summary(summary)
    
    print(f"\nAll results saved in '{output_dir}' directory")

def calculate_summary_statistics(all_results):
    """Calculate summary statistics across all experiments"""
    
    feasibility_rates = [r['feasibility_rate'] for r in all_results]
    best_probs = [r['best_solution_probability'] for r in all_results]
    approx_ratios = [r['approximation_ratio'] for r in all_results]
    
    summary = {
        'n_experiments': len(all_results),
        'timestamp': datetime.now().isoformat(),
        
        'feasibility_rate': {
            'mean': np.mean(feasibility_rates),
            'std': np.std(feasibility_rates),
            'min': np.min(feasibility_rates),
            'max': np.max(feasibility_rates)
        },
        
        'best_solution_probability': {
            'mean': np.mean(best_probs),
            'std': np.std(best_probs),
            'min': np.min(best_probs),
            'max': np.max(best_probs)
        },
        
        'approximation_ratio': {
            'mean': np.mean(approx_ratios),
            'std': np.std(approx_ratios),
            'min': np.min(approx_ratios),
            'max': np.max(approx_ratios)
        },
        
        # Check how many achieved different probability thresholds
        'probability_thresholds': {
            'above_0.1_percent': sum(1 for p in best_probs if p >= 0.001),
            'above_0.5_percent': sum(1 for p in best_probs if p >= 0.005),
            'above_1_percent': sum(1 for p in best_probs if p >= 0.01),
        },
        
        # All individual results for analysis
        'all_feasibility_rates': feasibility_rates,
        'all_best_probabilities': best_probs,
        'all_approximation_ratios': approx_ratios
    }
    
    return summary

def print_final_summary(summary):
    """Print final summary of results"""
    
    print("\n" + "="*80)
    print("FINAL SUMMARY - HONEST QAOA RESULTS")
    print("="*80)
    
    print(f"\nTotal Experiments: {summary['n_experiments']}")
    
    print("\nFeasibility Rate:")
    print(f"  Mean: {summary['feasibility_rate']['mean']*100:.2f}%")
    print(f"  Std:  {summary['feasibility_rate']['std']*100:.2f}%")
    print(f"  Range: {summary['feasibility_rate']['min']*100:.2f}% - {summary['feasibility_rate']['max']*100:.2f}%")
    
    print("\nBest Solution Probability:")
    print(f"  Mean: {summary['best_solution_probability']['mean']*100:.4f}%")
    print(f"  Std:  {summary['best_solution_probability']['std']*100:.4f}%")
    print(f"  Range: {summary['best_solution_probability']['min']*100:.4f}% - {summary['best_solution_probability']['max']*100:.4f}%")
    
    print("\nApproximation Ratio:")
    print(f"  Mean: {summary['approximation_ratio']['mean']:.4f}")
    print(f"  Std:  {summary['approximation_ratio']['std']:.4f}")
    print(f"  Range: {summary['approximation_ratio']['min']:.4f} - {summary['approximation_ratio']['max']:.4f}")
    
    print("\nProbability Threshold Achievement:")
    print(f"  ≥0.1%: {summary['probability_thresholds']['above_0.1_percent']}/{summary['n_experiments']}")
    print(f"  ≥0.5%: {summary['probability_thresholds']['above_0.5_percent']}/{summary['n_experiments']}")
    print(f"  ≥1.0%: {summary['probability_thresholds']['above_1_percent']}/{summary['n_experiments']}")
    
    # Analysis
    avg_prob = summary['best_solution_probability']['mean']
    if summary['probability_thresholds']['above_1_percent'] > 0:
        print("\n✓ Some experiments achieved ≥1% target!")
    elif summary['probability_thresholds']['above_0.5_percent'] > 0:
        print("\n△ Achieved 0.5% but not 1% - this is realistic QAOA performance")
    else:
        print("\n○ Below 0.5% - typical for this problem size with honest QAOA")
    
    # Concentration factor
    from math import comb
    uniform_prob = 1 / comb(15, 4)
    concentration = avg_prob / uniform_prob
    print(f"\nAverage concentration over uniform: {concentration:.1f}x")
    print(f"(Uniform probability: {uniform_prob*100:.6f}%)")

if __name__ == "__main__":
    run_all_experiments()