"""
Fast QAOA Test - 15 assets, select 4
Run 20 experiments with top 1000 solution tracking
Optimized for speed
"""

import numpy as np
import json
import os
from datetime import datetime
import time
from honest_qaoa import HonestQAOA
import matplotlib.pyplot as plt

def check_feasibility(state_str: str, budget: int) -> bool:
    """Check if a state string represents a feasible solution"""
    return state_str.count('1') == budget

def calculate_objective(state_str: str, expected_returns: np.ndarray, 
                        covariance: np.ndarray, risk_factor: float) -> float:
    """Calculate portfolio objective for a given state"""
    portfolio = np.array([int(bit) for bit in state_str])
    
    if portfolio.sum() == 0:
        return 0.0
    
    # Normalize
    weights = portfolio / portfolio.sum()
    
    # Calculate return and risk
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariance, weights)))
    
    # Objective
    return portfolio_return - risk_factor * portfolio_risk

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
    """Run a single QAOA experiment with top 1000 tracking"""
    
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
    expected_returns, covariance = generate_market_scenario(n_assets, scenario, seed=3000 + exp_id)
    
    # Use regular HonestQAOA for speed
    qaoa = HonestQAOA(n_assets, budget, risk_factor)
    
    # Reduced parameters for speed
    qaoa.p = 5  # Moderate depth
    qaoa.shots = 4096  # Reduced shots
    
    start_time = time.time()
    
    # Run optimization with fewer iterations
    results = qaoa.optimize(expected_returns, covariance, max_iterations=50)
    
    computation_time = time.time() - start_time
    
    # Extract top 1000 solutions with feasibility check
    top_1000_solutions = []
    
    # Get all measured states from final distribution
    final_counts = results.get('final_counts', {})
    total_counts = results.get('total_counts', sum(final_counts.values()))
    
    # Sort by count (highest first)
    sorted_states = sorted(final_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Take top 1000 (or all if less than 1000)
    for i, (state, count) in enumerate(sorted_states[:1000]):
        prob = count / total_counts
        obj = calculate_objective(state, expected_returns, covariance, risk_factor)
        is_feasible = check_feasibility(state, budget)
        
        solution_data = {
            'rank': i + 1,
            'state': state,
            'probability': prob,
            'count': count,
            'objective': obj,
            'is_feasible': is_feasible
        }
        top_1000_solutions.append(solution_data)
    
    # Calculate detailed metrics
    feasible_solutions = [s for s in top_1000_solutions if s['is_feasible']]
    total_feasible_prob = sum(s['probability'] for s in feasible_solutions)
    
    # Find best feasible solution
    if feasible_solutions:
        best_feasible = max(feasible_solutions, key=lambda x: x['objective'])
        best_feasible_prob = best_feasible['probability']
        best_feasible_obj = best_feasible['objective']
    else:
        best_feasible_prob = 0.0
        best_feasible_obj = 0.0
    
    # Prepare results
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
        
        # Detailed metrics from top 1000
        'total_feasible_probability_top1000': total_feasible_prob,
        'n_feasible_in_top1000': len(feasible_solutions),
        'best_feasible_probability': best_feasible_prob,
        'best_feasible_objective': best_feasible_obj,
        
        # Top 1000 solutions with feasibility indicators
        'top_1000_solutions': top_1000_solutions,
        
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

def create_feasibility_plot(results: dict, exp_id: int, output_dir: str):
    """Create simple feasibility distribution plot"""
    
    top_solutions = results['top_1000_solutions'][:30]  # Plot top 30
    
    if not top_solutions:
        return
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    indices = range(len(top_solutions))
    colors = ['green' if s['is_feasible'] else 'red' for s in top_solutions]
    probs = [s['probability'] * 100 for s in top_solutions]
    
    bars = plt.bar(indices, probs, color=colors, alpha=0.7)
    plt.xlabel('Solution Rank')
    plt.ylabel('Probability (%)')
    plt.title(f'Exp {exp_id}: Top 30 Solutions (Green=Feasible, Red=Infeasible)')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, f'run_{exp_id:02d}_distribution.png')
    plt.savefig(plot_file, dpi=100)
    plt.close()
    
    print(f"  Saved plot: {plot_file}")

def run_all_experiments():
    """Run all 20 experiments"""
    
    print("\n" + "="*80)
    print("QAOA TEST V5 - 15 Assets, Select 4")
    print("20 Experiments with Top 1000 Solution Tracking")
    print("="*80)
    
    # Create output directory
    output_dir = '20_by_15_v5'
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
            
            # Create plot
            create_feasibility_plot(results, exp_id, output_dir)
            
            all_results.append(results)
            
            # Print summary
            print(f"  Feasibility: {results['feasibility_rate']*100:.1f}%")
            print(f"  Best Prob: {results['best_solution_probability']*100:.3f}%")
            print(f"  Feasible in top 1000: {results['n_feasible_in_top1000']}")
            print(f"  Total feasible prob: {results['total_feasible_probability_top1000']*100:.2f}%")
            print(f"  Time: {results['computation_time']:.1f}s")
            
        except Exception as e:
            print(f"  Error in experiment {exp_id}: {str(e)}")
            continue
    
    # Calculate summary
    if all_results:
        summary = {
            'n_experiments': len(all_results),
            'timestamp': datetime.now().isoformat(),
            'avg_feasibility_rate': np.mean([r['feasibility_rate'] for r in all_results]),
            'avg_best_probability': np.mean([r['best_solution_probability'] for r in all_results]),
            'avg_feasible_in_top1000': np.mean([r['n_feasible_in_top1000'] for r in all_results]),
            'avg_feasible_prob_top1000': np.mean([r['total_feasible_probability_top1000'] for r in all_results])
        }
        
        # Save summary
        summary_file = os.path.join(output_dir, 'summary_statistics.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Experiments completed: {summary['n_experiments']}")
        print(f"Avg feasibility rate: {summary['avg_feasibility_rate']*100:.2f}%")
        print(f"Avg best probability: {summary['avg_best_probability']*100:.4f}%")
        print(f"Avg feasible in top 1000: {summary['avg_feasible_in_top1000']:.1f}")
        print(f"Avg feasible prob in top 1000: {summary['avg_feasible_prob_top1000']*100:.2f}%")
    
    print(f"\nAll results saved in '{output_dir}' directory")

if __name__ == "__main__":
    run_all_experiments()