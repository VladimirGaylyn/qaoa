"""
20x20 Test: Run ultra_optimized_v3 20 times with different asset mixes
Analyze probability distributions and performance metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os
from ultra_optimized_v3_advanced import UltraOptimizedQAOAv3
from classical_strategies import ClassicalPortfolioStrategies
from warm_start_feedback import WarmStartFeedback
import warnings
warnings.filterwarnings('ignore')

def generate_diverse_portfolio_data(n_assets=20, run_number=0, seed=None):
    """Generate diverse portfolio data with different characteristics for each run"""
    if seed is not None:
        np.random.seed(seed)
    
    # Different market scenarios for diversity
    scenarios = [
        {'trend': 'bull', 'volatility': 'low', 'correlation': 'positive'},
        {'trend': 'bear', 'volatility': 'high', 'correlation': 'negative'},
        {'trend': 'neutral', 'volatility': 'medium', 'correlation': 'mixed'},
        {'trend': 'bull', 'volatility': 'high', 'correlation': 'clustered'},
        {'trend': 'bear', 'volatility': 'low', 'correlation': 'uncorrelated'},
    ]
    
    scenario = scenarios[run_number % len(scenarios)]
    
    # Generate returns based on scenario
    if scenario['trend'] == 'bull':
        base_return = np.random.uniform(0.05, 0.25, n_assets)
    elif scenario['trend'] == 'bear':
        base_return = np.random.uniform(-0.15, 0.05, n_assets)
    else:
        base_return = np.random.uniform(-0.05, 0.15, n_assets)
    
    # Add sector-specific adjustments
    sectors = ['TECH', 'FIN', 'HEALTH', 'ENERGY', 'CONS']
    sector_returns = {}
    for i, sector in enumerate(sectors):
        sector_adjustment = np.random.normal(0, 0.02)
        sector_returns[sector] = base_return[i*4:(i+1)*4] + sector_adjustment
    
    expected_returns = np.concatenate(list(sector_returns.values()))[:n_assets]
    
    # Generate volatilities based on scenario
    if scenario['volatility'] == 'low':
        volatilities = np.random.uniform(0.05, 0.15, n_assets)
    elif scenario['volatility'] == 'high':
        volatilities = np.random.uniform(0.15, 0.35, n_assets)
    else:
        volatilities = np.random.uniform(0.10, 0.25, n_assets)
    
    # Create correlation matrix based on scenario
    correlation = np.eye(n_assets)
    
    if scenario['correlation'] == 'positive':
        # High positive correlations
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                correlation[i, j] = correlation[j, i] = np.random.uniform(0.3, 0.8)
    elif scenario['correlation'] == 'negative':
        # Mixed with negative correlations
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                correlation[i, j] = correlation[j, i] = np.random.uniform(-0.5, 0.5)
    elif scenario['correlation'] == 'clustered':
        # High correlation within sectors, low between
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                if i // 4 == j // 4:  # Same sector
                    correlation[i, j] = correlation[j, i] = np.random.uniform(0.5, 0.9)
                else:
                    correlation[i, j] = correlation[j, i] = np.random.uniform(-0.2, 0.3)
    else:
        # Uncorrelated
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                correlation[i, j] = correlation[j, i] = np.random.uniform(-0.2, 0.2)
    
    # Ensure positive definite
    min_eigenvalue = np.min(np.linalg.eigvalsh(correlation))
    if min_eigenvalue < 0:
        correlation += (-min_eigenvalue + 0.01) * np.eye(n_assets)
    
    # Create covariance matrix
    covariance = np.outer(volatilities, volatilities) * correlation
    
    # Generate asset names
    asset_names = []
    for i, sector in enumerate(sectors):
        for j in range(4):
            asset_names.append(f"{sector}{j+1}")
    asset_names = asset_names[:n_assets]
    
    return expected_returns, covariance, asset_names, scenario

def analyze_probability_distribution(counts, best_solution, n_assets, budget):
    """Analyze the probability distribution of quantum measurements"""
    
    # Get total shots
    total_shots = sum(counts.values())
    
    # Find best solution probability
    best_solution_str = ''.join(str(int(x)) for x in best_solution)
    best_solution_prob = counts.get(best_solution_str, 0) / total_shots
    
    # Get top 10 solutions
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Calculate feasibility rate
    feasible_count = 0
    for bitstring, count in counts.items():
        if sum(int(bit) for bit in bitstring) == budget:
            feasible_count += count
    feasibility_rate = feasible_count / total_shots
    
    # Get probability distribution of top solutions
    top_solutions = []
    top_probabilities = []
    for bitstring, count in sorted_counts:
        top_solutions.append(bitstring)
        top_probabilities.append(count / total_shots)
    
    return {
        'best_solution_prob': best_solution_prob,
        'feasibility_rate': feasibility_rate,
        'top_solutions': top_solutions,
        'top_probabilities': top_probabilities,
        'total_shots': total_shots
    }

def plot_probability_distribution(counts, best_solution, run_number, save_dir):
    """Plot probability distribution for a single run"""
    
    # Get top 20 solutions for plotting
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20]
    
    bitstrings = [item[0] for item in sorted_counts]
    probabilities = [item[1] / sum(counts.values()) for item in sorted_counts]
    
    # Check if best solution is in top 20
    best_solution_str = ''.join(str(int(x)) for x in best_solution)
    best_idx = None
    if best_solution_str in bitstrings:
        best_idx = bitstrings.index(best_solution_str)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot of probabilities
    colors = ['red' if i == best_idx else 'steelblue' for i in range(len(bitstrings))]
    bars = ax1.bar(range(len(probabilities)), probabilities, color=colors)
    ax1.set_xlabel('Solution Rank')
    ax1.set_ylabel('Probability')
    ax1.set_title(f'Run {run_number}: Top 20 Solution Probabilities')
    
    # Add best solution marker
    if best_idx is not None:
        ax1.annotate('Best\nSolution', xy=(best_idx, probabilities[best_idx]),
                    xytext=(best_idx, probabilities[best_idx] + 0.01),
                    ha='center', fontsize=8, color='red', fontweight='bold')
    
    # Hamming weight distribution
    hamming_weights = {}
    total = sum(counts.values())
    for bitstring, count in counts.items():
        weight = sum(int(bit) for bit in bitstring)
        if weight not in hamming_weights:
            hamming_weights[weight] = 0
        hamming_weights[weight] += count
    
    weights = sorted(hamming_weights.keys())
    weight_probs = [hamming_weights[w] / total for w in weights]
    
    bars2 = ax2.bar(weights, weight_probs, color='teal', alpha=0.7)
    ax2.axvline(x=5, color='red', linestyle='--', label='Target (k=5)')
    ax2.set_xlabel('Hamming Weight')
    ax2.set_ylabel('Probability')
    ax2.set_title(f'Run {run_number}: Hamming Weight Distribution')
    ax2.legend()
    
    # Highlight k=5 bar
    if 5 in weights:
        idx_5 = weights.index(5)
        bars2[idx_5].set_color('darkgreen')
        bars2[idx_5].set_alpha(1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'run_{run_number:02d}_distribution.png'), dpi=100)
    plt.close()

def run_single_test(run_number, n_assets=20, budget=5, risk_factor=0.5):
    """Run a single test with unique portfolio data"""
    
    print(f"\n{'='*60}")
    print(f"RUN {run_number}/20")
    print(f"{'='*60}")
    
    # Generate unique portfolio data
    expected_returns, covariance, asset_names, scenario = generate_diverse_portfolio_data(
        n_assets=n_assets, run_number=run_number, seed=42 + run_number*100
    )
    
    print(f"Scenario: {scenario['trend']} market, {scenario['volatility']} volatility, {scenario['correlation']} correlation")
    
    # Initialize optimizer with correct parameters
    optimizer = UltraOptimizedQAOAv3(
        n_assets=n_assets,
        budget=budget,
        risk_factor=risk_factor
    )
    
    # Run optimization with portfolio data
    result = optimizer.solve_ultra_optimized_v3(
        expected_returns=expected_returns,
        covariance=covariance,
        max_iterations=30
    )
    
    # Get measurement counts from dataclass
    counts = result.measurement_counts
    
    # Analyze probability distribution
    prob_analysis = analyze_probability_distribution(
        counts, result.solution, n_assets, budget
    )
    
    # Plot distribution
    plot_probability_distribution(
        counts, result.solution, run_number, '20_by_20'
    )
    
    # Compile results
    run_results = {
        'run_number': run_number,
        'scenario': scenario,
        'approximation_ratio': result.approximation_ratio,
        'best_solution_probability': prob_analysis['best_solution_prob'],
        'feasibility_rate': result.feasibility_rate,
        'circuit_depth': result.circuit_depth,
        'convergence_iterations': result.iterations_to_convergence,
        'objective_value': result.objective_value,
        'selected_assets': [asset_names[i] for i, x in enumerate(result.solution) if x == 1],
        'top_5_solutions': prob_analysis['top_solutions'][:5],
        'top_5_probabilities': prob_analysis['top_probabilities'][:5]
    }
    
    # Save individual run results
    with open(os.path.join('20_by_20', f'run_{run_number:02d}_results.json'), 'w') as f:
        json.dump(run_results, f, indent=2)
    
    print(f"Approximation Ratio: {result.approximation_ratio:.4f}")
    print(f"Best Solution Probability: {prob_analysis['best_solution_prob']:.4f}")
    print(f"Feasibility Rate: {result.feasibility_rate:.4f}")
    print(f"Circuit Depth: {result.circuit_depth}")
    
    return run_results

def create_final_summary(all_results):
    """Create final summary statistics and visualization"""
    
    # Calculate aggregate statistics
    ar_values = [r['approximation_ratio'] for r in all_results]
    best_prob_values = [r['best_solution_probability'] for r in all_results]
    feasibility_values = [r['feasibility_rate'] for r in all_results]
    depth_values = [r['circuit_depth'] for r in all_results]
    iterations_values = [r['convergence_iterations'] for r in all_results]
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_runs': 20,
        'problem_size': {'n_assets': 20, 'budget': 5},
        
        'approximation_ratio': {
            'mean': float(np.mean(ar_values)),
            'std': float(np.std(ar_values)),
            'min': float(np.min(ar_values)),
            'max': float(np.max(ar_values)),
            'median': float(np.median(ar_values))
        },
        
        'best_solution_probability': {
            'mean': float(np.mean(best_prob_values)),
            'std': float(np.std(best_prob_values)),
            'min': float(np.min(best_prob_values)),
            'max': float(np.max(best_prob_values)),
            'median': float(np.median(best_prob_values))
        },
        
        'feasibility_rate': {
            'mean': float(np.mean(feasibility_values)),
            'std': float(np.std(feasibility_values)),
            'min': float(np.min(feasibility_values)),
            'max': float(np.max(feasibility_values)),
            'median': float(np.median(feasibility_values))
        },
        
        'circuit_depth': {
            'value': int(depth_values[0]),  # Should be constant
            'all_same': all(d == depth_values[0] for d in depth_values)
        },
        
        'convergence': {
            'mean_iterations': float(np.mean(iterations_values)),
            'std_iterations': float(np.std(iterations_values)),
            'min_iterations': int(np.min(iterations_values)),
            'max_iterations': int(np.max(iterations_values))
        },
        
        'scenario_performance': {}
    }
    
    # Group by scenario
    for result in all_results:
        scenario_key = f"{result['scenario']['trend']}_{result['scenario']['volatility']}"
        if scenario_key not in summary['scenario_performance']:
            summary['scenario_performance'][scenario_key] = []
        summary['scenario_performance'][scenario_key].append(result['approximation_ratio'])
    
    # Average per scenario
    for scenario_key in summary['scenario_performance']:
        values = summary['scenario_performance'][scenario_key]
        summary['scenario_performance'][scenario_key] = {
            'mean_ar': float(np.mean(values)),
            'count': len(values)
        }
    
    # Save summary
    with open('20_by_20/final_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create summary visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('20x20 Test: Final Statistics', fontsize=16, fontweight='bold')
    
    # 1. Approximation Ratio Distribution
    ax = axes[0, 0]
    ax.hist(ar_values, bins=10, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(ar_values), color='red', linestyle='--', label=f'Mean: {np.mean(ar_values):.4f}')
    ax.set_xlabel('Approximation Ratio')
    ax.set_ylabel('Frequency')
    ax.set_title('Approximation Ratio Distribution')
    ax.legend()
    
    # 2. Best Solution Probability
    ax = axes[0, 1]
    ax.hist(best_prob_values, bins=10, color='coral', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(best_prob_values), color='red', linestyle='--', label=f'Mean: {np.mean(best_prob_values):.4f}')
    ax.set_xlabel('Best Solution Probability')
    ax.set_ylabel('Frequency')
    ax.set_title('Best Solution Probability Distribution')
    ax.legend()
    
    # 3. Feasibility Rate
    ax = axes[0, 2]
    ax.hist(feasibility_values, bins=10, color='teal', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(feasibility_values), color='red', linestyle='--', label=f'Mean: {np.mean(feasibility_values):.4f}')
    ax.set_xlabel('Feasibility Rate')
    ax.set_ylabel('Frequency')
    ax.set_title('Feasibility Rate Distribution')
    ax.legend()
    
    # 4. Performance over runs
    ax = axes[1, 0]
    ax.plot(range(1, 21), ar_values, 'o-', color='darkgreen', markersize=6)
    ax.set_xlabel('Run Number')
    ax.set_ylabel('Approximation Ratio')
    ax.set_title('Performance Across Runs')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([min(ar_values) - 0.01, max(ar_values) + 0.01])
    
    # 5. Scenario comparison
    ax = axes[1, 1]
    scenarios = list(summary['scenario_performance'].keys())
    scenario_means = [summary['scenario_performance'][s]['mean_ar'] for s in scenarios]
    bars = ax.bar(range(len(scenarios)), scenario_means, color='purple', alpha=0.7)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([s.replace('_', '\n') for s in scenarios], rotation=45, ha='right')
    ax.set_ylabel('Mean Approximation Ratio')
    ax.set_title('Performance by Market Scenario')
    for i, (bar, val) in enumerate(zip(bars, scenario_means)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 6. Convergence iterations
    ax = axes[1, 2]
    ax.boxplot([iterations_values], labels=['Iterations'])
    ax.set_ylabel('Number of Iterations')
    ax.set_title(f'Convergence Speed (Mean: {np.mean(iterations_values):.1f})')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('20_by_20/final_summary_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return summary

def main():
    """Run complete 20x20 test suite"""
    
    print("="*60)
    print("STARTING 20x20 TEST SUITE")
    print("Testing ultra_optimized_v3 with 20 different asset mixes")
    print("="*60)
    
    all_results = []
    
    # Run 20 tests
    for run_number in range(1, 21):
        try:
            result = run_single_test(run_number)
            all_results.append(result)
        except Exception as e:
            print(f"Error in run {run_number}: {e}")
            continue
    
    # Create final summary
    print("\n" + "="*60)
    print("GENERATING FINAL SUMMARY")
    print("="*60)
    
    summary = create_final_summary(all_results)
    
    # Print key statistics
    print("\n" + "="*60)
    print("FINAL STATISTICS (20 RUNS)")
    print("="*60)
    print(f"\n1. APPROXIMATION RATIO:")
    print(f"   Mean: {summary['approximation_ratio']['mean']:.4f} ± {summary['approximation_ratio']['std']:.4f}")
    print(f"   Range: [{summary['approximation_ratio']['min']:.4f}, {summary['approximation_ratio']['max']:.4f}]")
    
    print(f"\n2. BEST SOLUTION PROBABILITY:")
    print(f"   Mean: {summary['best_solution_probability']['mean']:.4f} ± {summary['best_solution_probability']['std']:.4f}")
    print(f"   Range: [{summary['best_solution_probability']['min']:.4f}, {summary['best_solution_probability']['max']:.4f}]")
    
    print(f"\n3. FEASIBILITY RATE:")
    print(f"   Mean: {summary['feasibility_rate']['mean']:.4f} ± {summary['feasibility_rate']['std']:.4f}")
    print(f"   Range: [{summary['feasibility_rate']['min']:.4f}, {summary['feasibility_rate']['max']:.4f}]")
    
    print(f"\n4. CIRCUIT DEPTH:")
    print(f"   Constant: {summary['circuit_depth']['value']}")
    
    print(f"\n5. CONVERGENCE:")
    print(f"   Mean iterations: {summary['convergence']['mean_iterations']:.1f} ± {summary['convergence']['std_iterations']:.1f}")
    
    print("\n" + "="*60)
    print("TEST SUITE COMPLETE")
    print(f"Results saved in '20_by_20' directory")
    print("="*60)

if __name__ == "__main__":
    main()