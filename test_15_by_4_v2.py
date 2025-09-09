"""
Test Suite v2: 15 assets, select 4 with Fixed Implementation
Run 20 experiments and store results in '20_by_15_v2' folder
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

def generate_diverse_portfolio_data(n_assets=15, run_number=0, seed=None):
    """Generate diverse portfolio data for 15 assets"""
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
    
    # Add sector-specific adjustments (3 sectors for 15 assets)
    sectors = ['TECH', 'FIN', 'HEALTH']
    sector_returns = {}
    for i, sector in enumerate(sectors):
        sector_adjustment = np.random.normal(0, 0.02)
        sector_returns[sector] = base_return[i*5:(i+1)*5] + sector_adjustment
    
    expected_returns = np.concatenate(list(sector_returns.values()))[:n_assets]
    
    # Generate volatilities based on scenario
    if scenario['volatility'] == 'high':
        volatilities = np.random.uniform(0.15, 0.35, n_assets)
    elif scenario['volatility'] == 'low':
        volatilities = np.random.uniform(0.05, 0.15, n_assets)
    else:
        volatilities = np.random.uniform(0.10, 0.25, n_assets)
    
    # Generate correlation matrix based on scenario
    if scenario['correlation'] == 'positive':
        base_corr = 0.6
    elif scenario['correlation'] == 'negative':
        base_corr = -0.3
    elif scenario['correlation'] == 'clustered':
        base_corr = 0.4
    else:
        base_corr = 0.1
    
    # Create correlation matrix with sector clustering
    correlation_matrix = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            # Same sector correlation
            if i // 5 == j // 5:
                corr = base_corr + np.random.uniform(0.2, 0.4)
            else:
                corr = base_corr + np.random.uniform(-0.2, 0.2)
            
            # Ensure valid correlation
            corr = np.clip(corr, -0.99, 0.99)
            correlation_matrix[i, j] = corr
            correlation_matrix[j, i] = corr
    
    # Make correlation matrix positive definite
    eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
    eigenvalues = np.maximum(eigenvalues, 0.01)
    correlation_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    # Convert correlation to covariance
    D = np.diag(volatilities)
    covariance_matrix = D @ correlation_matrix @ D
    
    # Generate asset names
    asset_names = []
    for i, sector in enumerate(sectors):
        for j in range(5):
            asset_names.append(f"{sector}{j+1}")
    asset_names = asset_names[:n_assets]
    
    return expected_returns, covariance_matrix, asset_names, scenario

def run_single_test(run_number, output_dir):
    """Run a single test with 15 assets, select 4"""
    
    print(f"\n{'='*60}")
    print(f"RUN {run_number}/20")
    print(f"{'='*60}")
    
    # Generate diverse data for this run
    seed = 42 + run_number * 100
    expected_returns, covariance_matrix, asset_names, scenario = generate_diverse_portfolio_data(
        n_assets=15, 
        run_number=run_number-1,
        seed=seed
    )
    
    print(f"Scenario: {scenario['trend']} market, {scenario['volatility']} volatility, {scenario['correlation']} correlation")
    
    # Parameters
    budget = 4  # Select 4 assets
    risk_factor = 0.3
    
    # Initialize optimizer with v3
    optimizer = UltraOptimizedQAOAv3(
        n_assets=15,
        budget=budget,
        risk_factor=risk_factor
    )
    
    # Run optimization
    result_obj = optimizer.solve_ultra_optimized_v3(
        expected_returns=expected_returns,
        covariance=covariance_matrix,
        max_iterations=50
    )
    
    # Convert result to dict format
    result = {
        'best_solution': format(result_obj.solution.astype(int).dot(2**np.arange(result_obj.solution.size)[::-1]), f'0{len(result_obj.solution)}b'),
        'objective_value': result_obj.objective_value,
        'approximation_ratio': result_obj.approximation_ratio,
        'counts': result_obj.measurement_counts,
        'circuit_depth': result_obj.circuit_depth,
        'convergence_iteration': result_obj.iterations_to_convergence,
        'selected_assets': [asset_names[i] for i in range(len(result_obj.solution)) if result_obj.solution[i] == 1],
        'feasibility_rate': result_obj.feasibility_rate,
        'initial_feasibility_rate': result_obj.initial_feasibility_rate
    }
    
    # Get probability distribution
    counts = result['counts']
    total_shots = sum(counts.values())
    
    # Analyze results
    feasible_count = 0
    best_solution_count = 0
    best_value = result['objective_value']
    
    # Track feasible solutions
    feasible_solutions = {}
    for bitstring, count in counts.items():
        if sum(int(b) for b in bitstring) == budget:
            feasible_count += count
            feasible_solutions[bitstring] = count
            # Check if this is the best solution
            if bitstring == result['best_solution']:
                best_solution_count = count
    
    feasibility_rate = feasible_count / total_shots
    best_solution_prob = best_solution_count / total_shots
    
    # Get top 10 solutions with probabilities
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top_10_solutions = []
    top_10_probabilities = []
    top_10_feasible = []
    
    for bitstring, count in sorted_counts:
        top_10_solutions.append(bitstring)
        top_10_probabilities.append(count/total_shots)
        top_10_feasible.append(sum(int(b) for b in bitstring) == budget)
    
    # Print summary
    print(f"Approximation Ratio: {result.get('approximation_ratio', 0):.4f}")
    print(f"Best Solution Probability: {best_solution_prob:.4f}")
    print(f"Feasibility Rate: {feasibility_rate:.4f}")
    print(f"Initial Feasibility: {result.get('initial_feasibility_rate', 0):.4f}")
    print(f"Circuit Depth: {result.get('circuit_depth', 0)}")
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Probability distribution (top 10)
    plt.subplot(1, 3, 1)
    colors = ['green' if f else 'red' for f in top_10_feasible]
    states_display = [s[-8:] for s in top_10_solutions]  # Show last 8 bits
    
    bars = plt.bar(range(len(top_10_solutions)), top_10_probabilities, color=colors)
    plt.xlabel('State (last 8 bits)')
    plt.ylabel('Probability')
    plt.title(f'Run {run_number}: Top 10 States')
    plt.xticks(range(len(states_display)), states_display, rotation=45, ha='right')
    plt.legend([plt.Rectangle((0,0),1,1,fc='green'), plt.Rectangle((0,0),1,1,fc='red')],
               ['Feasible (4 assets)', 'Infeasible'], loc='upper right')
    
    # Plot 2: Hamming weight distribution
    plt.subplot(1, 3, 2)
    hamming_weights = {}
    for bitstring, count in counts.items():
        weight = sum(int(b) for b in bitstring)
        hamming_weights[weight] = hamming_weights.get(weight, 0) + count
    
    weights = sorted(hamming_weights.keys())
    weight_probs = [hamming_weights[w]/total_shots for w in weights]
    colors = ['green' if w == budget else 'lightblue' for w in weights]
    
    plt.bar(weights, weight_probs, color=colors)
    plt.xlabel('Hamming Weight (# assets selected)')
    plt.ylabel('Probability')
    plt.title(f'Hamming Weight Distribution')
    plt.axvline(x=budget, color='red', linestyle='--', label=f'Target = {budget}')
    plt.legend()
    
    # Plot 3: Feasible solutions distribution
    plt.subplot(1, 3, 3)
    if feasible_solutions:
        # Get top feasible solutions
        sorted_feasible = sorted(feasible_solutions.items(), key=lambda x: x[1], reverse=True)[:10]
        feasible_states = [s[-8:] for s, _ in sorted_feasible]
        feasible_probs = [count/total_shots for _, count in sorted_feasible]
        
        plt.bar(range(len(feasible_states)), feasible_probs, color='green')
        plt.xlabel('Feasible State (last 8 bits)')
        plt.ylabel('Probability')
        plt.title(f'Top 10 Feasible Solutions')
        plt.xticks(range(len(feasible_states)), feasible_states, rotation=45, ha='right')
    else:
        plt.text(0.5, 0.5, 'No feasible solutions found', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Feasible Solutions')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/run_{run_number:02d}_distribution.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # Save detailed results
    results = {
        'run_number': run_number,
        'scenario': scenario,
        'n_assets': 15,
        'budget': 4,
        'approximation_ratio': float(result.get('approximation_ratio', 0)),
        'best_solution_probability': float(best_solution_prob),
        'feasibility_rate': float(feasibility_rate),
        'initial_feasibility_rate': float(result.get('initial_feasibility_rate', 0)),
        'circuit_depth': int(result.get('circuit_depth', 0)),
        'convergence_iterations': int(result.get('convergence_iteration', 0)),
        'objective_value': float(best_value),
        'selected_assets': result.get('selected_assets', []),
        'top_10_solutions': top_10_solutions,
        'top_10_probabilities': [float(p) for p in top_10_probabilities],
        'top_10_feasible': top_10_feasible,
        'hamming_weight_distribution': {str(k): float(v/total_shots) for k, v in hamming_weights.items()}
    }
    
    with open(f'{output_dir}/run_{run_number:02d}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def main():
    """Run complete test suite for 15 assets, select 4"""
    
    # Create output directory
    output_dir = '20_by_15_v2'
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("STARTING TEST SUITE V2: 15 ASSETS, SELECT 4")
    print("Using ultra_optimized_v3 with fixed improvements")
    print("="*60)
    
    # Run all tests
    all_results = []
    for run_number in range(1, 21):
        try:
            results = run_single_test(run_number, output_dir)
            all_results.append(results)
        except Exception as e:
            print(f"Error in run {run_number}: {e}")
            continue
    
    # Generate summary statistics
    print("\n" + "="*60)
    print("GENERATING FINAL SUMMARY")
    print("="*60)
    
    # Calculate aggregate statistics
    approximation_ratios = [r['approximation_ratio'] for r in all_results]
    best_solution_probs = [r['best_solution_probability'] for r in all_results]
    feasibility_rates = [r['feasibility_rate'] for r in all_results]
    initial_feasibility_rates = [r['initial_feasibility_rate'] for r in all_results]
    circuit_depths = [r['circuit_depth'] for r in all_results]
    convergence_iterations = [r['convergence_iterations'] for r in all_results]
    
    # Create summary visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Approximation Ratios
    axes[0, 0].bar(range(1, len(approximation_ratios)+1), approximation_ratios, color='blue', alpha=0.7)
    axes[0, 0].axhline(y=1.0, color='red', linestyle='--', label='Optimal')
    axes[0, 0].set_xlabel('Run Number')
    axes[0, 0].set_ylabel('Approximation Ratio')
    axes[0, 0].set_title('Approximation Ratios Across Runs')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Best Solution Probabilities
    axes[0, 1].bar(range(1, len(best_solution_probs)+1), best_solution_probs, color='green', alpha=0.7)
    axes[0, 1].set_xlabel('Run Number')
    axes[0, 1].set_ylabel('Probability')
    axes[0, 1].set_title('Best Solution Probability')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Feasibility Rates
    axes[0, 2].bar(range(1, len(feasibility_rates)+1), feasibility_rates, color='orange', alpha=0.7)
    axes[0, 2].bar(range(1, len(initial_feasibility_rates)+1), initial_feasibility_rates, 
                   color='lightblue', alpha=0.5, label='Initial')
    axes[0, 2].axhline(y=0.2, color='red', linestyle='--', label='Target 20%')
    axes[0, 2].set_xlabel('Run Number')
    axes[0, 2].set_ylabel('Feasibility Rate')
    axes[0, 2].set_title('Constraint Satisfaction Rate')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Distribution of Approximation Ratios
    axes[1, 0].hist(approximation_ratios, bins=10, color='blue', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Approximation Ratio')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Approximation Ratios')
    axes[1, 0].axvline(x=np.mean(approximation_ratios), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(approximation_ratios):.3f}')
    axes[1, 0].legend()
    
    # Plot 5: Distribution of Feasibility Rates
    axes[1, 1].hist(feasibility_rates, bins=10, color='orange', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Feasibility Rate')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Feasibility Rates')
    axes[1, 1].axvline(x=np.mean(feasibility_rates), color='red', linestyle='--',
                       label=f'Mean: {np.mean(feasibility_rates):.3f}')
    axes[1, 1].legend()
    
    # Plot 6: Feasibility Improvement
    improvements = [f - i for f, i in zip(feasibility_rates, initial_feasibility_rates)]
    axes[1, 2].bar(range(1, len(improvements)+1), improvements, color='purple', alpha=0.7)
    axes[1, 2].set_xlabel('Run Number')
    axes[1, 2].set_ylabel('Feasibility Improvement')
    axes[1, 2].set_title('Optimization Improvement (Final - Initial)')
    axes[1, 2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle('15 Assets, Select 4 - Test Suite V2 Summary', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/summary_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save summary statistics
    summary = {
        'test_date': datetime.now().isoformat(),
        'configuration': {
            'n_assets': 15,
            'budget': 4,
            'total_runs': len(all_results)
        },
        'statistics': {
            'approximation_ratio': {
                'mean': float(np.mean(approximation_ratios)),
                'std': float(np.std(approximation_ratios)),
                'min': float(np.min(approximation_ratios)),
                'max': float(np.max(approximation_ratios)),
                'median': float(np.median(approximation_ratios))
            },
            'best_solution_probability': {
                'mean': float(np.mean(best_solution_probs)),
                'std': float(np.std(best_solution_probs)),
                'min': float(np.min(best_solution_probs)),
                'max': float(np.max(best_solution_probs)),
                'median': float(np.median(best_solution_probs))
            },
            'feasibility_rate': {
                'mean': float(np.mean(feasibility_rates)),
                'std': float(np.std(feasibility_rates)),
                'min': float(np.min(feasibility_rates)),
                'max': float(np.max(feasibility_rates)),
                'median': float(np.median(feasibility_rates))
            },
            'initial_feasibility_rate': {
                'mean': float(np.mean(initial_feasibility_rates)),
                'std': float(np.std(initial_feasibility_rates)),
                'min': float(np.min(initial_feasibility_rates)),
                'max': float(np.max(initial_feasibility_rates))
            },
            'circuit_depth': {
                'mean': float(np.mean(circuit_depths)),
                'std': float(np.std(circuit_depths)),
                'min': float(np.min(circuit_depths)),
                'max': float(np.max(circuit_depths))
            },
            'convergence_iterations': {
                'mean': float(np.mean(convergence_iterations)),
                'std': float(np.std(convergence_iterations))
            }
        },
        'all_results': all_results
    }
    
    with open(f'{output_dir}/summary_statistics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final statistics
    print("\n" + "="*60)
    print("FINAL STATISTICS (20 RUNS) - 15 ASSETS, SELECT 4 (V2)")
    print("="*60)
    
    print(f"\n1. APPROXIMATION RATIO:")
    print(f"   Mean: {np.mean(approximation_ratios):.4f} ± {np.std(approximation_ratios):.4f}")
    print(f"   Range: [{np.min(approximation_ratios):.4f}, {np.max(approximation_ratios):.4f}]")
    print(f"   Median: {np.median(approximation_ratios):.4f}")
    
    print(f"\n2. BEST SOLUTION PROBABILITY:")
    print(f"   Mean: {np.mean(best_solution_probs):.6f} ± {np.std(best_solution_probs):.6f}")
    print(f"   Range: [{np.min(best_solution_probs):.6f}, {np.max(best_solution_probs):.6f}]")
    
    print(f"\n3. FEASIBILITY RATE:")
    print(f"   Mean: {np.mean(feasibility_rates):.4f} ± {np.std(feasibility_rates):.4f}")
    print(f"   Range: [{np.min(feasibility_rates):.4f}, {np.max(feasibility_rates):.4f}]")
    print(f"   Target Achievement: {'✓' if np.mean(feasibility_rates) >= 0.15 else '✗'} (Expected: ~15-25%)")
    print(f"   Improvement: {np.mean([f-i for f,i in zip(feasibility_rates, initial_feasibility_rates)]):.4f}")
    
    print(f"\n4. CIRCUIT DEPTH:")
    print(f"   Mean: {int(np.mean(circuit_depths))} ± {np.std(circuit_depths):.1f}")
    print(f"   Range: [{int(np.min(circuit_depths))}, {int(np.max(circuit_depths))}]")
    
    print(f"\n5. CONVERGENCE:")
    print(f"   Mean iterations: {np.mean(convergence_iterations):.1f} ± {np.std(convergence_iterations):.1f}")
    
    # Calculate success metrics
    successful_runs = sum(1 for r in feasibility_rates if r > 0.1)
    high_quality_runs = sum(1 for r in approximation_ratios if r > 0.8)
    
    print(f"\n6. SUCCESS METRICS:")
    print(f"   Runs with >10% feasibility: {successful_runs}/20 ({successful_runs*5}%)")
    print(f"   Runs with AR > 0.8: {high_quality_runs}/20 ({high_quality_runs*5}%)")
    
    # Compare with v1
    print("\n" + "="*60)
    print("COMPARISON WITH V1")
    print("="*60)
    print("Metric                  | V1          | V2          | Improvement")
    print("-" * 70)
    print(f"Approximation Ratio     | 0.846       | {np.mean(approximation_ratios):.3f}       | {np.mean(approximation_ratios) - 0.846:+.3f}")
    print(f"Feasibility Rate        | 0.186       | {np.mean(feasibility_rates):.3f}       | {np.mean(feasibility_rates) - 0.186:+.3f}")
    print(f"Best Solution Prob      | 0.00015     | {np.mean(best_solution_probs):.5f}     | {np.mean(best_solution_probs) - 0.00015:+.5f}")
    
    print("\n" + "="*60)
    print("TEST SUITE COMPLETE")
    print(f"Results saved in '{output_dir}' directory")
    print("="*60)

if __name__ == "__main__":
    main()