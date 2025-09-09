"""
Test suite for 15 assets, select 4 portfolio configuration
Using the FIXED v3 implementation
Stores results in '20_by_15_v3' folder
"""

import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from ultra_optimized_v3_fixed import UltraOptimizedQAOAv3Fixed
import warnings
warnings.filterwarnings('ignore')

def create_market_scenario(scenario_type: str, n_assets: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create different market scenarios for testing"""
    
    if scenario_type == "bull_low_vol":
        expected_returns = np.random.uniform(0.08, 0.15, n_assets)
        volatilities = np.random.uniform(0.05, 0.15, n_assets)
        corr_range = (0.2, 0.5)
    elif scenario_type == "bear_high_vol":
        expected_returns = np.random.uniform(-0.05, 0.05, n_assets)
        volatilities = np.random.uniform(0.2, 0.4, n_assets)
        corr_range = (-0.3, 0.3)
    elif scenario_type == "neutral_mixed":
        expected_returns = np.random.uniform(-0.02, 0.10, n_assets)
        volatilities = np.random.uniform(0.1, 0.25, n_assets)
        corr_range = (-0.2, 0.4)
    elif scenario_type == "high_return_high_risk":
        expected_returns = np.random.uniform(0.10, 0.25, n_assets)
        volatilities = np.random.uniform(0.25, 0.45, n_assets)
        corr_range = (-0.1, 0.6)
    else:  # random
        expected_returns = np.random.uniform(-0.05, 0.20, n_assets)
        volatilities = np.random.uniform(0.05, 0.35, n_assets)
        corr_range = (-0.3, 0.5)
    
    # Create correlation matrix
    correlation = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            corr = np.random.uniform(*corr_range)
            correlation[i, j] = corr
            correlation[j, i] = corr
    
    # Ensure positive definite
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalues = np.maximum(eigenvalues, 0.01)
    correlation = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    # Convert to covariance
    D = np.diag(volatilities)
    covariance = D @ correlation @ D
    
    return expected_returns, covariance

def analyze_solution_distribution(counts: Dict[str, int], n_assets: int, budget: int) -> Dict:
    """Analyze the probability distribution of solutions"""
    
    total_shots = sum(counts.values())
    
    # Get top 10 solutions
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    top_10_solutions = []
    top_10_probabilities = []
    top_10_feasible = []
    
    for bitstring, count in sorted_counts:
        solution = [int(b) for b in bitstring[::-1][:n_assets]]
        probability = count / total_shots
        is_feasible = sum(solution) == budget
        
        top_10_solutions.append(bitstring[:n_assets])
        top_10_probabilities.append(probability)
        top_10_feasible.append(is_feasible)
    
    # Calculate Hamming weight distribution
    hamming_dist = {}
    for bitstring, count in counts.items():
        hamming_weight = sum(int(b) for b in bitstring[::-1][:n_assets])
        if hamming_weight not in hamming_dist:
            hamming_dist[hamming_weight] = 0
        hamming_dist[hamming_weight] += count / total_shots
    
    return {
        'top_10_solutions': top_10_solutions,
        'top_10_probabilities': top_10_probabilities,
        'top_10_feasible': top_10_feasible,
        'hamming_weight_distribution': hamming_dist
    }

def visualize_distribution(distribution: Dict, run_number: int, output_dir: str):
    """Create visualization of probability distribution"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Top 10 solutions
    solutions = [f"Sol{i+1}" for i in range(len(distribution['top_10_solutions']))]
    probabilities = distribution['top_10_probabilities']
    colors = ['green' if f else 'red' for f in distribution['top_10_feasible']]
    
    ax1.bar(solutions, probabilities, color=colors)
    ax1.set_xlabel('Solution')
    ax1.set_ylabel('Probability')
    ax1.set_title(f'Top 10 Solutions - Run {run_number}')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add actual bitstrings as text
    for i, (sol, prob) in enumerate(zip(distribution['top_10_solutions'], probabilities)):
        ax1.text(i, prob, f"{sol}\n{prob:.4f}", ha='center', va='bottom', fontsize=8)
    
    # Hamming weight distribution
    hamming_dist = distribution['hamming_weight_distribution']
    weights = sorted(hamming_dist.keys())
    probs = [hamming_dist[w] for w in weights]
    
    ax2.bar(weights, probs)
    ax2.axvline(x=4, color='green', linestyle='--', label='Target (k=4)')
    ax2.set_xlabel('Hamming Weight (# assets selected)')
    ax2.set_ylabel('Probability')
    ax2.set_title(f'Hamming Weight Distribution - Run {run_number}')
    ax2.legend()
    ax2.set_xticks(weights)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'run_{run_number:02d}_distribution.png'))
    plt.close()

def run_experiment(n_assets: int, budget: int, risk_factor: float, 
                  scenario_type: str, run_number: int) -> Dict:
    """Run a single experiment"""
    
    print(f"\n{'='*60}")
    print(f"RUN {run_number}/20")
    print(f"{'='*60}")
    print(f"Scenario: {scenario_type}")
    
    # Generate market data
    np.random.seed(42 + run_number)  # Different seed for each run
    expected_returns, covariance = create_market_scenario(scenario_type, n_assets)
    
    # Initialize optimizer with fixed implementation
    optimizer = UltraOptimizedQAOAv3Fixed(
        n_assets=n_assets,
        budget=budget,
        risk_factor=risk_factor
    )
    
    # Run optimization
    result = optimizer.solve_ultra_optimized_v3_fixed(
        expected_returns=expected_returns,
        covariance=covariance,
        max_iterations=30
    )
    
    # Analyze solution distribution
    distribution = analyze_solution_distribution(
        result.measurement_counts, n_assets, budget
    )
    
    # Prepare results
    experiment_result = {
        'run_number': run_number,
        'scenario': scenario_type,
        'n_assets': n_assets,
        'budget': budget,
        'approximation_ratio': result.approximation_ratio,
        'best_solution_probability': result.best_solution_probability,
        'feasibility_rate': result.feasibility_rate,
        'initial_feasibility_rate': result.initial_feasibility_rate,
        'circuit_depth': result.circuit_depth,
        'convergence_iterations': result.iterations_to_convergence,
        'objective_value': result.objective_value,
        'selected_assets': result.solution.tolist(),
        'top_10_solutions': distribution['top_10_solutions'],
        'top_10_probabilities': distribution['top_10_probabilities'],
        'top_10_feasible': distribution['top_10_feasible'],
        'hamming_weight_distribution': distribution['hamming_weight_distribution']
    }
    
    # Print summary
    print(f"Approximation Ratio: {result.approximation_ratio:.4f}")
    print(f"Best Solution Probability: {result.best_solution_probability:.4f}")
    print(f"Feasibility Rate: {result.feasibility_rate:.4f}")
    print(f"Initial Feasibility: {result.initial_feasibility_rate:.4f}")
    print(f"Circuit Depth: {result.circuit_depth}")
    
    return experiment_result, distribution

def main():
    """Run 20 experiments and store results"""
    
    # Configuration
    n_assets = 15
    budget = 4
    risk_factor = 0.3
    n_experiments = 20
    
    # Create output directory
    output_dir = '20_by_15_v3'
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print(f"STARTING TEST SUITE V3: {n_assets} ASSETS, SELECT {budget}")
    print(f"Using ultra_optimized_v3_fixed implementation")
    print("="*60)
    
    # Define scenarios to test
    scenarios = [
        "bull_low_vol",
        "bear_high_vol",
        "neutral_mixed",
        "high_return_high_risk",
        "random"
    ]
    
    all_results = []
    feasibility_rates = []
    approximation_ratios = []
    best_solution_probs = []
    circuit_depths = []
    convergence_iterations = []
    
    # Run experiments
    for i in range(1, n_experiments + 1):
        # Cycle through scenarios
        scenario = scenarios[(i-1) % len(scenarios)]
        
        # Run experiment
        result, distribution = run_experiment(
            n_assets, budget, risk_factor, scenario, i
        )
        
        # Store results
        all_results.append(result)
        feasibility_rates.append(result['feasibility_rate'])
        approximation_ratios.append(result['approximation_ratio'])
        best_solution_probs.append(result['best_solution_probability'])
        circuit_depths.append(result['circuit_depth'])
        convergence_iterations.append(result['convergence_iterations'])
        
        # Save individual result
        with open(os.path.join(output_dir, f'run_{i:02d}_results.json'), 'w') as f:
            json.dump(result, f, indent=2)
        
        # Create visualization
        visualize_distribution(distribution, i, output_dir)
    
    # Calculate statistics
    print("\n" + "="*60)
    print("GENERATING FINAL SUMMARY")
    print("="*60)
    
    summary_stats = {
        'test_date': datetime.now().isoformat(),
        'configuration': {
            'n_assets': n_assets,
            'budget': budget,
            'total_runs': n_experiments
        },
        'statistics': {
            'approximation_ratio': {
                'mean': np.mean(approximation_ratios),
                'std': np.std(approximation_ratios),
                'min': np.min(approximation_ratios),
                'max': np.max(approximation_ratios),
                'median': np.median(approximation_ratios)
            },
            'best_solution_probability': {
                'mean': np.mean(best_solution_probs),
                'std': np.std(best_solution_probs),
                'min': np.min(best_solution_probs),
                'max': np.max(best_solution_probs),
                'median': np.median(best_solution_probs)
            },
            'feasibility_rate': {
                'mean': np.mean(feasibility_rates),
                'std': np.std(feasibility_rates),
                'min': np.min(feasibility_rates),
                'max': np.max(feasibility_rates),
                'median': np.median(feasibility_rates)
            },
            'initial_feasibility_rate': {
                'mean': np.mean([r['initial_feasibility_rate'] for r in all_results]),
                'std': np.std([r['initial_feasibility_rate'] for r in all_results]),
                'min': np.min([r['initial_feasibility_rate'] for r in all_results]),
                'max': np.max([r['initial_feasibility_rate'] for r in all_results])
            },
            'circuit_depth': {
                'mean': np.mean(circuit_depths),
                'std': np.std(circuit_depths),
                'min': np.min(circuit_depths),
                'max': np.max(circuit_depths)
            },
            'convergence_iterations': {
                'mean': np.mean(convergence_iterations),
                'std': np.std(convergence_iterations)
            }
        },
        'all_results': all_results
    }
    
    # Save summary
    with open(os.path.join(output_dir, 'summary_statistics.json'), 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # Create summary visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Feasibility rates
    axes[0, 0].plot(feasibility_rates, 'b-o')
    axes[0, 0].axhline(y=0.2, color='g', linestyle='--', label='Target (20%)')
    axes[0, 0].set_xlabel('Run')
    axes[0, 0].set_ylabel('Feasibility Rate')
    axes[0, 0].set_title('Feasibility Rate Across Runs')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Approximation ratios
    axes[0, 1].plot(approximation_ratios, 'r-o')
    axes[0, 1].axhline(y=0.85, color='g', linestyle='--', label='Target (0.85)')
    axes[0, 1].set_xlabel('Run')
    axes[0, 1].set_ylabel('Approximation Ratio')
    axes[0, 1].set_title('Approximation Ratio Across Runs')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Best solution probabilities
    axes[0, 2].plot(best_solution_probs, 'g-o')
    axes[0, 2].axhline(y=0.0001, color='r', linestyle='--', label='Target (0.01%)')
    axes[0, 2].set_xlabel('Run')
    axes[0, 2].set_ylabel('Best Solution Probability')
    axes[0, 2].set_title('Best Solution Probability Across Runs')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Circuit depths histogram
    axes[1, 0].hist(circuit_depths, bins=range(min(circuit_depths), max(circuit_depths)+2), edgecolor='black')
    axes[1, 0].axvline(x=7, color='g', linestyle='--', label='Target (≤7)')
    axes[1, 0].set_xlabel('Circuit Depth')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Circuit Depth Distribution')
    axes[1, 0].legend()
    
    # Convergence iterations
    axes[1, 1].plot(convergence_iterations, 'm-o')
    axes[1, 1].axhline(y=30, color='g', linestyle='--', label='Target (<30)')
    axes[1, 1].set_xlabel('Run')
    axes[1, 1].set_ylabel('Iterations to Convergence')
    axes[1, 1].set_title('Convergence Speed')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Initial vs Final feasibility
    initial_feas = [r['initial_feasibility_rate'] for r in all_results]
    x = range(1, len(initial_feas) + 1)
    axes[1, 2].plot(x, initial_feas, 'b--o', label='Initial', alpha=0.5)
    axes[1, 2].plot(x, feasibility_rates, 'b-o', label='Final')
    axes[1, 2].set_xlabel('Run')
    axes[1, 2].set_ylabel('Feasibility Rate')
    axes[1, 2].set_title('Initial vs Final Feasibility')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle(f'QAOA v3 Fixed Performance Summary - {n_assets} Assets, Select {budget}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_statistics.png'))
    plt.close()
    
    # Print final statistics
    print("\n" + "="*60)
    print(f"FINAL STATISTICS ({n_experiments} RUNS) - {n_assets} ASSETS, SELECT {budget} (V3 FIXED)")
    print("="*60)
    
    print("\n1. APPROXIMATION RATIO:")
    print(f"   Mean: {np.mean(approximation_ratios):.4f} ± {np.std(approximation_ratios):.4f}")
    print(f"   Range: [{np.min(approximation_ratios):.4f}, {np.max(approximation_ratios):.4f}]")
    print(f"   Median: {np.median(approximation_ratios):.4f}")
    
    print("\n2. BEST SOLUTION PROBABILITY:")
    print(f"   Mean: {np.mean(best_solution_probs):.6f} ± {np.std(best_solution_probs):.6f}")
    print(f"   Range: [{np.min(best_solution_probs):.6f}, {np.max(best_solution_probs):.6f}]")
    
    print("\n3. FEASIBILITY RATE:")
    print(f"   Mean: {np.mean(feasibility_rates):.4f} ± {np.std(feasibility_rates):.4f}")
    print(f"   Range: [{np.min(feasibility_rates):.4f}, {np.max(feasibility_rates):.4f}]")
    
    print("\n4. CIRCUIT DEPTH:")
    print(f"   Consistent: {np.min(circuit_depths)} (target: ≤7)")
    
    print("\n5. CONVERGENCE:")
    print(f"   Mean iterations: {np.mean(convergence_iterations):.1f} ± {np.std(convergence_iterations):.1f}")
    
    print("\n6. TARGET ACHIEVEMENT:")
    print(f"   Feasibility > 20%: {'YES' if np.mean(feasibility_rates) > 0.20 else 'NO'} ({np.mean(feasibility_rates):.1%})")
    print(f"   Approximation > 0.85: {'YES' if np.mean(approximation_ratios) > 0.85 else 'NO'} ({np.mean(approximation_ratios):.3f})")
    print(f"   Best Sol Prob > 0.01%: {'YES' if np.mean(best_solution_probs) > 0.0001 else 'NO'} ({np.mean(best_solution_probs):.4%})")
    print(f"   Circuit Depth ≤ 7: {'YES' if np.max(circuit_depths) <= 7 else 'NO'} (depth: {np.max(circuit_depths)})")
    
    print(f"\nResults saved in '{output_dir}/' directory")
    print("="*60)

if __name__ == "__main__":
    main()