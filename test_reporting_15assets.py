"""
Quick test of reporting functionality for 15-asset portfolio
"""

import numpy as np
from improved_qaoa_portfolio import ImprovedQAOAPortfolio
from qaoa_reporting import QAOAReporter
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for faster execution


def test_15_asset_reporting():
    """Test reporting for 15-asset portfolio"""
    
    print("="*70)
    print("15-ASSET PORTFOLIO - REPORTING TEST")
    print("="*70)
    
    # Configuration
    n_assets = 15
    budget = 7
    risk_factor = 0.5
    
    # Generate test data
    np.random.seed(42)
    expected_returns = np.random.uniform(0.05, 0.25, n_assets)
    
    # Simple covariance matrix
    covariances = np.eye(n_assets) * 0.04  # Simple diagonal covariance
    
    # Initialize optimizer
    optimizer = ImprovedQAOAPortfolio(n_assets=n_assets, budget=budget, risk_factor=risk_factor)
    reporter = QAOAReporter(output_dir="reports")
    
    # 1. Classical solution
    print("\n1. Classical Solution:")
    print("-" * 40)
    classical_result = optimizer.solve_classical_baseline(expected_returns, covariances)
    print(f"  Objective: {classical_result.objective_value:.6f}")
    print(f"  Selected: {np.where(classical_result.solution == 1)[0].tolist()}")
    
    # 2. QAOA solution (minimal iterations for speed)
    print("\n2. QAOA Solution:")
    print("-" * 40)
    qaoa_result, measurement_counts = optimizer.solve_improved_qaoa(
        expected_returns, covariances,
        n_layers=1,
        max_iterations=5,  # Very few iterations for quick test
        return_counts=True
    )
    print(f"  Objective: {qaoa_result.objective_value:.6f}")
    print(f"  Approximation Ratio: {qaoa_result.approximation_ratio:.3f}")
    print(f"  Selected: {np.where(qaoa_result.solution == 1)[0].tolist()}")
    
    # 3. Generate reports
    print("\n3. Generating Reports:")
    print("-" * 40)
    
    # Save probability distribution
    print("  Creating probability distribution visualization...")
    prob_fig = reporter.generate_probability_distribution(
        measurement_counts, n_assets, budget,
        save_path="reports/test_probability_dist.png"
    )
    
    # Save comparison report
    print("  Creating comparison visualization...")
    comp_fig = reporter.generate_comparison_report(
        classical_result, qaoa_result, measurement_counts, n_assets, budget,
        save_path="reports/test_comparison.png"
    )
    
    # Generate text report
    print("  Generating text report...")
    text_report = reporter.generate_text_report(
        classical_result, qaoa_result, measurement_counts, n_assets, budget
    )
    
    # Print key statistics from text report
    print("\n4. Key Statistics:")
    print("-" * 40)
    
    # Calculate feasibility rate
    feasible_count = 0
    total_count = sum(measurement_counts.values())
    
    for bitstring, count in measurement_counts.items():
        bitstring_clean = bitstring.replace(' ', '')
        solution = [int(b) for b in bitstring_clean[::-1][:n_assets]]
        if sum(solution) == budget:
            feasible_count += count
    
    print(f"  Constraint Satisfaction: {100*feasible_count/total_count:.1f}%")
    print(f"  Total Measurements: {total_count}")
    print(f"  Unique States: {len(measurement_counts)}")
    print(f"  Circuit Depth: {qaoa_result.circuit_depth}")
    print(f"  Gate Count: {qaoa_result.gate_count}")
    
    # Save text report
    with open("reports/test_report.txt", 'w') as f:
        f.write(text_report)
    
    print("\n5. Reports Saved:")
    print("-" * 40)
    print("  - reports/test_probability_dist.png")
    print("  - reports/test_comparison.png")
    print("  - reports/test_report.txt")
    
    return classical_result, qaoa_result, measurement_counts


if __name__ == "__main__":
    classical, qaoa, counts = test_15_asset_reporting()
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print(f"Successfully generated reports for 15-asset portfolio")
    print(f"QAOA achieved {qaoa.approximation_ratio:.1%} of classical optimum")
    print("="*70)