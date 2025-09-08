"""
Run comprehensive benchmark with full reporting
Compares classical and quantum algorithms with detailed visualizations
"""

import numpy as np
from improved_qaoa_portfolio import ImprovedQAOAPortfolio
from qaoa_reporting import QAOAReporter
import matplotlib.pyplot as plt


def run_benchmark_with_reporting():
    """
    Run benchmark for multiple portfolio sizes with comprehensive reporting
    """
    print("="*70)
    print("QAOA PORTFOLIO OPTIMIZATION - COMPREHENSIVE BENCHMARK")
    print("With Probability Distributions and Performance Analysis")
    print("="*70)
    
    # Initialize reporter
    reporter = QAOAReporter(output_dir="reports")
    
    # Test configurations
    test_cases = [
        {'n_assets': 8, 'budget': 4, 'risk_factor': 0.5},
        {'n_assets': 10, 'budget': 5, 'risk_factor': 0.5},
        {'n_assets': 15, 'budget': 7, 'risk_factor': 0.5},
    ]
    
    for i, config in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST CASE {i}: {config['n_assets']} Assets, Budget={config['budget']}")
        print('='*70)
        
        # Generate test data with fixed seed for reproducibility
        np.random.seed(42 + i)
        
        # Generate expected returns
        expected_returns = np.random.uniform(0.05, 0.25, config['n_assets'])
        
        # Generate covariance matrix
        correlation = np.random.uniform(-0.3, 0.7, 
                                      (config['n_assets'], config['n_assets']))
        correlation = (correlation + correlation.T) / 2
        np.fill_diagonal(correlation, 1)
        
        # Ensure positive definite
        eigenvalues, eigenvectors = np.linalg.eigh(correlation)
        eigenvalues[eigenvalues < 0.01] = 0.01
        correlation = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        std_devs = np.random.uniform(0.1, 0.3, config['n_assets'])
        covariances = np.outer(std_devs, std_devs) * correlation
        
        # Initialize optimizer
        optimizer = ImprovedQAOAPortfolio(
            n_assets=config['n_assets'],
            budget=config['budget'],
            risk_factor=config['risk_factor']
        )
        
        # 1. Solve with classical algorithm
        print("\n1. CLASSICAL ALGORITHM")
        print("-" * 40)
        classical_result = optimizer.solve_classical_baseline(expected_returns, covariances)
        
        print(f"Classical Results:")
        print(f"  Objective: {classical_result.objective_value:.6f}")
        print(f"  Expected Return: {classical_result.expected_return:.4f}")
        print(f"  Risk: {classical_result.risk:.4f}")
        print(f"  Time: {classical_result.execution_time:.3f}s")
        
        # 2. Solve with QAOA and get measurement counts
        print("\n2. QUANTUM ALGORITHM (QAOA)")
        print("-" * 40)
        
        # Adjust parameters based on problem size
        if config['n_assets'] <= 8:
            n_layers = 2
            max_iterations = 50
        elif config['n_assets'] <= 12:
            n_layers = 1
            max_iterations = 30
        else:
            n_layers = 1
            max_iterations = 20
        
        qaoa_result, measurement_counts = optimizer.solve_improved_qaoa(
            expected_returns, covariances,
            n_layers=n_layers,
            max_iterations=max_iterations,
            return_counts=True
        )
        
        print(f"\nQAOA Results:")
        print(f"  Objective: {qaoa_result.objective_value:.6f}")
        print(f"  Expected Return: {qaoa_result.expected_return:.4f}")
        print(f"  Risk: {qaoa_result.risk:.4f}")
        print(f"  Approximation Ratio: {qaoa_result.approximation_ratio:.3f}")
        print(f"  Time: {qaoa_result.execution_time:.3f}s")
        
        # 3. Generate comprehensive report
        print("\n3. GENERATING REPORT")
        print("-" * 40)
        
        report_name = f"portfolio_{config['n_assets']}assets_{config['budget']}budget"
        report_dir = reporter.save_full_report(
            classical_result=classical_result,
            qaoa_result=qaoa_result,
            measurement_counts=measurement_counts,
            n_assets=config['n_assets'],
            budget=config['budget'],
            report_name=report_name
        )
        
        print(f"\nReport saved to: {report_dir}")
        
        # 4. Print summary statistics
        print("\n4. SUMMARY STATISTICS")
        print("-" * 40)
        
        # Calculate feasibility rate
        feasible_count = 0
        total_count = sum(measurement_counts.values())
        
        for bitstring, count in measurement_counts.items():
            bitstring_clean = bitstring.replace(' ', '')
            solution = [int(b) for b in bitstring_clean[::-1][:config['n_assets']]]
            if sum(solution) == config['budget']:
                feasible_count += count
        
        feasibility_rate = feasible_count / total_count if total_count > 0 else 0
        
        print(f"Constraint Satisfaction Rate: {feasibility_rate:.1%}")
        print(f"Total Measurements: {total_count}")
        print(f"Unique States Measured: {len(measurement_counts)}")
        print(f"Circuit Depth: {qaoa_result.circuit_depth}")
        print(f"Gate Count: {qaoa_result.gate_count}")
        
        # Performance comparison
        if classical_result.objective_value > 0:
            performance_ratio = qaoa_result.objective_value / classical_result.objective_value
            print(f"Performance vs Classical: {performance_ratio:.1%}")
        
        print(f"Speedup Factor: {classical_result.execution_time / qaoa_result.execution_time:.2f}x")
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("All reports saved in 'reports' directory")
    print("="*70)


def run_single_15_asset_benchmark():
    """
    Run a single benchmark specifically for 15-asset portfolio with detailed analysis
    """
    print("="*70)
    print("15-ASSET PORTFOLIO OPTIMIZATION - DETAILED ANALYSIS")
    print("="*70)
    
    # Configuration
    n_assets = 15
    budget = 7
    risk_factor = 0.5
    
    # Generate test data
    np.random.seed(42)
    expected_returns = np.random.uniform(0.05, 0.25, n_assets)
    
    # Generate covariance matrix
    correlation = np.random.uniform(-0.3, 0.7, (n_assets, n_assets))
    correlation = (correlation + correlation.T) / 2
    np.fill_diagonal(correlation, 1)
    
    # Ensure positive definite
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalues[eigenvalues < 0.01] = 0.01
    correlation = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    std_devs = np.random.uniform(0.1, 0.3, n_assets)
    covariances = np.outer(std_devs, std_devs) * correlation
    
    # Initialize optimizer and reporter
    optimizer = ImprovedQAOAPortfolio(n_assets=n_assets, budget=budget, risk_factor=risk_factor)
    reporter = QAOAReporter(output_dir="reports")
    
    # Solve with classical
    print("\nClassical Solution:")
    classical_result = optimizer.solve_classical_baseline(expected_returns, covariances)
    
    # Solve with QAOA
    print("\nQAOA Solution:")
    qaoa_result, measurement_counts = optimizer.solve_improved_qaoa(
        expected_returns, covariances,
        n_layers=1,
        max_iterations=20,
        return_counts=True
    )
    
    # Generate and display reports
    print("\nGenerating comprehensive report...")
    
    # Generate probability distribution
    prob_fig = reporter.generate_probability_distribution(
        measurement_counts, n_assets, budget
    )
    plt.show()
    
    # Generate comparison report
    comp_fig = reporter.generate_comparison_report(
        classical_result, qaoa_result, measurement_counts, n_assets, budget
    )
    plt.show()
    
    # Print text report
    text_report = reporter.generate_text_report(
        classical_result, qaoa_result, measurement_counts, n_assets, budget
    )
    print("\n" + text_report)
    
    # Save full report
    report_dir = reporter.save_full_report(
        classical_result, qaoa_result, measurement_counts, n_assets, budget,
        report_name="15_asset_detailed_analysis"
    )
    
    print(f"\nFull report saved to: {report_dir}")


if __name__ == "__main__":
    # Run comprehensive benchmark
    run_benchmark_with_reporting()
    
    # Optionally run detailed 15-asset analysis
    print("\n" + "="*70)
    print("Running detailed 15-asset analysis...")
    print("="*70)
    run_single_15_asset_benchmark()