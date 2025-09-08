"""
Quick test of improved QAOA for 15-asset portfolio
"""

import numpy as np
from improved_qaoa_portfolio import ImprovedQAOAPortfolio
import time

def test_15_asset_portfolio():
    """Test improved QAOA with 15-asset portfolio"""
    
    print("="*60)
    print("TESTING IMPROVED QAOA WITH 15-ASSET PORTFOLIO")
    print("="*60)
    
    # Configuration for 15 assets
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
    
    # Initialize optimizer
    optimizer = ImprovedQAOAPortfolio(
        n_assets=n_assets,
        budget=budget,
        risk_factor=risk_factor
    )
    
    # 1. Classical baseline
    print("\n1. Classical Baseline:")
    print("-" * 40)
    classical = optimizer.solve_classical_baseline(expected_returns, covariances)
    print(f"  Objective: {classical.objective_value:.6f}")
    print(f"  Expected Return: {classical.expected_return:.4f}")
    print(f"  Risk: {classical.risk:.4f}")
    print(f"  Selected Assets: {int(classical.n_selected)}")
    print(f"  Time: {classical.execution_time:.3f}s")
    
    # 2. Improved QAOA with single layer for efficiency
    print("\n2. Improved QAOA (Dicke + Grover):")
    print("-" * 40)
    
    # Force single layer and fewer iterations for 15-asset portfolio
    qaoa = optimizer.solve_improved_qaoa(
        expected_returns, 
        covariances,
        n_layers=1,  # Single layer for 15 assets
        max_iterations=20  # Fewer iterations for quick test
    )
    
    print(f"\nResults Summary:")
    print("=" * 40)
    print(f"Constraint Satisfied: {qaoa.constraint_satisfied}")
    print(f"Selected Assets: {qaoa.n_selected} (target: {budget})")
    print(f"Approximation Ratio: {qaoa.approximation_ratio:.3f}")
    print(f"Circuit Depth: {qaoa.circuit_depth}")
    print(f"Gate Count: {qaoa.gate_count}")
    
    # Show selected assets
    if qaoa.constraint_satisfied:
        selected_indices = np.where(qaoa.solution == 1)[0]
        print(f"Selected Asset Indices: {selected_indices.tolist()}")
    
    return classical, qaoa


if __name__ == "__main__":
    classical, qaoa = test_15_asset_portfolio()
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    
    if qaoa.constraint_satisfied:
        print(f"SUCCESS: Successfully optimized 15-asset portfolio!")
        print(f"SUCCESS: Achieved {qaoa.approximation_ratio:.1%} of classical optimum")
    else:
        print("FAILED: Failed to find feasible solution")
    
    print("="*60)