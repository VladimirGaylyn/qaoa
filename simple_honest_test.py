"""
Simple Test - Show HONEST QAOA Performance
"""

import numpy as np
from honest_qaoa import HonestQAOA

# Problem setup
n_assets = 15
budget = 4
risk_factor = 0.5

# Generate test data
np.random.seed(42)
expected_returns = np.random.uniform(0.05, 0.15, n_assets)

# Covariance matrix
correlation = np.eye(n_assets)
for i in range(n_assets):
    for j in range(i + 1, n_assets):
        correlation[i, j] = correlation[j, i] = np.random.uniform(-0.3, 0.6)

std_devs = np.random.uniform(0.1, 0.4, n_assets)
covariance = np.outer(std_devs, std_devs) * correlation

# Ensure positive definite
min_eig = np.min(np.linalg.eigvals(covariance))
if min_eig < 0:
    covariance += np.eye(n_assets) * (-min_eig + 0.01)

print("\n" + "="*70)
print("HONEST QAOA - REAL Quantum Performance (No Tricks!)")
print("="*70)

# Run honest QAOA
qaoa = HonestQAOA(n_assets, budget, risk_factor)
results = qaoa.optimize(expected_returns, covariance, max_iterations=100)

print("\n" + "="*70)
print("FINAL HONEST RESULTS")
print("="*70)
print(f"Best Solution Probability: {results['best_solution_probability']*100:.4f}%")
print(f"Feasibility Rate: {results['feasibility_rate']*100:.2f}%")
print(f"Approximation Ratio: {results['approximation_ratio']:.4f}")
print(f"Circuit Executions: {results['circuit_executions']}")
print(f"Total Quantum Shots: {results['total_shots']:,}")

# Compare with theory
from math import comb
n_feasible = comb(n_assets, budget)
uniform_prob = 1/n_feasible
concentration = results['best_solution_probability'] / uniform_prob

print(f"\nConcentration achieved: {concentration:.1f}x over uniform")
print(f"Uniform probability: {uniform_prob*100:.6f}%")

if results['best_solution_probability'] >= 0.01:
    print("\n✓ ACHIEVED 1% TARGET WITH HONEST QAOA!")
else:
    gap = 0.01 / results['best_solution_probability']
    print(f"\nBelow 1% target by factor of {gap:.1f}x")
    print("This is the REAL performance limit of current QAOA")
    
print("\nTop 5 Solutions:")
for i, sol in enumerate(results['top_10_solutions'][:5]):
    print(f"  {i+1}. Probability: {sol['probability']*100:.4f}%, "
          f"Count: {sol['count']}/{results['total_counts']}")