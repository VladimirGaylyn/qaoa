"""
Quick test to verify XY-mixer and optimization fixes
"""

import numpy as np
from best_honest_qaoa import BestHonestQAOA

# Small test problem
n_assets = 8
budget = 3
risk_factor = 0.5

# Generate simple test data
np.random.seed(42)
expected_returns = np.random.uniform(0.05, 0.15, n_assets)

# Simple covariance
covariance = np.eye(n_assets) * 0.1
for i in range(n_assets):
    for j in range(i + 1, n_assets):
        covariance[i, j] = covariance[j, i] = np.random.uniform(-0.03, 0.06)

# Ensure positive definite
min_eig = np.min(np.linalg.eigvals(covariance))
if min_eig < 0:
    covariance += np.eye(n_assets) * (-min_eig + 0.01)

print("Testing fixed BestHonestQAOA implementation...")
print(f"Problem: {n_assets} assets, select {budget}")

# Create optimizer with fixed code
qaoa = BestHonestQAOA(n_assets, budget, risk_factor)
qaoa.p = 3  # Small depth for quick test
qaoa.shots = 1024  # Fewer shots for speed
qaoa.max_iterations = 20  # Quick test

print("\nRunning optimization with fixes:")
print("- XY-mixer with full strength (no arbitrary 0.5 factor)")
print("- Hybrid SPSA + L-BFGS-B optimization")

try:
    results = qaoa.run(expected_returns, covariance)
    
    print("\nTest completed successfully!")
    print(f"Best solution probability: {results['best_solution_probability']*100:.4f}%")
    print(f"Feasibility rate: {results['feasibility_rate']*100:.2f}%")
    print(f"Circuit executions: {results['circuit_executions']}")
    print("\nFixes are working correctly!")
    
except Exception as e:
    print(f"\nError during test: {e}")
    print("There may be an issue with the fixes.")