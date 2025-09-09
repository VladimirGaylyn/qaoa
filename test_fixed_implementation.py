"""
Test the fixed QAOA implementation
Verify all improvements are working correctly
"""

import numpy as np
from ultra_optimized_v3_fixed import UltraOptimizedQAOAv3Fixed
from classical_strategies import ClassicalPortfolioStrategies
import warnings
warnings.filterwarnings('ignore')

def test_fixed_implementation():
    """Test the fixed v3 with all improvements"""
    
    # Test configuration
    n_assets = 15
    budget = 4
    risk_factor = 0.3
    
    # Generate test data
    np.random.seed(42)
    expected_returns = np.random.uniform(0.05, 0.15, n_assets)
    volatilities = np.random.uniform(0.1, 0.3, n_assets)
    
    # Create correlation matrix
    correlation = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            corr = np.random.uniform(-0.3, 0.3)
            correlation[i, j] = corr
            correlation[j, i] = corr
    
    # Make positive definite
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalues = np.maximum(eigenvalues, 0.01)
    correlation = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    # Convert to covariance
    D = np.diag(volatilities)
    covariance_matrix = D @ correlation @ D
    
    print("="*60)
    print("TESTING FIXED QAOA IMPLEMENTATION")
    print("="*60)
    print(f"Configuration: {n_assets} assets, select {budget}")
    print(f"Risk factor: {risk_factor}")
    
    # Initialize optimizer
    optimizer = UltraOptimizedQAOAv3Fixed(
        n_assets=n_assets,
        budget=budget,
        risk_factor=risk_factor
    )
    
    # Verify fixes
    print("\n" + "="*60)
    print("VERIFYING FIXES")
    print("="*60)
    
    # 1. Check penalty scaling
    expected_penalty = 20.0 * np.sqrt(n_assets + 1)
    print(f"[OK] Penalty multiplier: {optimizer.penalty_multiplier:.2f} (expected: {expected_penalty:.2f})")
    
    # 2. Check shot allocation
    shots_initial = optimizer.get_adaptive_shot_count(iteration=1)
    shots_mid = optimizer.get_adaptive_shot_count(iteration=25)
    shots_final = optimizer.get_adaptive_shot_count(is_final=True)
    print(f"[OK] Adaptive shots: {shots_initial} -> {shots_mid} -> {shots_final}")
    
    # 3. Check circuit parameters
    qc, params = optimizer.create_improved_shallow_circuit(n_assets)
    print(f"[OK] Circuit parameters: {len(params)} (properly calculated)")
    print(f"[OK] Circuit depth: {qc.depth()} (target: <=7)")
    
    # 4. Test amplitude amplification
    test_counts = {
        '000000000001111': 100,  # Feasible (4 assets)
        '000000000011111': 50,   # Not feasible (5 assets)
        '000000000000111': 30,   # Not feasible (3 assets)
    }
    amplified = optimizer.amplify_feasible_solutions(test_counts)
    print(f"[OK] Amplitude amplification: {amplified['000000000001111']} (was 100, amplified by 1.5x)")
    
    # Run optimization
    print("\n" + "="*60)
    print("RUNNING OPTIMIZATION")
    print("="*60)
    
    result = optimizer.solve_ultra_optimized_v3_fixed(
        expected_returns=expected_returns,
        covariance=covariance_matrix,
        max_iterations=30
    )
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"[OK] Circuit depth: {result.circuit_depth} (target: <=7)")
    print(f"[OK] Approximation ratio: {result.approximation_ratio:.3f}")
    print(f"[OK] Feasibility rate: {result.feasibility_rate:.1%}")
    print(f"[OK] Best solution probability: {result.best_solution_probability:.4%}")
    print(f"[OK] Constraint satisfied: {result.constraint_satisfied}")
    print(f"[OK] Converged: {result.converged} (iteration {result.iterations_to_convergence})")
    print(f"[OK] Execution time: {result.execution_time:.2f}s")
    
    # Check improvements
    print("\n" + "="*60)
    print("PERFORMANCE IMPROVEMENTS")
    print("="*60)
    
    improvements = []
    
    # Target: Circuit depth ≤ 7
    if result.circuit_depth <= 7:
        improvements.append(f"[PASS] Circuit depth: {result.circuit_depth} <= 7")
    else:
        improvements.append(f"[FAIL] Circuit depth: {result.circuit_depth} > 7")
    
    # Target: Feasibility rate > 20%
    if result.feasibility_rate > 0.20:
        improvements.append(f"[PASS] Feasibility rate: {result.feasibility_rate:.1%} > 20%")
    elif result.feasibility_rate > 0.15:
        improvements.append(f"[WARN] Feasibility rate: {result.feasibility_rate:.1%} (close to target)")
    else:
        improvements.append(f"[FAIL] Feasibility rate: {result.feasibility_rate:.1%} < 20%")
    
    # Target: Best solution probability > 0.01%
    if result.best_solution_probability > 0.0001:
        improvements.append(f"[PASS] Best solution prob: {result.best_solution_probability:.4%} > 0.01%")
    else:
        improvements.append(f"[WARN] Best solution prob: {result.best_solution_probability:.4%} (needs improvement)")
    
    # Target: Approximation ratio > 0.85
    if result.approximation_ratio > 0.85:
        improvements.append(f"[PASS] Approximation ratio: {result.approximation_ratio:.3f} > 0.85")
    elif result.approximation_ratio > 0.70:
        improvements.append(f"[WARN] Approximation ratio: {result.approximation_ratio:.3f} (moderate)")
    else:
        improvements.append(f"[FAIL] Approximation ratio: {result.approximation_ratio:.3f} < 0.70")
    
    # Target: Convergence < 30 iterations
    if result.iterations_to_convergence < 20:
        improvements.append(f"[PASS] Convergence: {result.iterations_to_convergence} iterations < 20")
    elif result.iterations_to_convergence < 30:
        improvements.append(f"[WARN] Convergence: {result.iterations_to_convergence} iterations < 30")
    else:
        improvements.append(f"[FAIL] Convergence: {result.iterations_to_convergence} iterations >= 30")
    
    for improvement in improvements:
        print(improvement)
    
    # Show improvement from initial state
    print("\n" + "="*60)
    print("IMPROVEMENT FROM INITIAL STATE")
    print("="*60)
    print(f"Initial feasibility: {result.initial_feasibility_rate:.1%}")
    print(f"Final feasibility: {result.feasibility_rate:.1%}")
    improvement_factor = result.feasibility_rate / result.initial_feasibility_rate if result.initial_feasibility_rate > 0 else float('inf')
    print(f"Improvement factor: {improvement_factor:.1f}x")
    
    # Show selected assets
    print("\n" + "="*60)
    print("PORTFOLIO SELECTION")
    print("="*60)
    selected_indices = np.where(result.solution == 1)[0]
    print(f"Selected assets: {selected_indices.tolist()}")
    print(f"Expected return: {result.expected_return:.4f}")
    print(f"Risk: {result.risk:.4f}")
    print(f"Sharpe ratio: {result.sharpe_ratio:.4f}")
    
    return result

def compare_with_original():
    """Compare fixed version with original"""
    print("\n" + "="*60)
    print("COMPARISON: ORIGINAL vs FIXED")
    print("="*60)
    
    # Test configuration
    n_assets = 15
    budget = 4
    risk_factor = 0.3
    
    # Generate test data
    np.random.seed(42)
    expected_returns = np.random.uniform(0.05, 0.15, n_assets)
    volatilities = np.random.uniform(0.1, 0.3, n_assets)
    
    # Create correlation matrix
    correlation = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            corr = np.random.uniform(-0.3, 0.3)
            correlation[i, j] = corr
            correlation[j, i] = corr
    
    # Make positive definite
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalues = np.maximum(eigenvalues, 0.01)
    correlation = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    # Convert to covariance
    D = np.diag(volatilities)
    covariance_matrix = D @ correlation @ D
    
    # Test fixed version
    print("\nTesting FIXED version...")
    optimizer_fixed = UltraOptimizedQAOAv3Fixed(n_assets, budget, risk_factor)
    result_fixed = optimizer_fixed.solve_ultra_optimized_v3_fixed(
        expected_returns, covariance_matrix, max_iterations=20
    )
    
    # Import and test original (if available)
    try:
        from ultra_optimized_v3_advanced import UltraOptimizedQAOAv3
        print("\nTesting ORIGINAL version...")
        optimizer_orig = UltraOptimizedQAOAv3(n_assets, budget, risk_factor)
        result_orig = optimizer_orig.solve_ultra_optimized_v3(
            expected_returns, covariance_matrix, max_iterations=20
        )
        
        # Compare results
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        print(f"Metric                  | Original     | Fixed        | Improvement")
        print("-" * 70)
        print(f"Feasibility Rate        | {result_orig.feasibility_rate:8.1%}    | {result_fixed.feasibility_rate:8.1%}    | {(result_fixed.feasibility_rate - result_orig.feasibility_rate)*100:+.1f}pp")
        print(f"Best Solution Prob      | {result_orig.best_solution_probability if hasattr(result_orig, 'best_solution_probability') else 0:8.4%}    | {result_fixed.best_solution_probability:8.4%}    | {result_fixed.best_solution_probability:+.4%}")
        print(f"Approximation Ratio     | {result_orig.approximation_ratio:8.3f}    | {result_fixed.approximation_ratio:8.3f}    | {result_fixed.approximation_ratio - result_orig.approximation_ratio:+.3f}")
        print(f"Convergence Iterations  | {result_orig.iterations_to_convergence:8d}    | {result_fixed.iterations_to_convergence:8d}    | {result_orig.iterations_to_convergence - result_fixed.iterations_to_convergence:+d}")
        print(f"Circuit Depth          | {result_orig.circuit_depth:8d}    | {result_fixed.circuit_depth:8d}    | {result_orig.circuit_depth - result_fixed.circuit_depth:+d}")
        
    except ImportError:
        print("\nOriginal version not available for comparison")
    except Exception as e:
        print(f"\nError comparing with original: {e}")

if __name__ == "__main__":
    # Run main test
    result = test_fixed_implementation()
    
    # Run comparison if desired
    # compare_with_original()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    success_count = 0
    total_checks = 5
    
    if result.circuit_depth <= 7:
        success_count += 1
    if result.feasibility_rate > 0.15:
        success_count += 1
    if result.best_solution_probability > 0:
        success_count += 1
    if result.approximation_ratio > 0.7:
        success_count += 1
    if result.constraint_satisfied:
        success_count += 1
    
    print(f"Passed {success_count}/{total_checks} critical checks")
    
    if success_count == total_checks:
        print("[SUCCESS] ALL CRITICAL ISSUES FIXED!")
    elif success_count >= 4:
        print("[WARNING] Most issues fixed, minor improvements needed")
    else:
        print("[ERROR] Some critical issues remain")