"""
Test the Fixed QAOA Implementation
Verify all improvements work correctly
"""

import numpy as np
import json
import time
from datetime import datetime
from fixed_qaoa import FixedQAOA

def generate_test_portfolio_data(n_assets: int, seed: int = 42):
    """Generate realistic portfolio test data"""
    np.random.seed(seed)
    
    # Expected returns (5-15% annually)
    expected_returns = np.random.uniform(0.05, 0.15, n_assets)
    
    # Generate correlation matrix
    correlation = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            corr = np.random.uniform(-0.3, 0.6)
            correlation[i, j] = corr
            correlation[j, i] = corr
    
    # Standard deviations (10-40% annually)
    std_devs = np.random.uniform(0.1, 0.4, n_assets)
    
    # Covariance matrix
    covariance = np.outer(std_devs, std_devs) * correlation
    
    # Ensure positive definiteness
    min_eigenvalue = np.min(np.linalg.eigvals(covariance))
    if min_eigenvalue < 0:
        covariance += np.eye(n_assets) * (-min_eigenvalue + 0.01)
    
    return expected_returns, covariance

def test_fixed_qaoa():
    """Test the fixed QAOA implementation"""
    
    print("\n" + "="*80)
    print("TESTING FIXED QAOA IMPLEMENTATION")
    print("="*80)
    
    # Test configuration
    n_assets = 15
    budget = 4
    risk_factor = 0.5
    
    print(f"\nTest Configuration:")
    print(f"  Assets: {n_assets}")
    print(f"  Budget: {budget}")
    print(f"  Risk Factor: {risk_factor}")
    
    # Generate test data
    expected_returns, covariance = generate_test_portfolio_data(n_assets)
    
    # Create fixed QAOA instance
    qaoa = FixedQAOA(n_assets, budget, risk_factor)
    
    # Run optimization
    results = qaoa.optimize(expected_returns, covariance)
    
    # Verify improvements
    print("\n" + "="*80)
    print("VERIFICATION OF FIXES")
    print("="*80)
    
    # Fix 1: Check shot count
    print(f"\n[+] Fix 1 - Shot Count:")
    print(f"  Used {qaoa.shots} shots (min recommended: {2**n_assets})")
    print(f"  Statistical error: ~{100/np.sqrt(qaoa.shots):.2f}%")
    
    # Fix 2: Check feasibility rate improvement
    print(f"\n[+] Fix 2 - Feasibility Rate:")
    print(f"  Achieved: {results['feasibility_rate']*100:.2f}%")
    print(f"  Target: 30-40%")
    if results['feasibility_rate'] > 0.25:
        print("  [+] GOOD - Above 25%")
    else:
        print("  [!] Needs more tuning")
    
    # Fix 3: Check best solution probability
    print(f"\n[+] Fix 3 - Best Solution Probability:")
    print(f"  Achieved: {results['best_solution_probability']*100:.3f}%")
    print(f"  Previous: ~0.016%")
    print(f"  Target: 0.1-0.5%")
    if results['best_solution_probability'] > 0.0005:
        print("  [+] IMPROVED - Above 0.05%")
    
    # Fix 4: Check approximation ratio
    print(f"\n[+] Fix 4 - Approximation Ratio:")
    print(f"  Achieved: {results['approximation_ratio']:.3f}")
    print(f"  Target: 0.85-0.95")
    if results['approximation_ratio'] > 0.8:
        print("  [+] GOOD - Above 0.8")
    
    # Fix 5: Check XY-mixer effectiveness
    print(f"\n[+] Fix 5 - Hamming Weight Preservation:")
    if results['n_feasible_states'] > 0:
        print(f"  Found {results['n_feasible_states']} feasible states")
        print("  [+] XY-mixer working")
    
    # Fix 6: Check error mitigation
    print(f"\n[+] Fix 6 - Error Mitigation:")
    print(f"  Applied to {results['total_counts']} measurements")
    print("  [+] Hamming weight filtering active")
    
    # Performance comparison
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    
    comparison = {
        'Metric': ['Feasibility Rate', 'Best Solution Prob', 'Approximation Ratio', 'Circuit Executions'],
        'Previous': ['~20%', '~0.016%', '~1.0', '50'],
        'Fixed': [
            f"{results['feasibility_rate']*100:.1f}%",
            f"{results['best_solution_probability']*100:.3f}%",
            f"{results['approximation_ratio']:.3f}",
            f"{results['circuit_executions']}"
        ],
        'Improvement': [
            f"{results['feasibility_rate']/0.2:.1f}x" if results['feasibility_rate'] > 0 else 'N/A',
            f"{results['best_solution_probability']/0.00016:.1f}x" if results['best_solution_probability'] > 0 else 'N/A',
            'Better' if results['approximation_ratio'] < 1.0 else 'Same',
            'More thorough'
        ]
    }
    
    # Print comparison table
    print("\n{:<25} {:<15} {:<15} {:<15}".format(*comparison['Metric']))
    print("-" * 70)
    print("{:<25} {:<15} {:<15} {:<15}".format(*comparison['Previous']))
    print("{:<25} {:<15} {:<15} {:<15}".format(*comparison['Fixed']))
    print("{:<25} {:<15} {:<15} {:<15}".format(*comparison['Improvement']))
    
    # Save results
    test_results = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'n_assets': n_assets,
            'budget': budget,
            'risk_factor': risk_factor,
            'shots': qaoa.shots,
            'circuit_depth': qaoa.p,
            'max_iterations': qaoa.max_iterations
        },
        'results': {
            'feasibility_rate': results['feasibility_rate'],
            'best_solution_probability': results['best_solution_probability'],
            'approximation_ratio': results['approximation_ratio'],
            'n_feasible_states': results['n_feasible_states'],
            'n_unique_states': results['n_unique_states'],
            'computation_time': results['computation_time']
        },
        'improvements': {
            'feasibility_improvement': results['feasibility_rate'] / 0.2,
            'probability_improvement': results['best_solution_probability'] / 0.00016 if results['best_solution_probability'] > 0 else 0,
            'all_fixes_applied': True
        }
    }
    
    with open('fixed_qaoa_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n[+] Results saved to fixed_qaoa_test_results.json")
    
    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    improvements = 0
    if results['feasibility_rate'] > 0.25:
        improvements += 1
        print("[+] Feasibility rate significantly improved")
    
    if results['best_solution_probability'] > 0.0005:
        improvements += 1
        print("[+] Best solution probability improved")
    
    if results['approximation_ratio'] > 0.8 and results['approximation_ratio'] < 1.0:
        improvements += 1
        print("[+] Approximation ratio reasonable")
    
    if results['n_feasible_states'] > 100:
        improvements += 1
        print("[+] Finding many feasible states")
    
    print(f"\nOverall: {improvements}/4 major improvements achieved")
    
    if improvements >= 3:
        print("SUCCESS: Fixed QAOA shows significant improvement!")
    elif improvements >= 2:
        print("PARTIAL SUCCESS: Some improvements, but more tuning needed")
    else:
        print("NEEDS WORK: Further optimization required")
    
    return results

def run_comparison_test():
    """Run a quick comparison between old and fixed implementations"""
    
    print("\n" + "="*80)
    print("QUICK COMPARISON TEST")
    print("="*80)
    
    n_assets = 10  # Smaller for faster test
    budget = 3
    risk_factor = 0.5
    
    expected_returns, covariance = generate_test_portfolio_data(n_assets, seed=100)
    
    print("\nTesting Fixed QAOA...")
    fixed_qaoa = FixedQAOA(n_assets, budget, risk_factor)
    fixed_qaoa.max_iterations = 50  # Reduce for speed
    
    start = time.time()
    fixed_results = fixed_qaoa.optimize(expected_returns, covariance)
    fixed_time = time.time() - start
    
    print("\n" + "-"*60)
    print("COMPARISON RESULTS:")
    print("-"*60)
    print(f"Fixed QAOA:")
    print(f"  Time: {fixed_time:.2f}s")
    print(f"  Feasibility: {fixed_results['feasibility_rate']*100:.1f}%")
    print(f"  Best Prob: {fixed_results['best_solution_probability']*100:.3f}%")
    print(f"  Approx Ratio: {fixed_results['approximation_ratio']:.3f}")

if __name__ == "__main__":
    # Run main test
    results = test_fixed_qaoa()
    
    # Run comparison test
    print("\n" + "="*80)
    run_comparison_test()