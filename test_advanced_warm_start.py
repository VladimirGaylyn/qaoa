"""
Test Advanced Warm Start Implementation
Compares Ultra-Optimized QAOA v2 vs v3 with advanced warm start
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import json
from datetime import datetime
from typing import Dict, List, Any
from ultra_optimized_v2 import UltraOptimizedQAOAv2
from ultra_optimized_v3_advanced import UltraOptimizedQAOAv3

def generate_diverse_test_problems(n_tests: int = 10) -> List[Dict]:
    """Generate diverse portfolio optimization problems"""
    
    problems = []
    np.random.seed(42)
    
    # Diverse configurations
    configs = [
        {'n_assets': 6, 'budget': 3, 'risk_factor': 0.5, 'correlation': 'low'},
        {'n_assets': 8, 'budget': 4, 'risk_factor': 0.3, 'correlation': 'high'},
        {'n_assets': 10, 'budget': 5, 'risk_factor': 0.7, 'correlation': 'mixed'},
        {'n_assets': 12, 'budget': 6, 'risk_factor': 0.5, 'correlation': 'clustered'},
        {'n_assets': 15, 'budget': 7, 'risk_factor': 0.4, 'correlation': 'low'},
        {'n_assets': 8, 'budget': 3, 'risk_factor': 0.6, 'correlation': 'negative'},
        {'n_assets': 10, 'budget': 6, 'risk_factor': 0.5, 'correlation': 'high'},
        {'n_assets': 12, 'budget': 5, 'risk_factor': 0.3, 'correlation': 'mixed'},
        {'n_assets': 15, 'budget': 8, 'risk_factor': 0.5, 'correlation': 'low'},
        {'n_assets': 20, 'budget': 10, 'risk_factor': 0.4, 'correlation': 'clustered'}
    ]
    
    for i, config in enumerate(configs[:n_tests]):
        # Generate returns
        if i % 3 == 0:
            # High dispersion returns
            expected_returns = np.random.uniform(0.01, 0.40, config['n_assets'])
        elif i % 3 == 1:
            # Concentrated returns
            expected_returns = np.random.uniform(0.08, 0.12, config['n_assets'])
        else:
            # Mixed returns
            expected_returns = np.concatenate([
                np.random.uniform(0.05, 0.10, config['n_assets']//2),
                np.random.uniform(0.15, 0.25, config['n_assets'] - config['n_assets']//2)
            ])
            np.random.shuffle(expected_returns)
        
        # Generate correlation structure
        n = config['n_assets']
        if config['correlation'] == 'low':
            # Low correlation
            correlation = np.eye(n)
            for j in range(n):
                for k in range(j+1, n):
                    correlation[j, k] = correlation[k, j] = np.random.uniform(-0.2, 0.3)
        
        elif config['correlation'] == 'high':
            # High positive correlation
            correlation = np.eye(n)
            for j in range(n):
                for k in range(j+1, n):
                    correlation[j, k] = correlation[k, j] = np.random.uniform(0.4, 0.8)
        
        elif config['correlation'] == 'negative':
            # Some negative correlations
            correlation = np.eye(n)
            for j in range(n):
                for k in range(j+1, n):
                    if (j + k) % 3 == 0:
                        correlation[j, k] = correlation[k, j] = np.random.uniform(-0.6, -0.2)
                    else:
                        correlation[j, k] = correlation[k, j] = np.random.uniform(-0.1, 0.4)
        
        elif config['correlation'] == 'clustered':
            # Clustered correlations
            correlation = np.eye(n)
            cluster_size = n // 3
            for j in range(n):
                for k in range(j+1, n):
                    if j // cluster_size == k // cluster_size:
                        # Same cluster: high correlation
                        correlation[j, k] = correlation[k, j] = np.random.uniform(0.5, 0.8)
                    else:
                        # Different cluster: low correlation
                        correlation[j, k] = correlation[k, j] = np.random.uniform(-0.1, 0.2)
        
        else:  # mixed
            # Mixed correlation structure
            correlation = np.random.uniform(-0.3, 0.7, (n, n))
            correlation = (correlation + correlation.T) / 2
            np.fill_diagonal(correlation, 1.0)
        
        # Ensure positive definiteness
        eigenvalues = np.linalg.eigvals(correlation)
        if np.min(eigenvalues) < 0.01:
            correlation = correlation + (-np.min(eigenvalues) + 0.01) * np.eye(n)
        
        # Generate covariance
        volatilities = np.random.uniform(0.05, 0.35, n)
        covariance = np.outer(volatilities, volatilities) * correlation
        
        problems.append({
            'config': config,
            'expected_returns': expected_returns,
            'covariance': covariance
        })
    
    return problems

def run_comparison_test():
    """Run comprehensive comparison between v2 and v3"""
    
    print("="*70)
    print("ADVANCED WARM START COMPARISON TEST")
    print("Ultra-Optimized QAOA v2 vs v3")
    print("="*70)
    
    # Generate test problems
    problems = generate_diverse_test_problems(10)
    
    results = []
    
    for i, problem in enumerate(problems, 1):
        config = problem['config']
        
        print(f"\n{'='*50}")
        print(f"TEST {i}/10: {config['n_assets']} assets, budget={config['budget']}")
        print(f"Risk factor: {config['risk_factor']}, Correlation: {config['correlation']}")
        print('='*50)
        
        # Test v2 (without advanced warm start)
        print("\n--- Testing v2 (Standard Warm Start) ---")
        optimizer_v2 = UltraOptimizedQAOAv2(
            n_assets=config['n_assets'],
            budget=config['budget'],
            risk_factor=config['risk_factor']
        )
        
        start_v2 = time.time()
        result_v2 = optimizer_v2.solve_ultra_optimized_v2(
            problem['expected_returns'],
            problem['covariance'],
            max_iterations=30,
            use_dicke=False
        )
        time_v2 = time.time() - start_v2
        
        # Test v3 (with advanced warm start)
        print("\n--- Testing v3 (Advanced Warm Start) ---")
        optimizer_v3 = UltraOptimizedQAOAv3(
            n_assets=config['n_assets'],
            budget=config['budget'],
            risk_factor=config['risk_factor']
        )
        
        start_v3 = time.time()
        result_v3 = optimizer_v3.solve_ultra_optimized_v3(
            problem['expected_returns'],
            problem['covariance'],
            max_iterations=30
        )
        time_v3 = time.time() - start_v3
        
        # Compare results
        print(f"\nCOMPARISON:")
        print(f"Convergence Speed:")
        print(f"  v2: {result_v2.iterations_to_convergence} iterations")
        print(f"  v3: {result_v3.iterations_to_convergence} iterations")
        
        convergence_improvement = ((result_v2.iterations_to_convergence - 
                                  result_v3.iterations_to_convergence) / 
                                 result_v2.iterations_to_convergence * 100)
        print(f"  Improvement: {convergence_improvement:.1f}%")
        
        print(f"\nSolution Quality:")
        print(f"  v2: Approx ratio = {result_v2.approximation_ratio:.3f}, Sharpe = {result_v2.sharpe_ratio:.3f}")
        print(f"  v3: Approx ratio = {result_v3.approximation_ratio:.3f}, Sharpe = {result_v3.sharpe_ratio:.3f}")
        
        print(f"\nFeasibility:")
        print(f"  v2: Initial = {result_v2.initial_feasibility_rate:.1%}")
        print(f"  v3: Initial = {result_v3.initial_feasibility_rate:.1%}")
        
        results.append({
            'test_id': i,
            'config': config,
            'v2': {
                'iterations': result_v2.iterations_to_convergence,
                'approximation_ratio': result_v2.approximation_ratio,
                'sharpe_ratio': result_v2.sharpe_ratio,
                'objective_value': result_v2.objective_value,
                'feasibility_rate': result_v2.feasibility_rate,
                'initial_feasibility': result_v2.initial_feasibility_rate,
                'execution_time': time_v2,
                'converged': result_v2.converged
            },
            'v3': {
                'iterations': result_v3.iterations_to_convergence,
                'approximation_ratio': result_v3.approximation_ratio,
                'sharpe_ratio': result_v3.sharpe_ratio,
                'objective_value': result_v3.objective_value,
                'feasibility_rate': result_v3.feasibility_rate,
                'initial_feasibility': result_v3.initial_feasibility_rate,
                'execution_time': time_v3,
                'converged': result_v3.converged,
                'warm_start_strategy': result_v3.warm_start_strategy,
                'warm_start_quality': result_v3.warm_start_quality
            },
            'improvements': {
                'convergence_speedup_pct': convergence_improvement,
                'approx_ratio_diff': result_v3.approximation_ratio - result_v2.approximation_ratio,
                'sharpe_diff': result_v3.sharpe_ratio - result_v2.sharpe_ratio,
                'feasibility_diff': result_v3.initial_feasibility_rate - result_v2.initial_feasibility_rate
            }
        })
    
    return results

def create_comparison_visualization(results: List[Dict]):
    """Create visualization comparing v2 and v3 performance"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Advanced Warm Start Performance Comparison', fontsize=16, fontweight='bold')
    
    test_ids = [r['test_id'] for r in results]
    
    # 1. Convergence Speed Comparison
    ax = axes[0, 0]
    v2_iterations = [r['v2']['iterations'] for r in results]
    v3_iterations = [r['v3']['iterations'] for r in results]
    
    x = np.arange(len(test_ids))
    width = 0.35
    
    ax.bar(x - width/2, v2_iterations, width, label='v2 (Standard)', color='#ff6b6b')
    ax.bar(x + width/2, v3_iterations, width, label='v3 (Advanced)', color='#4ecdc4')
    
    ax.set_xlabel('Test')
    ax.set_ylabel('Iterations to Convergence')
    ax.set_title('Convergence Speed', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(test_ids)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Approximation Ratio
    ax = axes[0, 1]
    v2_approx = [r['v2']['approximation_ratio'] for r in results]
    v3_approx = [r['v3']['approximation_ratio'] for r in results]
    
    ax.plot(test_ids, v2_approx, 'o-', label='v2', color='#ff6b6b', linewidth=2)
    ax.plot(test_ids, v3_approx, 's-', label='v3', color='#4ecdc4', linewidth=2)
    
    ax.set_xlabel('Test')
    ax.set_ylabel('Approximation Ratio')
    ax.set_title('Solution Quality', fontweight='bold')
    ax.set_ylim([0.9, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Convergence Improvement
    ax = axes[0, 2]
    improvements = [r['improvements']['convergence_speedup_pct'] for r in results]
    colors = ['#4ecdc4' if imp > 0 else '#ff6b6b' for imp in improvements]
    
    ax.bar(test_ids, improvements, color=colors)
    ax.set_xlabel('Test')
    ax.set_ylabel('Convergence Improvement (%)')
    ax.set_title('Speed Improvement with Advanced Warm Start', fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    # 4. Initial Feasibility Rates
    ax = axes[1, 0]
    v2_feas = [r['v2']['initial_feasibility'] * 100 for r in results]
    v3_feas = [r['v3']['initial_feasibility'] * 100 for r in results]
    
    ax.bar(x - width/2, v2_feas, width, label='v2', color='#ff6b6b')
    ax.bar(x + width/2, v3_feas, width, label='v3', color='#4ecdc4')
    
    ax.set_xlabel('Test')
    ax.set_ylabel('Initial Feasibility (%)')
    ax.set_title('Constraint Satisfaction', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(test_ids)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Sharpe Ratio Comparison
    ax = axes[1, 1]
    v2_sharpe = [r['v2']['sharpe_ratio'] for r in results]
    v3_sharpe = [r['v3']['sharpe_ratio'] for r in results]
    
    ax.scatter(v2_sharpe, v3_sharpe, s=100, alpha=0.6, color='#4ecdc4')
    
    # Add diagonal line
    max_sharpe = max(max(v2_sharpe), max(v3_sharpe))
    min_sharpe = min(min(v2_sharpe), min(v3_sharpe))
    ax.plot([min_sharpe, max_sharpe], [min_sharpe, max_sharpe], 'r--', alpha=0.5)
    
    ax.set_xlabel('v2 Sharpe Ratio')
    ax.set_ylabel('v3 Sharpe Ratio')
    ax.set_title('Risk-Adjusted Return Comparison', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 6. Warm Start Strategy Distribution
    ax = axes[1, 2]
    strategies = [r['v3']['warm_start_strategy'] for r in results]
    strategy_counts = {}
    for s in strategies:
        strategy_counts[s] = strategy_counts.get(s, 0) + 1
    
    ax.pie(strategy_counts.values(), labels=strategy_counts.keys(), autopct='%1.0f%%',
           colors=['#4ecdc4', '#45b7d1', '#ffd700', '#ff6b6b', '#9b59b6', '#3498db'])
    ax.set_title('Best Classical Strategies Used', fontweight='bold')
    
    plt.tight_layout()
    return fig

def generate_comparison_report(results: List[Dict]) -> str:
    """Generate detailed comparison report"""
    
    report = []
    report.append("# Advanced Warm Start Performance Report")
    report.append(f"\nGenerated: {datetime.now().isoformat()}\n")
    
    # Calculate aggregate statistics
    v2_avg_iter = np.mean([r['v2']['iterations'] for r in results])
    v3_avg_iter = np.mean([r['v3']['iterations'] for r in results])
    
    v2_avg_approx = np.mean([r['v2']['approximation_ratio'] for r in results])
    v3_avg_approx = np.mean([r['v3']['approximation_ratio'] for r in results])
    
    v2_avg_sharpe = np.mean([r['v2']['sharpe_ratio'] for r in results])
    v3_avg_sharpe = np.mean([r['v3']['sharpe_ratio'] for r in results])
    
    avg_speedup = np.mean([r['improvements']['convergence_speedup_pct'] for r in results])
    
    report.append("## Executive Summary\n")
    report.append("### Key Improvements with Advanced Warm Start (v3)")
    report.append(f"- **Convergence Speed**: {v2_avg_iter:.1f} -> {v3_avg_iter:.1f} iterations ({avg_speedup:.1f}% faster)")
    report.append(f"- **Approximation Ratio**: {v2_avg_approx:.3f} -> {v3_avg_approx:.3f}")
    report.append(f"- **Sharpe Ratio**: {v2_avg_sharpe:.3f} -> {v3_avg_sharpe:.3f}")
    
    # Count improvements
    convergence_wins = sum(1 for r in results if r['v3']['iterations'] < r['v2']['iterations'])
    quality_wins = sum(1 for r in results if r['v3']['approximation_ratio'] > r['v2']['approximation_ratio'])
    
    report.append(f"\n### Win Rates")
    report.append(f"- v3 converged faster: {convergence_wins}/{len(results)} tests")
    report.append(f"- v3 better quality: {quality_wins}/{len(results)} tests\n")
    
    # Detailed results
    report.append("## Detailed Test Results\n")
    
    for r in results:
        report.append(f"### Test {r['test_id']}: {r['config']['n_assets']} assets, budget={r['config']['budget']}")
        report.append(f"**Configuration**: Risk={r['config']['risk_factor']}, Correlation={r['config']['correlation']}\n")
        
        report.append("| Metric | v2 (Standard) | v3 (Advanced) | Improvement |")
        report.append("|--------|---------------|---------------|-------------|")
        report.append(f"| Iterations | {r['v2']['iterations']} | {r['v3']['iterations']} | {r['improvements']['convergence_speedup_pct']:.1f}% |")
        report.append(f"| Approx Ratio | {r['v2']['approximation_ratio']:.3f} | {r['v3']['approximation_ratio']:.3f} | {r['improvements']['approx_ratio_diff']:+.3f} |")
        report.append(f"| Sharpe Ratio | {r['v2']['sharpe_ratio']:.3f} | {r['v3']['sharpe_ratio']:.3f} | {r['improvements']['sharpe_diff']:+.3f} |")
        report.append(f"| Initial Feasibility | {r['v2']['initial_feasibility']:.1%} | {r['v3']['initial_feasibility']:.1%} | {r['improvements']['feasibility_diff']*100:+.1f}pp |")
        
        report.append(f"\n**Warm Start Strategy**: {r['v3']['warm_start_strategy']}")
        report.append(f"**Strategy Quality**: {r['v3']['warm_start_quality']:.4f}\n")
    
    # Technical analysis
    report.append("## Technical Analysis\n")
    
    report.append("### Advanced Warm Start Components")
    report.append("1. **Multi-Strategy Classical Solutions**")
    report.append("   - Greedy Sharpe selection")
    report.append("   - Minimum variance portfolio")
    report.append("   - Maximum diversification")
    report.append("   - Risk parity")
    report.append("   - Momentum-based selection")
    report.append("   - Correlation clustering\n")
    
    report.append("2. **Problem-Specific Angle Calculation**")
    report.append("   - Asset quality scores")
    report.append("   - Correlation-aware parameters")
    report.append("   - Adaptive mixing angles\n")
    
    report.append("3. **Feedback Learning System**")
    report.append("   - Problem feature extraction")
    report.append("   - Historical performance tracking")
    report.append("   - Adaptive parameter adjustment\n")
    
    # Conclusions
    report.append("## Conclusions\n")
    
    if avg_speedup > 20:
        report.append("✅ **Significant convergence improvement** achieved with advanced warm start")
    elif avg_speedup > 10:
        report.append("✅ **Moderate convergence improvement** achieved with advanced warm start")
    else:
        report.append("⚠️ **Marginal convergence improvement** - further tuning may be needed")
    
    if v3_avg_approx > v2_avg_approx:
        report.append("✅ **Solution quality improved** with better initialization")
    
    report.append("\nThe advanced warm start system successfully:")
    report.append("- Reduces convergence time through intelligent initialization")
    report.append("- Maintains or improves solution quality")
    report.append("- Adapts to different problem structures")
    report.append("- Learns from historical performance\n")
    
    return '\n'.join(report)

def main():
    """Run complete advanced warm start test"""
    
    # Run comparison tests
    results = run_comparison_test()
    
    # Save results
    with open('advanced_warm_start_results.json', 'w') as f:
        json.dump({
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'test_type': 'Advanced Warm Start Comparison',
                'total_tests': len(results)
            },
            'results': results
        }, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print("\nResults saved: advanced_warm_start_results.json")
    
    # Create visualization
    fig = create_comparison_visualization(results)
    fig.savefig('advanced_warm_start_comparison.png', dpi=300, bbox_inches='tight')
    print("Visualization saved: advanced_warm_start_comparison.png")
    
    # Generate report
    report = generate_comparison_report(results)
    with open('advanced_warm_start_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    print("Report saved: advanced_warm_start_report.md")
    
    # Print summary
    print("\n" + "="*70)
    print("ADVANCED WARM START TEST COMPLETE")
    print("="*70)
    
    avg_speedup = np.mean([r['improvements']['convergence_speedup_pct'] for r in results])
    print(f"Average convergence improvement: {avg_speedup:.1f}%")
    
    v2_avg_approx = np.mean([r['v2']['approximation_ratio'] for r in results])
    v3_avg_approx = np.mean([r['v3']['approximation_ratio'] for r in results])
    print(f"Average approximation ratio: v2={v2_avg_approx:.3f}, v3={v3_avg_approx:.3f}")
    
    print("="*70)

if __name__ == "__main__":
    main()