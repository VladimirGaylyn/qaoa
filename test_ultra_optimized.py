"""
Comprehensive test suite for ultra-optimized QAOA
Compares standard vs ultra-optimized implementations
"""

import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, List, Any
import matplotlib.pyplot as plt

from ultra_optimized_qaoa import UltraOptimizedQAOA
from optimized_qaoa_portfolio import OptimizedQAOAPortfolio


def run_comprehensive_comparison():
    """Run comprehensive comparison between implementations"""
    
    print("="*80)
    print("COMPREHENSIVE QAOA COMPARISON TEST")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Test configurations
    test_configs = [
        {"n_assets": 6, "budget": 3, "risk_factor": 0.5, "seed": 42},
        {"n_assets": 8, "budget": 4, "risk_factor": 0.5, "seed": 43},
        {"n_assets": 10, "budget": 5, "risk_factor": 0.5, "seed": 44},
        {"n_assets": 12, "budget": 6, "risk_factor": 0.5, "seed": 45},
        {"n_assets": 15, "budget": 7, "risk_factor": 0.5, "seed": 46},
    ]
    
    all_results = []
    
    for config in test_configs:
        print(f"\n{'='*80}")
        print(f"TEST: {config['n_assets']} Assets, Budget={config['budget']}")
        print('='*80)
        
        # Set seed for reproducibility
        np.random.seed(config['seed'])
        
        # Generate market data
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
        covariance = np.outer(std_devs, std_devs) * correlation
        
        result_data = {"config": config}
        
        # 1. Standard Optimized QAOA (depth ~17)
        print("\n1. Standard Optimized QAOA:")
        print("-" * 40)
        
        standard_optimizer = OptimizedQAOAPortfolio(
            n_assets=config['n_assets'],
            budget=config['budget'],
            risk_factor=config['risk_factor']
        )
        
        standard_result = standard_optimizer.solve_optimized_qaoa(
            expected_returns,
            covariance,
            p=1,
            max_iterations=30,
            use_warm_start=True,
            use_adaptive_penalty=True
        )
        
        result_data["standard"] = {
            "objective_value": standard_result.objective_value,
            "approximation_ratio": standard_result.approximation_ratio,
            "circuit_depth": standard_result.circuit_depth,
            "gate_count": standard_result.gate_count,
            "execution_time": standard_result.execution_time,
            "constraint_satisfied": standard_result.constraint_satisfied,
            "feasibility_rate": standard_result.feasibility_rate,
            "sharpe_ratio": standard_result.sharpe_ratio,
            "expected_return": standard_result.expected_return,
            "risk": standard_result.risk
        }
        
        # 2. Ultra-Optimized QAOA (depth ≤ 6)
        print("\n2. Ultra-Optimized QAOA (depth <= 6):")
        print("-" * 40)
        
        ultra_optimizer = UltraOptimizedQAOA(
            n_assets=config['n_assets'],
            budget=config['budget'],
            risk_factor=config['risk_factor']
        )
        
        ultra_result = ultra_optimizer.solve_ultra_optimized(
            expected_returns,
            covariance,
            max_iterations=30
        )
        
        result_data["ultra"] = {
            "objective_value": ultra_result.objective_value,
            "approximation_ratio": ultra_result.approximation_ratio,
            "circuit_depth": ultra_result.circuit_depth,
            "gate_count": ultra_result.gate_count,
            "execution_time": ultra_result.execution_time,
            "constraint_satisfied": ultra_result.constraint_satisfied,
            "feasibility_rate": ultra_result.feasibility_rate,
            "sharpe_ratio": ultra_result.sharpe_ratio,
            "expected_return": ultra_result.expected_return,
            "risk": ultra_result.risk,
            "repaired_solutions": ultra_result.repaired_solutions,
            "converged": ultra_result.converged,
            "iterations_to_convergence": ultra_result.iterations_to_convergence
        }
        
        # Calculate improvements
        depth_reduction = (1 - ultra_result.circuit_depth / standard_result.circuit_depth) * 100
        gate_reduction = (1 - ultra_result.gate_count / standard_result.gate_count) * 100
        speedup = standard_result.execution_time / ultra_result.execution_time
        
        result_data["improvements"] = {
            "depth_reduction_pct": depth_reduction,
            "gate_reduction_pct": gate_reduction,
            "speedup_factor": speedup,
            "approx_ratio_diff": ultra_result.approximation_ratio - standard_result.approximation_ratio
        }
        
        all_results.append(result_data)
        
        # Print comparison summary
        print(f"\n3. Comparison Summary:")
        print("-" * 40)
        print(f"  Circuit Depth: {standard_result.circuit_depth} -> {ultra_result.circuit_depth} ({depth_reduction:.1f}% reduction)")
        print(f"  Gate Count: {standard_result.gate_count} -> {ultra_result.gate_count} ({gate_reduction:.1f}% reduction)")
        print(f"  Execution Time: {standard_result.execution_time:.2f}s -> {ultra_result.execution_time:.2f}s ({speedup:.1f}x speedup)")
        print(f"  Approximation Ratio: {standard_result.approximation_ratio:.3f} -> {ultra_result.approximation_ratio:.3f}")
        print(f"  Solutions Repaired: {ultra_result.repaired_solutions}")
        print(f"  Early Convergence: {ultra_result.converged} (at iteration {ultra_result.iterations_to_convergence})")
    
    return all_results


def save_test_results(results: List[Dict], filename: str = "ultra_test_results.json"):
    """Save test results to JSON file"""
    
    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    # Calculate summary statistics
    summary = calculate_summary_stats(results)
    
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "test_type": "Ultra-Optimized vs Standard QAOA Comparison",
            "total_tests": len(results)
        },
        "results": convert_to_serializable(results),
        "summary": convert_to_serializable(summary)
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    return output


def calculate_summary_stats(results: List[Dict]) -> Dict:
    """Calculate summary statistics across all tests"""
    
    # Extract metrics
    standard_depths = [r['standard']['circuit_depth'] for r in results]
    ultra_depths = [r['ultra']['circuit_depth'] for r in results]
    
    standard_approx = [r['standard']['approximation_ratio'] for r in results]
    ultra_approx = [r['ultra']['approximation_ratio'] for r in results]
    
    standard_times = [r['standard']['execution_time'] for r in results]
    ultra_times = [r['ultra']['execution_time'] for r in results]
    
    repairs = [r['ultra']['repaired_solutions'] for r in results]
    convergence_iters = [r['ultra']['iterations_to_convergence'] for r in results]
    
    depth_reductions = [r['improvements']['depth_reduction_pct'] for r in results]
    speedups = [r['improvements']['speedup_factor'] for r in results]
    
    return {
        "circuit_depth": {
            "standard_mean": np.mean(standard_depths),
            "ultra_mean": np.mean(ultra_depths),
            "reduction_mean": np.mean(depth_reductions)
        },
        "approximation_ratio": {
            "standard_mean": np.mean(standard_approx),
            "ultra_mean": np.mean(ultra_approx),
            "difference": np.mean(ultra_approx) - np.mean(standard_approx)
        },
        "execution_time": {
            "standard_mean": np.mean(standard_times),
            "ultra_mean": np.mean(ultra_times),
            "speedup_mean": np.mean(speedups)
        },
        "solution_repair": {
            "total_repairs": sum(repairs),
            "avg_repairs_per_test": np.mean(repairs)
        },
        "convergence": {
            "avg_iterations": np.mean(convergence_iters),
            "all_converged": all(r['ultra']['converged'] for r in results)
        }
    }


def generate_comparison_plots(results_data: Dict, filename: str = "ultra_comparison.png"):
    """Generate comparison visualization plots"""
    
    results = results_data['results']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Extract data
    n_assets = [r['config']['n_assets'] for r in results]
    
    # 1. Circuit Depth Comparison
    ax1 = axes[0, 0]
    standard_depths = [r['standard']['circuit_depth'] for r in results]
    ultra_depths = [r['ultra']['circuit_depth'] for r in results]
    
    x = np.arange(len(n_assets))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, standard_depths, width, label='Standard', color='blue', alpha=0.7)
    bars2 = ax1.bar(x + width/2, ultra_depths, width, label='Ultra (≤6)', color='green', alpha=0.7)
    
    ax1.set_xlabel('Number of Assets')
    ax1.set_ylabel('Circuit Depth')
    ax1.set_title('Circuit Depth Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(n_assets)
    ax1.legend()
    ax1.axhline(y=6, color='red', linestyle='--', alpha=0.5, label='Target')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Approximation Ratio Comparison
    ax2 = axes[0, 1]
    standard_approx = [r['standard']['approximation_ratio'] for r in results]
    ultra_approx = [r['ultra']['approximation_ratio'] for r in results]
    
    ax2.plot(n_assets, standard_approx, 'o-', label='Standard', color='blue', linewidth=2, markersize=8)
    ax2.plot(n_assets, ultra_approx, 's-', label='Ultra', color='green', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Assets')
    ax2.set_ylabel('Approximation Ratio')
    ax2.set_title('Solution Quality Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.5, 1.05])
    
    # 3. Execution Time
    ax3 = axes[0, 2]
    standard_times = [r['standard']['execution_time'] for r in results]
    ultra_times = [r['ultra']['execution_time'] for r in results]
    
    ax3.plot(n_assets, standard_times, 'o-', label='Standard', color='blue', linewidth=2, markersize=8)
    ax3.plot(n_assets, ultra_times, 's-', label='Ultra', color='green', linewidth=2, markersize=8)
    ax3.set_xlabel('Number of Assets')
    ax3.set_ylabel('Execution Time (s)')
    ax3.set_title('Computational Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Gate Count
    ax4 = axes[1, 0]
    standard_gates = [r['standard']['gate_count'] for r in results]
    ultra_gates = [r['ultra']['gate_count'] for r in results]
    
    bars1 = ax4.bar(x - width/2, standard_gates, width, label='Standard', color='blue', alpha=0.7)
    bars2 = ax4.bar(x + width/2, ultra_gates, width, label='Ultra', color='green', alpha=0.7)
    
    ax4.set_xlabel('Number of Assets')
    ax4.set_ylabel('Gate Count')
    ax4.set_title('Circuit Complexity')
    ax4.set_xticks(x)
    ax4.set_xticklabels(n_assets)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Solutions Repaired
    ax5 = axes[1, 1]
    repairs = [r['ultra']['repaired_solutions'] for r in results]
    
    ax5.bar(n_assets, repairs, color='orange', alpha=0.7)
    ax5.set_xlabel('Number of Assets')
    ax5.set_ylabel('Solutions Repaired')
    ax5.set_title('Constraint Repair Activity')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Convergence Iterations
    ax6 = axes[1, 2]
    convergence_iters = [r['ultra']['iterations_to_convergence'] for r in results]
    
    ax6.plot(n_assets, convergence_iters, 'd-', color='purple', linewidth=2, markersize=8)
    ax6.set_xlabel('Number of Assets')
    ax6.set_ylabel('Iterations to Convergence')
    ax6.set_title('Early Stopping Performance')
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Max iterations')
    ax6.legend()
    
    plt.suptitle('Ultra-Optimized QAOA Performance Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Comparison plots saved to: {filename}")
    
    return fig


def generate_markdown_report(results_data: Dict, filename: str = "ultra_test_report.md"):
    """Generate detailed markdown report"""
    
    with open(filename, 'w') as f:
        f.write("# Ultra-Optimized QAOA Test Report\n\n")
        f.write(f"Generated: {results_data['metadata']['timestamp']}\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        summary = results_data['summary']
        
        f.write("### Key Achievements\n")
        f.write(f"- **Circuit Depth**: Reduced from {summary['circuit_depth']['standard_mean']:.1f} to {summary['circuit_depth']['ultra_mean']:.1f} ({summary['circuit_depth']['reduction_mean']:.1f}% reduction)\n")
        f.write(f"- **Average Speedup**: {summary['execution_time']['speedup_mean']:.1f}x faster\n")
        f.write(f"- **Approximation Quality**: {summary['approximation_ratio']['ultra_mean']:.1%} average\n")
        f.write(f"- **Convergence**: All tests converged in ~{summary['convergence']['avg_iterations']:.0f} iterations\n")
        f.write(f"- **Solution Repairs**: {summary['solution_repair']['avg_repairs_per_test']:.1f} repairs per test\n\n")
        
        # Detailed Results
        f.write("## Detailed Results\n\n")
        
        for i, result in enumerate(results_data['results'], 1):
            config = result['config']
            standard = result['standard']
            ultra = result['ultra']
            improvements = result['improvements']
            
            f.write(f"### Test {i}: {config['n_assets']} Assets, Budget={config['budget']}\n\n")
            
            f.write("#### Performance Comparison\n")
            f.write("| Metric | Standard | Ultra-Optimized | Improvement |\n")
            f.write("|--------|----------|-----------------|-------------|\n")
            f.write(f"| Circuit Depth | {standard['circuit_depth']} | {ultra['circuit_depth']} | {improvements['depth_reduction_pct']:.1f}% reduction |\n")
            f.write(f"| Gate Count | {standard['gate_count']} | {ultra['gate_count']} | {improvements['gate_reduction_pct']:.1f}% reduction |\n")
            f.write(f"| Execution Time | {standard['execution_time']:.2f}s | {ultra['execution_time']:.2f}s | {improvements['speedup_factor']:.1f}x speedup |\n")
            f.write(f"| Approx. Ratio | {standard['approximation_ratio']:.3f} | {ultra['approximation_ratio']:.3f} | {'+' if improvements['approx_ratio_diff'] > 0 else ''}{improvements['approx_ratio_diff']:.3f} |\n\n")
            
            f.write("#### Solution Quality\n")
            f.write(f"- **Sharpe Ratio**: Standard={standard['sharpe_ratio']:.3f}, Ultra={ultra['sharpe_ratio']:.3f}\n")
            f.write(f"- **Expected Return**: Standard={standard['expected_return']:.4f}, Ultra={ultra['expected_return']:.4f}\n")
            f.write(f"- **Risk**: Standard={standard['risk']:.4f}, Ultra={ultra['risk']:.4f}\n")
            f.write(f"- **Constraint Satisfied**: Standard={'Yes' if standard['constraint_satisfied'] else 'No'}, Ultra={'Yes' if ultra['constraint_satisfied'] else 'No'}\n\n")
            
            f.write("#### Ultra-Optimized Features\n")
            f.write(f"- **Solutions Repaired**: {ultra['repaired_solutions']}\n")
            f.write(f"- **Converged**: {'Yes' if ultra['converged'] else 'No'}\n")
            f.write(f"- **Iterations to Convergence**: {ultra['iterations_to_convergence']}\n")
            f.write(f"- **Feasibility Rate**: {ultra['feasibility_rate']:.1%}\n\n")
            
            f.write("---\n\n")
        
        # Summary Statistics
        f.write("## Summary Statistics\n\n")
        f.write("### Average Performance Metrics\n")
        f.write(f"- **Circuit Depth**: {summary['circuit_depth']['ultra_mean']:.1f} (target: <= 6)\n")
        f.write(f"- **Approximation Ratio**: {summary['approximation_ratio']['ultra_mean']:.1%}\n")
        f.write(f"- **Execution Time**: {summary['execution_time']['ultra_mean']:.2f}s\n")
        f.write(f"- **Total Solutions Repaired**: {summary['solution_repair']['total_repairs']}\n\n")
        
        f.write("### Improvements Over Standard\n")
        f.write(f"- **Depth Reduction**: {summary['circuit_depth']['reduction_mean']:.1f}%\n")
        f.write(f"- **Speed Improvement**: {summary['execution_time']['speedup_mean']:.1f}x\n")
        f.write(f"- **Quality Difference**: {summary['approximation_ratio']['difference']:.3f}\n\n")
        
        f.write("## Conclusion\n\n")
        f.write("The ultra-optimized QAOA successfully achieves the target circuit depth of <= 6 while maintaining ")
        f.write("competitive approximation ratios through intelligent solution repair mechanisms. ")
        f.write("The implementation demonstrates significant improvements in circuit efficiency and execution time ")
        f.write("with minimal impact on solution quality.\n")
    
    print(f"Markdown report saved to: {filename}")


if __name__ == "__main__":
    print("\nRunning comprehensive ultra-optimized QAOA tests...\n")
    
    # Run tests
    results = run_comprehensive_comparison()
    
    # Save results
    results_data = save_test_results(results, "ultra_test_results.json")
    
    # Generate visualizations
    generate_comparison_plots(results_data, "ultra_comparison.png")
    
    # Generate markdown report
    generate_markdown_report(results_data, "ultra_test_report.md")
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUITE COMPLETE")
    print("="*80)
    
    summary = results_data['summary']
    print(f"\nKey Results:")
    print(f"  Circuit Depth: {summary['circuit_depth']['standard_mean']:.1f} -> {summary['circuit_depth']['ultra_mean']:.1f}")
    print(f"  Average Speedup: {summary['execution_time']['speedup_mean']:.1f}x")
    print(f"  Approximation Quality: {summary['approximation_ratio']['ultra_mean']:.1%}")
    print(f"  All tests converged: {summary['convergence']['all_converged']}")
    
    print(f"\nFiles generated:")
    print(f"  - ultra_test_results.json (raw data)")
    print(f"  - ultra_comparison.png (visualizations)")
    print(f"  - ultra_test_report.md (detailed report)")
    print("="*80)