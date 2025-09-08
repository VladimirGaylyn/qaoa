"""
Run comprehensive test and save results to file
"""

import numpy as np
import json
import time
from datetime import datetime
from optimized_qaoa_portfolio import OptimizedQAOAPortfolio
# from qaoa_reporting import QAOAReporter  # Not needed for this test
import os


def run_comprehensive_test():
    """Run comprehensive test across multiple portfolio sizes"""
    
    print("="*70)
    print("COMPREHENSIVE QAOA TEST - FINAL VALIDATION")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Test configurations
    test_configs = [
        {"n_assets": 8, "budget": 4, "risk_factor": 0.5, "seed": 42},
        {"n_assets": 10, "budget": 5, "risk_factor": 0.5, "seed": 43},
        {"n_assets": 12, "budget": 6, "risk_factor": 0.5, "seed": 44},
        {"n_assets": 15, "budget": 7, "risk_factor": 0.5, "seed": 45},
    ]
    
    all_results = []
    
    for config in test_configs:
        print(f"\n{'='*70}")
        print(f"TEST: {config['n_assets']} Assets, Budget={config['budget']}")
        print('='*70)
        
        # Set seed for reproducibility
        np.random.seed(config['seed'])
        
        # Initialize optimizer
        optimizer = OptimizedQAOAPortfolio(
            n_assets=config['n_assets'],
            budget=config['budget'],
            risk_factor=config['risk_factor']
        )
        
        # Generate market data
        expected_returns = np.random.uniform(0.05, 0.25, config['n_assets'])
        covariance = optimizer.generate_valid_covariance_matrix(config['n_assets'])
        
        # Get classical baseline
        print("\n1. Classical Baseline:")
        classical_start = time.time()
        classical_value = optimizer.solve_classical_exact(expected_returns, covariance)
        classical_solution = optimizer.solve_classical_quick(expected_returns, covariance)
        classical_time = time.time() - classical_start
        
        # Calculate classical metrics
        if np.sum(classical_solution) > 0:
            weights = classical_solution / config['budget']
            classical_return = np.dot(weights, expected_returns)
            classical_risk = np.sqrt(np.dot(weights, np.dot(covariance, weights)))
            classical_sharpe = classical_return / classical_risk if classical_risk > 0 else 0
        else:
            classical_return = 0
            classical_risk = 0
            classical_sharpe = 0
        
        print(f"   Objective: {classical_value:.6f}")
        print(f"   Sharpe Ratio: {classical_sharpe:.3f}")
        print(f"   Time: {classical_time:.3f}s")
        
        # Run QAOA
        print("\n2. Optimized QAOA:")
        
        # Determine parameters based on size
        if config['n_assets'] <= 10:
            p = 2
            max_iterations = 40
        else:
            p = 1
            max_iterations = 30
        
        qaoa_result = optimizer.solve_optimized_qaoa(
            expected_returns,
            covariance,
            p=p,
            max_iterations=max_iterations,
            use_warm_start=True,
            use_adaptive_penalty=True,
            use_adaptive_sampling=False
        )
        
        # Compile results
        result_data = {
            "config": config,
            "classical": {
                "objective_value": classical_value,
                "expected_return": classical_return,
                "risk": classical_risk,
                "sharpe_ratio": classical_sharpe,
                "solution": classical_solution.tolist(),
                "execution_time": classical_time
            },
            "qaoa": {
                "objective_value": qaoa_result.objective_value,
                "expected_return": qaoa_result.expected_return,
                "risk": qaoa_result.risk,
                "sharpe_ratio": qaoa_result.sharpe_ratio,
                "constraint_satisfied": qaoa_result.constraint_satisfied,
                "n_selected": qaoa_result.n_selected,
                "solution": qaoa_result.solution.tolist(),
                "execution_time": qaoa_result.execution_time,
                "circuit_depth": qaoa_result.circuit_depth,
                "gate_count": qaoa_result.gate_count,
                "approximation_ratio": qaoa_result.approximation_ratio,
                "feasibility_rate": qaoa_result.feasibility_rate,
                "convergence_iterations": len(qaoa_result.convergence_history)
            },
            "performance_metrics": {
                "speedup": classical_time / qaoa_result.execution_time if qaoa_result.execution_time > 0 else 0,
                "circuit_efficiency": qaoa_result.gate_count / config['n_assets'],
                "constraint_success": qaoa_result.constraint_satisfied,
                "quality_ratio": qaoa_result.approximation_ratio
            }
        }
        
        all_results.append(result_data)
        
        # Print summary
        print(f"\n3. Summary:")
        print(f"   Approximation Ratio: {qaoa_result.approximation_ratio:.1%}")
        print(f"   Circuit Depth: {qaoa_result.circuit_depth}")
        print(f"   Feasibility Rate: {qaoa_result.feasibility_rate:.1%}")
        print(f"   Performance vs Classical: {qaoa_result.approximation_ratio:.1%}")
    
    return all_results


def save_results(results, filename="test_results.json"):
    """Save test results to JSON file"""
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    # Add metadata
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(results),
            "test_type": "Comprehensive QAOA Validation",
            "implementation": "Optimized QAOA with Hardware-Efficient Ansatz"
        },
        "results": convert_to_serializable(results),
        "summary": convert_to_serializable(calculate_summary(results))
    }
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    
    return output


def calculate_summary(results):
    """Calculate summary statistics across all tests"""
    
    approximation_ratios = [r['qaoa']['approximation_ratio'] for r in results]
    execution_times = [r['qaoa']['execution_time'] for r in results]
    circuit_depths = [r['qaoa']['circuit_depth'] for r in results]
    gate_counts = [r['qaoa']['gate_count'] for r in results]
    feasibility_rates = [r['qaoa']['feasibility_rate'] for r in results]
    sharpe_ratios = [r['qaoa']['sharpe_ratio'] for r in results]
    constraint_satisfied = [r['qaoa']['constraint_satisfied'] for r in results]
    
    summary = {
        "approximation_ratio": {
            "mean": np.mean(approximation_ratios),
            "std": np.std(approximation_ratios),
            "min": np.min(approximation_ratios),
            "max": np.max(approximation_ratios)
        },
        "execution_time": {
            "mean": np.mean(execution_times),
            "std": np.std(execution_times),
            "min": np.min(execution_times),
            "max": np.max(execution_times)
        },
        "circuit_depth": {
            "mean": np.mean(circuit_depths),
            "std": np.std(circuit_depths),
            "min": np.min(circuit_depths),
            "max": np.max(circuit_depths)
        },
        "gate_count": {
            "mean": np.mean(gate_counts),
            "std": np.std(gate_counts),
            "min": np.min(gate_counts),
            "max": np.max(gate_counts)
        },
        "feasibility_rate": {
            "mean": np.mean(feasibility_rates),
            "std": np.std(feasibility_rates),
            "min": np.min(feasibility_rates),
            "max": np.max(feasibility_rates)
        },
        "sharpe_ratio": {
            "mean": np.mean(sharpe_ratios),
            "std": np.std(sharpe_ratios),
            "min": np.min(sharpe_ratios),
            "max": np.max(sharpe_ratios)
        },
        "constraint_satisfaction_rate": sum(constraint_satisfied) / len(constraint_satisfied)
    }
    
    return summary


def generate_markdown_report(results_data, filename="test_report.md"):
    """Generate a markdown report from test results"""
    
    with open(filename, 'w') as f:
        f.write("# QAOA Portfolio Optimization - Test Results\n\n")
        f.write(f"Generated: {results_data['metadata']['timestamp']}\n\n")
        
        f.write("## Executive Summary\n\n")
        summary = results_data['summary']
        f.write(f"- **Average Approximation Ratio**: {summary['approximation_ratio']['mean']:.1%}\n")
        f.write(f"- **Average Circuit Depth**: {summary['circuit_depth']['mean']:.1f}\n")
        f.write(f"- **Average Execution Time**: {summary['execution_time']['mean']:.2f}s\n")
        f.write(f"- **Constraint Satisfaction Rate**: {summary['constraint_satisfaction_rate']:.0%}\n")
        f.write(f"- **Average Sharpe Ratio**: {summary['sharpe_ratio']['mean']:.3f}\n\n")
        
        f.write("## Detailed Results\n\n")
        
        for i, result in enumerate(results_data['results'], 1):
            config = result['config']
            qaoa = result['qaoa']
            classical = result['classical']
            
            f.write(f"### Test {i}: {config['n_assets']} Assets, Budget={config['budget']}\n\n")
            
            f.write("#### Configuration\n")
            f.write(f"- Assets: {config['n_assets']}\n")
            f.write(f"- Budget: {config['budget']}\n")
            f.write(f"- Risk Factor: {config['risk_factor']}\n\n")
            
            f.write("#### Performance Metrics\n")
            f.write(f"- **Approximation Ratio**: {qaoa['approximation_ratio']:.1%}\n")
            f.write(f"- **Circuit Depth**: {qaoa['circuit_depth']}\n")
            f.write(f"- **Gate Count**: {qaoa['gate_count']}\n")
            f.write(f"- **Execution Time**: {qaoa['execution_time']:.2f}s\n")
            f.write(f"- **Feasibility Rate**: {qaoa['feasibility_rate']:.1%}\n\n")
            
            f.write("#### Financial Metrics\n")
            f.write(f"- **QAOA Sharpe Ratio**: {qaoa['sharpe_ratio']:.3f}\n")
            f.write(f"- **Classical Sharpe Ratio**: {classical['sharpe_ratio']:.3f}\n")
            f.write(f"- **Expected Return**: {qaoa['expected_return']:.4f}\n")
            f.write(f"- **Risk**: {qaoa['risk']:.4f}\n\n")
            
            f.write("#### Solution Quality\n")
            f.write(f"- **Constraint Satisfied**: {'Yes' if qaoa['constraint_satisfied'] else 'No'}\n")
            f.write(f"- **Assets Selected**: {qaoa['n_selected']}\n")
            f.write(f"- **Selected Indices**: {[i for i, x in enumerate(qaoa['solution']) if x == 1]}\n\n")
            
            f.write("---\n\n")
        
        f.write("## Summary Statistics\n\n")
        f.write("| Metric | Mean | Std | Min | Max |\n")
        f.write("|--------|------|-----|-----|-----|\n")
        
        for metric, values in summary.items():
            if isinstance(values, dict):
                f.write(f"| {metric.replace('_', ' ').title()} | "
                       f"{values['mean']:.3f} | "
                       f"{values['std']:.3f} | "
                       f"{values['min']:.3f} | "
                       f"{values['max']:.3f} |\n")
    
    print(f"Markdown report saved to: {filename}")


if __name__ == "__main__":
    # Run comprehensive test
    print("\nRunning comprehensive test suite...\n")
    results = run_comprehensive_test()
    
    # Save results to JSON
    results_data = save_results(results, "test_results.json")
    
    # Generate markdown report
    generate_markdown_report(results_data, "test_report.md")
    
    # Print final summary
    print("\n" + "="*70)
    print("TEST SUITE COMPLETE")
    print("="*70)
    
    summary = results_data['summary']
    print(f"\nOverall Performance:")
    print(f"  Average Approximation Ratio: {summary['approximation_ratio']['mean']:.1%}")
    print(f"  Average Circuit Depth: {summary['circuit_depth']['mean']:.1f}")
    print(f"  Average Execution Time: {summary['execution_time']['mean']:.2f}s")
    print(f"  Constraint Satisfaction: {summary['constraint_satisfaction_rate']:.0%}")
    
    print(f"\nResults saved to:")
    print(f"  - test_results.json (raw data)")
    print(f"  - test_report.md (formatted report)")
    print("="*70)