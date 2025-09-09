"""
Test Suite for Enhanced QAOA V4
Runs 20 experiments with 15 assets (selecting 4) and stores results
"""

import numpy as np
import json
import os
from datetime import datetime
import time
from typing import Dict, List
import yfinance as yf
import pandas as pd

from enhanced_qaoa_v4 import EnhancedQAOAV4
from exhaustive_classical_solver import ExhaustiveClassicalSolver

def generate_market_data(n_assets: int, scenario: str = 'balanced') -> tuple:
    """Generate realistic market data for testing"""
    
    np.random.seed(int(time.time() * 1000) % 2**32)  # Different seed each time
    
    if scenario == 'bull':
        # Bull market - higher returns, lower volatility
        mean_return = 0.15
        return_std = 0.08
        correlation_factor = 0.3
    elif scenario == 'bear':
        # Bear market - lower returns, higher volatility
        mean_return = 0.02
        return_std = 0.25
        correlation_factor = 0.6
    elif scenario == 'volatile':
        # High volatility market
        mean_return = 0.08
        return_std = 0.35
        correlation_factor = 0.4
    else:  # balanced
        # Balanced market
        mean_return = 0.08
        return_std = 0.15
        correlation_factor = 0.4
    
    # Generate expected returns
    expected_returns = np.random.normal(mean_return, return_std, n_assets)
    expected_returns = np.clip(expected_returns, -0.5, 0.5)  # Reasonable bounds
    
    # Generate correlation matrix
    correlation = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            corr = np.random.uniform(-correlation_factor, correlation_factor)
            correlation[i, j] = corr
            correlation[j, i] = corr
    
    # Generate standard deviations
    std_devs = np.random.uniform(0.1, 0.4, n_assets)
    
    # Calculate covariance matrix
    covariance = np.outer(std_devs, std_devs) * correlation
    
    # Ensure positive definiteness
    min_eigenvalue = np.min(np.linalg.eigvals(covariance))
    if min_eigenvalue < 0:
        covariance += np.eye(n_assets) * (-min_eigenvalue + 0.01)
    
    return expected_returns, covariance

def run_single_experiment(experiment_id: int, n_assets: int = 15, budget: int = 4) -> Dict:
    """Run a single experiment with Enhanced QAOA V4"""
    
    print(f"\n{'='*60}")
    print(f"Experiment {experiment_id}")
    print(f"{'='*60}")
    
    # Generate market scenario
    scenarios = ['balanced', 'bull', 'bear', 'volatile']
    scenario = scenarios[experiment_id % len(scenarios)]
    
    print(f"Market Scenario: {scenario}")
    print(f"Assets: {n_assets}, Portfolio Size: {budget}")
    
    # Generate market data
    expected_returns, covariance = generate_market_data(n_assets, scenario)
    
    # Risk factors to test
    risk_factors = [0.3, 0.5, 0.7]
    risk_factor = risk_factors[experiment_id % len(risk_factors)]
    
    print(f"Risk Factor: {risk_factor}")
    
    # Initialize enhanced QAOA
    qaoa = EnhancedQAOAV4(n_assets, budget, risk_factor)
    
    # Run optimization
    try:
        result = qaoa.optimize(expected_returns, covariance)
        
        # Extract key metrics
        experiment_result = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'scenario': scenario,
            'n_assets': n_assets,
            'budget': budget,
            'risk_factor': risk_factor,
            'best_objective': result['best_objective'],
            'best_solution_probability': result['best_solution_probability'],
            'feasibility_rate': result['feasibility_rate'],
            'approximation_ratio': result['approximation_ratio'],
            'classical_best': result['classical_best'],
            'top_10_solutions': result['top_10_solutions'],
            'computation_time': result['computation_time'],
            'circuit_depth': result['circuit_depth'],
            'n_parameters': result['n_parameters'],
            'optimal_parameters': result['optimal_parameters'],
            'n_feasible_solutions': result['n_feasible_solutions']
        }
        
        # Print summary
        print(f"\nResults Summary:")
        print(f"  Best Objective: {result['best_objective']:.6f}")
        print(f"  Best Solution Probability: {result['best_solution_probability']*100:.3f}%")
        print(f"  Feasibility Rate: {result['feasibility_rate']*100:.2f}%")
        print(f"  Approximation Ratio: {result['approximation_ratio']:.4f}")
        print(f"  Circuit Depth: {result['circuit_depth']}")
        print(f"  Computation Time: {result['computation_time']:.2f}s")
        
        # Probability distribution for top 10
        if result['top_10_solutions']:
            print(f"\nTop 10 Solutions Distribution:")
            for sol in result['top_10_solutions'][:5]:  # Show top 5
                print(f"  Rank {sol['rank']}: {sol['state']} -> "
                      f"Obj={sol['objective']:.4f}, Prob={sol['probability']*100:.3f}%")
        
        return experiment_result
        
    except Exception as e:
        print(f"Error in experiment {experiment_id}: {str(e)}")
        return {
            'experiment_id': experiment_id,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def run_all_experiments(n_experiments: int = 20):
    """Run all experiments and save results"""
    
    print("\n" + "="*80)
    print("Enhanced QAOA V4 - Test Suite")
    print("Target: >1% Best Solution Probability")
    print("Configuration: 15 Assets, Select 4")
    print(f"Running {n_experiments} experiments")
    print("="*80)
    
    # Create results directory
    results_dir = "20_by_15_v4"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Run experiments
    all_results = []
    successful_experiments = 0
    
    for i in range(1, n_experiments + 1):
        try:
            result = run_single_experiment(i, n_assets=15, budget=4)
            all_results.append(result)
            
            if 'error' not in result:
                successful_experiments += 1
                
                # Save individual result
                result_file = os.path.join(results_dir, f"experiment_{i:02d}.json")
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
            
        except Exception as e:
            print(f"Failed to complete experiment {i}: {str(e)}")
            all_results.append({
                'experiment_id': i,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
    
    # Calculate aggregate statistics
    if successful_experiments > 0:
        valid_results = [r for r in all_results if 'error' not in r]
        
        avg_best_prob = np.mean([r['best_solution_probability'] for r in valid_results])
        avg_feasibility = np.mean([r['feasibility_rate'] for r in valid_results])
        avg_ar = np.mean([r['approximation_ratio'] for r in valid_results])
        avg_time = np.mean([r['computation_time'] for r in valid_results])
        
        # Count experiments achieving >1% best solution probability
        achieving_target = sum(1 for r in valid_results 
                              if r['best_solution_probability'] > 0.01)
        
        summary = {
            'total_experiments': n_experiments,
            'successful_experiments': successful_experiments,
            'experiments_achieving_target': achieving_target,
            'target_achievement_rate': achieving_target / successful_experiments,
            'average_best_solution_probability': avg_best_prob,
            'average_feasibility_rate': avg_feasibility,
            'average_approximation_ratio': avg_ar,
            'average_computation_time': avg_time,
            'best_solution_probabilities': [r['best_solution_probability'] for r in valid_results],
            'feasibility_rates': [r['feasibility_rate'] for r in valid_results],
            'approximation_ratios': [r['approximation_ratio'] for r in valid_results],
            'timestamp': datetime.now().isoformat()
        }
        
        # Print final summary
        print("\n" + "="*80)
        print("FINAL SUMMARY - Enhanced QAOA V4")
        print("="*80)
        print(f"Total Experiments: {n_experiments}")
        print(f"Successful: {successful_experiments}")
        print(f"Achieving >1% Target: {achieving_target} ({achieving_target/successful_experiments*100:.1f}%)")
        print(f"\nAverage Metrics:")
        print(f"  Best Solution Probability: {avg_best_prob*100:.3f}%")
        print(f"  Feasibility Rate: {avg_feasibility*100:.2f}%")
        print(f"  Approximation Ratio: {avg_ar:.4f}")
        print(f"  Computation Time: {avg_time:.2f}s")
        
        if avg_best_prob > 0.01:
            print(f"\n[SUCCESS] Target of >1% best solution probability ACHIEVED!")
            print(f"  Average: {avg_best_prob*100:.3f}%")
            print(f"  Improvement: {avg_best_prob/0.01:.1f}x target")
        else:
            print(f"\n[INFO] Average best solution probability: {avg_best_prob*100:.3f}%")
            print(f"  Still below 1% target, but {achieving_target} experiments achieved it")
        
        # Save summary
        summary_file = os.path.join(results_dir, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save all results
        all_results_file = os.path.join(results_dir, "all_results.json")
        with open(all_results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nResults saved to '{results_dir}' directory")
        
    else:
        print("\nNo successful experiments completed")

if __name__ == "__main__":
    # Run the test suite
    run_all_experiments(n_experiments=20)