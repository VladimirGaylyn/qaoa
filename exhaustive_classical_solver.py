"""
Exhaustive Classical Solver for QAOA Warm Start
Enumerates all possible portfolio combinations for small problems
"""

import numpy as np
from itertools import combinations
from typing import List, Tuple, Dict
import time

class ExhaustiveClassicalSolver:
    """Complete enumeration of all portfolio combinations"""
    
    def __init__(self, n_assets: int, budget: int, risk_factor: float):
        self.n_assets = n_assets
        self.budget = budget
        self.risk_factor = risk_factor
        
    def solve(self, expected_returns: np.ndarray, 
              covariance: np.ndarray, 
              top_k: int = 5) -> Dict:
        """
        Enumerate all possible portfolios and return top solutions
        
        Args:
            expected_returns: Expected returns for each asset
            covariance: Covariance matrix
            top_k: Number of top solutions to return
            
        Returns:
            Dictionary with solutions, objectives, and timing
        """
        start_time = time.time()
        
        # Generate all possible combinations
        all_combinations = list(combinations(range(self.n_assets), self.budget))
        n_combinations = len(all_combinations)
        
        print(f"  Evaluating {n_combinations} possible portfolios (C({self.n_assets},{self.budget}))")
        
        # Evaluate each combination
        results = []
        for combo in all_combinations:
            # Create binary solution vector
            solution = np.zeros(self.n_assets)
            solution[list(combo)] = 1
            
            # Calculate objective
            weights = solution / self.budget
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(covariance, weights))
            objective = portfolio_return - self.risk_factor * portfolio_variance
            
            results.append((objective, solution, combo))
        
        # Sort by objective (descending)
        results.sort(key=lambda x: x[0], reverse=True)
        
        # Extract top solutions
        top_solutions = []
        top_objectives = []
        top_combinations = []
        
        for i in range(min(top_k, len(results))):
            obj, sol, combo = results[i]
            top_solutions.append(sol)
            top_objectives.append(obj)
            top_combinations.append(combo)
        
        # Calculate solution distribution
        solution_distribution = {}
        for obj, sol, combo in results[:100]:  # Top 100 for distribution
            sol_str = ''.join(str(int(x)) for x in sol)
            if sol_str not in solution_distribution:
                solution_distribution[sol_str] = 0
            # Normalize objectives to probabilities (softmax-like)
            prob = np.exp((obj - results[0][0]) * 10)  # Scale factor for differentiation
            solution_distribution[sol_str] = prob
        
        # Normalize distribution
        total_prob = sum(solution_distribution.values())
        if total_prob > 0:
            for key in solution_distribution:
                solution_distribution[key] /= total_prob
        
        elapsed_time = time.time() - start_time
        
        return {
            'best_solution': top_solutions[0],
            'best_objective': top_objectives[0],
            'top_solutions': top_solutions,
            'top_objectives': top_objectives,
            'top_combinations': top_combinations,
            'solution_distribution': solution_distribution,
            'n_evaluated': n_combinations,
            'computation_time': elapsed_time,
            'all_objectives': [r[0] for r in results]  # For analysis
        }
    
    def get_warm_start_parameters(self, top_solutions: List[np.ndarray],
                                  top_objectives: List[float],
                                  p: int) -> np.ndarray:
        """
        Convert top classical solutions to QAOA warm start parameters
        
        Args:
            top_solutions: List of top binary solutions
            top_objectives: Corresponding objective values
            p: QAOA circuit depth
            
        Returns:
            Initial parameter vector for QAOA
        """
        # Use weighted average of solution characteristics
        n_params = 2 * p
        params = np.zeros(n_params)
        
        # Weight solutions by their relative objectives
        weights = np.array(top_objectives)
        if len(weights) > 1:
            weights = weights - np.min(weights)
            weights = weights / (np.max(weights) + 1e-10)
        else:
            weights = np.array([1.0])
        
        # Extract features from solutions
        for i, (sol, weight) in enumerate(zip(top_solutions, weights)):
            # Use solution structure to inform parameters
            # Higher objective -> smaller beta (less mixing)
            # More concentrated solutions -> larger gamma
            
            # Calculate solution concentration (how clustered selected assets are)
            selected_indices = np.where(sol > 0)[0]
            if len(selected_indices) > 1:
                concentration = 1.0 / (np.std(selected_indices) / self.n_assets + 0.1)
            else:
                concentration = 1.0
            
            # Set parameters based on solution quality and structure
            for layer in range(p):
                # Gamma parameters (cost Hamiltonian)
                gamma_idx = 2 * layer
                params[gamma_idx] += weight * concentration * np.pi / 4
                
                # Beta parameters (mixing Hamiltonian)
                beta_idx = 2 * layer + 1
                # Less mixing for better solutions
                params[beta_idx] += weight * (1 - weight * 0.5) * np.pi / 8
        
        # Normalize by number of solutions considered
        params /= len(top_solutions)
        
        # Add progressive structure (deeper layers get smaller angles)
        for layer in range(p):
            decay_factor = np.exp(-layer * 0.3)
            params[2 * layer] *= decay_factor  # Gamma
            params[2 * layer + 1] *= decay_factor  # Beta
        
        return params
    
    def analyze_solution_landscape(self, expected_returns: np.ndarray,
                                  covariance: np.ndarray) -> Dict:
        """
        Analyze the solution landscape for insights
        
        Returns:
            Dictionary with landscape statistics
        """
        result = self.solve(expected_returns, covariance, top_k=10)
        
        all_objectives = result['all_objectives']
        
        # Calculate statistics
        obj_mean = np.mean(all_objectives)
        obj_std = np.std(all_objectives)
        obj_min = np.min(all_objectives)
        obj_max = np.max(all_objectives)
        
        # Gap analysis
        gap_to_second = all_objectives[0] - all_objectives[1] if len(all_objectives) > 1 else 0
        gap_to_median = all_objectives[0] - np.median(all_objectives)
        
        # Solution diversity in top 10
        top_10 = result['top_solutions'][:10]
        diversity_scores = []
        for i in range(len(top_10)):
            for j in range(i + 1, len(top_10)):
                # Hamming distance
                distance = np.sum(np.abs(top_10[i] - top_10[j]))
                diversity_scores.append(distance)
        
        avg_diversity = np.mean(diversity_scores) if diversity_scores else 0
        
        return {
            'objective_stats': {
                'mean': obj_mean,
                'std': obj_std,
                'min': obj_min,
                'max': obj_max,
                'range': obj_max - obj_min
            },
            'gap_analysis': {
                'gap_to_second': gap_to_second,
                'gap_to_median': gap_to_median,
                'relative_gap': gap_to_second / (obj_max - obj_min + 1e-10)
            },
            'solution_diversity': {
                'avg_hamming_distance': avg_diversity,
                'max_distance': self.budget * 2  # Maximum possible Hamming distance
            },
            'landscape_ruggedness': obj_std / (obj_max - obj_min + 1e-10)
        }