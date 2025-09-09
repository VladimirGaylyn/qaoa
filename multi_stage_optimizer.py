"""
Multi-Stage Optimization Strategy for QAOA
Implements progressive refinement with 400 total evaluations
"""

import numpy as np
from scipy.optimize import minimize
from typing import Callable, Dict, List, Tuple, Optional
import time

class MultiStageOptimizer:
    """Multi-stage optimization with progressive refinement"""
    
    def __init__(self, total_evaluations: int = 400):
        self.total_evaluations = total_evaluations
        self.evaluation_count = 0
        self.stage_results = []
        self.all_evaluations = []
        
    def optimize(self, objective_function: Callable,
                initial_params: np.ndarray,
                p: int,
                bounds: Optional[List[Tuple[float, float]]] = None) -> Dict:
        """
        Run multi-stage optimization
        
        Args:
            objective_function: Function to minimize (returns negative for maximization)
            initial_params: Initial parameter vector
            p: QAOA circuit depth
            bounds: Parameter bounds
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        if bounds is None:
            bounds = [(-np.pi, np.pi)] * len(initial_params)
        
        # Reset counters
        self.evaluation_count = 0
        self.stage_results = []
        self.all_evaluations = []
        
        # Wrap objective to track evaluations
        def tracked_objective(params):
            self.evaluation_count += 1
            value = objective_function(params)
            self.all_evaluations.append({
                'iteration': self.evaluation_count,
                'params': params.copy(),
                'value': value
            })
            return value
        
        # Stage 1: Global exploration (30% of budget)
        stage1_evals = int(0.3 * self.total_evaluations)
        print(f"  Stage 1: Global exploration ({stage1_evals} evaluations)")
        
        stage1_result = self._stage1_global_exploration(
            tracked_objective, initial_params, bounds, stage1_evals
        )
        self.stage_results.append(stage1_result)
        
        # Stage 2: Local refinement (40% of budget)
        stage2_evals = int(0.4 * self.total_evaluations)
        print(f"  Stage 2: Local refinement ({stage2_evals} evaluations)")
        
        stage2_result = self._stage2_local_refinement(
            tracked_objective, stage1_result['best_params'], bounds, stage2_evals
        )
        self.stage_results.append(stage2_result)
        
        # Stage 3: Fine-tuning (30% of budget)
        stage3_evals = self.total_evaluations - self.evaluation_count
        print(f"  Stage 3: Fine-tuning ({stage3_evals} evaluations)")
        
        stage3_result = self._stage3_fine_tuning(
            tracked_objective, stage2_result['best_params'], bounds, stage3_evals
        )
        self.stage_results.append(stage3_result)
        
        # Find overall best
        best_eval = min(self.all_evaluations, key=lambda x: x['value'])
        
        elapsed_time = time.time() - start_time
        
        return {
            'best_params': best_eval['params'],
            'best_value': best_eval['value'],
            'n_evaluations': self.evaluation_count,
            'convergence_history': self._get_convergence_history(),
            'stage_results': self.stage_results,
            'computation_time': elapsed_time,
            'improvement_per_stage': self._calculate_stage_improvements()
        }
    
    def _stage1_global_exploration(self, objective: Callable,
                                  initial_params: np.ndarray,
                                  bounds: List[Tuple[float, float]],
                                  max_evals: int) -> Dict:
        """
        Stage 1: Global exploration with multiple random starts
        """
        stage_start = self.evaluation_count
        best_value = float('inf')
        best_params = initial_params.copy()
        
        # Number of random starts
        n_starts = min(5, max(2, max_evals // 20))
        evals_per_start = max_evals // n_starts
        
        for i in range(n_starts):
            # Generate random start point
            if i == 0:
                start_point = initial_params
            else:
                start_point = np.random.uniform(
                    [b[0] for b in bounds],
                    [b[1] for b in bounds]
                )
            
            # Run COBYLA with limited iterations
            remaining_evals = max_evals - (self.evaluation_count - stage_start)
            if remaining_evals <= 0:
                break
            
            result = minimize(
                objective,
                start_point,
                method='COBYLA',
                options={
                    'maxiter': min(evals_per_start, remaining_evals),
                    'rhobeg': 0.5,  # Larger initial step for exploration
                    'tol': 1e-3
                }
            )
            
            if result.fun < best_value:
                best_value = result.fun
                best_params = result.x.copy()
        
        return {
            'best_params': best_params,
            'best_value': best_value,
            'n_starts': n_starts,
            'evaluations_used': self.evaluation_count - stage_start
        }
    
    def _stage2_local_refinement(self, objective: Callable,
                                initial_params: np.ndarray,
                                bounds: List[Tuple[float, float]],
                                max_evals: int) -> Dict:
        """
        Stage 2: Local refinement with adaptive step sizes
        """
        stage_start = self.evaluation_count
        
        # Use L-BFGS-B for better local convergence
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': max_evals,
                'maxfun': max_evals,
                'ftol': 1e-6,
                'gtol': 1e-6
            }
        )
        
        # If evaluations remain, try Nelder-Mead for robustness
        remaining_evals = max_evals - (self.evaluation_count - stage_start)
        if remaining_evals > 10:
            result2 = minimize(
                objective,
                result.x,
                method='Nelder-Mead',
                options={
                    'maxiter': remaining_evals,
                    'xatol': 1e-6,
                    'fatol': 1e-8
                }
            )
            if result2.fun < result.fun:
                result = result2
        
        return {
            'best_params': result.x.copy(),
            'best_value': result.fun,
            'method': 'L-BFGS-B + Nelder-Mead',
            'evaluations_used': self.evaluation_count - stage_start
        }
    
    def _stage3_fine_tuning(self, objective: Callable,
                           initial_params: np.ndarray,
                           bounds: List[Tuple[float, float]],
                           max_evals: int) -> Dict:
        """
        Stage 3: Fine-tuning with small perturbations
        """
        stage_start = self.evaluation_count
        best_params = initial_params.copy()
        best_value = objective(best_params)
        
        # Adaptive perturbation search
        n_params = len(initial_params)
        perturbation_scale = 0.1
        
        while self.evaluation_count - stage_start < max_evals:
            # Try small perturbations in each direction
            for i in range(n_params):
                if self.evaluation_count - stage_start >= max_evals:
                    break
                
                # Positive perturbation
                perturbed = best_params.copy()
                perturbed[i] += perturbation_scale
                perturbed[i] = np.clip(perturbed[i], bounds[i][0], bounds[i][1])
                
                value = objective(perturbed)
                if value < best_value:
                    best_value = value
                    best_params = perturbed.copy()
                    perturbation_scale *= 1.1  # Increase step if improving
                
                if self.evaluation_count - stage_start >= max_evals:
                    break
                
                # Negative perturbation
                perturbed = best_params.copy()
                perturbed[i] -= perturbation_scale
                perturbed[i] = np.clip(perturbed[i], bounds[i][0], bounds[i][1])
                
                value = objective(perturbed)
                if value < best_value:
                    best_value = value
                    best_params = perturbed.copy()
                    perturbation_scale *= 1.1
            
            # Reduce perturbation scale if no improvement
            perturbation_scale *= 0.95
            
            # Try random perturbation occasionally
            if np.random.random() < 0.1:
                perturbed = best_params + np.random.normal(0, perturbation_scale, n_params)
                for i in range(n_params):
                    perturbed[i] = np.clip(perturbed[i], bounds[i][0], bounds[i][1])
                
                value = objective(perturbed)
                if value < best_value:
                    best_value = value
                    best_params = perturbed.copy()
        
        return {
            'best_params': best_params,
            'best_value': best_value,
            'final_perturbation_scale': perturbation_scale,
            'evaluations_used': self.evaluation_count - stage_start
        }
    
    def _get_convergence_history(self) -> List[float]:
        """Get best value at each iteration"""
        history = []
        current_best = float('inf')
        
        for eval_data in self.all_evaluations:
            if eval_data['value'] < current_best:
                current_best = eval_data['value']
            history.append(current_best)
        
        return history
    
    def _calculate_stage_improvements(self) -> Dict:
        """Calculate improvement achieved in each stage"""
        improvements = {}
        
        for i, stage in enumerate(self.stage_results):
            if i == 0:
                # First stage improvement from initial
                if self.all_evaluations:
                    initial_value = self.all_evaluations[0]['value']
                    improvement = initial_value - stage['best_value']
                else:
                    improvement = 0
            else:
                # Improvement from previous stage
                improvement = self.stage_results[i-1]['best_value'] - stage['best_value']
            
            improvements[f'stage_{i+1}'] = {
                'absolute_improvement': improvement,
                'relative_improvement': improvement / (abs(self.stage_results[i-1]['best_value']) + 1e-10) if i > 0 else 0,
                'final_value': stage['best_value']
            }
        
        return improvements
    
    def get_parameter_evolution(self) -> np.ndarray:
        """Get evolution of parameters through optimization"""
        if not self.all_evaluations:
            return np.array([])
        
        params_history = np.array([eval_data['params'] for eval_data in self.all_evaluations])
        return params_history