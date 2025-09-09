"""
Warm Start Feedback System for Adaptive Parameter Learning
Tracks performance history and adapts warm start parameters based on problem similarity
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import os

class WarmStartFeedback:
    """Track and improve warm start performance through learning"""
    
    def __init__(self, memory_size: int = 100, save_path: str = "warm_start_history.json"):
        self.memory_size = memory_size
        self.save_path = save_path
        self.performance_history = []
        self.parameter_history = []
        self.problem_features_history = []
        
        # Load existing history if available
        self.load_history()
        
    def extract_problem_features(self, expected_returns: np.ndarray,
                                covariance: np.ndarray,
                                n_assets: int, budget: int, 
                                risk_factor: float) -> np.ndarray:
        """Extract features that characterize the problem"""
        
        features = []
        
        # Basic return statistics
        features.extend([
            np.mean(expected_returns),
            np.std(expected_returns),
            np.min(expected_returns),
            np.max(expected_returns),
            np.median(expected_returns),
            expected_returns[np.argmax(expected_returns)] / (expected_returns[np.argmin(expected_returns)] + 1e-10)
        ])
        
        # Risk metrics
        volatilities = np.sqrt(np.diag(covariance))
        features.extend([
            np.mean(volatilities),
            np.std(volatilities),
            np.min(volatilities),
            np.max(volatilities),
            np.median(volatilities)
        ])
        
        # Sharpe ratio statistics
        sharpe_ratios = []
        for i in range(n_assets):
            if volatilities[i] > 0:
                sharpe = expected_returns[i] / volatilities[i]
            else:
                sharpe = expected_returns[i]
            sharpe_ratios.append(sharpe)
        
        features.extend([
            np.mean(sharpe_ratios),
            np.std(sharpe_ratios),
            np.max(sharpe_ratios),
            np.min(sharpe_ratios)
        ])
        
        # Correlation structure
        corr_matrix = np.zeros((n_assets, n_assets))
        for i in range(n_assets):
            for j in range(n_assets):
                if volatilities[i] > 0 and volatilities[j] > 0:
                    corr_matrix[i, j] = covariance[i, j] / (volatilities[i] * volatilities[j])
                else:
                    corr_matrix[i, j] = 0 if i != j else 1
        
        upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        features.extend([
            np.mean(upper_tri),
            np.std(upper_tri),
            np.min(upper_tri),
            np.max(upper_tri),
            np.sum(upper_tri > 0.5) / max(len(upper_tri), 1),  # Fraction of high correlations
            np.sum(upper_tri < -0.3) / max(len(upper_tri), 1),  # Fraction of negative correlations
            np.sum(np.abs(upper_tri) > 0.7) / max(len(upper_tri), 1)  # Fraction of strong correlations
        ])
        
        # Eigenvalue features (covariance structure)
        try:
            eigenvalues = np.linalg.eigvalsh(covariance)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
            eigenvalues = np.abs(eigenvalues)  # Handle numerical issues
            
            features.extend([
                eigenvalues[0] / (eigenvalues[-1] + 1e-10),  # Condition number
                np.sum(eigenvalues[:3]) / (np.sum(eigenvalues) + 1e-10),  # Concentration in top 3
                eigenvalues[-1] / (np.sum(eigenvalues) + 1e-10),  # Smallest eigenvalue fraction
                np.sum(eigenvalues[:budget]) / (np.sum(eigenvalues) + 1e-10)  # Budget concentration
            ])
        except:
            features.extend([1000, 0.5, 0.001, 0.5])  # Default values if eigendecomposition fails
        
        # Problem configuration
        features.extend([
            n_assets,
            budget,
            budget / n_assets,
            risk_factor,
            n_assets * budget,  # Problem scale
            (n_assets - budget) / n_assets  # Selection sparsity
        ])
        
        return np.array(features)
    
    def update(self, problem_features: np.ndarray, 
              initial_params: np.ndarray,
              final_performance: float,
              convergence_speed: int,
              approximation_ratio: float):
        """Update feedback system with new result"""
        
        # Store in history
        self.problem_features_history.append(problem_features.tolist())
        self.parameter_history.append(initial_params.tolist())
        self.performance_history.append({
            'final_objective': float(final_performance),
            'convergence_speed': int(convergence_speed),
            'approximation_ratio': float(approximation_ratio),
            'timestamp': np.datetime64('now').astype(str)
        })
        
        # Keep only recent history
        if len(self.performance_history) > self.memory_size:
            self.problem_features_history.pop(0)
            self.parameter_history.pop(0)
            self.performance_history.pop(0)
        
        # Save history periodically
        if len(self.performance_history) % 10 == 0:
            self.save_history()
    
    def get_adaptive_parameters(self, current_features: np.ndarray,
                               base_params: np.ndarray,
                               p: int) -> np.ndarray:
        """Adapt parameters based on historical performance"""
        
        if len(self.performance_history) < 5:
            # Not enough history, return base parameters
            return base_params
        
        # Find similar problems in history
        similarities = []
        for i, features in enumerate(self.problem_features_history):
            # Calculate similarity (inverse of normalized distance)
            features_array = np.array(features)
            
            # Normalize features for fair comparison
            norm_current = (current_features - np.mean(current_features)) / (np.std(current_features) + 1e-10)
            norm_historical = (features_array - np.mean(features_array)) / (np.std(features_array) + 1e-10)
            
            distance = np.linalg.norm(norm_current - norm_historical)
            similarity = 1.0 / (1.0 + distance)
            
            # Weight by performance
            perf = self.performance_history[i]
            quality_score = perf['approximation_ratio'] * (1.0 / (1.0 + perf['convergence_speed'] / 30))
            
            similarities.append((similarity * quality_score, i))
        
        # Get top K most similar and successful problems
        similarities.sort(reverse=True)
        K = min(5, len(similarities))
        top_similar = similarities[:K]
        
        # Weighted average of successful parameters
        adjusted_params = base_params.copy()
        total_weight = 0
        
        for weighted_sim, idx in top_similar:
            if weighted_sim > 0.1:  # Only use sufficiently similar problems
                historical_params = np.array(self.parameter_history[idx])
                
                # Ensure parameter arrays have same length
                if len(historical_params) == len(adjusted_params):
                    # Weighted contribution
                    weight = weighted_sim
                    param_diff = historical_params - base_params
                    
                    # Adaptive learning rate based on confidence
                    learning_rate = 0.3 * (1 - np.exp(-len(self.performance_history) / 20))
                    
                    adjusted_params += weight * param_diff * learning_rate
                    total_weight += weight
        
        if total_weight > 0:
            # Normalize the adjustment
            adjustment = (adjusted_params - base_params) / total_weight
            adjusted_params = base_params + adjustment * 0.5  # Conservative adjustment
            
            # Add small exploration noise
            noise_level = 0.02 * (1 - total_weight / K)  # Less noise when confident
            adjusted_params += np.random.normal(0, noise_level, len(adjusted_params))
            
            # Clip to valid range
            adjusted_params = np.clip(adjusted_params, -np.pi, np.pi)
            
            print(f"    Adapted parameters using {K} similar problems (avg similarity: {total_weight/K:.3f})")
        
        return adjusted_params
    
    def get_performance_statistics(self) -> Dict:
        """Get statistics about warm start performance"""
        if not self.performance_history:
            return {}
        
        recent = self.performance_history[-20:] if len(self.performance_history) >= 20 else self.performance_history
        
        stats = {
            'total_problems': len(self.performance_history),
            'avg_convergence': np.mean([p['convergence_speed'] for p in recent]),
            'avg_approximation': np.mean([p['approximation_ratio'] for p in recent]),
            'best_approximation': np.max([p['approximation_ratio'] for p in recent]),
            'improvement_trend': self._calculate_improvement_trend()
        }
        
        return stats
    
    def _calculate_improvement_trend(self) -> float:
        """Calculate if performance is improving over time"""
        if len(self.performance_history) < 10:
            return 0.0
        
        # Compare recent vs older performance
        mid = len(self.performance_history) // 2
        old_perf = [p['approximation_ratio'] for p in self.performance_history[:mid]]
        new_perf = [p['approximation_ratio'] for p in self.performance_history[mid:]]
        
        old_avg = np.mean(old_perf)
        new_avg = np.mean(new_perf)
        
        if old_avg > 0:
            trend = (new_avg - old_avg) / old_avg
        else:
            trend = 0.0
        
        return trend
    
    def save_history(self):
        """Save history to file"""
        try:
            data = {
                'performance': self.performance_history,
                'features': self.problem_features_history,
                'parameters': self.parameter_history
            }
            
            with open(self.save_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save warm start history: {e}")
    
    def load_history(self):
        """Load history from file"""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r') as f:
                    data = json.load(f)
                
                self.performance_history = data.get('performance', [])
                self.problem_features_history = data.get('features', [])
                self.parameter_history = data.get('parameters', [])
                
                print(f"    Loaded warm start history: {len(self.performance_history)} problems")
            except Exception as e:
                print(f"Warning: Could not load warm start history: {e}")
    
    def clear_history(self):
        """Clear all history"""
        self.performance_history = []
        self.parameter_history = []
        self.problem_features_history = []
        
        if os.path.exists(self.save_path):
            os.remove(self.save_path)