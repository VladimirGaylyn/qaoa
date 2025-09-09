"""
Amplitude Amplification Post-Processing for QAOA
Enhances probability of good solutions through iterative amplification
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import Counter

class AmplitudeAmplifier:
    """Post-process QAOA results to amplify good solutions"""
    
    def __init__(self, amplification_rounds: int = 3, 
                 amplification_factor: float = 1.5):
        self.amplification_rounds = amplification_rounds
        self.amplification_factor = amplification_factor
        
    def amplify(self, counts: Dict[str, int], 
                objective_function: callable,
                n_assets: int,
                budget: int) -> Dict[str, float]:
        """
        Amplify probabilities of good solutions
        
        Args:
            counts: Raw measurement counts from QAOA
            objective_function: Function to evaluate solution quality
            n_assets: Number of assets
            budget: Number of assets to select
            
        Returns:
            Amplified probability distribution
        """
        # Convert counts to probabilities
        total_counts = sum(counts.values())
        probabilities = {state: count/total_counts for state, count in counts.items()}
        
        # Evaluate each solution
        solution_qualities = {}
        for state in probabilities:
            # Convert bitstring to solution vector
            solution = np.array([int(bit) for bit in state])
            
            # Check feasibility
            if np.sum(solution) == budget:
                # Feasible solution - evaluate quality
                quality = objective_function(solution)
            else:
                # Infeasible - assign penalty
                constraint_violation = abs(np.sum(solution) - budget)
                quality = -1000 * constraint_violation  # Heavy penalty
            
            solution_qualities[state] = quality
        
        # Find threshold for "good" solutions
        all_qualities = list(solution_qualities.values())
        if not all_qualities:
            return probabilities
        
        # Use top 10% as "good" solutions for more aggressive amplification
        sorted_qualities = sorted(all_qualities, reverse=True)
        threshold_idx = max(1, min(10, len(sorted_qualities) // 10))  # At least 1, at most 10
        quality_threshold = sorted_qualities[threshold_idx - 1]
        
        # Iterative amplification
        amplified_probs = probabilities.copy()
        
        for round_idx in range(self.amplification_rounds):
            # Calculate amplification weights
            weights = {}
            for state, prob in amplified_probs.items():
                quality = solution_qualities[state]
                
                if quality >= quality_threshold:
                    # Good solution - amplify
                    weight = self.amplification_factor
                else:
                    # Poor solution - suppress
                    weight = 1.0 / self.amplification_factor
                
                # Additional boost for best solutions
                if quality >= sorted_qualities[0] * 0.95:
                    weight *= 1.5  # Stronger boost for top solutions
                
                weights[state] = weight
            
            # Apply weights and renormalize
            for state in amplified_probs:
                amplified_probs[state] *= weights[state]
            
            # Renormalize
            total_prob = sum(amplified_probs.values())
            if total_prob > 0:
                for state in amplified_probs:
                    amplified_probs[state] /= total_prob
            
            # Keep amplification strong throughout
            pass  # Don't reduce amplification factor
        
        return amplified_probs
    
    def concentrate_probability(self, counts: Dict[str, int],
                              top_k: int = 5) -> Dict[str, float]:
        """
        Concentrate probability mass on top-k solutions
        
        Args:
            counts: Raw measurement counts
            top_k: Number of top solutions to concentrate on
            
        Returns:
            Concentrated probability distribution
        """
        # Sort solutions by count
        sorted_states = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        # Take top-k
        top_states = sorted_states[:top_k]
        
        # Calculate concentration weights
        concentrated_probs = {}
        total_top_counts = sum(count for _, count in top_states)
        
        for state, count in top_states:
            # Give extra weight to top solutions
            rank = sorted_states.index((state, count)) + 1
            rank_weight = 1.0 / np.sqrt(rank)
            
            # Base probability from counts
            base_prob = count / total_top_counts
            
            # Weighted probability
            concentrated_probs[state] = base_prob * rank_weight
        
        # Add small probability for other states
        other_states = sorted_states[top_k:]
        if other_states:
            remaining_prob = 0.1  # Reserve 10% for other states
            n_other = len(other_states)
            for state, _ in other_states:
                concentrated_probs[state] = remaining_prob / n_other
        
        # Renormalize
        total_prob = sum(concentrated_probs.values())
        if total_prob > 0:
            for state in concentrated_probs:
                concentrated_probs[state] /= total_prob
        
        return concentrated_probs
    
    def apply_grover_operator(self, probabilities: Dict[str, float],
                             good_states: List[str],
                             iterations: int = 2) -> Dict[str, float]:
        """
        Apply Grover-like operator to amplify good states
        
        Args:
            probabilities: Initial probability distribution
            good_states: List of states to amplify
            iterations: Number of Grover iterations
            
        Returns:
            Amplified distribution
        """
        # Convert to amplitude representation
        states = list(probabilities.keys())
        amplitudes = np.array([np.sqrt(probabilities[s]) for s in states])
        
        for _ in range(iterations):
            # Oracle: flip phase of good states
            for i, state in enumerate(states):
                if state in good_states:
                    amplitudes[i] *= -1
            
            # Inversion about average
            avg_amplitude = np.mean(amplitudes)
            amplitudes = 2 * avg_amplitude - amplitudes
        
        # Convert back to probabilities
        probabilities_new = {}
        amplitudes_squared = amplitudes ** 2
        total = np.sum(amplitudes_squared)
        
        if total > 0:
            for i, state in enumerate(states):
                probabilities_new[state] = amplitudes_squared[i] / total
        else:
            # Fallback to uniform
            for state in states:
                probabilities_new[state] = 1.0 / len(states)
        
        return probabilities_new
    
    def smart_sampling(self, probabilities: Dict[str, float],
                      n_samples: int,
                      bias_factor: float = 2.0) -> List[str]:
        """
        Smart sampling that biases toward high-probability states
        
        Args:
            probabilities: Probability distribution
            n_samples: Number of samples to generate
            bias_factor: How much to bias toward high-probability states
            
        Returns:
            List of sampled states
        """
        states = list(probabilities.keys())
        probs = np.array([probabilities[s] for s in states])
        
        # Apply bias
        biased_probs = probs ** bias_factor
        biased_probs /= np.sum(biased_probs)
        
        # Sample
        samples = np.random.choice(states, size=n_samples, p=biased_probs)
        
        return list(samples)
    
    def analyze_amplification_effect(self, original_counts: Dict[str, int],
                                    amplified_probs: Dict[str, float]) -> Dict:
        """
        Analyze the effect of amplification
        
        Returns:
            Dictionary with amplification statistics
        """
        # Original probabilities
        total_original = sum(original_counts.values())
        original_probs = {s: c/total_original for s, c in original_counts.items()}
        
        # Find top solution
        top_state = max(amplified_probs.items(), key=lambda x: x[1])[0]
        
        # Calculate amplification metrics
        original_top_prob = original_probs.get(top_state, 0)
        amplified_top_prob = amplified_probs.get(top_state, 0)
        
        # Entropy calculation
        def entropy(probs):
            return -sum(p * np.log(p + 1e-10) for p in probs.values())
        
        original_entropy = entropy(original_probs)
        amplified_entropy = entropy(amplified_probs)
        
        # Concentration metric (Gini coefficient)
        def gini_coefficient(probs):
            sorted_probs = sorted(probs.values())
            n = len(sorted_probs)
            cumsum = np.cumsum(sorted_probs)
            return (2 * np.sum((np.arange(1, n+1) * sorted_probs))) / (n * cumsum[-1]) - (n + 1) / n
        
        original_gini = gini_coefficient(original_probs)
        amplified_gini = gini_coefficient(amplified_probs)
        
        return {
            'top_solution_amplification': amplified_top_prob / (original_top_prob + 1e-10),
            'entropy_reduction': (original_entropy - amplified_entropy) / original_entropy,
            'concentration_increase': amplified_gini - original_gini,
            'original_top_prob': original_top_prob,
            'amplified_top_prob': amplified_top_prob,
            'n_states_before': len(original_probs),
            'n_states_after': len([p for p in amplified_probs.values() if p > 0.001])
        }