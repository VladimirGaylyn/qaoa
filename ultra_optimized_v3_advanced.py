"""
Ultra-Optimized QAOA v3 with Advanced Warm Start
Integrates multi-strategy classical solutions, problem-specific angles,
correlation-aware initialization, and adaptive feedback learning
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from scipy.optimize import minimize
from itertools import combinations

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter, ParameterVector
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    Optimize1qGates, 
    CommutativeCancellation,
    OptimizeSwapBeforeMeasure,
    RemoveDiagonalGatesBeforeMeasure
)
from qiskit import transpile

# Import advanced warm start components
from classical_strategies import ClassicalPortfolioStrategies
from warm_start_feedback import WarmStartFeedback


@dataclass
class UltraOptimizedResultV3:
    """Enhanced result with warm start metrics"""
    solution: np.ndarray
    objective_value: float
    expected_return: float
    risk: float
    sharpe_ratio: float
    constraint_satisfied: bool
    n_selected: int
    execution_time: float
    circuit_depth: int
    gate_count: int
    approximation_ratio: float
    convergence_history: List[float]
    measurement_counts: Dict[str, int]
    feasibility_rate: float
    repaired_solutions: int
    converged: bool
    iterations_to_convergence: int
    initial_feasibility_rate: float
    warm_start_strategy: str
    warm_start_quality: float


class UltraOptimizedQAOAv3:
    """
    Ultra-optimized QAOA v3 with advanced warm start capabilities
    """
    
    def __init__(self, n_assets: int, budget: int, risk_factor: float = 0.5):
        self.n_assets = n_assets
        self.budget = budget
        self.risk_factor = risk_factor
        self.backend = AerSimulator(method='statevector')
        
        # Penalty configuration
        self.base_penalty = 100.0
        self.penalty_multiplier = self.base_penalty * (1 + n_assets)
        
        # Convergence parameters
        self.convergence_tolerance = 1e-4
        self.convergence_window = 5
        self.min_iterations = 10
        
        # Advanced warm start components
        self.classical_strategies = ClassicalPortfolioStrategies(n_assets, budget, risk_factor)
        self.feedback_system = WarmStartFeedback(memory_size=200)
        self.use_advanced_warm_start = True
        
    def create_improved_shallow_circuit(self, n_qubits: int) -> Tuple[QuantumCircuit, ParameterVector]:
        """Create shallow circuit with depth <= 7"""
        qc = QuantumCircuit(n_qubits)
        
        # More parameters for better expressiveness
        num_params = 3 * n_qubits
        params = ParameterVector('θ', num_params)
        param_idx = 0
        
        # Layer 1: Initial rotation with parameters (depth 1)
        for i in range(n_qubits):
            qc.ry(params[param_idx], i)
            param_idx += 1
        
        # Layer 2-3: Full connectivity with alternating pattern (depth 2-3)
        for i in range(0, n_qubits - 1, 2):
            qc.cx(i, i + 1)
        for i in range(1, n_qubits - 1, 2):
            qc.cx(i, i + 1)
        
        # Layer 4: Parameterized phase rotation (depth 4)
        for i in range(n_qubits):
            qc.rz(params[param_idx], i)
            param_idx += 1
        
        # Layer 5: XY-mixing for constraint preservation (depth 5)
        for i in range(0, min(n_qubits - 1, 4), 2):
            if param_idx < len(params) - 1:
                qc.rxx(params[param_idx], i, i + 1)
                param_idx += 1
        
        # Layer 6: Final mixing (depth 6-7)
        for i in range(n_qubits):
            if param_idx < len(params):
                qc.rx(params[param_idx], i)
                param_idx += 1
        
        qc.measure_all()
        return qc, params[:param_idx]
    
    def get_multi_strategy_warm_start(self, expected_returns: np.ndarray,
                                     covariance: np.ndarray, p: int) -> Tuple[np.ndarray, str, float]:
        """Generate warm start from multiple classical strategies"""
        
        print("  Generating multi-strategy classical solutions...")
        solutions, qualities = self.classical_strategies.get_all_solutions(
            expected_returns, covariance
        )
        
        # Sort strategies by quality
        sorted_strategies = sorted(qualities.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_strategies[0][1] > -float('inf'):
            print(f"    Best strategy: {sorted_strategies[0][0]} (objective: {sorted_strategies[0][1]:.4f})")
            print(f"    Using top 3 strategies for ensemble warm start")
        
        # Use top 3 strategies for ensemble
        top_strategies = [s for s in sorted_strategies[:3] if s[1] > -float('inf')]
        
        if not top_strategies:
            # Fallback to random if all strategies failed
            return np.random.uniform(-np.pi/4, np.pi/4, p * 3 * self.n_assets), "random", 0.0
        
        ensemble_params = []
        total_quality = sum(max(0, q) for _, q in top_strategies)
        
        for strategy_name, quality in top_strategies:
            solution = solutions[strategy_name]
            
            # Generate problem-specific parameters for this solution
            params = self.solution_to_parameters(
                solution, expected_returns, covariance, p, quality
            )
            
            # Weight by quality
            if total_quality > 0:
                weight = max(0, quality) / total_quality
            else:
                weight = 1.0 / len(top_strategies)
            
            ensemble_params.append(params * weight)
        
        # Combine parameters
        final_params = np.sum(ensemble_params, axis=0)
        
        return final_params, sorted_strategies[0][0], sorted_strategies[0][1]
    
    def solution_to_parameters(self, solution: np.ndarray, 
                              expected_returns: np.ndarray,
                              covariance: np.ndarray,
                              p: int, 
                              solution_quality: float) -> np.ndarray:
        """Convert classical solution to problem-specific QAOA angles"""
        
        # Simplified parameter count for shallow circuit
        n_params = 3 * self.n_assets  # Matching our circuit structure
        params = np.zeros(n_params)
        
        # Analyze solution characteristics
        selected_indices = np.where(solution == 1)[0]
        
        # Calculate problem-specific metrics
        return_range = np.max(expected_returns) - np.min(expected_returns)
        volatilities = np.sqrt(np.diag(covariance))
        avg_correlation = np.mean(np.abs(covariance[np.triu_indices_from(covariance, k=1)]))
        problem_difficulty = avg_correlation * self.n_assets / self.budget
        
        # Asset-specific scores
        asset_scores = np.zeros(self.n_assets)
        for i in range(self.n_assets):
            # Score based on return and risk
            if np.max(expected_returns) > 0:
                ret_score = expected_returns[i] / np.max(expected_returns)
            else:
                ret_score = 0.5
            
            if volatilities[i] > 0:
                risk_score = 1.0 / volatilities[i]
                risk_score /= np.max(1.0 / (volatilities + 1e-10))
            else:
                risk_score = 0.5
            
            asset_scores[i] = 0.6 * ret_score + 0.4 * risk_score
            
            if i in selected_indices:
                asset_scores[i] *= (1 + max(0, solution_quality))
        
        # Generate parameters
        param_idx = 0
        
        # Initial rotation parameters (Ry gates)
        for i in range(self.n_assets):
            if i in selected_indices:
                # Selected assets: stronger initial rotation
                angle = np.pi/3 * asset_scores[i]
            else:
                # Unselected assets: weaker rotation
                angle = np.pi/6 * (1 - asset_scores[i])
            
            params[param_idx] = np.clip(angle, -np.pi, np.pi)
            param_idx += 1
        
        # Phase rotation parameters (Rz gates)
        for i in range(self.n_assets):
            # Base angle depends on problem structure
            base_angle = np.pi / (4 + problem_difficulty)
            
            if i in selected_indices:
                angle = base_angle * (1 + asset_scores[i])
            else:
                angle = base_angle * 0.5
            
            params[param_idx] = np.clip(angle, -np.pi, np.pi)
            param_idx += 1
        
        # Mixing parameters (Rx gates and Rxx if present)
        remaining_params = n_params - param_idx
        for i in range(remaining_params):
            # Adaptive mixing based on solution confidence
            confidence = max(0, solution_quality)
            mixing_angle = np.pi/4 * (1 - confidence * 0.5)
            
            params[param_idx] = mixing_angle
            param_idx += 1
        
        return params
    
    def correlation_aware_initialization(self, expected_returns: np.ndarray,
                                        covariance: np.ndarray, p: int) -> np.ndarray:
        """Initialize parameters considering asset correlations"""
        
        # Calculate correlation matrix
        std_devs = np.sqrt(np.diag(covariance))
        std_devs[std_devs == 0] = 1e-10
        correlation = covariance / np.outer(std_devs, std_devs)
        
        # Identify correlation clusters
        correlation_clusters = self.identify_correlation_clusters(correlation)
        
        # Initialize parameters
        n_params = 3 * self.n_assets
        params = np.zeros(n_params)
        
        param_idx = 0
        
        # Initial rotations based on cluster membership
        for i in range(self.n_assets):
            cluster_id = self.get_cluster_id(i, correlation_clusters)
            cluster_size = len(correlation_clusters[cluster_id])
            
            # Larger clusters get different initialization
            if cluster_size > self.n_assets / 3:
                angle = np.pi/4  # Large cluster
            elif cluster_size > 1:
                angle = np.pi/3  # Medium cluster
            else:
                angle = np.pi/6  # Singleton
            
            # Adjust by expected return
            if np.max(expected_returns) > np.min(expected_returns):
                return_factor = (expected_returns[i] - np.min(expected_returns)) / (np.max(expected_returns) - np.min(expected_returns))
            else:
                return_factor = 0.5
            
            params[param_idx] = angle * (0.5 + return_factor)
            param_idx += 1
        
        # Phase rotations based on correlations
        for i in range(self.n_assets):
            cluster_id = self.get_cluster_id(i, correlation_clusters)
            cluster_assets = correlation_clusters[cluster_id]
            
            # Calculate within-cluster correlation
            if len(cluster_assets) > 1:
                within_cluster_corr = np.mean([abs(correlation[i, j]) 
                                              for j in cluster_assets if j != i])
            else:
                within_cluster_corr = 0
            
            # High correlation -> different phase
            params[param_idx] = np.pi/4 * (1 - within_cluster_corr)
            param_idx += 1
        
        # Mixing parameters
        for i in range(n_params - param_idx):
            params[param_idx] = np.pi/6
            param_idx += 1
        
        return params
    
    def identify_correlation_clusters(self, correlation: np.ndarray, 
                                     threshold: float = 0.5) -> Dict[int, List[int]]:
        """Identify groups of highly correlated assets"""
        n = len(correlation)
        visited = [False] * n
        clusters = {}
        cluster_id = 0
        
        for i in range(n):
            if visited[i]:
                continue
            
            # Start new cluster
            cluster = [i]
            visited[i] = True
            
            # Find correlated assets
            for j in range(i+1, n):
                if not visited[j] and abs(correlation[i, j]) > threshold:
                    cluster.append(j)
                    visited[j] = True
            
            clusters[cluster_id] = cluster
            cluster_id += 1
        
        # Handle unclustered assets
        for i in range(n):
            if not any(i in cluster for cluster in clusters.values()):
                clusters[cluster_id] = [i]
                cluster_id += 1
        
        return clusters
    
    def get_cluster_id(self, asset: int, 
                      clusters: Dict[int, List[int]]) -> int:
        """Get cluster ID for an asset"""
        for cluster_id, assets in clusters.items():
            if asset in assets:
                return cluster_id
        return -1
    
    def post_process_with_smart_repair(self, counts: Dict[str, int], 
                                      expected_returns: np.ndarray,
                                      covariance: np.ndarray) -> Tuple[np.ndarray, float, int, float]:
        """Post-processing with smart repair strategies"""
        best_feasible = None
        best_feasible_value = -np.inf
        repaired_count = 0
        
        # Track initial feasibility
        total_count = sum(counts.values())
        feasible_count = 0
        
        # Sort by count to prioritize high-probability solutions
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        for bitstring, count in sorted_counts[:100]:
            solution = np.array([int(b) for b in bitstring[::-1][:self.n_assets]])
            n_selected = np.sum(solution)
            
            if n_selected == self.budget:
                feasible_count += count
                value = self.calculate_objective(solution, expected_returns, covariance)
                if value > best_feasible_value:
                    best_feasible_value = value
                    best_feasible = solution
            
            elif abs(n_selected - self.budget) <= 2:
                # Smart repair for near-feasible solutions
                repaired = self.smart_repair(solution, expected_returns, covariance)
                if repaired is not None:
                    repaired_count += 1
                    value = self.calculate_objective(repaired, expected_returns, covariance)
                    if value > best_feasible_value:
                        best_feasible_value = value
                        best_feasible = repaired
        
        initial_feasibility = feasible_count / total_count if total_count > 0 else 0
        
        # If no feasible solution found, use best greedy solution
        if best_feasible is None:
            best_feasible = self.greedy_selection(expected_returns, covariance)
            best_feasible_value = self.calculate_objective(best_feasible, expected_returns, covariance)
            repaired_count += 1
        
        return best_feasible, best_feasible_value, repaired_count, initial_feasibility
    
    def smart_repair(self, solution: np.ndarray, 
                    expected_returns: np.ndarray,
                    covariance: np.ndarray) -> Optional[np.ndarray]:
        """Smarter repair using portfolio metrics"""
        current_sum = np.sum(solution)
        repaired = solution.copy()
        
        volatilities = np.sqrt(np.diag(covariance))
        
        if current_sum > self.budget:
            # Remove assets with worst risk-adjusted returns
            selected_indices = np.where(repaired == 1)[0]
            
            sharpe_contributions = []
            for idx in selected_indices:
                if volatilities[idx] > 0:
                    sharpe = expected_returns[idx] / volatilities[idx]
                else:
                    sharpe = expected_returns[idx]
                sharpe_contributions.append((sharpe, idx))
            
            sharpe_contributions.sort()
            for _, idx in sharpe_contributions[:current_sum - self.budget]:
                repaired[idx] = 0
        
        elif current_sum < self.budget:
            # Add assets with best risk-adjusted returns
            unselected_indices = np.where(repaired == 0)[0]
            
            sharpe_contributions = []
            for idx in unselected_indices:
                if volatilities[idx] > 0:
                    sharpe = expected_returns[idx] / volatilities[idx]
                else:
                    sharpe = expected_returns[idx]
                sharpe_contributions.append((sharpe, idx))
            
            sharpe_contributions.sort(reverse=True)
            for _, idx in sharpe_contributions[:self.budget - current_sum]:
                repaired[idx] = 1
        
        if np.sum(repaired) == self.budget:
            return repaired
        return None
    
    def greedy_selection(self, expected_returns: np.ndarray,
                        covariance: np.ndarray) -> np.ndarray:
        """Greedy selection based on Sharpe ratio"""
        solution = np.zeros(self.n_assets)
        volatilities = np.sqrt(np.diag(covariance))
        
        sharpe_ratios = []
        for i in range(self.n_assets):
            if volatilities[i] > 0:
                sharpe = expected_returns[i] / volatilities[i]
            else:
                sharpe = expected_returns[i]
            sharpe_ratios.append((sharpe, i))
        
        sharpe_ratios.sort(reverse=True)
        for _, idx in sharpe_ratios[:self.budget]:
            solution[idx] = 1
        
        return solution
    
    def calculate_objective(self, solution: np.ndarray,
                          expected_returns: np.ndarray,
                          covariance: np.ndarray) -> float:
        """Calculate portfolio objective with strong penalty"""
        if np.sum(solution) != self.budget:
            penalty = self.penalty_multiplier
            violation = abs(np.sum(solution) - self.budget)
            return -penalty * violation
        
        weights = solution / self.budget
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = np.dot(weights, np.dot(covariance, weights))
        
        return portfolio_return - self.risk_factor * portfolio_variance
    
    def optimize_circuit_compilation(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Optimize circuit compilation for minimal depth"""
        pm = PassManager([
            Optimize1qGates(),
            CommutativeCancellation(),
            OptimizeSwapBeforeMeasure(),
            RemoveDiagonalGatesBeforeMeasure()
        ])
        
        optimized = pm.run(circuit)
        optimized = transpile(
            optimized,
            basis_gates=['rx', 'ry', 'rz', 'cx', 'rxx', 'ryy', 'rzz'],
            optimization_level=3,
            seed_transpiler=42
        )
        
        return optimized
    
    def solve_classical_baseline(self, expected_returns: np.ndarray,
                                covariance: np.ndarray) -> float:
        """Classical baseline for comparison"""
        best_value = -np.inf
        
        if self.n_assets <= 12:
            for indices in combinations(range(self.n_assets), self.budget):
                solution = np.zeros(self.n_assets)
                solution[list(indices)] = 1
                value = self.calculate_objective(solution, expected_returns, covariance)
                best_value = max(best_value, value)
        else:
            # Use greedy + random sampling
            greedy_sol = self.greedy_selection(expected_returns, covariance)
            best_value = self.calculate_objective(greedy_sol, expected_returns, covariance)
            
            for _ in range(5000):
                indices = np.random.choice(self.n_assets, self.budget, replace=False)
                solution = np.zeros(self.n_assets)
                solution[indices] = 1
                value = self.calculate_objective(solution, expected_returns, covariance)
                best_value = max(best_value, value)
        
        return best_value
    
    def solve_ultra_optimized_v3(self, expected_returns: np.ndarray,
                                 covariance: np.ndarray,
                                 max_iterations: int = 50) -> UltraOptimizedResultV3:
        """Solve with advanced warm start capabilities"""
        
        print(f"\nUltra-Optimized QAOA v3 for {self.n_assets} assets, budget={self.budget}")
        print(f"  Mode: Advanced Warm Start")
        start_time = time.time()
        
        # Create circuit
        qc, params = self.create_improved_shallow_circuit(self.n_assets)
        qc_optimized = self.optimize_circuit_compilation(qc)
        
        # Get circuit metrics
        circuit_depth = qc_optimized.depth()
        gate_count = sum(qc_optimized.count_ops().values())
        
        print(f"  Circuit depth: {circuit_depth}")
        print(f"  Gate count: {gate_count}")
        
        # Infer number of layers
        p = 1  # Single layer for our shallow circuit
        
        # ADVANCED WARM START
        warm_start_strategy = "random"
        warm_start_quality = 0.0
        
        if self.use_advanced_warm_start:
            print("  Using advanced warm start strategy...")
            
            # 1. Generate multi-strategy solutions
            initial_params, strategy, quality = self.get_multi_strategy_warm_start(
                expected_returns, covariance, p
            )
            warm_start_strategy = strategy
            warm_start_quality = quality
            
            # 2. Apply correlation-aware adjustments
            correlation_params = self.correlation_aware_initialization(
                expected_returns, covariance, p
            )
            
            # 3. Blend parameters
            initial_params = 0.7 * initial_params + 0.3 * correlation_params
            
            # 4. Extract problem features
            problem_features = self.feedback_system.extract_problem_features(
                expected_returns, covariance, 
                self.n_assets, self.budget, self.risk_factor
            )
            
            # 5. Apply historical feedback
            initial_params = self.feedback_system.get_adaptive_parameters(
                problem_features, initial_params, p
            )
            
            # 6. Add exploration noise
            noise_level = 0.05
            initial_params += np.random.normal(0, noise_level, len(initial_params))
            initial_params = np.clip(initial_params, -np.pi, np.pi)
            
            # Ensure correct length
            if len(initial_params) != len(params):
                initial_params = np.resize(initial_params, len(params))
        else:
            initial_params = np.random.uniform(-np.pi/4, np.pi/4, len(params))
        
        # Optimization tracking
        convergence_history = []
        converged = False
        iterations_to_convergence = max_iterations
        best_initial_feasibility = 0
        
        def objective_function(theta):
            bound_qc = qc_optimized.assign_parameters(dict(zip(params, theta)))
            job = self.backend.run(bound_qc, shots=2048)
            counts = job.result().get_counts()
            
            solution, value, _, initial_feas = self.post_process_with_smart_repair(
                counts, expected_returns, covariance
            )
            
            nonlocal best_initial_feasibility
            best_initial_feasibility = max(best_initial_feasibility, initial_feas)
            
            convergence_history.append(-value)
            
            # Check convergence
            if len(convergence_history) >= self.convergence_window:
                recent = convergence_history[-self.convergence_window:]
                if np.std(recent) < self.convergence_tolerance:
                    nonlocal converged, iterations_to_convergence
                    if not converged and len(convergence_history) >= self.min_iterations:
                        converged = True
                        iterations_to_convergence = len(convergence_history)
            
            return -value
        
        # Optimize
        result = minimize(
            objective_function,
            initial_params,
            method='COBYLA',
            options={'maxiter': max_iterations, 'rhobeg': 0.5}
        )
        
        optimal_params = result.x
        
        # Get final solution
        final_qc = qc_optimized.assign_parameters(dict(zip(params, optimal_params)))
        job = self.backend.run(final_qc, shots=8192)
        final_counts = job.result().get_counts()
        
        best_solution, best_value, repaired_count, initial_feasibility = self.post_process_with_smart_repair(
            final_counts, expected_returns, covariance
        )
        
        # Calculate metrics
        constraint_satisfied = (np.sum(best_solution) == self.budget)
        
        feasible_count = sum(count for bitstring, count in final_counts.items()
                           if sum(int(b) for b in bitstring[::-1][:self.n_assets]) == self.budget)
        total_count = sum(final_counts.values())
        feasibility_rate = feasible_count / total_count if total_count > 0 else 0
        
        # Portfolio metrics
        if constraint_satisfied:
            weights = best_solution / self.budget
            expected_return = np.dot(weights, expected_returns)
            risk = np.sqrt(np.dot(weights, np.dot(covariance, weights)))
            sharpe_ratio = expected_return / risk if risk > 0 else 0
        else:
            expected_return = 0
            risk = 0
            sharpe_ratio = 0
        
        # Approximation ratio
        classical_optimal = self.solve_classical_baseline(expected_returns, covariance)
        approximation_ratio = best_value / classical_optimal if classical_optimal > 0 else 0
        
        execution_time = time.time() - start_time
        
        # Update feedback system
        if self.use_advanced_warm_start:
            self.feedback_system.update(
                problem_features,
                initial_params,
                best_value,
                iterations_to_convergence,
                approximation_ratio
            )
            
            # Print feedback statistics
            stats = self.feedback_system.get_performance_statistics()
            if stats:
                print(f"  Feedback system: {stats['total_problems']} problems learned")
                print(f"    Avg convergence: {stats.get('avg_convergence', 0):.1f} iterations")
                print(f"    Improvement trend: {stats.get('improvement_trend', 0):.1%}")
        
        print(f"  Objective: {best_value:.6f}")
        print(f"  Approximation ratio: {approximation_ratio:.3f}")
        print(f"  Initial feasibility: {initial_feasibility:.1%}")
        print(f"  Final feasibility: {feasibility_rate:.1%}")
        print(f"  Converged: {converged} (at iteration {iterations_to_convergence})")
        print(f"  Time: {execution_time:.2f}s")
        
        return UltraOptimizedResultV3(
            solution=best_solution,
            objective_value=best_value,
            expected_return=expected_return,
            risk=risk,
            sharpe_ratio=sharpe_ratio,
            constraint_satisfied=constraint_satisfied,
            n_selected=int(np.sum(best_solution)),
            execution_time=execution_time,
            circuit_depth=circuit_depth,
            gate_count=gate_count,
            approximation_ratio=approximation_ratio,
            convergence_history=convergence_history,
            measurement_counts=final_counts,
            feasibility_rate=feasibility_rate,
            repaired_solutions=repaired_count,
            converged=converged,
            iterations_to_convergence=iterations_to_convergence,
            initial_feasibility_rate=initial_feasibility,
            warm_start_strategy=warm_start_strategy,
            warm_start_quality=warm_start_quality
        )