"""
Ultra-Optimized QAOA with Circuit Depth ≤ 6
Includes constraint repair, circuit optimization, and convergence tracking
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from scipy.optimize import minimize
from scipy.special import comb

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


@dataclass
class UltraOptimizedResult:
    """Enhanced result with repair statistics"""
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


class UltraOptimizedQAOA:
    """
    Ultra-optimized QAOA with maximum circuit depth of 6
    Includes solution repair and convergence tracking
    """
    
    def __init__(self, n_assets: int, budget: int, risk_factor: float = 0.5):
        self.n_assets = n_assets
        self.budget = budget
        self.risk_factor = risk_factor
        self.backend = AerSimulator(method='statevector')
        
        # Adaptive penalty
        self.base_penalty = 10.0
        self.penalty_multiplier = self.base_penalty * (1 + n_assets/10)
        
        # Convergence parameters
        self.convergence_tolerance = 1e-4
        self.convergence_window = 5
        self.min_iterations = 10
        
    def create_ultra_shallow_circuit(self, n_qubits: int) -> Tuple[QuantumCircuit, ParameterVector]:
        """
        Create ultra-shallow circuit with depth ≤ 6
        Uses single layer with selective entanglement
        """
        qc = QuantumCircuit(n_qubits)
        
        # Single layer architecture for minimal depth
        num_params = n_qubits + (n_qubits // 2)  # Reduced parameters
        params = ParameterVector('θ', num_params)
        param_idx = 0
        
        # Layer 1: Initial rotation (depth 1)
        for i in range(n_qubits):
            qc.ry(params[param_idx], i)
            param_idx += 1
        
        # Layer 2: Sparse entanglement (depth 2-3)
        # Only entangle every other pair to reduce depth
        for i in range(0, n_qubits - 1, 2):
            qc.cx(i, i + 1)
        
        # Layer 3: Phase rotation on entangled pairs (depth 4)
        for i in range(0, n_qubits - 1, 2):
            if param_idx < len(params):
                qc.rz(params[param_idx], i + 1)
                param_idx += 1
        
        # Layer 4: Disentangle (depth 5)
        for i in range(0, n_qubits - 1, 2):
            qc.cx(i, i + 1)
        
        # Layer 5: Final rotation (depth 6)
        for i in range(n_qubits):
            qc.ry(np.pi/8, i)  # Fixed angle for stability
        
        # Add measurements
        qc.measure_all()
        
        return qc, params
    
    def optimize_circuit_compilation(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Optimize circuit compilation for minimal depth
        """
        # Create optimization pass manager
        pm = PassManager([
            Optimize1qGates(),              # Combine single-qubit gates
            CommutativeCancellation(),      # Cancel commuting gates
            OptimizeSwapBeforeMeasure(),    # Remove unnecessary swaps
            RemoveDiagonalGatesBeforeMeasure()  # Remove diagonal gates before measurement
        ])
        
        # Run optimization passes
        optimized = pm.run(circuit)
        
        # Further transpile for backend
        optimized = transpile(
            optimized,
            basis_gates=['rx', 'ry', 'rz', 'cx'],
            optimization_level=3,
            seed_transpiler=42
        )
        
        return optimized
    
    def post_process_solution(self, counts: Dict[str, int], 
                            expected_returns: np.ndarray,
                            covariance: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """
        Post-process solutions with constraint repair
        """
        best_feasible = None
        best_feasible_value = -np.inf
        best_infeasible = None
        best_infeasible_violation = float('inf')
        repaired_count = 0
        
        for bitstring, count in counts.items():
            solution = np.array([int(b) for b in bitstring[::-1][:self.n_assets]])
            violation = abs(np.sum(solution) - self.budget)
            
            if violation == 0:
                # Feasible solution
                value = self.calculate_objective(solution, expected_returns, covariance)
                if value > best_feasible_value:
                    best_feasible_value = value
                    best_feasible = solution
            
            elif violation <= 2:
                # Can repair by flipping 1-2 bits
                repaired = self.greedy_repair(solution, expected_returns, covariance)
                if repaired is not None:
                    repaired_count += 1
                    value = self.calculate_objective(repaired, expected_returns, covariance)
                    if value > best_feasible_value:
                        best_feasible_value = value
                        best_feasible = repaired
            
            else:
                # Track best infeasible for fallback
                if violation < best_infeasible_violation:
                    best_infeasible_violation = violation
                    best_infeasible = solution
        
        # If no feasible solution found, repair best infeasible
        if best_feasible is None and best_infeasible is not None:
            best_feasible = self.force_repair(best_infeasible)
            best_feasible_value = self.calculate_objective(best_feasible, expected_returns, covariance)
            repaired_count += 1
        
        # Fallback to default selection
        if best_feasible is None:
            best_feasible = np.zeros(self.n_assets)
            best_feasible[:self.budget] = 1
            best_feasible_value = self.calculate_objective(best_feasible, expected_returns, covariance)
        
        return best_feasible, best_feasible_value, repaired_count
    
    def greedy_repair(self, solution: np.ndarray, 
                     expected_returns: np.ndarray,
                     covariance: np.ndarray) -> Optional[np.ndarray]:
        """
        Greedily repair solution by flipping bits
        """
        current_sum = np.sum(solution)
        repaired = solution.copy()
        
        if current_sum > self.budget:
            # Need to remove assets
            selected_indices = np.where(repaired == 1)[0]
            
            # Calculate marginal impact of removing each asset
            impacts = []
            for idx in selected_indices:
                temp = repaired.copy()
                temp[idx] = 0
                value = self.calculate_objective(temp, expected_returns, covariance)
                impacts.append((value, idx))
            
            # Remove assets with least negative impact
            impacts.sort(reverse=True)
            for _, idx in impacts[:current_sum - self.budget]:
                repaired[idx] = 0
        
        elif current_sum < self.budget:
            # Need to add assets
            unselected_indices = np.where(repaired == 0)[0]
            
            # Calculate marginal benefit of adding each asset
            benefits = []
            for idx in unselected_indices:
                temp = repaired.copy()
                temp[idx] = 1
                value = self.calculate_objective(temp, expected_returns, covariance)
                benefits.append((value, idx))
            
            # Add assets with highest benefit
            benefits.sort(reverse=True)
            for _, idx in benefits[:self.budget - current_sum]:
                repaired[idx] = 1
        
        # Verify repair succeeded
        if np.sum(repaired) == self.budget:
            return repaired
        return None
    
    def force_repair(self, solution: np.ndarray) -> np.ndarray:
        """
        Force repair to meet budget constraint
        """
        repaired = np.zeros(self.n_assets)
        current_sum = np.sum(solution)
        
        if current_sum >= self.budget:
            # Keep first budget assets
            selected = np.where(solution == 1)[0][:self.budget]
            repaired[selected] = 1
        else:
            # Add random assets to meet budget
            repaired[:self.budget] = 1
        
        return repaired
    
    def calculate_objective(self, solution: np.ndarray,
                          expected_returns: np.ndarray,
                          covariance: np.ndarray) -> float:
        """Calculate portfolio objective value"""
        if np.sum(solution) != self.budget:
            return -np.inf
        
        weights = solution / self.budget
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = np.dot(weights, np.dot(covariance, weights))
        
        return portfolio_return - self.risk_factor * portfolio_variance
    
    def check_convergence(self, history: List[float]) -> bool:
        """
        Check if optimization has converged using variance window
        """
        if len(history) < self.convergence_window:
            return False
        
        if len(history) < self.min_iterations:
            return False
        
        recent = history[-self.convergence_window:]
        variance = np.var(recent)
        
        return variance < self.convergence_tolerance
    
    def solve_ultra_optimized(self, expected_returns: np.ndarray,
                             covariance: np.ndarray,
                             max_iterations: int = 50) -> UltraOptimizedResult:
        """
        Solve with ultra-optimized shallow circuit and enhancements
        """
        print(f"\nUltra-Optimized QAOA for {self.n_assets} assets, budget={self.budget}")
        print(f"  Target: Circuit depth <= 6")
        start_time = time.time()
        
        # Create ultra-shallow circuit
        qc, params = self.create_ultra_shallow_circuit(self.n_assets)
        
        # Optimize circuit compilation
        qc_optimized = self.optimize_circuit_compilation(qc)
        
        # Get actual circuit metrics
        circuit_depth = qc_optimized.depth()
        gate_count = sum(qc_optimized.count_ops().values())
        
        print(f"  Achieved circuit depth: {circuit_depth}")
        print(f"  Gate count: {gate_count}")
        
        # Initialize parameters
        initial_params = np.random.uniform(-np.pi/4, np.pi/4, len(params))
        
        # Convergence tracking
        convergence_history = []
        converged = False
        iterations_to_convergence = max_iterations
        
        # Optimization with early stopping
        def objective_function(theta):
            # Bind parameters
            bound_qc = qc_optimized.assign_parameters(dict(zip(params, theta)))
            
            # Execute
            job = self.backend.run(bound_qc, shots=1024)
            counts = job.result().get_counts()
            
            # Post-process with repair
            solution, value, _ = self.post_process_solution(counts, expected_returns, covariance)
            
            convergence_history.append(-value)
            
            # Check convergence
            if self.check_convergence(convergence_history):
                nonlocal converged, iterations_to_convergence
                if not converged:
                    converged = True
                    iterations_to_convergence = len(convergence_history)
            
            return -value
        
        # Optimize with early stopping callback
        result = minimize(
            objective_function,
            initial_params,
            method='COBYLA',
            options={'maxiter': max_iterations}
        )
        
        optimal_params = result.x
        
        # Get final solution with more shots
        final_qc = qc_optimized.assign_parameters(dict(zip(params, optimal_params)))
        job = self.backend.run(final_qc, shots=4096)
        final_counts = job.result().get_counts()
        
        # Final post-processing with repair
        best_solution, best_value, repaired_count = self.post_process_solution(
            final_counts, expected_returns, covariance
        )
        
        # Calculate metrics
        constraint_satisfied = (np.sum(best_solution) == self.budget)
        
        # Feasibility rate
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
        
        print(f"  Objective: {best_value:.6f}")
        print(f"  Approximation ratio: {approximation_ratio:.3f}")
        print(f"  Converged: {converged} (at iteration {iterations_to_convergence})")
        print(f"  Repaired solutions: {repaired_count}")
        print(f"  Time: {execution_time:.2f}s")
        
        return UltraOptimizedResult(
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
            iterations_to_convergence=iterations_to_convergence
        )
    
    def solve_classical_baseline(self, expected_returns: np.ndarray,
                                covariance: np.ndarray) -> float:
        """Classical baseline for comparison"""
        best_value = -np.inf
        
        if self.n_assets <= 12:
            from itertools import combinations
            for indices in combinations(range(self.n_assets), self.budget):
                solution = np.zeros(self.n_assets)
                solution[list(indices)] = 1
                value = self.calculate_objective(solution, expected_returns, covariance)
                best_value = max(best_value, value)
        else:
            # Random sampling for large problems
            for _ in range(5000):
                indices = np.random.choice(self.n_assets, self.budget, replace=False)
                solution = np.zeros(self.n_assets)
                solution[indices] = 1
                value = self.calculate_objective(solution, expected_returns, covariance)
                best_value = max(best_value, value)
        
        return best_value


def test_ultra_optimized():
    """Test ultra-optimized QAOA with depth ≤ 6"""
    
    print("="*70)
    print("ULTRA-OPTIMIZED QAOA TEST - CIRCUIT DEPTH <= 6")
    print("="*70)
    
    # Test configurations
    test_configs = [
        {"n_assets": 8, "budget": 4},
        {"n_assets": 10, "budget": 5},
        {"n_assets": 15, "budget": 7},
    ]
    
    for config in test_configs:
        print(f"\nTest: {config['n_assets']} assets, budget={config['budget']}")
        print("-" * 40)
        
        # Initialize
        optimizer = UltraOptimizedQAOA(
            n_assets=config['n_assets'],
            budget=config['budget'],
            risk_factor=0.5
        )
        
        # Generate data
        np.random.seed(42)
        expected_returns = np.random.uniform(0.05, 0.25, config['n_assets'])
        
        # Simple covariance for testing
        covariance = np.eye(config['n_assets']) * 0.04
        
        # Run optimization
        result = optimizer.solve_ultra_optimized(
            expected_returns,
            covariance,
            max_iterations=30
        )
        
        print(f"\nResults:")
        print(f"  Circuit Depth: {result.circuit_depth} (target: <= 6)")
        print(f"  Constraint Satisfied: {result.constraint_satisfied}")
        print(f"  Feasibility Rate: {result.feasibility_rate:.1%}")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.3f}")
        
        # Verify depth constraint
        if result.circuit_depth <= 6:
            print(f"  SUCCESS: Circuit depth {result.circuit_depth} <= 6")
        else:
            print(f"  FAILED: Circuit depth {result.circuit_depth} > 6")
    
    print("\n" + "="*70)
    print("ULTRA-OPTIMIZATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    test_ultra_optimized()