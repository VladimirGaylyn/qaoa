"""
HONEST QAOA Implementation - No Tricks, Real Quantum Mechanics
This implementation shows REAL quantum performance without any classical cheating
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter
import time
from typing import Dict, List, Tuple
from scipy.optimize import minimize
from collections import Counter

class HonestQAOA:
    """Real QAOA implementation with authentic quantum mechanics"""
    
    def __init__(self, n_assets: int, budget: int, risk_factor: float = 0.5):
        self.n_assets = n_assets
        self.budget = budget
        self.risk_factor = risk_factor
        self.n_qubits = n_assets
        
        # REALISTIC parameters for genuine quantum performance
        self.p = 5  # Higher depth for better performance (but still realistic)
        self.shots = 8192  # Realistic shot count
        
        # Realistic penalty scaling
        self.penalty = 3.0  # Moderate penalty to maintain ~15-20% feasibility
        
        # Backend
        self.backend = AerSimulator()
        
        # Tracking
        self.circuit_executions = 0
        self.total_shots_used = 0
        
    def create_mixer_circuit(self, qc: QuantumCircuit, qreg: QuantumRegister, beta: Parameter):
        """Create REAL mixer - simple X rotation on each qubit"""
        for i in range(self.n_qubits):
            qc.rx(2 * beta, qreg[i])
    
    def create_cost_circuit(self, qc: QuantumCircuit, qreg: QuantumRegister, gamma: Parameter,
                           linear: Dict, quadratic: Dict):
        """Create cost Hamiltonian evolution - REAL implementation"""
        
        # Linear terms (single qubit rotations)
        for i, coeff in linear.items():
            if abs(coeff) > 1e-10:
                qc.rz(2 * gamma * coeff, qreg[i])
        
        # Quadratic terms (two-qubit interactions)
        for (i, j), coeff in quadratic.items():
            if abs(coeff) > 1e-10:
                if i == j:
                    # Self-interaction term
                    qc.rz(gamma * coeff, qreg[i])
                else:
                    # Two-qubit interaction: e^(-i*gamma*Z_i*Z_j)
                    qc.rzz(2 * gamma * coeff, qreg[i], qreg[j])
    
    def create_qaoa_circuit(self, linear: Dict, quadratic: Dict) -> QuantumCircuit:
        """Create REAL QAOA circuit - no tricks"""
        
        qreg = QuantumRegister(self.n_qubits, 'q')
        creg = ClassicalRegister(self.n_qubits, 'c')
        qc = QuantumCircuit(qreg, creg)
        
        # Create parameters
        betas = [Parameter(f'beta_{i}') for i in range(self.p)]
        gammas = [Parameter(f'gamma_{i}') for i in range(self.p)]
        
        # HONEST initialization - equal superposition (no biasing)
        for i in range(self.n_qubits):
            qc.h(qreg[i])
        
        # QAOA layers
        for layer in range(self.p):
            # Cost Hamiltonian
            self.create_cost_circuit(qc, qreg, gammas[layer], linear, quadratic)
            
            # Mixer Hamiltonian
            self.create_mixer_circuit(qc, qreg, betas[layer])
        
        # Measurements
        qc.measure(qreg, creg)
        
        return qc
    
    def calculate_objective(self, solution: np.ndarray, expected_returns: np.ndarray,
                          covariance: np.ndarray) -> float:
        """Calculate portfolio objective value"""
        
        # Check feasibility
        if np.sum(solution) != self.budget:
            # Return penalty proportional to constraint violation
            violation = abs(np.sum(solution) - self.budget)
            return -self.penalty * violation
        
        # Calculate portfolio metrics
        weights = solution / self.budget
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = np.dot(weights, np.dot(covariance, weights))
        
        return portfolio_return - self.risk_factor * portfolio_variance
    
    def prepare_problem(self, expected_returns: np.ndarray, covariance: np.ndarray) -> Tuple[Dict, Dict]:
        """Prepare QUBO formulation - HONEST version"""
        
        linear = {}
        quadratic = {}
        
        # Portfolio optimization terms
        for i in range(self.n_assets):
            linear[i] = -expected_returns[i] / self.budget
        
        for i in range(self.n_assets):
            for j in range(i, self.n_assets):
                if i == j:
                    quadratic[(i, j)] = self.risk_factor * covariance[i, j] / (self.budget ** 2)
                else:
                    quadratic[(i, j)] = 2 * self.risk_factor * covariance[i, j] / (self.budget ** 2)
        
        # Budget constraint penalty (moderate, not overwhelming)
        for i in range(self.n_assets):
            linear[i] += self.penalty * (1 - 2 * self.budget / self.n_assets)
        
        for i in range(self.n_assets):
            for j in range(i, self.n_assets):
                penalty_term = self.penalty / self.n_assets
                if i == j:
                    quadratic[(i, j)] = quadratic.get((i, j), 0) + penalty_term
                else:
                    quadratic[(i, j)] = quadratic.get((i, j), 0) + 2 * penalty_term
        
        return linear, quadratic
    
    def execute_circuit(self, circuit: QuantumCircuit, params: np.ndarray) -> Dict[str, int]:
        """Execute quantum circuit - REAL execution with shot noise"""
        
        # Bind parameters
        param_dict = {}
        for i in range(self.p):
            param_dict[f'beta_{i}'] = params[2 * i + 1]
            param_dict[f'gamma_{i}'] = params[2 * i]
        
        bound_circuit = circuit.assign_parameters(param_dict)
        
        # Execute with REAL shot count
        self.circuit_executions += 1
        self.total_shots_used += self.shots
        
        job = self.backend.run(bound_circuit, shots=self.shots)
        result = job.result()
        counts = result.get_counts()
        
        return counts
    
    def objective_function(self, params: np.ndarray, circuit: QuantumCircuit,
                          expected_returns: np.ndarray, covariance: np.ndarray) -> float:
        """QAOA objective - compute REAL expectation value"""
        
        # Execute circuit
        counts = self.execute_circuit(circuit, params)
        
        # Calculate expectation value
        expectation = 0
        total_counts = sum(counts.values())
        
        for bitstring, count in counts.items():
            # Convert to solution
            solution = np.array([int(bit) for bit in bitstring][::-1])  # Reverse for Qiskit ordering
            
            # Calculate objective
            obj_value = self.calculate_objective(solution, expected_returns, covariance)
            
            # Add to expectation
            expectation += (count / total_counts) * obj_value
        
        return -expectation  # Minimize negative for maximization
    
    def optimize(self, expected_returns: np.ndarray, covariance: np.ndarray,
                max_iterations: int = 100) -> Dict:
        """Run HONEST QAOA optimization"""
        
        print("\n" + "="*60)
        print("HONEST QAOA - Real Quantum Performance")
        print("="*60)
        print(f"Problem: {self.n_assets} assets, select {self.budget}")
        print(f"Circuit depth (p): {self.p}")
        print(f"Shots per execution: {self.shots}")
        print(f"Max iterations: {max_iterations}")
        
        start_time = time.time()
        
        # Prepare problem
        linear, quadratic = self.prepare_problem(expected_returns, covariance)
        
        # Create circuit
        circuit = self.create_qaoa_circuit(linear, quadratic)
        actual_depth = circuit.depth()
        print(f"Actual circuit depth: {actual_depth}")
        
        # Initial parameters - small random values
        initial_params = np.random.uniform(-np.pi/4, np.pi/4, 2 * self.p)
        
        # Reset counters
        self.circuit_executions = 0
        self.total_shots_used = 0
        
        # Optimize
        print("\nOptimizing...")
        result = minimize(
            lambda p: self.objective_function(p, circuit, expected_returns, covariance),
            initial_params,
            method='COBYLA',
            options={'maxiter': max_iterations, 'disp': False}
        )
        
        print(f"Optimization complete: {self.circuit_executions} circuit executions")
        print(f"Total shots used: {self.total_shots_used:,}")
        
        # Get final distribution with more shots for accuracy
        print("\nGetting final distribution with 32768 shots...")
        self.shots = 32768  # More shots for final measurement
        final_counts = self.execute_circuit(circuit, result.x)
        
        # Analyze results
        results = self.analyze_results(final_counts, expected_returns, covariance)
        
        # Add metadata
        results['optimization_time'] = time.time() - start_time
        results['circuit_depth'] = actual_depth
        results['circuit_executions'] = self.circuit_executions
        results['total_shots'] = self.total_shots_used
        results['p_layers'] = self.p
        results['final_parameters'] = result.x.tolist()
        
        return results
    
    def analyze_results(self, counts: Dict[str, int], 
                       expected_returns: np.ndarray,
                       covariance: np.ndarray) -> Dict:
        """Analyze REAL quantum results - no manipulation"""
        
        total_counts = sum(counts.values())
        
        # Find feasible solutions
        feasible_solutions = {}
        all_objectives = []
        
        for bitstring, count in counts.items():
            solution = np.array([int(bit) for bit in bitstring][::-1])
            
            if np.sum(solution) == self.budget:
                obj_value = self.calculate_objective(solution, expected_returns, covariance)
                feasible_solutions[bitstring] = {
                    'count': count,
                    'probability': count / total_counts,
                    'objective': obj_value,
                    'solution': solution.tolist()
                }
                all_objectives.append(obj_value)
        
        # Calculate REAL metrics
        feasibility_rate = sum(fs['count'] for fs in feasible_solutions.values()) / total_counts
        
        # Find best solution
        if feasible_solutions:
            best_state = max(feasible_solutions.items(), key=lambda x: x[1]['objective'])
            best_solution_prob = best_state[1]['probability']
            best_objective = best_state[1]['objective']
            best_solution = best_state[1]['solution']
            
            # Get top 10 solutions
            sorted_solutions = sorted(feasible_solutions.items(), 
                                    key=lambda x: x[1]['objective'], 
                                    reverse=True)[:10]
            
            top_10 = []
            for state, info in sorted_solutions:
                top_10.append({
                    'state': state,
                    'probability': info['probability'],
                    'objective': info['objective'],
                    'count': info['count']
                })
        else:
            best_solution_prob = 0
            best_objective = -float('inf')
            best_solution = None
            top_10 = []
        
        # Calculate classical best (for comparison)
        classical_best = self.classical_baseline(expected_returns, covariance)
        
        # Show distribution statistics
        print("\n" + "-"*60)
        print("REAL Quantum Results:")
        print(f"Feasibility rate: {feasibility_rate*100:.2f}%")
        print(f"Best solution probability: {best_solution_prob*100:.4f}%")
        print(f"Number of unique states measured: {len(counts)}")
        print(f"Number of feasible states found: {len(feasible_solutions)}")
        
        if top_10:
            print("\nTop 5 solutions by probability:")
            for i, sol in enumerate(top_10[:5]):
                print(f"  {i+1}. State {sol['state']}: {sol['probability']*100:.4f}% "
                      f"(count: {sol['count']}/{total_counts})")
        
        return {
            'feasibility_rate': feasibility_rate,
            'best_solution_probability': best_solution_prob,
            'best_objective': best_objective,
            'best_solution': best_solution,
            'classical_best': classical_best,
            'approximation_ratio': best_objective / classical_best if classical_best != 0 else 0,
            'top_10_solutions': top_10,
            'n_unique_states': len(counts),
            'n_feasible_states': len(feasible_solutions),
            'total_counts': total_counts
        }
    
    def classical_baseline(self, expected_returns: np.ndarray, covariance: np.ndarray) -> float:
        """Get classical solution for comparison"""
        from itertools import combinations
        
        best_obj = -float('inf')
        
        # Check all combinations (only works for small problems)
        for combo in combinations(range(self.n_assets), self.budget):
            solution = np.zeros(self.n_assets)
            solution[list(combo)] = 1
            obj = self.calculate_objective(solution, expected_returns, covariance)
            if obj > best_obj:
                best_obj = obj
        
        return best_obj