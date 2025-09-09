"""
Best Honest QAOA - Pushing toward 1% with REAL quantum mechanics
This uses every legitimate technique to maximize performance
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter
import time
from typing import Dict, Tuple
from scipy.optimize import minimize, differential_evolution
from collections import Counter
from optimization_strategies import (
    LegitimateEnhancements, AdvancedQAOACircuit, 
    ImprovedOptimizer, QAOAPerformanceBooster
)

class BestHonestQAOA:
    """The best honest QAOA implementation possible"""
    
    def __init__(self, n_assets: int, budget: int, risk_factor: float = 0.5):
        self.n_assets = n_assets
        self.budget = budget
        self.risk_factor = risk_factor
        
        # Get optimal configuration
        config = QAOAPerformanceBooster.get_best_configuration(n_assets, budget)
        
        self.p = config['p']
        self.shots = config['shots']
        self.use_xy_mixer = config['use_xy_mixer']
        self.use_weighted_init = config['use_weighted_init']
        self.penalty = config['penalty_scaling']
        self.max_iterations = config['max_iterations']
        
        # Backend
        self.backend = AerSimulator()
        
        # Tracking
        self.circuit_executions = 0
        self.total_shots = 0
        
        print(f"\nBest Honest QAOA Configuration:")
        print(f"  Circuit depth (p): {self.p}")
        print(f"  Shots: {self.shots}")
        print(f"  XY-Mixer: {self.use_xy_mixer}")
        print(f"  Weighted init: {self.use_weighted_init}")
        
    def create_optimized_circuit(self, linear: Dict, quadratic: Dict) -> QuantumCircuit:
        """Create the best honest QAOA circuit"""
        
        qreg = QuantumRegister(self.n_assets, 'q')
        creg = ClassicalRegister(self.n_assets, 'c')
        qc = QuantumCircuit(qreg, creg)
        
        # Parameters
        betas = [Parameter(f'beta_{i}') for i in range(self.p)]
        gammas = [Parameter(f'gamma_{i}') for i in range(self.p)]
        
        # INITIALIZATION STRATEGY
        if self.use_weighted_init:
            # Bias toward feasible states (legitimate technique)
            target_prob = self.budget / self.n_assets
            theta = 2 * np.arcsin(np.sqrt(target_prob))
            for i in range(self.n_assets):
                qc.ry(theta, qreg[i])
        else:
            # Standard uniform superposition
            for i in range(self.n_assets):
                qc.h(qreg[i])
        
        # QAOA LAYERS
        for layer in range(self.p):
            # COST HAMILTONIAN
            gamma = gammas[layer]
            
            # Apply linear terms
            for i, coeff in linear.items():
                if abs(coeff) > 1e-10:
                    qc.rz(2 * gamma * coeff, qreg[i])
            
            # Apply quadratic terms efficiently
            for (i, j), coeff in quadratic.items():
                if abs(coeff) > 1e-10:
                    if i == j:
                        qc.rz(gamma * coeff, qreg[i])
                    else:
                        # Efficient two-qubit gate
                        qc.rzz(2 * gamma * coeff, qreg[i], qreg[j])
            
            # MIXER HAMILTONIAN
            beta = betas[layer]
            
            if self.use_xy_mixer and layer % 2 == 1:
                # XY-mixer for better constraint preservation
                for i in range(self.n_assets - 1):
                    qc.rxx(beta * 0.5, qreg[i], qreg[i + 1])
                    qc.ryy(beta * 0.5, qreg[i], qreg[i + 1])
                # Close the ring with reduced strength
                qc.rxx(beta * 0.25, qreg[-1], qreg[0])
                qc.ryy(beta * 0.25, qreg[-1], qreg[0])
            else:
                # Standard X-mixer
                for i in range(self.n_assets):
                    qc.rx(2 * beta, qreg[i])
        
        # Measurement
        qc.measure(qreg, creg)
        
        return qc
    
    def prepare_qubo(self, expected_returns: np.ndarray, 
                    covariance: np.ndarray) -> Tuple[Dict, Dict]:
        """Prepare QUBO with balanced penalties"""
        
        linear = {}
        quadratic = {}
        
        # Portfolio optimization terms
        scale = 1.0 / self.budget
        
        for i in range(self.n_assets):
            linear[i] = -expected_returns[i] * scale
        
        for i in range(self.n_assets):
            for j in range(i, self.n_assets):
                coeff = self.risk_factor * covariance[i, j] * scale * scale
                if i == j:
                    quadratic[(i, j)] = coeff
                else:
                    quadratic[(i, j)] = 2 * coeff
        
        # Constraint penalty - carefully balanced
        # Too high: kills feasibility
        # Too low: no constraint satisfaction
        penalty_per_qubit = self.penalty / np.sqrt(self.n_assets)
        
        for i in range(self.n_assets):
            linear[i] += penalty_per_qubit * (1 - 2 * self.budget / self.n_assets)
        
        for i in range(self.n_assets):
            for j in range(i, self.n_assets):
                if i == j:
                    quadratic[(i, j)] = quadratic.get((i, j), 0) + penalty_per_qubit
                else:
                    quadratic[(i, j)] = quadratic.get((i, j), 0) + 2 * penalty_per_qubit
        
        return linear, quadratic
    
    def execute_and_measure(self, circuit: QuantumCircuit, params: np.ndarray) -> Dict[str, int]:
        """Execute circuit with realistic shot noise"""
        
        # Bind parameters
        param_dict = {}
        for i in range(self.p):
            param_dict[f'beta_{i}'] = params[2 * i + 1]
            param_dict[f'gamma_{i}'] = params[2 * i]
        
        bound_circuit = circuit.assign_parameters(param_dict)
        
        # Track execution
        self.circuit_executions += 1
        self.total_shots += self.shots
        
        # Run on simulator
        job = self.backend.run(bound_circuit, shots=self.shots, seed_simulator=None)
        result = job.result()
        counts = result.get_counts()
        
        return counts
    
    def objective_function(self, params: np.ndarray, circuit: QuantumCircuit,
                          expected_returns: np.ndarray, covariance: np.ndarray) -> float:
        """Compute expectation value"""
        
        counts = self.execute_and_measure(circuit, params)
        
        expectation = 0
        total = sum(counts.values())
        
        for bitstring, count in counts.items():
            # Convert to solution (reverse for Qiskit ordering)
            solution = np.array([int(bit) for bit in bitstring][::-1])
            
            # Compute objective
            if np.sum(solution) == self.budget:
                weights = solution / self.budget
                ret = np.dot(weights, expected_returns)
                risk = np.dot(weights, np.dot(covariance, weights))
                obj = ret - self.risk_factor * risk
            else:
                # Penalty for infeasible
                violation = abs(np.sum(solution) - self.budget)
                obj = -self.penalty * violation
            
            expectation += (count / total) * obj
        
        return -expectation  # Minimize negative
    
    def optimize_smart(self, circuit: QuantumCircuit, expected_returns: np.ndarray,
                       covariance: np.ndarray) -> np.ndarray:
        """Smart optimization using multiple strategies"""
        
        print("\nOptimization Strategy:")
        
        # Strategy 1: Get good initial parameters
        initial_params = LegitimateEnhancements.warm_start_parameters(
            self.n_assets, self.budget, self.p
        )
        
        # Strategy 2: Layer-wise optimization for first few layers
        if self.p > 4:
            print("  Phase 1: Layer-wise optimization...")
            params = ImprovedOptimizer.layerwise_optimization(
                circuit,
                lambda p: self.objective_function(p, circuit, expected_returns, covariance),
                min(4, self.p),
                max_evals=50
            )
            # Extend with good guesses for remaining layers
            if len(params) < 2 * self.p:
                remaining = 2 * self.p - len(params)
                params = np.concatenate([params, initial_params[-remaining:]])
        else:
            params = initial_params
        
        # Strategy 3: Global optimization
        print("  Phase 2: Global optimization...")
        
        # Use appropriate bounds for each parameter
        bounds = []
        for i in range(self.p):
            # Gammas - cost parameters
            gamma_scale = 1.0 / (1 + i * 0.2)  # Deeper layers use smaller angles
            bounds.append((-np.pi * gamma_scale, np.pi * gamma_scale))
            
            # Betas - mixer parameters  
            beta_scale = 1.0 / (1 + i * 0.15)
            bounds.append((-np.pi/2 * beta_scale, np.pi/2 * beta_scale))
        
        # Main optimization
        result = minimize(
            lambda p: self.objective_function(p, circuit, expected_returns, covariance),
            params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': self.max_iterations, 'ftol': 1e-6}
        )
        
        return result.x
    
    def run(self, expected_returns: np.ndarray, covariance: np.ndarray) -> Dict:
        """Run the best honest QAOA"""
        
        print("\n" + "="*70)
        print("BEST HONEST QAOA - Maximum Performance with Real Quantum Mechanics")
        print("="*70)
        
        start_time = time.time()
        
        # Show realistic expectations
        from optimization_strategies import show_realistic_expectations
        show_realistic_expectations(self.n_assets, self.budget)
        
        # Prepare problem
        linear, quadratic = self.prepare_qubo(expected_returns, covariance)
        
        # Create circuit
        print("\nCreating optimized circuit...")
        circuit = self.create_optimized_circuit(linear, quadratic)
        print(f"Circuit depth: {circuit.depth()}")
        print(f"Circuit width: {circuit.width()}")
        
        # Reset counters
        self.circuit_executions = 0
        self.total_shots = 0
        
        # Optimize
        print("\nRunning quantum optimization...")
        optimal_params = self.optimize_smart(circuit, expected_returns, covariance)
        
        print(f"\nOptimization complete:")
        print(f"  Circuit executions: {self.circuit_executions}")
        print(f"  Total shots: {self.total_shots:,}")
        
        # Final measurement with maximum shots
        print("\nFinal measurement with 65536 shots...")
        self.shots = 65536  # Maximum shots for final accuracy
        final_counts = self.execute_and_measure(circuit, optimal_params)
        
        # Analyze results
        results = self.analyze_final_results(final_counts, expected_returns, covariance)
        
        # Add metadata
        results['total_time'] = time.time() - start_time
        results['circuit_depth'] = circuit.depth()
        results['p_layers'] = self.p
        results['total_circuit_executions'] = self.circuit_executions
        results['total_shots_used'] = self.total_shots
        
        # Show achievement
        print("\n" + "="*70)
        if results['best_solution_probability'] >= 0.01:
            print("SUCCESS! Achieved ≥1% best solution probability!")
        else:
            print(f"Achieved {results['best_solution_probability']*100:.4f}% "
                  f"(target was 1%)")
        print("="*70)
        
        return results
    
    def analyze_final_results(self, counts: Dict[str, int],
                             expected_returns: np.ndarray,
                             covariance: np.ndarray) -> Dict:
        """Analyze results honestly"""
        
        total = sum(counts.values())
        
        # Find feasible solutions
        feasible = {}
        for bitstring, count in counts.items():
            solution = np.array([int(bit) for bit in bitstring][::-1])
            
            if np.sum(solution) == self.budget:
                weights = solution / self.budget
                ret = np.dot(weights, expected_returns)
                risk = np.dot(weights, np.dot(covariance, weights))
                obj = ret - self.risk_factor * risk
                
                feasible[bitstring] = {
                    'count': count,
                    'probability': count / total,
                    'objective': obj,
                    'return': ret,
                    'risk': np.sqrt(risk),
                    'solution': solution.tolist()
                }
        
        # Calculate metrics
        feasibility_rate = sum(f['count'] for f in feasible.values()) / total
        
        if feasible:
            # Sort by objective
            sorted_feasible = sorted(feasible.items(), 
                                   key=lambda x: x[1]['objective'],
                                   reverse=True)
            
            best = sorted_feasible[0][1]
            best_prob = best['probability']
            best_obj = best['objective']
            best_sol = best['solution']
            
            # Get top 10
            top_10 = []
            for state, info in sorted_feasible[:10]:
                top_10.append({
                    'state': state,
                    'probability': info['probability'],
                    'objective': info['objective'],
                    'count': info['count'],
                    'return': info['return'],
                    'risk': info['risk']
                })
        else:
            best_prob = 0
            best_obj = -float('inf')
            best_sol = None
            top_10 = []
        
        # Classical comparison
        from itertools import combinations
        classical_best = -float('inf')
        for combo in combinations(range(self.n_assets), self.budget):
            sol = np.zeros(self.n_assets)
            sol[list(combo)] = 1
            weights = sol / self.budget
            ret = np.dot(weights, expected_returns)
            risk = np.dot(weights, np.dot(covariance, weights))
            obj = ret - self.risk_factor * risk
            if obj > classical_best:
                classical_best = obj
        
        # Display results
        print("\n" + "-"*70)
        print("HONEST RESULTS:")
        print(f"Feasibility rate: {feasibility_rate*100:.2f}%")
        print(f"Best solution probability: {best_prob*100:.4f}%")
        print(f"Approximation ratio: {best_obj/classical_best:.4f}")
        print(f"Unique states measured: {len(counts)}")
        print(f"Feasible states found: {len(feasible)}")
        
        if top_10:
            print("\nTop 5 solutions:")
            for i, sol in enumerate(top_10[:5]):
                print(f"  {i+1}. Prob: {sol['probability']*100:.4f}%, "
                      f"Obj: {sol['objective']:.4f}, "
                      f"Count: {sol['count']}/{total}")
        
        return {
            'feasibility_rate': feasibility_rate,
            'best_solution_probability': best_prob,
            'best_objective': best_obj,
            'best_solution': best_sol,
            'classical_best': classical_best,
            'approximation_ratio': best_obj / classical_best if classical_best != 0 else 0,
            'top_10_solutions': top_10,
            'n_unique_states': len(counts),
            'n_feasible_states': len(feasible)
        }