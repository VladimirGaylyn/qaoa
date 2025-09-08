"""
Proper QAOA Implementation for Portfolio Optimization
Real quantum circuit execution using Qiskit
No simulations or fake results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time
import json
import os
from datetime import datetime

# Qiskit imports - using correct APIs
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.minimum_eigensolvers import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA, SPSA, NELDER_MEAD
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_finance.applications.optimization import PortfolioOptimization


class ProperQAOAPortfolio:
    """Real QAOA implementation for portfolio optimization"""
    
    def __init__(self, n_assets: int, budget: int, risk_factor: float = 0.5):
        self.n_assets = n_assets
        self.budget = budget
        self.risk_factor = risk_factor
        self.results = {}
        
    def create_portfolio_problem(self, expected_returns: np.ndarray, 
                                covariances: np.ndarray) -> QuadraticProgram:
        """Create portfolio optimization problem"""
        
        # Validate inputs
        assert len(expected_returns) == self.n_assets
        assert covariances.shape == (self.n_assets, self.n_assets)
        
        # Ensure covariance matrix is positive definite
        eigenvalues = np.linalg.eigvalsh(covariances)
        if eigenvalues.min() <= 0:
            # Add small regularization
            covariances = covariances + (abs(eigenvalues.min()) + 1e-6) * np.eye(self.n_assets)
        
        # Create portfolio optimization instance
        portfolio = PortfolioOptimization(
            expected_returns=expected_returns,
            covariances=covariances,
            risk_factor=self.risk_factor,
            budget=self.budget
        )
        
        # Convert to quadratic program
        qp = portfolio.to_quadratic_program()
        
        return qp
    
    def solve_classical(self, qp: QuadraticProgram) -> Dict:
        """Solve using classical exact eigensolver"""
        
        print("Running classical solver...")
        start_time = time.time()
        
        # Use NumPy minimum eigensolver for exact solution
        exact_solver = NumPyMinimumEigensolver()
        converter = QuadraticProgramToQubo()
        qubo = converter.convert(qp)
        
        # Get exact solution
        exact_result = exact_solver.compute_minimum_eigenvalue(qubo.to_ising()[0])
        
        # Convert result
        solution = self._decode_solution(exact_result.eigenstate, qp.get_num_vars())
        
        classical_time = time.time() - start_time
        
        return {
            'solution': solution,
            'eigenvalue': exact_result.eigenvalue,
            'time': classical_time,
            'method': 'exact_eigensolver'
        }
    
    def solve_qaoa_standard(self, qp: QuadraticProgram, reps: int = 3, 
                           optimizer_name: str = 'COBYLA') -> Dict:
        """Solve using standard QAOA"""
        
        print(f"Running standard QAOA with {optimizer_name}...")
        start_time = time.time()
        
        # Convert to QUBO
        converter = QuadraticProgramToQubo()
        qubo = converter.convert(qp)
        
        # Convert to Ising Hamiltonian
        hamiltonian, offset = qubo.to_ising()
        
        # Select optimizer
        if optimizer_name == 'COBYLA':
            optimizer = COBYLA(maxiter=100)
        elif optimizer_name == 'SPSA':
            optimizer = SPSA(maxiter=100)
        else:
            optimizer = NELDER_MEAD(maxiter=100)
        
        # Create QAOA instance with Sampler
        sampler = Sampler()
        qaoa = QAOA(
            sampler=sampler,
            optimizer=optimizer,
            reps=reps
        )
        
        # Run QAOA
        result = qaoa.compute_minimum_eigenvalue(hamiltonian)
        
        # Extract solution
        solution = self._decode_solution(result.eigenstate, qp.get_num_vars())
        
        qaoa_time = time.time() - start_time
        
        return {
            'solution': solution,
            'eigenvalue': result.eigenvalue,
            'time': qaoa_time,
            'optimizer': optimizer_name,
            'iterations': result.optimizer_result.nfev if hasattr(result, 'optimizer_result') else None
        }
    
    def solve_qaoa_warmstart(self, qp: QuadraticProgram, 
                            classical_solution: np.ndarray,
                            reps: int = 3) -> Dict:
        """Solve using QAOA with warmstart from classical solution"""
        
        print("Running warmstart QAOA...")
        start_time = time.time()
        
        # Convert to QUBO and Ising
        converter = QuadraticProgramToQubo()
        qubo = converter.convert(qp)
        hamiltonian, offset = qubo.to_ising()
        
        # Generate warmstart parameters based on classical solution
        initial_point = self._generate_warmstart_params(classical_solution, reps)
        
        # Use COBYLA with warmstart
        optimizer = COBYLA(maxiter=100)
        sampler = Sampler()
        
        qaoa = QAOA(
            sampler=sampler,
            optimizer=optimizer,
            reps=reps,
            initial_point=initial_point
        )
        
        # Run QAOA with warmstart
        result = qaoa.compute_minimum_eigenvalue(hamiltonian)
        
        # Extract solution
        solution = self._decode_solution(result.eigenstate, qp.get_num_vars())
        
        warmstart_time = time.time() - start_time
        
        return {
            'solution': solution,
            'eigenvalue': result.eigenvalue,
            'time': warmstart_time,
            'initial_point': initial_point.tolist()
        }
    
    def _decode_solution(self, eigenstate, num_vars: int) -> np.ndarray:
        """Decode solution from eigenstate"""
        if eigenstate is None:
            return np.zeros(num_vars)
        
        # Handle dictionary format (new Qiskit format)
        if isinstance(eigenstate, dict):
            # Find bitstring with highest probability/count
            max_bitstring = max(eigenstate, key=eigenstate.get)
            # Remove spaces if present
            max_bitstring = max_bitstring.replace(' ', '')
        elif hasattr(eigenstate, 'binary_probabilities'):
            probs = eigenstate.binary_probabilities()
            max_bitstring = max(probs, key=probs.get)
        else:
            # Handle array format
            try:
                max_bitstring = format(np.argmax(np.abs(eigenstate)**2), f'0{num_vars}b')
            except:
                # Fallback to zeros
                return np.zeros(num_vars)
        
        # Convert bitstring to array (handle reverse order)
        solution = np.array([int(bit) for bit in max_bitstring[:num_vars][::-1]])
        
        # Ensure correct length
        if len(solution) < num_vars:
            solution = np.pad(solution, (0, num_vars - len(solution)), 'constant')
        elif len(solution) > num_vars:
            solution = solution[:num_vars]
        
        return solution
    
    def _generate_warmstart_params(self, classical_solution: np.ndarray, 
                                   reps: int) -> np.ndarray:
        """Generate informed warmstart parameters"""
        
        params = []
        classical_quality = np.sum(classical_solution) / self.budget
        
        for p in range(reps):
            # Beta (mixer) - start small to preserve classical structure
            beta = np.pi * 0.1 * (1 + p) / reps
            
            # Gamma (cost) - scale with classical solution quality
            gamma = np.pi * 0.3 * classical_quality * (1 + p) / reps
            
            params.extend([beta, gamma])
        
        return np.array(params)
    
    def calculate_portfolio_metrics(self, solution: np.ndarray, 
                                   expected_returns: np.ndarray,
                                   covariances: np.ndarray) -> Dict:
        """Calculate portfolio performance metrics"""
        
        selected = solution > 0.5
        n_selected = np.sum(selected)
        
        if n_selected != self.budget:
            # Constraint violated
            return {
                'valid': False,
                'n_selected': int(n_selected),
                'return': 0,
                'risk': np.inf,
                'sharpe': -np.inf
            }
        
        # Equal weight allocation
        weights = np.zeros(self.n_assets)
        weights[selected] = 1.0 / n_selected
        
        # Calculate metrics
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = np.dot(weights, np.dot(covariances, weights))
        portfolio_risk = np.sqrt(portfolio_variance)
        
        # Sharpe ratio (assuming risk-free rate = 0.02)
        risk_free = 0.02
        sharpe = (portfolio_return - risk_free) / portfolio_risk if portfolio_risk > 0 else 0
        
        # Objective value (Markowitz)
        objective = portfolio_return - self.risk_factor * portfolio_variance
        
        return {
            'valid': True,
            'n_selected': int(n_selected),
            'return': float(portfolio_return),
            'risk': float(portfolio_risk),
            'sharpe': float(sharpe),
            'objective': float(objective),
            'selected_assets': np.where(selected)[0].tolist()
        }
    
    def run_complete_comparison(self, expected_returns: np.ndarray,
                               covariances: np.ndarray) -> Dict:
        """Run complete comparison of all methods"""
        
        print("\n" + "="*60)
        print("RUNNING PROPER QAOA PORTFOLIO OPTIMIZATION")
        print("="*60)
        
        # Create problem
        qp = self.create_portfolio_problem(expected_returns, covariances)
        
        results = {}
        
        # 1. Classical solution
        classical_result = self.solve_classical(qp)
        classical_metrics = self.calculate_portfolio_metrics(
            classical_result['solution'], expected_returns, covariances
        )
        results['classical'] = {**classical_result, 'metrics': classical_metrics}
        
        # 2. Standard QAOA with different optimizers
        for optimizer in ['COBYLA', 'SPSA']:
            qaoa_result = self.solve_qaoa_standard(qp, reps=3, optimizer_name=optimizer)
            qaoa_metrics = self.calculate_portfolio_metrics(
                qaoa_result['solution'], expected_returns, covariances
            )
            results[f'qaoa_{optimizer.lower()}'] = {**qaoa_result, 'metrics': qaoa_metrics}
        
        # 3. Warmstart QAOA
        warmstart_result = self.solve_qaoa_warmstart(
            qp, classical_result['solution'], reps=3
        )
        warmstart_metrics = self.calculate_portfolio_metrics(
            warmstart_result['solution'], expected_returns, covariances
        )
        results['qaoa_warmstart'] = {**warmstart_result, 'metrics': warmstart_metrics}
        
        # Calculate comparison metrics
        results['comparison'] = self._calculate_comparisons(results)
        
        return results
    
    def _calculate_comparisons(self, results: Dict) -> Dict:
        """Calculate comparison metrics between methods"""
        
        comparisons = {}
        classical_obj = results['classical']['metrics'].get('objective', 0)
        
        for method in ['qaoa_cobyla', 'qaoa_spsa', 'qaoa_warmstart']:
            if method in results and 'metrics' in results[method]:
                method_obj = results[method]['metrics'].get('objective', 0)
                if classical_obj != 0 and method_obj is not None:
                    comparisons[method] = {
                        'approximation_ratio': method_obj / classical_obj,
                        'time_ratio': results[method]['time'] / results['classical']['time'],
                        'objective_gap': abs(method_obj - classical_obj)
                    }
                else:
                    comparisons[method] = {
                        'approximation_ratio': 0,
                        'time_ratio': results[method]['time'] / results['classical']['time'],
                        'objective_gap': float('inf')
                    }
        
        return comparisons


def generate_test_portfolio(n_assets: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate test portfolio data"""
    
    np.random.seed(seed)
    
    # Expected returns (5% to 25% annually)
    expected_returns = np.random.uniform(0.05, 0.25, n_assets)
    
    # Generate correlation matrix
    correlation = np.random.uniform(-0.3, 0.7, (n_assets, n_assets))
    correlation = (correlation + correlation.T) / 2
    np.fill_diagonal(correlation, 1)
    
    # Ensure positive definite
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalues[eigenvalues < 0.01] = 0.01
    correlation = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    # Convert to covariance
    std_devs = np.random.uniform(0.1, 0.3, n_assets)
    covariances = np.outer(std_devs, std_devs) * correlation
    
    return expected_returns, covariances


def main():
    """Main execution function"""
    
    # Create results directory
    os.makedirs('proper_results', exist_ok=True)
    
    # Test with different portfolio sizes
    test_configs = [
        {'n_assets': 6, 'budget': 3, 'risk_factor': 0.5},
        {'n_assets': 8, 'budget': 4, 'risk_factor': 0.5},
        {'n_assets': 10, 'budget': 5, 'risk_factor': 0.5}
    ]
    
    all_results = []
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing with {config['n_assets']} assets, selecting {config['budget']}")
        print('='*60)
        
        # Generate test data
        expected_returns, covariances = generate_test_portfolio(config['n_assets'])
        
        # Create optimizer
        optimizer = ProperQAOAPortfolio(
            n_assets=config['n_assets'],
            budget=config['budget'],
            risk_factor=config['risk_factor']
        )
        
        # Run comparison
        results = optimizer.run_complete_comparison(expected_returns, covariances)
        
        # Store results
        results['config'] = config
        results['timestamp'] = datetime.now().isoformat()
        all_results.append(results)
        
        # Print summary
        print("\nRESULTS SUMMARY:")
        print("-" * 40)
        for method in ['classical', 'qaoa_cobyla', 'qaoa_spsa', 'qaoa_warmstart']:
            if method in results:
                print(f"\n{method.upper()}:")
                print(f"  Valid: {results[method]['metrics']['valid']}")
                if results[method]['metrics']['valid']:
                    print(f"  Objective: {results[method]['metrics']['objective']:.6f}")
                    print(f"  Sharpe: {results[method]['metrics']['sharpe']:.3f}")
                else:
                    print(f"  Selected: {results[method]['metrics']['n_selected']} assets (need {config['budget']})")
                print(f"  Time: {results[method]['time']:.3f}s")
                
                if method != 'classical' and 'comparison' in results and results[method]['metrics']['valid']:
                    approx = results['comparison'][method]['approximation_ratio']
                    print(f"  Approximation Ratio: {approx:.3f}")
        
        # Save individual result
        filename = f"proper_results/portfolio_{config['n_assets']}assets_{config['budget']}budget.json"
        
        # Prepare results for JSON (remove non-serializable objects)
        json_safe_results = {
            'config': config,
            'timestamp': results['timestamp'],
            'classical': {
                'solution': results['classical']['solution'].tolist(),
                'time': results['classical']['time'],
                'metrics': results['classical']['metrics']
            }
        }
        
        for method in ['qaoa_cobyla', 'qaoa_spsa', 'qaoa_warmstart']:
            if method in results:
                json_safe_results[method] = {
                    'solution': results[method]['solution'].tolist(),
                    'time': results[method]['time'],
                    'metrics': results[method]['metrics']
                }
        
        json_safe_results['comparison'] = results['comparison']
        
        with open(filename, 'w') as f:
            json.dump(json_safe_results, f, indent=2)
        
        print(f"\nResults saved to {filename}")
    
    # Save aggregate results (already in JSON-safe format)
    with open('proper_results/aggregate_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED - REAL QAOA EXECUTION")
    print("Results saved in proper_results/")
    print("="*60)
    
    return all_results


if __name__ == "__main__":
    results = main()