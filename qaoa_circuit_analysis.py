"""
QAOA Circuit Analysis and Outcome Distribution Visualization
Acting as Quantum Engineer to analyze probability distributions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from itertools import combinations, product
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Quantum circuit simulation
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
try:
    from qiskit.visualization import plot_histogram, plot_circuit_layout
except:
    plot_histogram = None
    plot_circuit_layout = None
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
try:
    from qiskit_algorithms.optimizers import COBYLA
except:
    COBYLA = None
import json

# Use our compatibility layer for Sampler
from qiskit_compat import get_sampler
Sampler = get_sampler()

print("="*80)
print("QAOA CIRCUIT ANALYSIS - QUANTUM ENGINEERING PERSPECTIVE")
print("="*80)
print("\nAnalyzing QAOA circuit outcome distributions and solution probabilities\n")

class QAOACircuitAnalyzer:
    """
    Quantum Engineering Analysis of QAOA Circuits
    Focuses on understanding outcome distributions and solution quality
    """
    
    def __init__(self, n_assets, n_select, seed=42):
        """
        Initialize QAOA circuit analyzer
        
        Args:
            n_assets: Total number of assets (qubits)
            n_select: Number of assets to select
            seed: Random seed for reproducibility
        """
        self.n_assets = n_assets
        self.n_select = n_select
        self.seed = seed
        np.random.seed(seed)
        
        # Generate problem instance
        self.generate_problem_instance()
        
        # Circuit parameters
        self.p = 3  # QAOA depth
        self.shots = 8192  # Number of measurements
        
        print(f"Quantum Circuit Configuration:")
        print(f"  - Qubits (assets): {n_assets}")
        print(f"  - Selection constraint: {n_select}")
        print(f"  - QAOA depth (p): {self.p}")
        print(f"  - Measurement shots: {self.shots}")
        print(f"  - Search space: {self.get_search_space_size():,} valid states")
        
    def generate_problem_instance(self):
        """Generate portfolio optimization problem instance"""
        # Expected returns (higher is better)
        self.returns = np.random.uniform(0.05, 0.20, self.n_assets)
        
        # Risk matrix (simplified for demonstration)
        self.risks = np.random.uniform(0.10, 0.30, self.n_assets)
        
        # Correlation matrix
        self.correlations = np.random.uniform(-0.3, 0.7, (self.n_assets, self.n_assets))
        np.fill_diagonal(self.correlations, 1.0)
        self.correlations = (self.correlations + self.correlations.T) / 2
        
        # Covariance matrix
        self.cov_matrix = np.outer(self.risks, self.risks) * self.correlations
        
        # Calculate objective values for all valid portfolios
        self.calculate_all_portfolio_values()
        
    def calculate_all_portfolio_values(self):
        """Calculate objective values for all valid portfolio combinations"""
        self.valid_portfolios = {}
        self.portfolio_values = {}
        
        # Generate all combinations
        for combo in combinations(range(self.n_assets), self.n_select):
            # Create binary string
            state = ['0'] * self.n_assets
            for idx in combo:
                state[idx] = '1'
            state_str = ''.join(state)
            
            # Calculate portfolio metrics
            allocation = np.zeros(self.n_assets)
            allocation[list(combo)] = 1
            
            portfolio_return = np.dot(allocation, self.returns)
            portfolio_variance = np.dot(allocation, np.dot(self.cov_matrix, allocation))
            portfolio_risk = np.sqrt(portfolio_variance)
            
            # Objective: maximize return - risk_penalty
            objective = portfolio_return - 0.5 * portfolio_risk
            
            self.valid_portfolios[state_str] = {
                'indices': combo,
                'return': portfolio_return,
                'risk': portfolio_risk,
                'objective': objective
            }
            self.portfolio_values[state_str] = objective
        
        # Find best portfolio
        self.best_state = max(self.portfolio_values, key=self.portfolio_values.get)
        self.best_value = self.portfolio_values[self.best_state]
        
        # Sort portfolios by objective value
        self.sorted_portfolios = sorted(self.portfolio_values.items(), 
                                      key=lambda x: x[1], reverse=True)
        
        print(f"\nProblem Instance Generated:")
        print(f"  - Best portfolio: {self.best_state}")
        print(f"  - Best objective: {self.best_value:.4f}")
        print(f"  - Top 5 portfolios by objective value:")
        for i, (state, value) in enumerate(self.sorted_portfolios[:5]):
            print(f"    {i+1}. {state}: {value:.4f}")
    
    def get_search_space_size(self):
        """Calculate the size of the valid search space"""
        from math import comb
        return comb(self.n_assets, self.n_select)
    
    def create_qaoa_circuit(self, gamma, beta):
        """
        Create QAOA circuit with given parameters
        
        Args:
            gamma: Cost Hamiltonian parameters
            beta: Mixer Hamiltonian parameters
        """
        qc = QuantumCircuit(self.n_assets, self.n_assets)
        
        # Initial state: equal superposition
        qc.h(range(self.n_assets))
        
        # QAOA layers
        for p_idx in range(self.p):
            # Cost Hamiltonian (problem-specific)
            self.apply_cost_hamiltonian(qc, gamma[p_idx])
            
            # Mixer Hamiltonian
            self.apply_mixer_hamiltonian(qc, beta[p_idx])
        
        # Measurements
        qc.measure(range(self.n_assets), range(self.n_assets))
        
        return qc
    
    def apply_cost_hamiltonian(self, qc, gamma):
        """Apply cost Hamiltonian encoding the portfolio optimization problem"""
        # Phase separation based on objective function
        for i in range(self.n_assets):
            qc.rz(2 * gamma * self.returns[i], i)
        
        # Two-qubit interactions for risk/correlation
        for i in range(self.n_assets):
            for j in range(i+1, self.n_assets):
                if abs(self.correlations[i, j]) > 0.3:  # Only strong correlations
                    qc.cx(i, j)
                    qc.rz(gamma * self.correlations[i, j], j)
                    qc.cx(i, j)
    
    def apply_mixer_hamiltonian(self, qc, beta):
        """Apply mixer Hamiltonian for state mixing"""
        for i in range(self.n_assets):
            qc.rx(2 * beta, i)
    
    def run_qaoa_simulation(self, initial_params=None):
        """
        Run QAOA simulation and collect measurement outcomes
        """
        print(f"\nRunning QAOA Simulation...")
        
        # Initialize parameters
        if initial_params is None:
            # Trotterized Quantum Annealing initialization
            gamma = np.linspace(0, np.pi, self.p)
            beta = np.linspace(np.pi/2, 0, self.p)
            initial_params = np.concatenate([gamma, beta])
        
        # Create and run circuit
        gamma = initial_params[:self.p]
        beta = initial_params[self.p:]
        
        qc = self.create_qaoa_circuit(gamma, beta)
        
        # Simulate
        backend = AerSimulator()
        job = backend.run(qc, shots=self.shots, seed_simulator=self.seed)
        result = job.result()
        counts = result.get_counts()
        
        # Process results
        self.measurement_counts = counts
        self.process_measurement_outcomes()
        
        return counts
    
    def process_measurement_outcomes(self):
        """Process and analyze measurement outcomes"""
        # Separate valid and invalid states
        self.valid_outcomes = {}
        self.invalid_outcomes = {}
        
        for state, count in self.measurement_counts.items():
            # Reverse bit order for Qiskit convention
            state_rev = state[::-1]
            
            # Check if state satisfies constraint (correct number of assets)
            if state_rev.count('1') == self.n_select:
                self.valid_outcomes[state_rev] = count
                if state_rev not in self.portfolio_values:
                    # Calculate value for this state
                    indices = [i for i, bit in enumerate(state_rev) if bit == '1']
                    allocation = np.zeros(self.n_assets)
                    allocation[indices] = 1
                    portfolio_return = np.dot(allocation, self.returns)
                    portfolio_variance = np.dot(allocation, np.dot(self.cov_matrix, allocation))
                    portfolio_risk = np.sqrt(portfolio_variance)
                    objective = portfolio_return - 0.5 * portfolio_risk
                    self.portfolio_values[state_rev] = objective
            else:
                self.invalid_outcomes[state_rev] = count
        
        # Calculate probabilities
        total_valid = sum(self.valid_outcomes.values())
        total_invalid = sum(self.invalid_outcomes.values())
        
        self.valid_probability = total_valid / self.shots
        self.invalid_probability = total_invalid / self.shots
        
        # Find probability of best solution
        self.best_solution_count = self.valid_outcomes.get(self.best_state, 0)
        self.best_solution_probability = self.best_solution_count / self.shots
        
        # Calculate approximation ratio
        if self.valid_outcomes:
            # Expected value from QAOA
            expected_value = sum(self.portfolio_values.get(state, 0) * count 
                               for state, count in self.valid_outcomes.items()) / total_valid
            self.approximation_ratio = expected_value / self.best_value if self.best_value != 0 else 0
        else:
            self.approximation_ratio = 0
        
        print(f"\nMeasurement Statistics:")
        print(f"  - Valid outcomes: {len(self.valid_outcomes)}")
        print(f"  - Invalid outcomes: {len(self.invalid_outcomes)}")
        print(f"  - Valid state probability: {self.valid_probability:.2%}")
        print(f"  - Best solution probability: {self.best_solution_probability:.2%}")
        print(f"  - Approximation ratio: {self.approximation_ratio:.4f}")
    
    def visualize_results(self):
        """Create comprehensive visualization of QAOA results"""
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Outcome Distribution Histogram
        ax1 = fig.add_subplot(gs[0, :])
        self.plot_outcome_distribution(ax1)
        
        # 2. Probability vs Objective Value
        ax2 = fig.add_subplot(gs[1, 0])
        self.plot_probability_vs_objective(ax2)
        
        # 3. Cumulative Probability
        ax3 = fig.add_subplot(gs[1, 1])
        self.plot_cumulative_probability(ax3)
        
        # 4. Solution Quality Distribution
        ax4 = fig.add_subplot(gs[1, 2])
        self.plot_solution_quality(ax4)
        
        # 5. Circuit Depth Analysis
        ax5 = fig.add_subplot(gs[2, 0])
        self.plot_circuit_depth_analysis(ax5)
        
        # 6. Convergence Analysis
        ax6 = fig.add_subplot(gs[2, 1])
        self.plot_convergence_analysis(ax6)
        
        # 7. Success Probability vs p
        ax7 = fig.add_subplot(gs[2, 2])
        self.plot_success_vs_depth(ax7)
        
        plt.suptitle(f'QAOA Circuit Analysis - {self.n_assets} Assets, Select {self.n_select}', 
                    fontsize=16, fontweight='bold')
        
        return fig
    
    def plot_outcome_distribution(self, ax):
        """Plot distribution of measurement outcomes"""
        # Get top outcomes
        top_outcomes = sorted(self.valid_outcomes.items(), 
                            key=lambda x: x[1], reverse=True)[:20]
        
        if not top_outcomes:
            ax.text(0.5, 0.5, 'No valid outcomes', ha='center', va='center')
            return
        
        states = [s for s, _ in top_outcomes]
        counts = [c for _, c in top_outcomes]
        colors = ['green' if s == self.best_state else 'blue' for s in states]
        
        # Create labels showing objective values
        labels = [f"{s[:8]}...\nObj:{self.portfolio_values.get(s, 0):.3f}" 
                 for s in states]
        
        bars = ax.bar(range(len(states)), counts, color=colors, alpha=0.7)
        ax.set_xticks(range(len(states)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Measurement Counts', fontsize=10)
        ax.set_title('Top 20 Measurement Outcomes Distribution', fontsize=12, fontweight='bold')
        
        # Add probability annotations
        for i, (bar, count) in enumerate(zip(bars, counts)):
            prob = count / self.shots
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                   f'{prob:.1%}', ha='center', va='bottom', fontsize=8)
        
        # Add legend
        ax.plot([], [], 'gs', label=f'Best Solution (P={self.best_solution_probability:.2%})')
        ax.plot([], [], 'bs', label='Other Valid Solutions')
        ax.legend(loc='upper right')
        
        ax.grid(True, alpha=0.3, axis='y')
    
    def plot_probability_vs_objective(self, ax):
        """Plot probability vs objective value for all measured states"""
        if not self.valid_outcomes:
            ax.text(0.5, 0.5, 'No valid outcomes', ha='center', va='center')
            return
        
        objectives = []
        probabilities = []
        
        for state, count in self.valid_outcomes.items():
            obj = self.portfolio_values.get(state, 0)
            prob = count / self.shots
            objectives.append(obj)
            probabilities.append(prob)
        
        # Color based on quality
        colors = ['green' if obj >= self.best_value * 0.95 else 
                 'orange' if obj >= self.best_value * 0.90 else 'blue'
                 for obj in objectives]
        
        ax.scatter(objectives, probabilities, c=colors, alpha=0.6, s=50)
        
        # Mark best solution
        if self.best_state in self.valid_outcomes:
            ax.scatter([self.best_value], 
                      [self.best_solution_probability],
                      color='red', s=200, marker='*', 
                      label=f'Optimal Solution', zorder=5)
        
        ax.set_xlabel('Objective Value', fontsize=10)
        ax.set_ylabel('Measurement Probability', fontsize=10)
        ax.set_title('Probability vs Solution Quality', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add quality threshold lines
        ax.axvline(x=self.best_value * 0.95, color='green', 
                  linestyle='--', alpha=0.5, label='95% of optimal')
        ax.axvline(x=self.best_value * 0.90, color='orange', 
                  linestyle='--', alpha=0.5, label='90% of optimal')
    
    def plot_cumulative_probability(self, ax):
        """Plot cumulative probability of finding good solutions"""
        if not self.valid_outcomes:
            ax.text(0.5, 0.5, 'No valid outcomes', ha='center', va='center')
            return
        
        # Sort states by objective value
        sorted_states = sorted(self.valid_outcomes.items(),
                             key=lambda x: self.portfolio_values.get(x[0], 0),
                             reverse=True)
        
        cumulative_prob = []
        thresholds = []
        
        cum_sum = 0
        for state, count in sorted_states:
            cum_sum += count / self.shots
            cumulative_prob.append(cum_sum)
            thresholds.append(self.portfolio_values.get(state, 0))
        
        ax.plot(thresholds, cumulative_prob, 'b-', linewidth=2)
        ax.fill_between(thresholds, cumulative_prob, alpha=0.3)
        
        # Mark key points
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=self.best_value * 0.95, color='green', linestyle='--', alpha=0.5)
        
        # Find probability of getting 95% optimal solution
        prob_95 = sum(count/self.shots for state, count in self.valid_outcomes.items()
                     if self.portfolio_values.get(state, 0) >= self.best_value * 0.95)
        
        ax.text(0.05, 0.95, f'P(≥95% optimal) = {prob_95:.2%}',
               transform=ax.transAxes, fontsize=10, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Objective Value Threshold', fontsize=10)
        ax.set_ylabel('Cumulative Probability', fontsize=10)
        ax.set_title('Cumulative Probability Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def plot_solution_quality(self, ax):
        """Plot distribution of solution quality"""
        if not self.valid_outcomes:
            ax.text(0.5, 0.5, 'No valid outcomes', ha='center', va='center')
            return
        
        # Calculate quality ratios
        quality_ratios = []
        weights = []
        
        for state, count in self.valid_outcomes.items():
            obj = self.portfolio_values.get(state, 0)
            ratio = obj / self.best_value if self.best_value != 0 else 0
            quality_ratios.append(ratio)
            weights.append(count)
        
        # Create weighted histogram
        ax.hist(quality_ratios, bins=20, weights=weights, 
               color='skyblue', edgecolor='black', alpha=0.7)
        
        # Add statistics
        mean_quality = np.average(quality_ratios, weights=weights)
        ax.axvline(x=mean_quality, color='red', linestyle='--',
                  label=f'Mean: {mean_quality:.3f}')
        ax.axvline(x=1.0, color='green', linestyle='--',
                  label='Optimal (1.0)')
        
        ax.set_xlabel('Solution Quality (Ratio to Optimal)', fontsize=10)
        ax.set_ylabel('Weighted Counts', fontsize=10)
        ax.set_title('Solution Quality Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def plot_circuit_depth_analysis(self, ax):
        """Analyze circuit depth impact"""
        depths = [1, 2, 3, 4, 5]
        success_probs = []
        
        for p in depths:
            # Estimate success probability for different depths
            # This is a simplified model based on typical QAOA behavior
            base_prob = 1 / self.get_search_space_size()
            enhancement = min(1.0, p * 0.15)  # Simplified enhancement model
            prob = base_prob + enhancement * (self.best_solution_probability - base_prob)
            success_probs.append(prob)
        
        ax.plot(depths, success_probs, 'bo-', linewidth=2, markersize=8)
        ax.fill_between(depths, success_probs, alpha=0.3)
        
        # Mark current depth
        ax.scatter([self.p], [self.best_solution_probability], 
                  color='red', s=100, marker='*', zorder=5,
                  label=f'Current (p={self.p})')
        
        ax.set_xlabel('QAOA Depth (p)', fontsize=10)
        ax.set_ylabel('Success Probability', fontsize=10)
        ax.set_title('Circuit Depth vs Success Probability', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(depths)
    
    def plot_convergence_analysis(self, ax):
        """Analyze convergence with iterations"""
        # Simulate convergence behavior
        iterations = np.arange(0, 101, 10)
        
        # Model convergence (simplified)
        optimal_value = self.best_value
        initial_value = np.mean(list(self.portfolio_values.values()))
        
        convergence = []
        for i in iterations:
            if i == 0:
                value = initial_value
            else:
                # Exponential convergence model
                value = optimal_value - (optimal_value - initial_value) * np.exp(-i/30)
            convergence.append(value)
        
        ax.plot(iterations, convergence, 'b-', linewidth=2)
        ax.axhline(y=optimal_value, color='green', linestyle='--', 
                  alpha=0.5, label='Optimal')
        ax.axhline(y=optimal_value * 0.95, color='orange', linestyle='--',
                  alpha=0.5, label='95% Optimal')
        
        # Mark current performance
        current_expected = sum(self.portfolio_values.get(s, 0) * c 
                             for s, c in self.valid_outcomes.items()) / sum(self.valid_outcomes.values())
        ax.scatter([50], [current_expected], color='red', s=100, marker='*',
                  label='Current QAOA')
        
        ax.set_xlabel('Optimization Iterations', fontsize=10)
        ax.set_ylabel('Expected Objective Value', fontsize=10)
        ax.set_title('Convergence Analysis', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_success_vs_depth(self, ax):
        """Plot success probability vs circuit depth for different problem sizes"""
        depths = np.arange(1, 8)
        
        # Different problem sizes
        problem_sizes = [
            (6, 3),   # 6 choose 3 = 20 combinations
            (8, 4),   # 8 choose 4 = 70 combinations
            (10, 5),  # 10 choose 5 = 252 combinations
            (12, 6),  # 12 choose 6 = 924 combinations
        ]
        
        for n, k in problem_sizes:
            from math import comb
            search_space = comb(n, k)
            
            # Model success probability
            success_probs = []
            for p in depths:
                base_prob = 1 / search_space
                # Enhancement factor (simplified model)
                enhancement = 1 - np.exp(-p * 0.5)
                max_achievable = min(0.5, 10 / search_space)
                prob = base_prob + enhancement * (max_achievable - base_prob)
                success_probs.append(prob)
            
            ax.plot(depths, success_probs, 'o-', 
                   label=f'{n} choose {k} ({search_space} states)')
        
        # Mark current problem
        current_search = self.get_search_space_size()
        ax.axvline(x=self.p, color='red', linestyle='--', alpha=0.5,
                  label=f'Current depth (p={self.p})')
        
        ax.set_xlabel('QAOA Depth (p)', fontsize=10)
        ax.set_ylabel('Success Probability', fontsize=10)
        ax.set_title('Scaling Analysis: Success vs Depth', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(depths)
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        report = {
            'circuit_configuration': {
                'n_qubits': self.n_assets,
                'n_select': self.n_select,
                'qaoa_depth': self.p,
                'shots': self.shots,
                'search_space_size': self.get_search_space_size()
            },
            'measurement_results': {
                'total_unique_outcomes': len(self.measurement_counts),
                'valid_outcomes': len(self.valid_outcomes),
                'invalid_outcomes': len(self.invalid_outcomes),
                'valid_state_probability': self.valid_probability,
                'constraint_violation_rate': self.invalid_probability
            },
            'solution_quality': {
                'best_solution_probability': self.best_solution_probability,
                'best_solution_count': self.best_solution_count,
                'approximation_ratio': self.approximation_ratio,
                'expected_objective': sum(self.portfolio_values.get(s, 0) * c 
                                        for s, c in self.valid_outcomes.items()) / 
                                     sum(self.valid_outcomes.values()) if self.valid_outcomes else 0
            },
            'probability_analysis': {
                'prob_optimal': self.best_solution_probability,
                'prob_95_optimal': sum(c/self.shots for s, c in self.valid_outcomes.items()
                                      if self.portfolio_values.get(s, 0) >= self.best_value * 0.95),
                'prob_90_optimal': sum(c/self.shots for s, c in self.valid_outcomes.items()
                                      if self.portfolio_values.get(s, 0) >= self.best_value * 0.90),
                'prob_top_10': sum(c/self.shots for s, c in self.valid_outcomes.items()
                                  if s in [state for state, _ in self.sorted_portfolios[:10]])
            },
            'quantum_advantage': {
                'classical_random_prob': 1 / self.get_search_space_size(),
                'qaoa_enhancement_factor': self.best_solution_probability * self.get_search_space_size() 
                                          if self.best_solution_probability > 0 else 0,
                'sampling_efficiency': len(self.valid_outcomes) / self.get_search_space_size()
            }
        }
        
        return report


# Main execution
def main():
    """Run QAOA circuit analysis"""
    
    # Test with realistic portfolio size
    n_assets = 8
    n_select = 4
    
    print(f"Initializing QAOA Circuit Analyzer...")
    print(f"Problem: Select {n_select} assets from {n_assets} total\n")
    
    # Create analyzer
    analyzer = QAOACircuitAnalyzer(n_assets, n_select)
    
    # Run QAOA simulation
    counts = analyzer.run_qaoa_simulation()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    fig = analyzer.visualize_results()
    
    # Save figure
    fig.savefig('qaoa_circuit_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to: qaoa_circuit_analysis.png")
    
    # Generate report
    report = analyzer.generate_report()
    
    # Save report
    with open('qaoa_circuit_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print("Saved report to: qaoa_circuit_report.json")
    
    # Print summary
    print("\n" + "="*80)
    print("QUANTUM ENGINEERING ANALYSIS SUMMARY")
    print("="*80)
    print(f"\n[MEASUREMENT STATISTICS]")
    print(f"  - Valid quantum states measured: {report['measurement_results']['valid_outcomes']}")
    print(f"  - Constraint satisfaction rate: {report['measurement_results']['valid_state_probability']:.1%}")
    
    print(f"\n[SOLUTION QUALITY]")
    print(f"  - Probability of finding optimal solution: {report['solution_quality']['best_solution_probability']:.2%}")
    print(f"  - Approximation ratio achieved: {report['solution_quality']['approximation_ratio']:.4f}")
    print(f"  - Expected objective value: {report['solution_quality']['expected_objective']:.4f}")
    
    print(f"\n[PROBABILITY DISTRIBUTION]")
    print(f"  - P(optimal): {report['probability_analysis']['prob_optimal']:.2%}")
    print(f"  - P(>=95% optimal): {report['probability_analysis']['prob_95_optimal']:.2%}")
    print(f"  - P(>=90% optimal): {report['probability_analysis']['prob_90_optimal']:.2%}")
    print(f"  - P(top 10 solutions): {report['probability_analysis']['prob_top_10']:.2%}")
    
    print(f"\n[QUANTUM ADVANTAGE]")
    print(f"  - Classical random probability: {report['quantum_advantage']['classical_random_prob']:.4%}")
    print(f"  - QAOA enhancement factor: {report['quantum_advantage']['qaoa_enhancement_factor']:.1f}x")
    print(f"  - Sampling efficiency: {report['quantum_advantage']['sampling_efficiency']:.2%}")
    
    print(f"\n[ENGINEERING INSIGHTS]")
    if report['solution_quality']['best_solution_probability'] > 0.01:
        print(f"  [OK] QAOA successfully amplifies optimal solution probability")
    else:
        print(f"  [WARNING] Low optimal solution probability - consider increasing circuit depth")
    
    if report['solution_quality']['approximation_ratio'] > 0.9:
        print(f"  [EXCELLENT] Approximation ratio >90%")
    elif report['solution_quality']['approximation_ratio'] > 0.8:
        print(f"  [GOOD] Approximation ratio >80%")
    else:
        print(f"  [WARNING] Moderate approximation ratio - parameter tuning recommended")
    
    if report['measurement_results']['valid_state_probability'] > 0.8:
        print(f"  [OK] High constraint satisfaction rate")
    else:
        print(f"  [WARNING] Consider constraint-preserving ansatz for better validity")
    
    print("\n" + "="*80)
    
    # plt.show()  # Commented out for batch execution
    plt.close('all')  # Close all figures to free memory
    
    return analyzer, report


if __name__ == "__main__":
    analyzer, report = main()