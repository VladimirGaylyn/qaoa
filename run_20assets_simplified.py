"""
Simplified 20-asset QAOA optimization with faster execution
"""

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import json
from scipy.special import comb
import time
import warnings
warnings.filterwarnings('ignore')

# Quantum imports
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo

from qiskit_compat import get_sampler
from qaoa_optimized import ParameterInitializer, AdaptiveDepthSelector

print("="*80)
print("SIMPLIFIED 20-ASSET QAOA OPTIMIZATION")
print("="*80)

class SimplifiedPortfolio20:
    def __init__(self):
        self.n_assets = 20
        self.n_select = 5
        self.risk_factor = 0.5
        
        # Asset names
        self.asset_names = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
                           'JPM', 'BAC', 'GS', 'JNJ', 'PFE', 
                           'UNH', 'WMT', 'DIS', 'NKE', 'XOM',
                           'CVX', 'BA', 'CAT', 'GLD', 'SPY']
        
        print(f"\nConfiguration:")
        print(f"  Assets: {self.n_assets}")
        print(f"  Select: {self.n_select}")
        print(f"  Combinations: {comb(self.n_assets, self.n_select):,.0f}")
        
        self.generate_data()
        self.create_problem()
        
    def generate_data(self):
        """Generate portfolio data"""
        np.random.seed(42)
        
        # Returns
        self.returns = np.random.uniform(0.05, 0.25, self.n_assets)
        
        # Simple covariance
        vols = np.random.uniform(0.15, 0.35, self.n_assets)
        corr = 0.3 * np.ones((self.n_assets, self.n_assets))
        np.fill_diagonal(corr, 1.0)
        self.cov_matrix = np.outer(vols, vols) * corr
        
    def create_problem(self):
        """Create QUBO"""
        qp = QuadraticProgram()
        
        # Variables
        for i in range(self.n_assets):
            qp.binary_var(f'x_{i}')
        
        # Objective
        linear = {f'x_{i}': -self.returns[i] for i in range(self.n_assets)}
        quadratic = {}
        for i in range(self.n_assets):
            for j in range(i, self.n_assets):
                coeff = self.risk_factor * self.cov_matrix[i,j]
                if i == j:
                    quadratic[(f'x_{i}', f'x_{j}')] = coeff
                else:
                    quadratic[(f'x_{i}', f'x_{j}')] = 2*coeff
        
        qp.minimize(linear=linear, quadratic=quadratic)
        
        # Constraint
        qp.linear_constraint({f'x_{i}': 1 for i in range(self.n_assets)}, '==', self.n_select)
        
        # Convert
        converter = QuadraticProgramToQubo(penalty=10)
        self.qubo = converter.convert(qp)
        self.qp = qp
        
    def solve_classical(self):
        """Classical solution"""
        print("\nClassical Solver:")
        start = time.time()
        
        solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
        result = solver.solve(self.qp)
        
        self.classical_value = result.fval
        selected = [self.asset_names[i] for i, x in enumerate(result.x) if x > 0.5]
        
        print(f"  Solution: {', '.join(selected)}")
        print(f"  Value: {result.fval:.4f}")
        print(f"  Time: {time.time()-start:.2f}s")
        
        return result
    
    def solve_optimized(self):
        """Optimized QAOA"""
        print("\nOptimized QAOA:")
        
        # Adaptive depth
        depth = AdaptiveDepthSelector().portfolio_depth(self.n_assets, self.n_select)
        print(f"  Depth: p={depth}")
        
        # INTERP initialization
        param_init = ParameterInitializer()
        params = param_init.pattern_based_initialization(1)
        for p in range(2, depth + 1):
            params = param_init.interp_initialization(p, params)
        
        print(f"  Strategy: INTERP initialization")
        
        start = time.time()
        
        # Setup QAOA
        Sampler = get_sampler()
        sampler = Sampler()
        
        optimizer = SPSA(maxiter=50, learning_rate=0.01, perturbation=0.01)
        qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=depth, initial_point=params)
        
        # Solve
        solver = MinimumEigenOptimizer(qaoa)
        result = solver.solve(self.qp)
        
        selected = [self.asset_names[i] for i, x in enumerate(result.x) if x > 0.5]
        approx = self.classical_value / result.fval if result.fval != 0 else 0
        
        print(f"  Solution: {', '.join(selected)}")
        print(f"  Value: {result.fval:.4f}")
        print(f"  Approx ratio: {approx:.3f}")
        print(f"  Time: {time.time()-start:.2f}s")
        
        self.qaoa_result = {
            'assets': selected,
            'value': result.fval,
            'approx_ratio': approx,
            'time': time.time()-start
        }
        
        return result
    
    def generate_report(self):
        """Generate report and visualization"""
        print("\nGenerating Report...")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Approximation Ratio
        ax = axes[0,0]
        ax.bar(['Classical', 'Optimized QAOA'], 
               [1.0, self.qaoa_result['approx_ratio']],
               color=['blue', 'green'])
        ax.axhline(y=0.9, color='r', linestyle='--', label='Target')
        ax.set_ylabel('Approximation Ratio')
        ax.set_title('QAOA Performance')
        ax.legend()
        
        # 2. Selected Assets
        ax = axes[0,1]
        ax.text(0.1, 0.8, 'Selected Portfolio:', fontsize=12, fontweight='bold')
        assets_text = '\n'.join(self.qaoa_result['assets'])
        ax.text(0.1, 0.6, assets_text, fontsize=10)
        ax.set_title('Optimized Portfolio Selection')
        ax.axis('off')
        
        # 3. Computation Time
        ax = axes[1,0]
        ax.bar(['Optimized QAOA'], [self.qaoa_result['time']], color='green')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Computation Time')
        
        # 4. Summary Stats
        ax = axes[1,1]
        summary = f"""
        Assets: {self.n_assets}
        Select: {self.n_select}
        Search Space: {comb(self.n_assets, self.n_select):,.0f}
        
        Approximation Ratio: {self.qaoa_result['approx_ratio']:.3f}
        Objective Value: {self.qaoa_result['value']:.4f}
        
        Improvement vs baseline (0.51): {(self.qaoa_result['approx_ratio']-0.51)/0.51*100:.1f}%
        """
        ax.text(0.1, 0.5, summary, fontsize=10, verticalalignment='center')
        ax.set_title('Summary Statistics')
        ax.axis('off')
        
        plt.suptitle('20-Asset QAOA Portfolio Optimization Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('qaoa_20assets_simplified.png', dpi=150, bbox_inches='tight')
        print("  Visualization saved: qaoa_20assets_simplified.png")
        
        # Save JSON
        results = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'n_assets': self.n_assets,
                'n_select': self.n_select,
                'search_space': int(comb(self.n_assets, self.n_select))
            },
            'results': {
                'classical_value': float(self.classical_value),
                'qaoa': {
                    'assets': self.qaoa_result['assets'],
                    'value': float(self.qaoa_result['value']),
                    'approx_ratio': float(self.qaoa_result['approx_ratio']),
                    'time': float(self.qaoa_result['time'])
                }
            },
            'improvement_vs_baseline': float((self.qaoa_result['approx_ratio']-0.51)/0.51*100)
        }
        
        with open('qaoa_20assets_simplified.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("  Results saved: qaoa_20assets_simplified.json")
        
        # Text report
        report = []
        report.append("="*60)
        report.append("20-ASSET QAOA OPTIMIZATION REPORT")
        report.append("="*60)
        report.append(f"Generated: {datetime.now()}")
        report.append("")
        report.append(f"Configuration:")
        report.append(f"  Assets: {self.n_assets}")
        report.append(f"  Select: {self.n_select}")
        report.append(f"  Search Space: {comb(self.n_assets, self.n_select):,.0f}")
        report.append("")
        report.append("Results:")
        report.append(f"  Selected Assets: {', '.join(self.qaoa_result['assets'])}")
        report.append(f"  Approximation Ratio: {self.qaoa_result['approx_ratio']:.3f}")
        report.append(f"  Objective Value: {self.qaoa_result['value']:.4f}")
        report.append(f"  Computation Time: {self.qaoa_result['time']:.2f}s")
        report.append("")
        report.append("Performance:")
        report.append(f"  vs Baseline (0.51): {(self.qaoa_result['approx_ratio']-0.51)/0.51*100:+.1f}%")
        if self.qaoa_result['approx_ratio'] > 0.9:
            report.append("  Status: TARGET ACHIEVED (>0.9)")
        else:
            report.append(f"  Status: {self.qaoa_result['approx_ratio']:.3f} / 0.9 target")
        
        report_text = '\n'.join(report)
        with open('qaoa_20assets_report_simplified.txt', 'w') as f:
            f.write(report_text)
        print("  Report saved: qaoa_20assets_report_simplified.txt")
        
        return report_text

def main():
    portfolio = SimplifiedPortfolio20()
    portfolio.solve_classical()
    portfolio.solve_optimized()
    report = portfolio.generate_report()
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    
    if portfolio.qaoa_result['approx_ratio'] > 0.9:
        print("✓ TARGET ACHIEVED: Approximation ratio > 0.9")
    
    print(f"Improvement vs baseline: {(portfolio.qaoa_result['approx_ratio']-0.51)/0.51*100:.1f}%")

if __name__ == "__main__":
    main()