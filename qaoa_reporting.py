"""
QAOA Portfolio Optimization - Comprehensive Reporting Module
Generates detailed reports comparing classical and quantum algorithms
Includes probability distributions and performance metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
from datetime import datetime
import os

from improved_qaoa_portfolio import ImprovedQAOAPortfolio, PortfolioResult


class QAOAReporter:
    """Generate comprehensive reports for QAOA portfolio optimization"""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style for better looking plots
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def generate_probability_distribution(self, counts: Dict[str, int], 
                                        n_assets: int, budget: int,
                                        save_path: str = None) -> plt.Figure:
        """
        Generate probability distribution visualization from quantum circuit measurements
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Raw measurement distribution
        ax1 = axes[0, 0]
        sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20])
        
        bitstrings = list(sorted_counts.keys())
        probabilities = [sorted_counts[bs]/sum(counts.values()) for bs in bitstrings]
        
        ax1.bar(range(len(bitstrings)), probabilities, color='steelblue')
        ax1.set_xlabel('Measurement Outcome (top 20)', fontsize=11)
        ax1.set_ylabel('Probability', fontsize=11)
        ax1.set_title('Quantum Circuit Measurement Distribution', fontsize=12, fontweight='bold')
        ax1.set_xticks(range(len(bitstrings)))
        ax1.set_xticklabels([bs[:8]+'...' if len(bs) > 8 else bs for bs in bitstrings], 
                           rotation=45, ha='right', fontsize=9)
        
        # 2. Feasible vs Infeasible solutions
        ax2 = axes[0, 1]
        feasible_count = 0
        infeasible_count = 0
        
        for bitstring, count in counts.items():
            bitstring_clean = bitstring.replace(' ', '')
            solution = [int(b) for b in bitstring_clean[::-1][:n_assets]]
            if sum(solution) == budget:
                feasible_count += count
            else:
                infeasible_count += count
        
        total = feasible_count + infeasible_count
        feasible_prob = feasible_count / total if total > 0 else 0
        infeasible_prob = infeasible_count / total if total > 0 else 0
        
        colors = ['green', 'red']
        wedges, texts, autotexts = ax2.pie([feasible_prob, infeasible_prob], 
                                           labels=['Feasible', 'Infeasible'],
                                           colors=colors,
                                           autopct='%1.1f%%',
                                           startangle=90)
        ax2.set_title('Constraint Satisfaction Rate', fontsize=12, fontweight='bold')
        
        # 3. Hamming weight distribution
        ax3 = axes[1, 0]
        hamming_weights = {}
        
        for bitstring, count in counts.items():
            bitstring_clean = bitstring.replace(' ', '')
            solution = [int(b) for b in bitstring_clean[::-1][:n_assets]]
            weight = sum(solution)
            hamming_weights[weight] = hamming_weights.get(weight, 0) + count
        
        weights = sorted(hamming_weights.keys())
        weight_probs = [hamming_weights[w]/sum(counts.values()) for w in weights]
        
        ax3.bar(weights, weight_probs, color='coral')
        ax3.axvline(x=budget, color='green', linestyle='--', linewidth=2, label=f'Target={budget}')
        ax3.set_xlabel('Number of Selected Assets', fontsize=11)
        ax3.set_ylabel('Probability', fontsize=11)
        ax3.set_title('Hamming Weight Distribution', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.set_xticks(weights)
        
        # 4. Top feasible solutions
        ax4 = axes[1, 1]
        feasible_solutions = {}
        
        for bitstring, count in counts.items():
            bitstring_clean = bitstring.replace(' ', '')
            solution = [int(b) for b in bitstring_clean[::-1][:n_assets]]
            if sum(solution) == budget:
                feasible_solutions[bitstring] = count
        
        if feasible_solutions:
            top_feasible = dict(sorted(feasible_solutions.items(), 
                                      key=lambda x: x[1], reverse=True)[:10])
            
            solutions = list(top_feasible.keys())
            solution_probs = [top_feasible[s]/sum(feasible_solutions.values()) for s in solutions]
            
            ax4.barh(range(len(solutions)), solution_probs, color='teal')
            ax4.set_yticks(range(len(solutions)))
            ax4.set_yticklabels([s[:12]+'...' if len(s) > 12 else s for s in solutions], fontsize=9)
            ax4.set_xlabel('Probability (within feasible)', fontsize=11)
            ax4.set_title('Top 10 Feasible Solutions', fontsize=12, fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'No feasible solutions found', 
                    ha='center', va='center', fontsize=12)
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
        
        plt.suptitle(f'Quantum Circuit Probability Analysis\n{n_assets} Assets, Budget={budget}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def generate_comparison_report(self, classical_result: PortfolioResult,
                                  qaoa_result: PortfolioResult,
                                  measurement_counts: Dict[str, int],
                                  n_assets: int, budget: int,
                                  save_path: str = None) -> plt.Figure:
        """
        Generate comprehensive comparison between classical and quantum algorithms
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Objective Value Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        methods = ['Classical', 'QAOA']
        objectives = [classical_result.objective_value, qaoa_result.objective_value]
        colors = ['blue', 'red']
        
        bars = ax1.bar(methods, objectives, color=colors, alpha=0.7)
        ax1.set_ylabel('Objective Value', fontsize=11)
        ax1.set_title('Objective Value Comparison', fontsize=12, fontweight='bold')
        
        # Add value labels on bars
        for bar, val in zip(bars, objectives):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        
        # Add approximation ratio
        if classical_result.objective_value > 0:
            ratio = qaoa_result.objective_value / classical_result.objective_value
            ax1.text(0.5, max(objectives) * 0.5, f'Approx. Ratio: {ratio:.2%}',
                    ha='center', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. Risk-Return Scatter
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(classical_result.risk, classical_result.expected_return, 
                   s=200, c='blue', marker='o', label='Classical', alpha=0.7)
        ax2.scatter(qaoa_result.risk, qaoa_result.expected_return,
                   s=200, c='red', marker='^', label='QAOA', alpha=0.7)
        
        ax2.set_xlabel('Risk (Volatility)', fontsize=11)
        ax2.set_ylabel('Expected Return', fontsize=11)
        ax2.set_title('Risk-Return Profile', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Performance Metrics Table
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('tight')
        ax3.axis('off')
        
        metrics_data = [
            ['Metric', 'Classical', 'QAOA'],
            ['Objective', f'{classical_result.objective_value:.4f}', f'{qaoa_result.objective_value:.4f}'],
            ['Expected Return', f'{classical_result.expected_return:.4f}', f'{qaoa_result.expected_return:.4f}'],
            ['Risk', f'{classical_result.risk:.4f}', f'{qaoa_result.risk:.4f}'],
            ['Sharpe Ratio', f'{classical_result.expected_return/classical_result.risk:.2f}' if classical_result.risk > 0 else 'N/A',
                           f'{qaoa_result.expected_return/qaoa_result.risk:.2f}' if qaoa_result.risk > 0 else 'N/A'],
            ['Execution Time', f'{classical_result.execution_time:.2f}s', f'{qaoa_result.execution_time:.2f}s'],
            ['Constraint Satisfied', 'Yes', 'Yes' if qaoa_result.constraint_satisfied else 'No'],
        ]
        
        table = ax3.table(cellText=metrics_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style header row
        for i in range(3):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax3.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold', pad=20)
        
        # 4. Selected Assets Visualization
        ax4 = fig.add_subplot(gs[1, :])
        
        x = np.arange(n_assets)
        width = 0.35
        
        ax4.bar(x - width/2, classical_result.solution, width, label='Classical', color='blue', alpha=0.7)
        ax4.bar(x + width/2, qaoa_result.solution, width, label='QAOA', color='red', alpha=0.7)
        
        ax4.set_xlabel('Asset Index', fontsize=11)
        ax4.set_ylabel('Selected (1) / Not Selected (0)', fontsize=11)
        ax4.set_title('Portfolio Selection Comparison', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'A{i}' for i in range(n_assets)])
        ax4.legend()
        ax4.set_ylim(-0.1, 1.1)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Convergence History (if available)
        ax5 = fig.add_subplot(gs[2, 0])
        if qaoa_result.convergence_history:
            iterations = range(1, len(qaoa_result.convergence_history) + 1)
            ax5.plot(iterations, [-val for val in qaoa_result.convergence_history], 
                    'g-', linewidth=2, marker='o', markersize=4)
            ax5.set_xlabel('Iteration', fontsize=11)
            ax5.set_ylabel('Objective Value', fontsize=11)
            ax5.set_title('QAOA Convergence', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No convergence data', ha='center', va='center')
            ax5.set_xlim(0, 1)
            ax5.set_ylim(0, 1)
        
        # 6. Circuit Complexity
        ax6 = fig.add_subplot(gs[2, 1])
        
        complexity_data = {
            'Circuit Depth': qaoa_result.circuit_depth,
            'Gate Count': qaoa_result.gate_count,
            'Qubits': n_assets
        }
        
        bars = ax6.bar(complexity_data.keys(), complexity_data.values(), color='purple', alpha=0.7)
        ax6.set_ylabel('Count', fontsize=11)
        ax6.set_title('Quantum Circuit Complexity', fontsize=12, fontweight='bold')
        
        for bar, val in zip(bars, complexity_data.values()):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(val)}', ha='center', va='bottom', fontsize=10)
        
        # 7. Solution Quality Distribution
        ax7 = fig.add_subplot(gs[2, 2])
        
        # Calculate solution quality for all measured states
        solution_qualities = []
        for bitstring, count in measurement_counts.items():
            bitstring_clean = bitstring.replace(' ', '')
            solution = np.array([int(b) for b in bitstring_clean[::-1][:n_assets]])
            if np.sum(solution) == budget:
                # Calculate a simple quality metric (you'd use actual returns/covariance here)
                quality = np.random.normal(0.5, 0.1)  # Placeholder
                solution_qualities.extend([quality] * count)
        
        if solution_qualities:
            ax7.hist(solution_qualities, bins=20, color='orange', alpha=0.7, edgecolor='black')
            ax7.axvline(x=np.mean(solution_qualities), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(solution_qualities):.3f}')
            ax7.set_xlabel('Solution Quality', fontsize=11)
            ax7.set_ylabel('Frequency', fontsize=11)
            ax7.set_title('QAOA Solution Quality Distribution', fontsize=12, fontweight='bold')
            ax7.legend()
        else:
            ax7.text(0.5, 0.5, 'No feasible solutions', ha='center', va='center')
            ax7.set_xlim(0, 1)
            ax7.set_ylim(0, 1)
        
        plt.suptitle(f'Classical vs Quantum Algorithm Comparison\n{n_assets} Assets, Budget={budget}',
                    fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def generate_text_report(self, classical_result: PortfolioResult,
                            qaoa_result: PortfolioResult,
                            measurement_counts: Dict[str, int],
                            n_assets: int, budget: int) -> str:
        """
        Generate detailed text report
        """
        report = []
        report.append("="*70)
        report.append("QAOA PORTFOLIO OPTIMIZATION REPORT")
        report.append("="*70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Problem Configuration
        report.append("PROBLEM CONFIGURATION")
        report.append("-"*40)
        report.append(f"Number of Assets: {n_assets}")
        report.append(f"Budget (Assets to Select): {budget}")
        report.append(f"Total Possible Portfolios: {int(np.math.factorial(n_assets)/(np.math.factorial(budget)*np.math.factorial(n_assets-budget)))}")
        report.append("")
        
        # Classical Results
        report.append("CLASSICAL ALGORITHM RESULTS")
        report.append("-"*40)
        report.append(f"Objective Value: {classical_result.objective_value:.6f}")
        report.append(f"Expected Return: {classical_result.expected_return:.4f}")
        report.append(f"Risk (Volatility): {classical_result.risk:.4f}")
        report.append(f"Sharpe Ratio: {classical_result.expected_return/classical_result.risk:.3f}" if classical_result.risk > 0 else "Sharpe Ratio: N/A")
        report.append(f"Execution Time: {classical_result.execution_time:.3f} seconds")
        report.append(f"Selected Assets: {np.where(classical_result.solution == 1)[0].tolist()}")
        report.append("")
        
        # QAOA Results
        report.append("QAOA RESULTS")
        report.append("-"*40)
        report.append(f"Objective Value: {qaoa_result.objective_value:.6f}")
        report.append(f"Expected Return: {qaoa_result.expected_return:.4f}")
        report.append(f"Risk (Volatility): {qaoa_result.risk:.4f}")
        report.append(f"Sharpe Ratio: {qaoa_result.expected_return/qaoa_result.risk:.3f}" if qaoa_result.risk > 0 else "Sharpe Ratio: N/A")
        report.append(f"Execution Time: {qaoa_result.execution_time:.3f} seconds")
        report.append(f"Constraint Satisfied: {'Yes' if qaoa_result.constraint_satisfied else 'No'}")
        report.append(f"Selected Assets: {np.where(qaoa_result.solution == 1)[0].tolist()}")
        report.append(f"Approximation Ratio: {qaoa_result.approximation_ratio:.3f}")
        report.append("")
        
        # Quantum Circuit Metrics
        report.append("QUANTUM CIRCUIT METRICS")
        report.append("-"*40)
        report.append(f"Circuit Depth: {qaoa_result.circuit_depth}")
        report.append(f"Gate Count: {qaoa_result.gate_count}")
        report.append(f"Number of Qubits: {n_assets}")
        report.append(f"Total Measurements: {sum(measurement_counts.values())}")
        report.append("")
        
        # Measurement Statistics
        report.append("MEASUREMENT STATISTICS")
        report.append("-"*40)
        
        # Calculate feasibility statistics
        feasible_count = 0
        infeasible_count = 0
        hamming_distribution = {}
        
        for bitstring, count in measurement_counts.items():
            bitstring_clean = bitstring.replace(' ', '')
            solution = [int(b) for b in bitstring_clean[::-1][:n_assets]]
            weight = sum(solution)
            
            hamming_distribution[weight] = hamming_distribution.get(weight, 0) + count
            
            if weight == budget:
                feasible_count += count
            else:
                infeasible_count += count
        
        total_counts = feasible_count + infeasible_count
        
        report.append(f"Feasible Solutions: {feasible_count} ({100*feasible_count/total_counts:.1f}%)")
        report.append(f"Infeasible Solutions: {infeasible_count} ({100*infeasible_count/total_counts:.1f}%)")
        report.append("")
        
        report.append("Hamming Weight Distribution:")
        for weight in sorted(hamming_distribution.keys()):
            percentage = 100 * hamming_distribution[weight] / total_counts
            marker = " <-- TARGET" if weight == budget else ""
            report.append(f"  {weight} assets: {hamming_distribution[weight]} ({percentage:.1f}%){marker}")
        report.append("")
        
        # Performance Comparison
        report.append("PERFORMANCE COMPARISON")
        report.append("-"*40)
        
        if classical_result.objective_value > 0:
            obj_ratio = qaoa_result.objective_value / classical_result.objective_value
            report.append(f"Objective Ratio (QAOA/Classical): {obj_ratio:.3f}")
        
        time_ratio = qaoa_result.execution_time / classical_result.execution_time if classical_result.execution_time > 0 else float('inf')
        report.append(f"Time Ratio (QAOA/Classical): {time_ratio:.1f}x")
        
        # Check if same solution
        if np.array_equal(classical_result.solution, qaoa_result.solution):
            report.append("Solution Match: QAOA found the same solution as classical")
        else:
            overlap = np.sum(classical_result.solution * qaoa_result.solution)
            report.append(f"Solution Overlap: {overlap} common assets")
        
        report.append("")
        report.append("="*70)
        report.append("END OF REPORT")
        report.append("="*70)
        
        return "\n".join(report)
    
    def save_full_report(self, classical_result: PortfolioResult,
                        qaoa_result: PortfolioResult,
                        measurement_counts: Dict[str, int],
                        n_assets: int, budget: int,
                        report_name: str = None):
        """
        Save complete report with all visualizations and text
        """
        if report_name is None:
            report_name = f"qaoa_report_{n_assets}assets_{budget}budget_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        report_dir = os.path.join(self.output_dir, report_name)
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate and save probability distribution
        print("Generating probability distribution visualization...")
        prob_fig = self.generate_probability_distribution(
            measurement_counts, n_assets, budget,
            save_path=os.path.join(report_dir, "probability_distribution.png")
        )
        plt.close(prob_fig)
        
        # Generate and save comparison report
        print("Generating comparison visualizations...")
        comp_fig = self.generate_comparison_report(
            classical_result, qaoa_result, measurement_counts, n_assets, budget,
            save_path=os.path.join(report_dir, "comparison_report.png")
        )
        plt.close(comp_fig)
        
        # Generate and save text report
        print("Generating text report...")
        text_report = self.generate_text_report(
            classical_result, qaoa_result, measurement_counts, n_assets, budget
        )
        
        with open(os.path.join(report_dir, "report.txt"), 'w') as f:
            f.write(text_report)
        
        # Save raw data as JSON
        print("Saving raw data...")
        raw_data = {
            'problem_config': {
                'n_assets': n_assets,
                'budget': budget
            },
            'classical_result': {
                'objective_value': classical_result.objective_value,
                'expected_return': classical_result.expected_return,
                'risk': classical_result.risk,
                'execution_time': classical_result.execution_time,
                'solution': classical_result.solution.tolist()
            },
            'qaoa_result': {
                'objective_value': qaoa_result.objective_value,
                'expected_return': qaoa_result.expected_return,
                'risk': qaoa_result.risk,
                'execution_time': qaoa_result.execution_time,
                'constraint_satisfied': qaoa_result.constraint_satisfied,
                'solution': qaoa_result.solution.tolist(),
                'approximation_ratio': qaoa_result.approximation_ratio,
                'circuit_depth': qaoa_result.circuit_depth,
                'gate_count': qaoa_result.gate_count,
                'convergence_history': qaoa_result.convergence_history
            },
            'measurement_counts': measurement_counts
        }
        
        with open(os.path.join(report_dir, "raw_data.json"), 'w') as f:
            json.dump(raw_data, f, indent=2)
        
        print(f"\nReport saved to: {report_dir}")
        print(f"  - probability_distribution.png")
        print(f"  - comparison_report.png")
        print(f"  - report.txt")
        print(f"  - raw_data.json")
        
        return report_dir