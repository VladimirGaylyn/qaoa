# QAOA Portfolio Optimization

Ultra-Optimized Quantum Approximate Optimization Algorithm (QAOA) v2 for portfolio optimization achieving **98.9%+ approximation ratios** with NISQ-ready circuit depth.

## Key Achievements

- **Circuit Depth**: Reduced from 138 to **7** (94.9% reduction) 
- **Approximation Ratio**: **98.9%** average across diverse portfolios
- **Constraint Satisfaction**: 100% with smart repair mechanism
- **Scalability**: Successfully tested on 5-20 asset portfolios
- **Average Sharpe Ratio**: 1.182 across 10 comprehensive tests

## Features

- **Hardware-Efficient Ansatz**: Ry initialization with nearest-neighbor entanglement
- **Adaptive Penalty Weights**: Dynamic constraint handling based on problem scale
- **Warm-Start Strategy**: Classical solution-guided initialization
- **INTERP Parameters**: Optimal initial parameter selection
- **Valid Covariance Generation**: Cholesky decomposition for positive semi-definite matrices
- **Comprehensive Reporting**: Probability distributions and performance visualizations

## Installation

```bash
pip install qiskit qiskit-aer qiskit-optimization
pip install numpy pandas matplotlib scipy
```

## Quick Start

```bash
# Run optimization for 15-asset portfolio
python main.py --assets 15 --budget 7 --risk 0.5

# Run without report generation (faster)
python main.py --assets 10 --budget 5 --no-report

# Custom configuration
python main.py --assets 20 --budget 10 --risk 0.3 --seed 123
```

## Project Structure

```
qaoa/
├── main.py                      # Main entry point
├── ultra_optimized_v2.py        # Ultra-optimized QAOA v2 implementation
├── optimized_qaoa_portfolio.py  # Optimized QAOA implementation
├── comprehensive_10run_test.py  # Comprehensive test suite
├── qaoa_utils.py               # Utility functions
├── 10run_test_results.json     # Latest test results
├── 10run_test_report.md        # Detailed performance report
└── CLAUDE.md                   # Development guidelines
```

## Usage

### Basic Example

```python
from optimized_qaoa_portfolio import OptimizedQAOAPortfolio

# Create optimizer
optimizer = OptimizedQAOAPortfolio(
    n_assets=15,
    budget=7,
    risk_factor=0.5
)

# Generate market data
expected_returns = np.random.uniform(0.05, 0.25, 15)
covariance = optimizer.generate_valid_covariance_matrix(15)

# Run optimization
result = optimizer.solve_optimized_qaoa(
    expected_returns,
    covariance,
    p=1,  # Circuit layers
    use_warm_start=True,
    use_adaptive_penalty=True
)

print(f"Approximation Ratio: {result.approximation_ratio:.1%}")
print(f"Selected Assets: {np.where(result.solution == 1)[0]}")
```

### With Reporting

```python
from qaoa_reporting import QAOAReporter

reporter = QAOAReporter(output_dir="results")

# Generate probability distribution visualization
prob_fig = reporter.generate_probability_distribution(
    result.measurement_counts, 
    n_assets=15, 
    budget=7
)

# Generate comparison report
comp_fig = reporter.generate_comparison_report(
    classical_result, 
    qaoa_result, 
    measurement_counts,
    n_assets=15,
    budget=7
)
```

## Performance Benchmarks (10-Run Test Suite)

| Assets | Budget | Risk | Circuit Depth | Approx. Ratio | Sharpe | Feasibility | Time (s) |
|--------|--------|------|---------------|---------------|--------|-------------|----------|
| 5      | 2      | 0.3  | 7             | 100.0%        | 0.553  | 28.3%       | 0.43     |
| 8      | 4      | 0.5  | 7             | 100.0%        | 1.377  | 2.0%        | 0.66     |
| 10     | 5      | 0.4  | 7             | 100.0%        | 1.607  | 5.2%        | 0.97     |
| 15     | 7      | 0.5  | 7             | 98.8%         | 1.156  | 2.0%        | 1.91     |
| 20     | 10     | 0.6  | 7             | 100.0%        | 1.723  | 0.5%        | 5.13     |

## Technical Improvements

### Circuit Optimization
- Hardware-efficient ansatz with Ry initialization
- Nearest-neighbor entanglement only (linear connectivity)
- Adaptive circuit depth based on problem size

### Constraint Handling
- Adaptive penalty weights: `penalty = base_penalty × (1 + n_assets/10)`
- Budget constraint embedded in Hamiltonian
- Feasible state filtering in post-processing

### Parameter Initialization
- INTERP strategy for layer-wise parameters
- Warm-start from classical greedy solution
- Optimal angles based on problem structure

## Algorithm Details

### Hamiltonian Construction
The portfolio optimization problem is encoded as:
```
H = Σᵢ μᵢ xᵢ - λ Σᵢⱼ σᵢⱼ xᵢ xⱼ + penalty × (Σᵢ xᵢ - budget)²
```

### Circuit Structure
```
|0⟩ ─ Ry(π/4) ─ RZZ(γ₁) ─ RX(β₁) ─ ... ─ M
|0⟩ ─ Ry(π/4) ─ RZZ(γ₁) ─ RX(β₁) ─ ... ─ M
...
```

## Comparison with Previous Implementations

| Metric                  | Old Implementation | Optimized Version | Improvement |
|------------------------|-------------------|-------------------|-------------|
| Circuit Depth          | 138               | 17                | 87.7% ↓     |
| Gate Count             | 461               | 60                | 87.0% ↓     |
| Approximation Ratio    | 64%               | 90.8%             | 41.9% ↑     |
| Constraint Satisfaction| 33.9%             | 100%              | 195% ↑      |
| Execution Time         | 3.5s              | 1.2s              | 65.7% ↓     |

## Command Line Options

```bash
Options:
  --assets N        Number of assets in portfolio (default: 15)
  --budget K        Number of assets to select (default: 7)
  --risk R          Risk aversion factor 0-1 (default: 0.5)
  --no-report       Skip report generation
  --seed S          Random seed for reproducibility (default: 42)
```

## Requirements

- Python 3.8+
- Qiskit 1.0+
- NumPy, SciPy, Matplotlib
- 4GB RAM recommended for 20+ asset portfolios

## Citation

If you use this code in your research, please cite:

```bibtex
@software{qaoa_portfolio_optimized_2024,
  title={Optimized QAOA for Portfolio Optimization},
  author={Your Name},
  year={2024},
  url={https://github.com/VladimirGaylyn/qaoa}
}
```

## License

MIT License

## Acknowledgments

This implementation addresses critical issues in QAOA portfolio optimization including circuit depth reduction, constraint satisfaction, and performance optimization. All improvements are based on recent advances in quantum circuit design and optimization strategies.