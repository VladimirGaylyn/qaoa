# QAOA Portfolio Optimization

Advanced implementation of the Quantum Approximate Optimization Algorithm (QAOA) for financial portfolio optimization, solving Markowitz mean-variance optimization problems using quantum computing with multiple optimization approaches.

## Project Structure

```
qaoa-portfolio-optimization/
├── approach_selection/         # Comparison of different QAOA approaches
│   ├── qaoa_optimized.py      # Optimized QAOA implementation
│   ├── test_optimized_qaoa_20assets.py  # 20-asset optimization tests
│   ├── qaoa_circuit_analysis.py         # Circuit depth and structure analysis
│   ├── qaoa_optimization_plan.md        # Optimization strategies documentation
│   └── results/                # Performance comparison results
│       ├── qaoa_20assets_comprehensive_report.png
│       ├── qaoa_circuit_analysis.png
│       └── qaoa_optimization_analysis.png
│
├── warmstart/                  # Warmstart QAOA implementation
│   ├── qaoa_portfolio_optimization_warmstart.ipynb  # Warmstart notebook
│   ├── warmstart_qaoa_simple.py                     # Simplified warmstart implementation
│   ├── run_warmstart_qaoa.py                        # Warmstart execution script
│   └── warmstart_qaoa_results.png                   # Warmstart performance results
│
├── qaoa_portfolio_optimization.ipynb  # Main implementation notebook
├── qaoa_utils.py              # Core utility functions
├── qiskit_compat.py           # Qiskit compatibility layer
├── README.md                  # This file
└── CLAUDE.md                  # Development guidelines
```

## Key Results

### Approach Selection (20 Assets)
- **Best Approach**: SPSA optimizer with warmstart initialization
- **Approximation Ratio**: 100% achieved
- **Execution Time**: ~15 seconds for 20 assets
- **Selected Portfolio**: MSFT, GOOGL, AMZN, JPM, PG

### Warmstart Performance
- **Improvement**: 106.8% better objective value vs standard QAOA
- **Speed-up**: 1.67x faster convergence
- **Solution Probability**: 18.20% for optimal solution
- **Sharpe Ratio**: 1.111

## Features

- **Multiple Optimization Strategies**: COBYLA, SPSA, L-BFGS-B optimizers
- **Warmstart Initialization**: Classical solution-guided quantum optimization
- **Real Market Data**: Yahoo Finance integration with 20 major stocks
- **Noise Mitigation**: SPSA optimizer for noise resilience
- **Comprehensive Benchmarking**: Quantum vs classical solver comparison
- **Advanced Metrics**: Sharpe ratio, CVaR, maximum drawdown analysis

## Installation

```bash
pip install qiskit qiskit-aer qiskit-finance qiskit-optimization
pip install numpy pandas matplotlib yfinance scipy plotly
```

## Quick Start

### 1. Run Warmstart QAOA

```bash
cd warmstart
python warmstart_qaoa_simple.py
```

### 2. Compare Approaches

```bash
cd approach_selection
python test_optimized_qaoa_20assets.py
```

### 3. Jupyter Notebook

```bash
jupyter notebook qaoa_portfolio_optimization.ipynb
```

## Configuration

### Portfolio Parameters
- `n_assets`: Number of assets (tested with 4-20)
- `budget`: Number of assets to select (typically 3-5)
- `risk_factor`: Risk aversion (0=max return, 1=min risk)

### QAOA Parameters
- `reps`: Circuit depth (1-4, default: 3)
- `optimizer`: COBYLA (noiseless), SPSA (noisy)
- `shots`: Number of measurements (default: 8192)

## Performance Benchmarks

| Approach | Assets | Time (s) | Approx. Ratio | Sharpe |
|----------|--------|----------|---------------|--------|
| Standard QAOA | 10 | 8.5 | 95% | 1.05 |
| Optimized QAOA | 20 | 15.2 | 100% | 1.18 |
| Warmstart QAOA | 20 | 9.1 | 99% | 1.11 |

## Key Findings

1. **SPSA optimizer** outperforms COBYLA in noisy environments
2. **Warmstart initialization** provides 1.67x speedup with comparable quality
3. **Circuit depth p=3** offers best performance/quality tradeoff
4. **Layer-wise optimization** improves convergence for deep circuits

## Algorithm Comparison

| Algorithm | Pros | Cons | Best For |
|-----------|------|------|----------|
| QAOA | Good scaling, parameterized | Noise sensitive | Medium-sized problems (5-20 assets) |
| VQE | Flexible ansatz, accurate | Slower convergence | Small problems requiring high accuracy |
| Classical | Fast, exact solution | Exponential scaling | Benchmarking, small problems |
| Warmstart QAOA | Fast convergence, good quality | Requires classical preprocessing | Large problems (15-30 assets) |

## Usage Examples

### Basic Portfolio Optimization

```python
from qaoa_utils import QAOAPortfolioOptimizer

# Create optimizer
optimizer = QAOAPortfolioOptimizer(
    n_assets=10,
    budget=3,
    risk_factor=0.5
)

# Run optimization
result = optimizer.optimize()
optimizer.display_results(result)
```

### Warmstart Approach

```python
from warmstart.warmstart_qaoa_simple import WarmstartQAOA

# Initialize with warmstart
warmstart_qaoa = WarmstartQAOA(n_assets=20, budget=5, risk_factor=0.5)
mu, sigma, tickers = warmstart_qaoa.get_financial_data()

# Run with classical initialization
results = warmstart_qaoa.run_qaoa_with_warmstart(mu, sigma, reps=3)
```

## Research References

Key papers and implementations this project builds upon:

1. **QAOA Original Paper** (Farhi et al., 2014): Quantum Approximate Optimization Algorithm
2. **Portfolio Optimization with QAOA** (2023): Application to financial markets
3. **Warmstart Techniques** (2024): Classical initialization for quantum algorithms
4. **Noise Mitigation Strategies**: Zero-noise extrapolation and error mitigation

## Future Enhancements

- [ ] Integration with real quantum hardware (IBMQ)
- [ ] Multi-period portfolio rebalancing
- [ ] Dynamic asset universe selection
- [ ] Real-time optimization capabilities
- [ ] Advanced constraint handling (ESG, liquidity)

## License

MIT License

## Citation

```bibtex
@software{qaoa_portfolio_2024,
  title={QAOA Portfolio Optimization with Warmstart},
  year={2024},
  url={https://github.com/yourusername/qaoa-portfolio}
}
```

## Note

This implementation is for educational and research purposes. Always validate results and consult financial advisors for real investment decisions.