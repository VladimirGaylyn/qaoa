# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Advanced implementation of QAOA (Quantum Approximate Optimization Algorithm) for financial portfolio optimization. The project focuses on solving Markowitz mean-variance optimization problems using quantum computing with noise mitigation and real-world constraints.

## Development Setup

### Requirements
- Python 3.8+
- Jupyter Notebook or Google Colab
- Key dependencies: qiskit, qiskit-finance, qiskit-optimization, numpy, pandas, yfinance

### Installation
```bash
pip install qiskit qiskit-aer qiskit-finance qiskit-optimization
pip install numpy pandas matplotlib yfinance scipy plotly scikit-learn
```

## Project Structure

```
qaoa/
├── qaoa_portfolio_optimization.ipynb  # Main notebook with complete implementation
├── qaoa_utils.py                      # Utility functions and helper classes
├── README.md                           # User documentation
└── CLAUDE.md                           # This file
```

## Common Commands

### Running the notebook in Google Colab
1. Upload `qaoa_portfolio_optimization.ipynb` to Google Colab
2. Run the installation cell first
3. Execute cells sequentially

### Testing locally
```python
python qaoa_utils.py  # Run utility module tests
jupyter notebook qaoa_portfolio_optimization.ipynb  # Open main notebook
```

### Import pattern
```python
from qaoa_utils import *
```

## Architecture Notes

### Core Components

1. **QAOAPortfolioOptimizer**: Main class for portfolio optimization
   - Supports QAOA, VQE, and classical solvers
   - Handles Markowitz mean-variance optimization
   - Includes noise mitigation strategies

2. **FinancialDataManager**: Data acquisition and preprocessing
   - Fetches real market data via yfinance
   - Calculates returns and covariance matrices
   - Handles missing data with synthetic generation

3. **AdvancedQAOAPortfolio**: Advanced features
   - CVaR risk measures
   - Custom QAOA circuits
   - Sector constraints

4. **NoiseAnalysis**: Noise handling
   - Creates realistic noise models
   - Implements zero-noise extrapolation
   - Compares noise resilience

### Key Design Patterns

- **Strategy Pattern**: Different optimizers (COBYLA, SPSA, L-BFGS-B)
- **Factory Pattern**: Circuit creation methods
- **Builder Pattern**: Constraint builders
- **Data Class Pattern**: PortfolioResult for structured results

### Algorithm Parameters

- **reps (p)**: QAOA circuit depth, typically 1-4
- **optimizer_type**: COBYLA for noiseless, SPSA for noisy
- **risk_factor**: 0-1, balances return vs risk
- **budget**: Number of assets to select

## Development Guidelines

### Code Style
- Use type hints for function parameters
- Document all classes and methods
- Follow PEP 8 conventions
- No unnecessary comments in implementation

### Performance Optimization
- Prefer COBYLA optimizer for noiseless simulations
- Use SPSA for noise-resilient optimization
- Keep circuit depth (p) ≤ 3 for practical execution times
- Batch operations when possible

### Error Handling
- Validate covariance matrix positive-definiteness
- Check for NaN/Inf in financial data
- Handle API failures with synthetic data fallback

### Testing Approach
- Compare QAOA results against classical exact solver
- Verify approximation ratio ≥ 0.9
- Test with different portfolio sizes (4-20 assets)
- Validate constraint satisfaction

## Common Issues and Solutions

1. **Import errors in Colab**: Run installation cell first
2. **Data fetch failures**: Uses synthetic data as fallback
3. **Poor convergence**: Adjust optimizer and increase iterations
4. **Memory issues**: Reduce number of assets or circuit depth

## Key Improvements in This Implementation

1. **Noise Mitigation**: SPSA optimizer, zero-noise extrapolation
2. **Advanced Risk Metrics**: CVaR, Sortino ratio, maximum drawdown
3. **Real Data Integration**: Yahoo Finance API with fallback
4. **Comprehensive Benchmarking**: Compare QAOA, VQE, and classical
5. **Visualization**: Interactive Plotly charts for results