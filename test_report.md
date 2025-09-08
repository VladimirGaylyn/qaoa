# QAOA Portfolio Optimization - Test Results

Generated: 2025-09-08T18:05:42.370918

## Executive Summary

- **Average Approximation Ratio**: 95.8%
- **Average Circuit Depth**: 14.8
- **Average Execution Time**: 1.21s
- **Constraint Satisfaction Rate**: 100%
- **Average Sharpe Ratio**: 2.272

## Detailed Results

### Test 1: 8 Assets, Budget=4

#### Configuration
- Assets: 8
- Budget: 4
- Risk Factor: 0.5

#### Performance Metrics
- **Approximation Ratio**: 100.0%
- **Circuit Depth**: 13
- **Gate Count**: 47
- **Execution Time**: 0.64s
- **Feasibility Rate**: 35.8%

#### Financial Metrics
- **QAOA Sharpe Ratio**: 2.139
- **Classical Sharpe Ratio**: 2.139
- **Expected Return**: 0.2074
- **Risk**: 0.0969

#### Solution Quality
- **Constraint Satisfied**: Yes
- **Assets Selected**: 4
- **Selected Indices**: [1, 2, 3, 7]

---

### Test 2: 10 Assets, Budget=5

#### Configuration
- Assets: 10
- Budget: 5
- Risk Factor: 0.5

#### Performance Metrics
- **Approximation Ratio**: 95.8%
- **Circuit Depth**: 15
- **Gate Count**: 59
- **Execution Time**: 0.92s
- **Feasibility Rate**: 24.3%

#### Financial Metrics
- **QAOA Sharpe Ratio**: 2.365
- **Classical Sharpe Ratio**: 3.886
- **Expected Return**: 0.1778
- **Risk**: 0.0752

#### Solution Quality
- **Constraint Satisfied**: Yes
- **Assets Selected**: 5
- **Selected Indices**: [1, 4, 5, 6, 9]

---

### Test 3: 12 Assets, Budget=6

#### Configuration
- Assets: 12
- Budget: 6
- Risk Factor: 0.5

#### Performance Metrics
- **Approximation Ratio**: 100.0%
- **Circuit Depth**: 14
- **Gate Count**: 48
- **Execution Time**: 1.15s
- **Feasibility Rate**: 23.7%

#### Financial Metrics
- **QAOA Sharpe Ratio**: 2.135
- **Classical Sharpe Ratio**: 2.331
- **Expected Return**: 0.1939
- **Risk**: 0.0908

#### Solution Quality
- **Constraint Satisfied**: Yes
- **Assets Selected**: 6
- **Selected Indices**: [0, 2, 5, 9, 10, 11]

---

### Test 4: 15 Assets, Budget=7

#### Configuration
- Assets: 15
- Budget: 7
- Risk Factor: 0.5

#### Performance Metrics
- **Approximation Ratio**: 87.6%
- **Circuit Depth**: 17
- **Gate Count**: 60
- **Execution Time**: 2.14s
- **Feasibility Rate**: 1.0%

#### Financial Metrics
- **QAOA Sharpe Ratio**: 2.450
- **Classical Sharpe Ratio**: 2.571
- **Expected Return**: 0.1756
- **Risk**: 0.0717

#### Solution Quality
- **Constraint Satisfied**: Yes
- **Assets Selected**: 7
- **Selected Indices**: [1, 2, 5, 9, 10, 12, 14]

---

## Summary Statistics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Approximation Ratio | 0.958 | 0.051 | 0.876 | 1.000 |
| Execution Time | 1.213 | 0.564 | 0.645 | 2.140 |
| Circuit Depth | 14.750 | 1.479 | 13.000 | 17.000 |
| Gate Count | 53.500 | 6.021 | 47.000 | 60.000 |
| Feasibility Rate | 0.212 | 0.126 | 0.010 | 0.358 |
| Sharpe Ratio | 2.272 | 0.139 | 2.135 | 2.450 |
