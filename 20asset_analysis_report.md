# Full-Scale QAOA Analysis Report: 20 Assets Portfolio Optimization

Generated: 2025-09-08 20:44:24

## Executive Summary

**Problem**: Select 5 assets from a 20-asset portfolio
**Risk Factor**: 0.5
**Test Runs**: 2
**Quantum Shots per Run**: 8192

### Key Findings

1. **Approximation Ratio**
   - v2: 1.008 average
   - v3: 1.008 average
   - Difference: -0.000

2. **Feasibility Rate**
   - v2: 12.7% of quantum measurements satisfy constraints
   - v3: 16.6% of quantum measurements satisfy constraints
   - Improvement: 3.9 percentage points

3. **Convergence Speed**
   - v2: 10.0 iterations average
   - v3: 10.5 iterations average
   - Speed difference: 5.0%

4. **Execution Time**
   - v2: 16.68s average
   - v3: 15.24s average

## Classical Baseline Solution

**Best Strategy**: max_diversification
**Objective Value**: 0.148546
**Expected Return**: 0.1627
**Risk (Volatility)**: 0.1685
**Sharpe Ratio**: 0.966

**Selected Assets**: TECH1, TECH3, TECH4, FIN3, HEALTH2

## Quantum Circuit Analysis

### Circuit Architecture
- **Circuit Depth**: 7 (both v2 and v3)
- **Gate Count**: 100 (v2), 100 (v3)
- **Qubit Count**: 20
- **Parameter Count**: ~60

### Quantum Measurement Distribution

**Unique Feasible Solutions Found**
- v2: 770 total across 2 runs
- v3: 1992 total across 2 runs

**Hamming Weight Distribution** (aggregated across all runs)

| Hamming Weight | v2 Count | v2 % | v3 Count | v3 % |
|----------------|----------|------|----------|------|
| 0 | 1801 | 11.0% | 306 | 1.9% |
| 1 | 1516 | 9.3% | 926 | 5.7% |
| 2 | 2430 | 14.8% | 1704 | 10.4% |
| 3 | 2843 | 17.4% | 2452 | 15.0% |
| 4 | 2327 | 14.2% | 2793 | 17.0% |
| **5 (target)** | **2078** | **12.7%** | **2712** | **16.6%** |
| 6 | 1492 | 9.1% | 2234 | 13.6% |
| 7 | 912 | 5.6% | 1505 | 9.2% |
| 8 | 523 | 3.2% | 932 | 5.7% |
| 9 | 266 | 1.6% | 499 | 3.0% |
| 10 | 122 | 0.7% | 200 | 1.2% |
| 11 | 57 | 0.3% | 85 | 0.5% |
| 12 | 14 | 0.1% | 26 | 0.2% |
| 13 | 2 | 0.0% | 10 | 0.1% |
| 15 | 1 | 0.0% | 0 | 0.0% |

## Solution Quality Analysis

### Statistical Comparison

| Metric | v2 | v3 | Classical |
|--------|----|----|-----------|
| Mean Objective | 0.144988 | 0.144962 | 0.148546 |
| Std Dev | 0.000026 | 0.000000 | 0.000000 |
| Min | 0.144962 | 0.144962 | - |
| Max | 0.145014 | 0.144962 | - |

## Asset Selection Analysis

### Most Frequently Selected Assets

| Asset | v2 Selection Rate | v3 Selection Rate |
|-------|-------------------|-------------------|
| TECH1 | 50.0% | 100.0% |
| TECH2 | 100.0% | 100.0% |
| TECH3 | 100.0% | 100.0% |
| TECH4 | 100.0% | 100.0% |
| FIN3 | 100.0% | 100.0% |
| FIN1 | 0.0% | 0.0% |
| FIN2 | 0.0% | 0.0% |
| FIN4 | 0.0% | 0.0% |
| HEALTH1 | 0.0% | 0.0% |
| HEALTH2 | 50.0% | 0.0% |

## Advanced Warm Start Analysis (v3)

### Warm Start Strategy Usage

| Strategy | Count | Percentage |
|----------|-------|------------|
| max_diversification | 2 | 100% |

## Sector Analysis

### Sector Representation in Solutions

| Sector | v2 Average | v3 Average | Classical |
|--------|------------|------------|-----------|
| TECH | 87.5% | 100.0% | 75.0% |
| FIN | 25.0% | 25.0% | 25.0% |
| HEALTH | 16.7% | 0.0% | 33.3% |
| ENERGY | 0.0% | 0.0% | 0.0% |
| CONS | 0.0% | 0.0% | 0.0% |
| UTIL | 0.0% | 0.0% | 0.0% |

## Conclusions

### Key Findings

1. **v2 shows better approximation ratio** (+0.000)
2. **v3 achieves higher feasibility rates** (+3.9pp)
3. **Circuit depth of 7 is maintained** for both versions (NISQ-ready)
4. **Both versions find near-optimal solutions** with >100.8% approximation ratio

### Recommendations

1. **For production use**: v3 with advanced warm start shows promise for complex problems
2. **For simple problems**: v2 may be sufficient with lower computational overhead
3. **Circuit depth of 7**: Suitable for current NISQ devices
4. **Feasibility rates**: Post-processing repair mechanisms remain important
5. **Sector diversification**: Both quantum approaches maintain reasonable diversification

---

*This analysis demonstrates that QAOA can successfully handle 20-asset portfolio optimization
problems with reasonable approximation ratios and feasibility rates on NISQ-ready circuits.*