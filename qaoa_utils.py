"""
Utility functions for QAOA Portfolio Optimization
Advanced helpers for quantum finance applications
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PortfolioResult:
    """Data class for portfolio optimization results"""
    allocation: np.ndarray
    expected_return: float
    risk: float
    sharpe_ratio: float
    objective_value: float
    algorithm: str
    execution_time: float
    metadata: Dict = None


class RiskMetrics:
    """Advanced risk metrics for portfolio analysis"""
    
    @staticmethod
    def calculate_var(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk (VaR)"""
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    @staticmethod
    def calculate_cvar(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (CVaR)"""
        var = RiskMetrics.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, target_return: float = 0) -> float:
        """Calculate Sortino ratio (downside risk-adjusted return)"""
        excess_returns = returns - target_return
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = np.sqrt(np.mean(downside_returns**2))
        
        if downside_deviation == 0:
            return np.inf
        
        return excess_returns.mean() / downside_deviation
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series, prices: pd.Series) -> float:
        """Calculate Calmar ratio (return / max drawdown)"""
        annual_return = returns.mean() * 252
        max_dd = abs(RiskMetrics.calculate_max_drawdown(prices))
        
        if max_dd == 0:
            return np.inf
        
        return annual_return / max_dd


class ConstraintBuilder:
    """Build complex constraints for portfolio optimization"""
    
    @staticmethod
    def cardinality_constraint(n_assets: int, min_assets: int, max_assets: int) -> Dict:
        """Create cardinality constraints (number of assets in portfolio)"""
        return {
            'type': 'cardinality',
            'min': min_assets,
            'max': max_assets,
            'n_assets': n_assets
        }
    
    @staticmethod
    def sector_constraint(sectors: Dict[str, List[int]], 
                         sector_limits: Dict[str, Tuple[float, float]]) -> Dict:
        """Create sector allocation constraints"""
        return {
            'type': 'sector',
            'sectors': sectors,
            'limits': sector_limits
        }
    
    @staticmethod
    def turnover_constraint(current_portfolio: np.ndarray, 
                           max_turnover: float) -> Dict:
        """Create turnover constraint to limit trading"""
        return {
            'type': 'turnover',
            'current': current_portfolio,
            'max_turnover': max_turnover
        }
    
    @staticmethod
    def esg_constraint(esg_scores: np.ndarray, min_score: float) -> Dict:
        """Create ESG (Environmental, Social, Governance) constraint"""
        return {
            'type': 'esg',
            'scores': esg_scores,
            'min_score': min_score
        }


class QAOACircuitOptimizer:
    """Optimize QAOA circuit parameters and structure"""
    
    @staticmethod
    def suggest_optimal_depth(n_qubits: int, noise_level: float = 0.01) -> int:
        """Suggest optimal circuit depth based on problem size and noise"""
        if noise_level > 0.05:
            return min(2, n_qubits // 4)
        elif noise_level > 0.01:
            return min(3, n_qubits // 3)
        else:
            return min(4, n_qubits // 2)
    
    @staticmethod
    def initialize_parameters(p: int, strategy: str = 'tqa') -> np.ndarray:
        """
        Initialize QAOA parameters using different strategies
        
        Args:
            p: Number of QAOA layers
            strategy: 'random', 'tqa' (Trotterized Quantum Annealing), or 'interp'
        """
        if strategy == 'random':
            betas = np.random.uniform(0, np.pi, p)
            gammas = np.random.uniform(0, 2*np.pi, p)
        elif strategy == 'tqa':
            # Trotterized Quantum Annealing schedule
            s = np.linspace(0, 1, p)
            betas = np.pi * (1 - s)
            gammas = np.pi * s
        elif strategy == 'interp':
            # Interpolation-based initialization
            betas = np.linspace(np.pi/4, np.pi/8, p)
            gammas = np.linspace(0, np.pi, p)
        else:
            raise ValueError(f"Unknown initialization strategy: {strategy}")
        
        return np.concatenate([betas, gammas])
    
    @staticmethod
    def adapt_parameters(current_params: np.ndarray, 
                        gradient: np.ndarray, 
                        iteration: int) -> np.ndarray:
        """Adaptive parameter update with momentum"""
        learning_rate = 0.1 / (1 + 0.01 * iteration)
        momentum = 0.9
        
        if not hasattr(adapt_parameters, 'velocity'):
            adapt_parameters.velocity = np.zeros_like(current_params)
        
        adapt_parameters.velocity = (momentum * adapt_parameters.velocity - 
                                    learning_rate * gradient)
        
        return current_params + adapt_parameters.velocity


class PortfolioBenchmarks:
    """Standard portfolio benchmarks for comparison"""
    
    @staticmethod
    def equal_weight_portfolio(n_assets: int, budget: int) -> np.ndarray:
        """Create equal-weight portfolio"""
        portfolio = np.zeros(n_assets)
        selected = np.random.choice(n_assets, budget, replace=False)
        portfolio[selected] = 1
        return portfolio
    
    @staticmethod
    def market_cap_weighted(market_caps: np.ndarray, budget: int) -> np.ndarray:
        """Create market-cap weighted portfolio"""
        n_assets = len(market_caps)
        weights = market_caps / market_caps.sum()
        top_indices = np.argsort(weights)[-budget:]
        
        portfolio = np.zeros(n_assets)
        portfolio[top_indices] = 1
        return portfolio
    
    @staticmethod
    def minimum_variance_portfolio(cov_matrix: np.ndarray, budget: int) -> np.ndarray:
        """Create minimum variance portfolio (classical)"""
        n_assets = cov_matrix.shape[0]
        variances = np.diag(cov_matrix)
        selected = np.argsort(variances)[:budget]
        
        portfolio = np.zeros(n_assets)
        portfolio[selected] = 1
        return portfolio
    
    @staticmethod
    def risk_parity_portfolio(cov_matrix: np.ndarray, budget: int) -> np.ndarray:
        """Create risk parity portfolio"""
        n_assets = cov_matrix.shape[0]
        inv_vol = 1 / np.sqrt(np.diag(cov_matrix))
        weights = inv_vol / inv_vol.sum()
        
        selected = np.argsort(weights)[-budget:]
        portfolio = np.zeros(n_assets)
        portfolio[selected] = 1
        return portfolio


class PerformanceAnalyzer:
    """Analyze and compare portfolio performance"""
    
    def __init__(self):
        self.results = []
    
    def add_result(self, result: PortfolioResult):
        """Add optimization result for comparison"""
        self.results.append(result)
    
    def compare_algorithms(self) -> pd.DataFrame:
        """Compare all stored results"""
        comparison = []
        
        for result in self.results:
            comparison.append({
                'Algorithm': result.algorithm,
                'Return': result.expected_return,
                'Risk': result.risk,
                'Sharpe': result.sharpe_ratio,
                'Objective': result.objective_value,
                'Time': result.execution_time
            })
        
        return pd.DataFrame(comparison)
    
    def calculate_improvement(self, baseline: str = 'Classical') -> pd.DataFrame:
        """Calculate improvement over baseline algorithm"""
        df = self.compare_algorithms()
        baseline_row = df[df['Algorithm'] == baseline].iloc[0]
        
        improvements = []
        for _, row in df.iterrows():
            if row['Algorithm'] != baseline:
                improvements.append({
                    'Algorithm': row['Algorithm'],
                    'Return_Improvement': (row['Return'] - baseline_row['Return']) / baseline_row['Return'] * 100,
                    'Risk_Reduction': (baseline_row['Risk'] - row['Risk']) / baseline_row['Risk'] * 100,
                    'Sharpe_Improvement': (row['Sharpe'] - baseline_row['Sharpe']) / baseline_row['Sharpe'] * 100,
                    'Speed_Ratio': baseline_row['Time'] / row['Time']
                })
        
        return pd.DataFrame(improvements)
    
    @staticmethod
    def calculate_approximation_ratio(quantum_obj: float, classical_obj: float) -> float:
        """Calculate approximation ratio for quantum solution"""
        if classical_obj == 0:
            return 1.0
        return quantum_obj / classical_obj


class DataValidator:
    """Validate input data for portfolio optimization"""
    
    @staticmethod
    def validate_returns(returns: np.ndarray) -> bool:
        """Validate expected returns array"""
        if not isinstance(returns, (np.ndarray, list)):
            raise ValueError("Returns must be array-like")
        
        returns = np.array(returns)
        
        if returns.ndim != 1:
            raise ValueError("Returns must be 1-dimensional")
        
        if np.any(np.isnan(returns)):
            raise ValueError("Returns contain NaN values")
        
        if np.any(np.isinf(returns)):
            raise ValueError("Returns contain infinite values")
        
        return True
    
    @staticmethod
    def validate_covariance(cov_matrix: np.ndarray) -> bool:
        """Validate covariance matrix"""
        if not isinstance(cov_matrix, np.ndarray):
            raise ValueError("Covariance must be numpy array")
        
        if cov_matrix.ndim != 2:
            raise ValueError("Covariance must be 2-dimensional")
        
        if cov_matrix.shape[0] != cov_matrix.shape[1]:
            raise ValueError("Covariance must be square matrix")
        
        if not np.allclose(cov_matrix, cov_matrix.T):
            raise ValueError("Covariance must be symmetric")
        
        eigenvalues = np.linalg.eigvals(cov_matrix)
        if np.any(eigenvalues < -1e-10):
            raise ValueError("Covariance must be positive semi-definite")
        
        return True
    
    @staticmethod
    def validate_constraints(constraints: Dict, n_assets: int) -> bool:
        """Validate portfolio constraints"""
        if 'budget' in constraints:
            if constraints['budget'] > n_assets or constraints['budget'] < 1:
                raise ValueError(f"Budget must be between 1 and {n_assets}")
        
        if 'bounds' in constraints:
            bounds = constraints['bounds']
            if len(bounds) != n_assets:
                raise ValueError("Bounds must match number of assets")
        
        return True


def create_sample_data(n_assets: int = 8, n_days: int = 252) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Create sample financial data for testing
    
    Returns:
        prices: DataFrame of asset prices
        expected_returns: Array of expected returns
        cov_matrix: Covariance matrix
    """
    np.random.seed(42)
    
    # Generate synthetic returns
    mu = np.random.uniform(0.05, 0.20, n_assets)
    sigma = np.random.uniform(0.15, 0.35, n_assets)
    
    # Create correlation matrix
    correlation = np.random.uniform(-0.3, 0.7, (n_assets, n_assets))
    correlation = (correlation + correlation.T) / 2
    np.fill_diagonal(correlation, 1)
    
    # Convert to covariance
    cov_matrix = np.outer(sigma, sigma) * correlation
    
    # Generate price paths
    returns = np.random.multivariate_normal(mu/252, cov_matrix/252, n_days)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(returns, axis=0)),
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )
    
    # Annualized statistics
    expected_returns = mu
    annual_cov = cov_matrix
    
    return prices, expected_returns, annual_cov


if __name__ == "__main__":
    print("QAOA Portfolio Optimization Utilities")
    print("=" * 50)
    print("\nAvailable modules:")
    print("- PortfolioResult: Data class for results")
    print("- RiskMetrics: Advanced risk calculations")
    print("- ConstraintBuilder: Create portfolio constraints")
    print("- QAOACircuitOptimizer: Optimize QAOA parameters")
    print("- PortfolioBenchmarks: Standard benchmark portfolios")
    print("- PerformanceAnalyzer: Compare algorithm performance")
    print("- DataValidator: Validate input data")
    print("\nUtility functions:")
    print("- create_sample_data(): Generate test data")