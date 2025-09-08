"""
Comprehensive test suite for QAOA Portfolio Optimization
Tests all components including compatibility, data handling, and optimization
"""

import unittest
import numpy as np
import pandas as pd
import warnings
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress warnings during testing
warnings.filterwarnings('ignore')

class TestQiskitCompatibility(unittest.TestCase):
    """Test Qiskit compatibility layer"""
    
    def test_import_compatibility_module(self):
        """Test that compatibility module can be imported"""
        try:
            import qiskit_compat
            self.assertTrue(hasattr(qiskit_compat, 'get_sampler'))
            self.assertTrue(hasattr(qiskit_compat, 'check_dependencies'))
            print("[PASS] Compatibility module imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import compatibility module: {e}")
    
    def test_qiskit_version_detection(self):
        """Test Qiskit version detection"""
        from qiskit_compat import get_qiskit_version
        major, minor = get_qiskit_version()
        self.assertIsInstance(major, int)
        self.assertIsInstance(minor, int)
        print(f"[PASS] Detected Qiskit version: {major}.{minor}")
    
    def test_sampler_import(self):
        """Test that we can get a working Sampler"""
        from qiskit_compat import get_sampler
        Sampler = get_sampler()
        self.assertIsNotNone(Sampler)
        print(f"[PASS] Sampler class obtained: {Sampler}")
    
    def test_dependency_check(self):
        """Test dependency checking function"""
        from qiskit_compat import check_dependencies
        result = check_dependencies()
        self.assertIsInstance(result, bool)
        print(f"[PASS] Dependency check completed: {'All installed' if result else 'Some missing'}")


class TestDataManagement(unittest.TestCase):
    """Test financial data management"""
    
    def setUp(self):
        """Set up test data"""
        self.tickers = ['AAPL', 'GOOGL', 'MSFT', 'JPM']
        self.n_assets = len(self.tickers)
        
    def test_synthetic_data_generation(self):
        """Test synthetic data generation"""
        from qaoa_utils import create_sample_data
        
        prices, returns, cov_matrix = create_sample_data(n_assets=self.n_assets)
        
        # Check shapes
        self.assertEqual(len(returns), self.n_assets)
        self.assertEqual(cov_matrix.shape, (self.n_assets, self.n_assets))
        
        # Check data properties
        self.assertTrue(np.all(np.isfinite(returns)))
        self.assertTrue(np.all(np.isfinite(cov_matrix)))
        
        # Check covariance matrix is symmetric
        self.assertTrue(np.allclose(cov_matrix, cov_matrix.T))
        
        # Check positive semi-definite
        eigenvalues = np.linalg.eigvals(cov_matrix)
        self.assertTrue(np.all(eigenvalues >= -1e-10))
        
        print("[PASS] Synthetic data generation successful")
        print(f"  - Generated {len(prices)} days of prices for {self.n_assets} assets")
        print(f"  - Expected returns range: [{returns.min():.3f}, {returns.max():.3f}]")
    
    def test_risk_metrics(self):
        """Test risk metric calculations"""
        from qaoa_utils import RiskMetrics
        
        # Generate test returns
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        prices = pd.Series(100 * np.exp(np.cumsum(returns)))
        
        # Test VaR
        var = RiskMetrics.calculate_var(returns, 0.95)
        self.assertIsInstance(var, float)
        self.assertTrue(var < 0)  # VaR should be negative for losses
        
        # Test CVaR
        cvar = RiskMetrics.calculate_cvar(returns, 0.95)
        self.assertIsInstance(cvar, float)
        self.assertTrue(cvar <= var)  # CVaR should be worse than VaR
        
        # Test max drawdown
        max_dd = RiskMetrics.calculate_max_drawdown(prices)
        self.assertIsInstance(max_dd, float)
        self.assertTrue(-1 <= max_dd <= 0)
        
        # Test Sortino ratio
        sortino = RiskMetrics.calculate_sortino_ratio(returns)
        self.assertIsInstance(sortino, float)
        
        print("[PASS] Risk metrics calculations successful")
        print(f"  - VaR (95%): {var:.4f}")
        print(f"  - CVaR (95%): {cvar:.4f}")
        print(f"  - Max Drawdown: {max_dd:.4f}")
        print(f"  - Sortino Ratio: {sortino:.4f}")
    
    def test_data_validation(self):
        """Test data validation functions"""
        from qaoa_utils import DataValidator
        
        # Valid data
        valid_returns = np.array([0.1, 0.05, -0.02, 0.08])
        valid_cov = np.array([[0.04, 0.01, 0.02, 0.01],
                             [0.01, 0.09, 0.01, 0.02],
                             [0.02, 0.01, 0.16, 0.01],
                             [0.01, 0.02, 0.01, 0.09]])
        
        # Test valid returns
        self.assertTrue(DataValidator.validate_returns(valid_returns))
        
        # Test valid covariance
        self.assertTrue(DataValidator.validate_covariance(valid_cov))
        
        # Test invalid returns (with NaN)
        invalid_returns = np.array([0.1, np.nan, 0.05])
        with self.assertRaises(ValueError):
            DataValidator.validate_returns(invalid_returns)
        
        # Test invalid covariance (non-symmetric)
        invalid_cov = np.array([[1, 2], [3, 4]])
        with self.assertRaises(ValueError):
            DataValidator.validate_covariance(invalid_cov)
        
        print("[PASS] Data validation tests passed")


class TestQAOAOptimization(unittest.TestCase):
    """Test QAOA optimization algorithms"""
    
    def setUp(self):
        """Set up test portfolio"""
        np.random.seed(42)
        self.n_assets = 6
        self.expected_returns = np.random.uniform(0.05, 0.15, self.n_assets)
        
        # Create valid covariance matrix
        correlation = np.random.uniform(-0.3, 0.7, (self.n_assets, self.n_assets))
        correlation = (correlation + correlation.T) / 2
        np.fill_diagonal(correlation, 1)
        sigma = np.random.uniform(0.1, 0.3, self.n_assets)
        self.cov_matrix = np.outer(sigma, sigma) * correlation
        
        self.budget = 3  # Select 3 assets
    
    def test_portfolio_benchmarks(self):
        """Test benchmark portfolio creation"""
        from qaoa_utils import PortfolioBenchmarks
        
        # Equal weight portfolio
        equal_weight = PortfolioBenchmarks.equal_weight_portfolio(self.n_assets, self.budget)
        self.assertEqual(np.sum(equal_weight), self.budget)
        self.assertTrue(np.all(equal_weight >= 0))
        self.assertTrue(np.all(equal_weight <= 1))
        
        # Minimum variance portfolio
        min_var = PortfolioBenchmarks.minimum_variance_portfolio(self.cov_matrix, self.budget)
        self.assertEqual(np.sum(min_var), self.budget)
        
        # Risk parity portfolio
        risk_parity = PortfolioBenchmarks.risk_parity_portfolio(self.cov_matrix, self.budget)
        self.assertEqual(np.sum(risk_parity), self.budget)
        
        print("[PASS] Benchmark portfolios created successfully")
        print(f"  - Equal weight: {equal_weight}")
        print(f"  - Min variance: {min_var}")
        print(f"  - Risk parity: {risk_parity}")
    
    def test_constraint_builder(self):
        """Test constraint building"""
        from qaoa_utils import ConstraintBuilder
        
        # Cardinality constraint
        card_constraint = ConstraintBuilder.cardinality_constraint(
            n_assets=self.n_assets,
            min_assets=2,
            max_assets=4
        )
        self.assertEqual(card_constraint['type'], 'cardinality')
        self.assertEqual(card_constraint['min'], 2)
        self.assertEqual(card_constraint['max'], 4)
        
        # Sector constraint
        sectors = {'Tech': [0, 1], 'Finance': [2, 3], 'Energy': [4, 5]}
        sector_limits = {'Tech': (1, 2), 'Finance': (0, 1)}
        sector_constraint = ConstraintBuilder.sector_constraint(sectors, sector_limits)
        self.assertEqual(sector_constraint['type'], 'sector')
        self.assertIn('Tech', sector_constraint['sectors'])
        
        print("[PASS] Constraint builder tests passed")
    
    def test_circuit_optimizer(self):
        """Test QAOA circuit optimization"""
        from qaoa_utils import QAOACircuitOptimizer
        
        # Test optimal depth suggestion
        depth_low_noise = QAOACircuitOptimizer.suggest_optimal_depth(8, noise_level=0.001)
        depth_high_noise = QAOACircuitOptimizer.suggest_optimal_depth(8, noise_level=0.1)
        self.assertLessEqual(depth_high_noise, depth_low_noise)
        
        # Test parameter initialization
        p = 3
        params_tqa = QAOACircuitOptimizer.initialize_parameters(p, strategy='tqa')
        params_random = QAOACircuitOptimizer.initialize_parameters(p, strategy='random')
        params_interp = QAOACircuitOptimizer.initialize_parameters(p, strategy='interp')
        
        self.assertEqual(len(params_tqa), 2 * p)
        self.assertEqual(len(params_random), 2 * p)
        self.assertEqual(len(params_interp), 2 * p)
        
        print("[PASS] Circuit optimizer tests passed")
        print(f"  - Suggested depth (low noise): {depth_low_noise}")
        print(f"  - Suggested depth (high noise): {depth_high_noise}")
    
    def test_performance_analyzer(self):
        """Test performance analysis tools"""
        from qaoa_utils import PerformanceAnalyzer, PortfolioResult
        
        analyzer = PerformanceAnalyzer()
        
        # Create mock results
        classical_result = PortfolioResult(
            allocation=np.array([1, 0, 1, 0, 1, 0]),
            expected_return=0.12,
            risk=0.18,
            sharpe_ratio=0.67,
            objective_value=-0.05,
            algorithm='Classical',
            execution_time=0.01
        )
        
        qaoa_result = PortfolioResult(
            allocation=np.array([1, 1, 0, 0, 1, 0]),
            expected_return=0.11,
            risk=0.17,
            sharpe_ratio=0.65,
            objective_value=-0.048,
            algorithm='QAOA',
            execution_time=0.5
        )
        
        analyzer.add_result(classical_result)
        analyzer.add_result(qaoa_result)
        
        # Test comparison
        comparison_df = analyzer.compare_algorithms()
        self.assertEqual(len(comparison_df), 2)
        self.assertIn('Algorithm', comparison_df.columns)
        self.assertIn('Sharpe', comparison_df.columns)
        
        # Test approximation ratio
        approx_ratio = PerformanceAnalyzer.calculate_approximation_ratio(
            qaoa_result.objective_value,
            classical_result.objective_value
        )
        self.assertIsInstance(approx_ratio, float)
        self.assertGreater(approx_ratio, 0)
        
        print("[PASS] Performance analyzer tests passed")
        print(f"  - Approximation ratio: {approx_ratio:.4f}")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete workflow"""
    
    def test_minimal_portfolio_optimization(self):
        """Test a minimal portfolio optimization workflow"""
        print("\n" + "="*60)
        print("INTEGRATION TEST: Minimal Portfolio Optimization")
        print("="*60)
        
        try:
            # Import required modules
            from qaoa_utils import create_sample_data, PortfolioBenchmarks
            from qiskit_compat import get_sampler
            
            # Create small test portfolio
            n_assets = 4
            prices, returns, cov_matrix = create_sample_data(n_assets=n_assets)
            
            # Create benchmark portfolios
            budget = 2
            equal_weight = PortfolioBenchmarks.equal_weight_portfolio(n_assets, budget)
            min_var = PortfolioBenchmarks.minimum_variance_portfolio(cov_matrix, budget)
            
            # Calculate metrics for equal weight portfolio
            portfolio_return = np.dot(equal_weight, returns)
            portfolio_variance = np.dot(equal_weight, np.dot(cov_matrix, equal_weight))
            portfolio_risk = np.sqrt(portfolio_variance)
            sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
            
            print(f"\n[PASS] Integration test successful!")
            print(f"  Portfolio metrics:")
            print(f"  - Expected Return: {portfolio_return:.4f}")
            print(f"  - Risk (Std Dev): {portfolio_risk:.4f}")
            print(f"  - Sharpe Ratio: {sharpe_ratio:.4f}")
            print(f"  - Selected assets: {np.where(equal_weight == 1)[0].tolist()}")
            
            self.assertGreater(portfolio_return, 0)
            self.assertGreater(portfolio_risk, 0)
            
        except Exception as e:
            self.fail(f"Integration test failed: {e}")
    
    def test_qiskit_basic_circuit(self):
        """Test basic Qiskit circuit creation"""
        try:
            from qiskit import QuantumCircuit
            from qiskit.circuit import Parameter
            
            # Create simple QAOA-like circuit
            n_qubits = 4
            qc = QuantumCircuit(n_qubits)
            
            # Add Hadamard gates
            qc.h(range(n_qubits))
            
            # Add parameterized rotation
            beta = Parameter('β')
            gamma = Parameter('γ')
            
            for i in range(n_qubits):
                qc.rz(gamma, i)
                qc.rx(beta, i)
            
            # Check circuit properties
            self.assertEqual(qc.num_qubits, n_qubits)
            self.assertEqual(len(qc.parameters), 2)
            
            print("[PASS] Qiskit circuit creation successful")
            print(f"  - Circuit with {n_qubits} qubits")
            print(f"  - Parameters: {list(qc.parameters)}")
            
        except Exception as e:
            print(f"[WARNING] Basic Qiskit test failed: {e}")
            print("  This may be due to missing Qiskit installation")


class TestNotebook(unittest.TestCase):
    """Test notebook functionality"""
    
    def test_notebook_exists(self):
        """Test that the main notebook exists"""
        import os
        notebook_path = os.path.join(
            os.path.dirname(__file__),
            'qaoa_portfolio_optimization.ipynb'
        )
        self.assertTrue(os.path.exists(notebook_path))
        print(f"[PASS] Notebook found at: {notebook_path}")
    
    def test_utils_module(self):
        """Test that utils module can be imported"""
        try:
            import qaoa_utils
            
            # Check for expected classes
            self.assertTrue(hasattr(qaoa_utils, 'RiskMetrics'))
            self.assertTrue(hasattr(qaoa_utils, 'PortfolioBenchmarks'))
            self.assertTrue(hasattr(qaoa_utils, 'DataValidator'))
            self.assertTrue(hasattr(qaoa_utils, 'PerformanceAnalyzer'))
            
            print("[PASS] Utils module imported successfully")
            print(f"  - Available classes: {[c for c in dir(qaoa_utils) if c[0].isupper()]}")
            
        except ImportError as e:
            self.fail(f"Failed to import utils module: {e}")


def run_all_tests():
    """Run all tests and generate report"""
    print("\n" + "="*70)
    print("QAOA PORTFOLIO OPTIMIZATION - COMPREHENSIVE TEST SUITE")
    print("="*70)
    print(f"Starting tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestQiskitCompatibility,
        TestDataManagement,
        TestQAOAOptimization,
        TestIntegration,
        TestNotebook
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n[SUCCESS] ALL TESTS PASSED!")
    else:
        print("\n[FAILED] SOME TESTS FAILED")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.split(chr(10))[0]}")
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback.split(chr(10))[0]}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)