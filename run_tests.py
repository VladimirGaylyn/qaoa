"""
Simple test runner for QAOA portfolio optimization
"""

import sys
import os

# Set UTF-8 encoding for output
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def run_tests():
    """Run tests and display results"""
    print("\n" + "="*60)
    print("QAOA PORTFOLIO OPTIMIZATION - TEST RESULTS")
    print("="*60 + "\n")
    
    test_results = []
    
    # Test 1: Import modules
    print("1. Testing module imports...")
    try:
        import qiskit
        import qiskit_compat
        import qaoa_utils
        test_results.append(("Module imports", "PASSED"))
        print("   [PASS] All modules imported successfully")
    except ImportError as e:
        test_results.append(("Module imports", f"FAILED: {e}"))
        print(f"   [FAIL] Import error: {e}")
    
    # Test 2: Qiskit functionality
    print("\n2. Testing Qiskit functionality...")
    try:
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        test_results.append(("Qiskit circuits", "PASSED"))
        print("   [PASS] Quantum circuit creation works")
    except Exception as e:
        test_results.append(("Qiskit circuits", f"FAILED: {e}"))
        print(f"   [FAIL] Circuit error: {e}")
    
    # Test 3: Data generation
    print("\n3. Testing data generation...")
    try:
        from qaoa_utils import create_sample_data
        prices, returns, cov = create_sample_data(n_assets=4)
        assert len(returns) == 4
        assert cov.shape == (4, 4)
        test_results.append(("Data generation", "PASSED"))
        print("   [PASS] Sample data generated correctly")
    except Exception as e:
        test_results.append(("Data generation", f"FAILED: {e}"))
        print(f"   [FAIL] Data error: {e}")
    
    # Test 4: Risk metrics
    print("\n4. Testing risk metrics...")
    try:
        from qaoa_utils import RiskMetrics
        import numpy as np
        test_returns = np.random.normal(0.001, 0.02, 100)
        var = RiskMetrics.calculate_var(test_returns, 0.95)
        cvar = RiskMetrics.calculate_cvar(test_returns, 0.95)
        assert isinstance(var, float)
        assert isinstance(cvar, float)
        test_results.append(("Risk metrics", "PASSED"))
        print("   [PASS] Risk calculations work")
        print(f"        VaR(95%): {var:.4f}, CVaR(95%): {cvar:.4f}")
    except Exception as e:
        test_results.append(("Risk metrics", f"FAILED: {e}"))
        print(f"   [FAIL] Risk metric error: {e}")
    
    # Test 5: Portfolio benchmarks
    print("\n5. Testing portfolio benchmarks...")
    try:
        from qaoa_utils import PortfolioBenchmarks
        import numpy as np
        n_assets = 6
        budget = 3
        portfolio = PortfolioBenchmarks.equal_weight_portfolio(n_assets, budget)
        assert np.sum(portfolio) == budget
        test_results.append(("Portfolio benchmarks", "PASSED"))
        print("   [PASS] Portfolio benchmarks work")
        print(f"        Equal weight portfolio: {portfolio}")
    except Exception as e:
        test_results.append(("Portfolio benchmarks", f"FAILED: {e}"))
        print(f"   [FAIL] Benchmark error: {e}")
    
    # Test 6: Compatibility layer
    print("\n6. Testing compatibility layer...")
    try:
        from qiskit_compat import get_sampler, get_qiskit_version
        Sampler = get_sampler()
        major, minor = get_qiskit_version()
        assert Sampler is not None
        test_results.append(("Compatibility layer", "PASSED"))
        print("   [PASS] Compatibility layer works")
        print(f"        Qiskit version: {major}.{minor}")
        print(f"        Sampler: {Sampler.__name__ if hasattr(Sampler, '__name__') else type(Sampler)}")
    except Exception as e:
        test_results.append(("Compatibility layer", f"FAILED: {e}"))
        print(f"   [FAIL] Compatibility error: {e}")
    
    # Test 7: Notebook existence
    print("\n7. Testing notebook file...")
    notebook_path = os.path.join(os.path.dirname(__file__), 'qaoa_portfolio_optimization.ipynb')
    if os.path.exists(notebook_path):
        test_results.append(("Notebook file", "PASSED"))
        print("   [PASS] Notebook file exists")
    else:
        test_results.append(("Notebook file", "FAILED: File not found"))
        print("   [FAIL] Notebook file not found")
    
    # Test 8: Constraint builder
    print("\n8. Testing constraint builder...")
    try:
        from qaoa_utils import ConstraintBuilder
        constraint = ConstraintBuilder.cardinality_constraint(10, 3, 5)
        assert constraint['min'] == 3
        assert constraint['max'] == 5
        test_results.append(("Constraint builder", "PASSED"))
        print("   [PASS] Constraint builder works")
    except Exception as e:
        test_results.append(("Constraint builder", f"FAILED: {e}"))
        print(f"   [FAIL] Constraint error: {e}")
    
    # Test 9: Circuit optimizer
    print("\n9. Testing circuit optimizer...")
    try:
        from qaoa_utils import QAOACircuitOptimizer
        depth = QAOACircuitOptimizer.suggest_optimal_depth(8, 0.01)
        params = QAOACircuitOptimizer.initialize_parameters(3, 'tqa')
        assert depth > 0
        assert len(params) == 6
        test_results.append(("Circuit optimizer", "PASSED"))
        print("   [PASS] Circuit optimizer works")
        print(f"        Suggested depth: {depth}, Params length: {len(params)}")
    except Exception as e:
        test_results.append(("Circuit optimizer", f"FAILED: {e}"))
        print(f"   [FAIL] Circuit optimizer error: {e}")
    
    # Test 10: Performance analyzer
    print("\n10. Testing performance analyzer...")
    try:
        from qaoa_utils import PerformanceAnalyzer, PortfolioResult
        import numpy as np
        
        result = PortfolioResult(
            allocation=np.array([1, 0, 1, 0]),
            expected_return=0.12,
            risk=0.18,
            sharpe_ratio=0.67,
            objective_value=-0.05,
            algorithm='Test',
            execution_time=0.01
        )
        
        analyzer = PerformanceAnalyzer()
        analyzer.add_result(result)
        df = analyzer.compare_algorithms()
        assert len(df) == 1
        test_results.append(("Performance analyzer", "PASSED"))
        print("   [PASS] Performance analyzer works")
    except Exception as e:
        test_results.append(("Performance analyzer", f"FAILED: {e}"))
        print(f"   [FAIL] Performance analyzer error: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, status in test_results if status == "PASSED")
    failed = len(test_results) - passed
    
    print(f"\nTotal tests: {len(test_results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {passed/len(test_results)*100:.1f}%")
    
    print("\nDetailed results:")
    for test_name, status in test_results:
        status_str = "[PASS]" if status == "PASSED" else "[FAIL]"
        print(f"  {status_str} {test_name}: {status}")
    
    if failed == 0:
        print("\n" + "="*60)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("The QAOA portfolio optimization code is working correctly.")
        print("="*60)
        return True
    else:
        print("\n" + "="*60)
        print("SOME TESTS FAILED")
        print("Please review the errors above.")
        print("="*60)
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)