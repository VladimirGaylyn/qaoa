"""
Qiskit compatibility layer for different versions
Handles import differences across Qiskit versions
"""

import sys
import importlib.util
import warnings

def get_qiskit_version():
    """Get installed Qiskit version"""
    try:
        import qiskit
        version_parts = qiskit.__version__.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
        return major, minor
    except:
        return 0, 0

def get_sampler():
    """Get appropriate Sampler class based on installed Qiskit version"""
    major, minor = get_qiskit_version()
    
    # Try different import paths in order of preference
    sampler_imports = [
        ('qiskit.primitives', 'StatevectorSampler'),
        ('qiskit.primitives', 'Sampler'),
        ('qiskit_aer.primitives', 'Sampler'),
        ('qiskit.primitives', 'BackendSampler'),
    ]
    
    for module_name, class_name in sampler_imports:
        try:
            module = importlib.import_module(module_name)
            sampler_class = getattr(module, class_name, None)
            if sampler_class:
                print(f"Using {class_name} from {module_name}")
                return sampler_class
        except (ImportError, AttributeError):
            continue
    
    # If all imports fail, create a basic wrapper
    print("Creating compatibility Sampler wrapper")
    return create_sampler_wrapper()

def create_sampler_wrapper():
    """Create a basic Sampler wrapper for compatibility"""
    try:
        from qiskit_aer import AerSimulator
        from qiskit import transpile
        
        class SamplerWrapper:
            """Basic Sampler wrapper for compatibility"""
            
            def __init__(self, backend=None, options=None):
                self.backend = backend or AerSimulator()
                self.options = options or {}
                
            def run(self, circuits, parameter_values=None, **kwargs):
                """Run circuits with optional parameter values"""
                if parameter_values is not None:
                    # Bind parameters if provided
                    if hasattr(circuits, '__iter__'):
                        bound_circuits = []
                        for circuit, params in zip(circuits, parameter_values):
                            if hasattr(circuit, 'bind_parameters'):
                                bound_circuits.append(circuit.bind_parameters(params))
                            else:
                                bound_circuits.append(circuit)
                        circuits = bound_circuits
                    else:
                        if hasattr(circuits, 'bind_parameters'):
                            circuits = circuits.bind_parameters(parameter_values)
                
                # Transpile and run
                transpiled = transpile(circuits, self.backend)
                job = self.backend.run(transpiled, shots=self.options.get('shots', 1024))
                return job
            
            def __call__(self, *args, **kwargs):
                return self.run(*args, **kwargs)
        
        return SamplerWrapper
    
    except ImportError:
        warnings.warn("Could not create Sampler wrapper. Some features may not work.")
        return None

def get_estimator():
    """Get appropriate Estimator class based on installed Qiskit version"""
    estimator_imports = [
        ('qiskit.primitives', 'StatevectorEstimator'),
        ('qiskit.primitives', 'Estimator'),
        ('qiskit_aer.primitives', 'Estimator'),
        ('qiskit.primitives', 'BackendEstimator'),
    ]
    
    for module_name, class_name in estimator_imports:
        try:
            module = importlib.import_module(module_name)
            estimator_class = getattr(module, class_name, None)
            if estimator_class:
                print(f"Using {class_name} from {module_name}")
                return estimator_class
        except (ImportError, AttributeError):
            continue
    
    return None

def check_dependencies():
    """Check and report status of required dependencies"""
    dependencies = {
        'qiskit': 'Core quantum computing framework',
        'qiskit_aer': 'High-performance simulators',
        'qiskit_algorithms': 'Quantum algorithms including QAOA',
        'qiskit_finance': 'Finance-specific applications',
        'qiskit_optimization': 'Optimization algorithms',
        'numpy': 'Numerical computing',
        'pandas': 'Data manipulation',
        'yfinance': 'Market data fetching',
        'plotly': 'Interactive visualizations'
    }
    
    missing = []
    installed = []
    
    for package, description in dependencies.items():
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing.append(f"{package}: {description}")
        else:
            installed.append(package)
    
    print("Dependency Check Results:")
    print("=" * 50)
    
    if installed:
        print(f"[OK] Installed ({len(installed)}): {', '.join(installed)}")
    
    if missing:
        print(f"\n[X] Missing ({len(missing)}):")
        for item in missing:
            print(f"  - {item}")
        print("\nTo install missing dependencies, run:")
        print("!pip install " + " ".join([m.split(':')[0] for m in missing]))
    else:
        print("\n[OK] All dependencies are installed!")
    
    return len(missing) == 0

def install_missing_dependencies():
    """Automatically install missing dependencies"""
    import subprocess
    
    packages = [
        'qiskit',
        'qiskit-aer', 
        'qiskit-algorithms',
        'qiskit-finance',
        'qiskit-optimization',
        'numpy',
        'pandas',
        'yfinance',
        'plotly',
        'matplotlib',
        'scipy'
    ]
    
    for package in packages:
        try:
            importlib.import_module(package.replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
    
    print("All dependencies installed!")

# Compatibility exports
__all__ = [
    'get_sampler',
    'get_estimator',
    'check_dependencies',
    'install_missing_dependencies',
    'get_qiskit_version'
]

if __name__ == "__main__":
    print("Qiskit Compatibility Module")
    print("=" * 50)
    major, minor = get_qiskit_version()
    print(f"Detected Qiskit version: {major}.{minor}")
    print("\nChecking dependencies...")
    check_dependencies()