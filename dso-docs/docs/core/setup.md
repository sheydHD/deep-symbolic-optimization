# Setup & Installation

> Version: 2.0 • Last updated: 2025-09-15

This guide provides detailed setup instructions for DSO, including troubleshooting and advanced configuration options.

## 🔧 **Installation Methods**

### **Method 1: Automated Setup (Recommended)**

The simplest way to get started with DSO:

```bash
# Clone the repository
git clone https://github.com/your-org/dso.git
cd dso

# Run modern automated setup
./main.sh modern setup

# Activate the environment
source .venv/bin/activate

# Verify installation
python tools/python/run.py test
```

### **Method 2: Manual Setup**

For more control over the installation process:

```bash
# Clone repository
git clone https://github.com/your-org/dso.git
cd dso

# Create virtual environment using uv
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Compile Cython extensions
python setup.py build_ext --inplace

# Run tests
pytest dso_pkg/dso/test/
```

### **Method 3: Development Setup**

For contributors and advanced users:

```bash
# Clone with development tools
git clone https://github.com/your-org/dso.git
cd dso

# Install development dependencies
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install

# Run full test suite
python tools/python/run.py test -v
```

## 📋 **System Requirements**

### **Minimum Requirements**
- **Python**: 3.10 or 3.11
- **RAM**: 4 GB minimum, 8 GB recommended
- **Storage**: 2 GB for installation + datasets
- **OS**: Linux, macOS, or Windows

### **Recommended Requirements**
- **Python**: 3.11 (best performance)
- **RAM**: 16 GB or more for large problems
- **CPU**: Multi-core processor (4+ cores recommended)
- **Storage**: SSD for faster I/O

### **Dependencies**

Core dependencies are automatically installed:

```
numpy >= 1.21.0
scipy >= 1.7.0
pandas >= 1.3.0
tensorflow >= 2.8.0
scikit-learn >= 1.0.0
cython >= 0.29.0
click >= 8.0.0
tqdm >= 4.60.0
```

Optional dependencies for advanced features:

```
# Visualization
matplotlib >= 3.5.0
seaborn >= 0.11.0

# Development
pytest >= 6.0.0
black >= 22.0.0
ruff >= 0.1.0
pre-commit >= 2.15.0

# Documentation
mkdocs >= 1.4.0
mkdocs-material >= 8.0.0
```

## 🚀 **Quick Verification**

After installation, verify everything works:

### **1. Basic Functionality**
```bash
# Test imports
python -c "from dso import DeepSymbolicOptimizer; print('✓ Basic import works')"

# Test sklearn interface
python -c "from dso import DeepSymbolicRegressor; print('✓ Sklearn interface works')"

# Test unified interface
python -c "from dso.unified_dso import UnifiedDSO; print('✓ Unified interface works')"
```

### **2. Run Test Suite**
```bash
# Quick test
pytest dso_pkg/dso/test/ -q

# Comprehensive test with MIMO
python tools/python/run.py test

# Expected output: All tests pass
```

### **3. Run Simple Benchmark**
```bash
# Run Nguyen-2 benchmark (should complete in ~5 minutes)
python tools/python/run.py bench-miso dso_pkg/dso/config/examples/regression/Nguyen-2.json

# Expected: Discovers x1^4 + x1^3 + x1^2 + x1
```

### **4. Test MIMO Functionality**
```bash
# Run MIMO benchmark
python tools/python/run.py bench-mimo

# Expected: Discovers multiple related expressions
```

## 🔧 **Configuration**

### **Environment Variables**

You can customize DSO behavior with environment variables:

```bash
# TensorFlow configuration
export TF_CPP_MIN_LOG_LEVEL=2          # Reduce TF logging
export TF_DETERMINISTIC_OPS=1          # Deterministic operations

# DSO configuration
export DSO_LOG_LEVEL=INFO              # Logging level
export DSO_CACHE_DIR=/path/to/cache    # Cache directory
export DSO_NUM_THREADS=4               # Thread count

# Memory configuration
export DSO_MAX_MEMORY_GB=8             # Memory limit
```

### **Configuration Files**

DSO uses JSON configuration files. Key locations:

```
dso_pkg/dso/config/
├── config_regression.json         # Default regression config
├── config_common.json            # Common settings
└── examples/
    └── regression/
        ├── Nguyen-1.json         # Simple polynomial
        ├── Nguyen-2.json         # Multi-term polynomial
        ├── MIMO-simple.json      # Basic MIMO
        └── MIMO-benchmark.json   # Complex MIMO
```

### **Custom Configuration**

Create custom configs for your problems:

```json
{
  "task": {
    "task_type": "regression",
    "dataset": "path/to/your/data.csv",
    "function_set": ["add", "sub", "mul", "div", "sin", "cos"],
    "metric": "inv_nrmse"
  },
  "training": {
    "n_samples": 2000000,
    "batch_size": 1000,
    "epsilon": 0.05
  },
  "policy": {
    "hidden_size": 128,
    "num_layers": 1
  }
}
```

## 🐛 **Troubleshooting**

### **Common Issues**

#### **Issue: ModuleNotFoundError**
```bash
# Problem: DSO modules not found
# Solution: Ensure virtual environment is activated
source .venv/bin/activate

# Verify installation
pip list | grep dso
```

#### **Issue: Cython Compilation Errors**
```bash
# Problem: C compiler not found
# Solution (Ubuntu/Debian):
sudo apt-get install build-essential

# Solution (macOS):
xcode-select --install

# Solution (Windows):
# Install Visual Studio Build Tools
```

#### **Issue: TensorFlow Warnings**
```bash
# Problem: Excessive TensorFlow logging
# Solution: Set environment variable
export TF_CPP_MIN_LOG_LEVEL=2

# Or in Python:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

#### **Issue: Memory Errors**
```bash
# Problem: Out of memory during training
# Solution: Reduce batch size and samples
{
  "training": {
    "n_samples": 500000,     # Reduced from 2M
    "batch_size": 500        # Reduced from 1000
  }
}
```

#### **Issue: Slow Training**
```bash
# Problem: Training takes too long
# Solution: Use multiple cores
{
  "training": {
    "n_cores_batch": 4       # Use 4 CPU cores
  }
}

# Or reduce problem complexity
{
  "prior": {
    "length": {
      "max_": 32             # Reduced from 64
    }
  }
}
```

### **Performance Optimization**

#### **CPU Optimization**
```bash
# Use all available cores for batch processing
export OMP_NUM_THREADS=$(nproc)

# Configure in config file
{
  "training": {
    "n_cores_batch": 8       # Adjust to your CPU count
  }
}
```

#### **Memory Optimization**
```bash
# For large datasets
{
  "training": {
    "batch_evaluation": true,
    "max_memory_gb": 16,
    "dataset_subsample": 0.8
  }
}
```

#### **GPU Support**
```bash
# Verify GPU availability
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"

# Enable GPU memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

### **Advanced Debugging**

#### **Enable Verbose Logging**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or in config
{
  "training": {
    "verbose": true
  },
  "logging": {
    "level": "DEBUG"
  }
}
```

#### **Profile Performance**
```python
# Time specific operations
import time

start = time.time()
result = model.train()
print(f"Training took: {time.time() - start:.2f}s")

# Memory profiling
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
```

### **Getting Help**

If you encounter issues not covered here:

1. **Check GitHub Issues**: [github.com/your-org/dso/issues](https://github.com/your-org/dso/issues)
2. **Review Documentation**: Browse these comprehensive guides
3. **Enable Debug Mode**: Add `"verbose": true` to your config
4. **Minimal Example**: Create a simple test case that reproduces the issue
5. **System Information**: Include Python version, OS, and error messages

### **Testing Your Installation**

Run this comprehensive test to verify your installation:

```python
#!/usr/bin/env python3
"""Comprehensive DSO installation test."""

def test_installation():
    print("🔧 Testing DSO Installation...")
    
    # Test 1: Basic imports
    try:
        from dso import DeepSymbolicOptimizer, DeepSymbolicRegressor
        from dso.unified_dso import UnifiedDSO
        print("✓ All imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 2: Simple regression
    try:
        import numpy as np
        X = np.random.rand(100, 2)
        y = X[:, 0] + X[:, 1]
        
        model = DeepSymbolicRegressor()
        # Quick test with minimal samples
        model.config_training['n_samples'] = 1000
        model.fit(X, y)
        print("✓ Basic regression test passed")
    except Exception as e:
        print(f"❌ Regression test failed: {e}")
        return False
    
    # Test 3: MIMO capability
    try:
        y_mimo = np.column_stack([X[:, 0] + X[:, 1], X[:, 0] * X[:, 1]])
        dso = UnifiedDSO()
        # Quick MIMO test
        results = dso.fit((X[:50], y_mimo[:50]))  # Small sample for speed
        print("✓ MIMO test passed")
    except Exception as e:
        print(f"❌ MIMO test failed: {e}")
        return False
    
    print("🎉 All tests passed! DSO is ready to use.")
    return True

if __name__ == "__main__":
    test_installation()
```

Save this as `test_installation.py` and run with:
```bash
python test_installation.py
```

## 🚀 **Next Steps**

Once DSO is installed and verified:

1. **Start with Tutorial**: Follow the [Getting Started Guide](getting_started.md)
2. **Try Examples**: Run built-in benchmarks with `python tools/python/run.py`
3. **Read Documentation**: Explore [Core Concepts](concept.md) and [Architecture](architecture.md)
4. **Custom Problems**: Apply DSO to your own datasets
5. **Advanced Features**: Explore [MIMO capabilities](mimo.md) and [constraints](constraints.md)

Your DSO installation is now complete and ready for mathematical discovery! 🧮✨