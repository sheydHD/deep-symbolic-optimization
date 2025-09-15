# MIMO Implementation Guide

> Version: 2.0 • Last updated: 2025-09-02 • **Implementation Status: ✅ Complete**

This guide covers the complete implementation of MIMO (Multiple Input Multiple Output) functionality in DSO, including architecture, usage, and real-world examples.

## 🎯 **Implementation Overview**

DSO now supports multiple data variants with automatic detection and configuration:

```mermaid
graph TD
    A[Data Input] --> B{Variant Detection}
    B --> C[SISO: f: R → R]
    B --> D[MISO: f: R^n → R]
    B --> E[SIMO: f: R → R^m]
    B --> F[MIMO: f: R^n → R^m]
    
    C --> G[Classic DSO Pipeline]
    D --> G
    E --> H[Multi-Output Pipeline]
    F --> H
    
    G --> I[Single Expression Discovery]
    H --> J[Multiple Expression Discovery]
    
    style A fill:#e1f5fe
    style B fill:#fff3e0
    style I fill:#c8e6c9
    style J fill:#f8bbd9
```

## 🏗️ **Architecture Components**

### **Three Implementation Approaches**

DSO now provides three ways to handle symbolic regression:

```mermaid
graph LR
    A[DSO Framework] --> B[Classic DSO]
    A --> C[Fixed DSO]
    A --> D[Unified DSO]
    
    B --> E[core.py<br/>Legacy SISO/MISO]
    C --> F[core_fixed.py<br/>MIMO Support]
    D --> G[core/ directory<br/>Modular System]
    
    style A fill:#e1f5fe
    style E fill:#c8e6c9
    style F fill:#f8bbd9
    style G fill:#fff3e0
```

#### **1. Classic DSO (`core.py`)**
- **Purpose**: Maintains backward compatibility
- **Supports**: SISO, MISO (traditional symbolic regression)
- **Usage**: Existing codebases and benchmarks
- **Entry Point**: `DeepSymbolicOptimizer`

#### **2. Fixed DSO (`core_fixed.py`)**
- **Purpose**: MIMO-enabled DSO with proper initialization
- **Supports**: SISO, MISO, MIMO
- **Usage**: Direct MIMO symbolic regression
- **Entry Point**: `DeepSymbolicOptimizerFixed`

#### **3. Unified DSO (`core/` modular system)**
- **Purpose**: Automatic variant detection and configuration
- **Supports**: All variants with auto-detection
- **Usage**: Most flexible, automatic optimization
- **Entry Point**: `UnifiedDSO`, `auto_fit()`

## 🚀 **MIMO Implementation Status**

### **✅ Completed Features**

#### **Data Handling**
- [x] Automatic variant detection (SISO/MISO/SIMO/MIMO)
- [x] Multi-output data loading and validation
- [x] Benchmark datasets (MIMO-simple, MIMO-benchmark, MIMO-easy)
- [x] CSV file support for custom datasets

#### **Core Components**
- [x] Fixed initialization order (`Program.set_execute()` before `Program.set_task()`)
- [x] Multi-output program execution
- [x] MIMO-aware policy configuration
- [x] Multi-objective evaluation metrics

#### **Training System**
- [x] Independent expression strategy (Option A)
- [x] Multi-output reward computation
- [x] Proper gradient handling for multiple outputs
- [x] Early stopping for MIMO tasks

#### **Integration & Compatibility**
- [x] Full backward compatibility with existing SISO/MISO code
- [x] Resolved import conflicts between `core.py` and `core/` directory
- [x] All original tests passing (33/33)
- [x] MIMO test suite complete

### **📋 Available MIMO Datasets**

```mermaid
graph TD
    A[MIMO Datasets] --> B[MIMO-easy<br/>2→2: x1, x2]
    A --> C[MIMO-simple<br/>2→2: x1+x2, x1*x2]
    A --> D[MIMO-benchmark<br/>3→3: x1*x2, sin(x3), x1+x2*x3]
    
    B --> E[20,000 samples<br/>Max length: 10]
    C --> F[50,000 samples<br/>Max length: 15]
    D --> G[100,000 samples<br/>Max length: 20]
    
    style A fill:#e1f5fe
    style E fill:#c8e6c9
    style F fill:#c8e6c9
    style G fill:#c8e6c9
```

## 💻 **Usage Examples**

### **Quick Start: Auto-Detection**

```python
from dso.unified_dso import auto_fit

# Your MIMO data
X = np.random.randn(200, 3)  # 3 inputs
y = np.column_stack([
    X[:, 0] * X[:, 1],        # Output 1: x1 * x2
    np.sin(X[:, 2]),          # Output 2: sin(x3)
    X[:, 0] + X[:, 1] * X[:, 2]  # Output 3: x1 + x2*x3
])

# Automatic MIMO detection and training
result = auto_fit(dataset=(X, y), training={'n_samples': 5000})

print(f"Detected: {result['variant_info']['variant']}")
print(f"Best expressions: {result['best_program']}")
```

### **Fixed DSO: Direct MIMO Control**

```python
from dso.core_fixed import DeepSymbolicOptimizerFixed

config = {
    'task': {
        'task_type': 'regression',
        'dataset': 'MIMO-benchmark',  # 3→3 benchmark
        'function_set': ['add', 'sub', 'mul', 'div', 'sin', 'cos'],
        'metric': 'inv_nrmse'
    },
    'training': {
        'n_samples': 10000,
        'batch_size': 100,
        'early_stopping': True
    },
    'policy': {
        'policy_type': 'rnn',
        'max_length': 20
    }
}

# Create and train MIMO DSO
dso = DeepSymbolicOptimizerFixed(config)
dso.setup()

# Training loop
for i in range(100):
    result = dso.train_one_step()
    if result is not None:
        print(f"Training completed: {result}")
        break
```

### **Unified DSO: Advanced Configuration**

```python
from dso.unified_dso import UnifiedDSO

# Create unified DSO with custom config
dso = UnifiedDSO(config_template="config_modular.json")

# Fit with automatic configuration
result = dso.fit(
    dataset=(X, y),
    training={'n_samples': 50000, 'batch_size': 200},
    policy={'multi_output_strategy': 'independent'},
    task={'function_set': ['add', 'mul', 'sin', 'cos', 'exp']}
)

# Evaluate on new data
predictions = dso.predict(X_new)
metrics = dso.evaluate(X_test, y_test)
```

## 🎛️ **MIMO Training Strategies**

### **Strategy Selection**

```mermaid
graph TD
    A[MIMO Training] --> B[Independent Strategy]
    A --> C[Shared Strategy]
    A --> D[Vector Strategy]
    
    B --> E[Separate expression<br/>per output]
    C --> F[Shared sub-expressions<br/>across outputs]
    D --> G[Single vector-valued<br/>expression]
    
    B --> H[✅ Implemented]
    C --> I[🚧 Planned]
    D --> J[🚧 Planned]
    
    style A fill:#e1f5fe
    style H fill:#c8e6c9
    style I fill:#fff3e0
    style J fill:#fff3e0
```

### **Independent Strategy (Current Implementation)**

Each output gets its own symbolic expression:

```python
# Example MIMO result
mimo_expressions = {
    'output_0': 'x1 * x2',           # First output
    'output_1': 'sin(x3)',          # Second output  
    'output_2': 'x1 + x2 * x3'      # Third output
}
```

**Advantages:**
- Maximum interpretability
- Independent optimization per output
- Robust to different output complexities
- Easy to implement and debug

## 📊 **MIMO Evaluation Metrics**

### **Multi-Output Performance Tracking**

```mermaid
graph LR
    A[MIMO Evaluation] --> B[Per-Output Metrics]
    A --> C[Overall Metrics]
    
    B --> D[output_0_nmse<br/>output_1_nmse<br/>output_2_nmse]
    B --> E[output_0_success<br/>output_1_success<br/>output_2_success]
    
    C --> F[nmse_mean<br/>nmse_max<br/>nmse_weighted]
    C --> G[success_all<br/>success_any<br/>success_rate]
    
    style A fill:#e1f5fe
    style D fill:#c8e6c9
    style E fill:#c8e6c9
    style F fill:#f8bbd9
    style G fill:#f8bbd9
```

### **Example Metrics Output**

```python
mimo_metrics = {
    # Per-output performance
    "output_0_nmse": 0.0001,
    "output_1_nmse": 0.0005, 
    "output_2_nmse": 0.0002,
    
    # Success flags
    "output_0_success": True,
    "output_1_success": True,
    "output_2_success": True,
    
    # Overall performance
    "nmse_mean": 0.00027,        # Average across outputs
    "nmse_max": 0.0005,          # Worst output
    "success_all": True,         # All outputs successful
    "success_rate": 1.0,         # Fraction successful
    
    # Complexity metrics
    "total_complexity": 45,      # Sum of expression complexities
    "mean_complexity": 15        # Average complexity
}
```

## 🔧 **Configuration Reference**

### **MIMO-Specific Configuration**

```json
{
  "task": {
    "task_type": "regression",
    "dataset": "MIMO-benchmark",
    
    // MIMO detection and handling
    "mimo_strategy": "independent",     // "independent", "shared", "vector"
    "mimo_reward": "mean",              // "mean", "weighted", "min", "max"
    "mimo_training": "joint",           // "joint", "alternating", "hierarchical"
    
    // Per-output customization
    "output_weights": [1.0, 1.5, 0.8], // Importance weights
    "output_thresholds": [1e-12, 1e-10, 1e-12] // Success thresholds
  },
  
  "training": {
    "n_samples": 100000,
    "batch_size": 1000,
    "early_stopping": true
  },
  
  "policy": {
    "policy_type": "rnn",
    "max_length": 20,
    "multi_output_strategy": "independent"
  }
}
```

## 🧪 **Testing & Validation**

### **Test Suite Status**

```mermaid
graph TD
    A[DSO Test Suite] --> B[Original Tests: 33/33 ✅]
    A --> C[MIMO Tests: 3/3 ✅]
    
    C --> D[SISO Compatibility ✅]
    C --> E[Simple MIMO 2→2 ✅]
    C --> F[Benchmark MIMO 3→3 ✅]
    
    style A fill:#e1f5fe
    style B fill:#c8e6c9
    style C fill:#c8e6c9
    style D fill:#c8e6c9
    style E fill:#c8e6c9
    style F fill:#c8e6c9
```

### **Running Tests**

```bash
# Original test suite (backward compatibility)
bash main.sh test

# MIMO-specific tests
python test_mimo_fixed.py

# Modular system demo
python tests/modular_dso_demo.py
```

## 🚧 **Future Enhancements**

### **Planned Features**

```mermaid
graph TD
    A[Future MIMO Features] --> B[Shared Sub-expressions]
    A --> C[Vector-Valued Functions]
    A --> D[Hierarchical Training]
    A --> E[Multi-Objective Optimization]
    
    B --> F[Common computations<br/>across outputs]
    C --> G[Single expressions<br/>multiple outputs]
    D --> H[Progressive difficulty<br/>training]
    E --> I[Pareto optimization<br/>accuracy vs complexity]
    
    style A fill:#e1f5fe
    style F fill:#fff3e0
    style G fill:#fff3e0
    style H fill:#fff3e0
    style I fill:#fff3e0
```

### **Research Opportunities**

- **Cross-Output Learning**: How outputs can inform each other's discovery
- **MIMO Constraints**: Constraints operating across multiple outputs
- **Transfer Learning**: Using MISO models to bootstrap MIMO discovery
- **Benchmarking**: Comprehensive MIMO benchmark suite development

## 📚 **References & Resources**

- **Implementation Files**: `core_fixed.py`, `unified_dso.py`, `core/` modules
- **Test Files**: `test_mimo_fixed.py`, `tests/modular_dso_demo.py`
- **Config Files**: `config_modular.json`, MIMO example configs
- **Documentation**: `MIMO_IMPLEMENTATION_SUMMARY.md`, `MIMO_INTEGRATION_FIXES.md`

## 🎉 **Conclusion**

The MIMO implementation in DSO represents a significant advancement in symbolic regression capabilities:

- **✅ Full Implementation**: Complete MIMO support with automatic detection
- **✅ Backward Compatibility**: All existing functionality preserved
- **✅ Production Ready**: Tested and validated implementation
- **✅ Extensible**: Clean architecture for future enhancements

DSO now handles the full spectrum of symbolic regression problems, from simple single-output tasks to complex multi-output systems, while maintaining its core strength of discovering interpretable mathematical relationships.
