# Regression Features & Capabilities

> Version: 2.0 • Last updated: 2025-09-15

This document details DSO's comprehensive regression capabilities, with a focus on the new multi-output features and modular architecture.

## 🎯 **Regression Overview**

DSO transforms symbolic regression from a single-output, manually configured process into a comprehensive, automatically adapting system that handles diverse data structures.

### **Supported Data Variants**

DSO automatically detects and optimally handles different regression scenarios:

```mermaid
graph LR
    A[Input Data] --> B{Auto-Detect Variant}
    B --> C[SISO: x → y]
    B --> D[MISO: [x1,x2,x3] → y]
    B --> E[SIMO: x → [y1,y2,y3]]
    B --> F[MIMO: [x1,x2] → [y1,y2]]
    
    C --> G[Single Expression]
    D --> G
    E --> H[Multiple Expressions]
    F --> H
    
    style A fill:#e1f5fe
    style G fill:#c8e6c9
    style H fill:#f8bbd9
```

## 📊 **Data Variant Details**

### **SISO: Single Input Single Output**
```python
# Example: Simple polynomial relationship
X = np.array([[1], [2], [3], [4]])  # Single input feature
y = np.array([1, 4, 9, 16])         # Single output
# Expected: f(x) = x^2

from dso import DeepSymbolicRegressor
model = DeepSymbolicRegressor()
model.fit(X, y)
print(model.program_.pretty())  # Output: "x1^2"
```

**Use Cases:**
- Univariate function discovery
- Time series analysis (with lag features)
- Simple physical relationships

### **MISO: Multiple Input Single Output**
```python
# Example: Multi-variable physics equation
X = np.random.rand(100, 3)  # Three input features
y = np.sin(X[:,0]) + X[:,1] * X[:,2]  # Combined relationship
# Expected: f(x1,x2,x3) = sin(x1) + x2*x3

model = DeepSymbolicRegressor()
model.fit(X, y)
print(model.program_.pretty())  # Output: "sin(x1) + mul(x2, x3)"
```

**Use Cases:**
- Traditional symbolic regression
- Multi-variable scientific modeling
- Engineering design equations

### **SIMO: Single Input Multiple Output**
```python
# Example: Multiple transformations of single input
X = np.array([[1], [2], [3], [4]])  # Single input
y = np.column_stack([
    X.flatten() ** 2,          # x^2
    np.sin(X.flatten()),       # sin(x)
    np.exp(X.flatten())        # exp(x)
])
# Expected: [f1(x)=x^2, f2(x)=sin(x), f3(x)=exp(x)]

from dso.unified_dso import UnifiedDSO
dso = UnifiedDSO()
results = dso.fit((X, y))
for i, expr in enumerate(results["expressions"]):
    print(f"Output {i+1}: {expr}")
```

**Use Cases:**
- Signal processing (multiple frequency components)
- Feature engineering (creating derived features)
- Mathematical function decomposition

### **MIMO: Multiple Input Multiple Output**
```python
# Example: System of equations
X = np.random.rand(100, 2)  # Two input features  
y = np.column_stack([
    X[:,0] * X[:,1],           # x1 * x2
    X[:,0] + X[:,1]**2,        # x1 + x2^2
    np.sin(X[:,0]) - X[:,1]    # sin(x1) - x2
])
# Expected: Multiple related expressions

dso = UnifiedDSO()
results = dso.fit((X, y))
print("MIMO system discovered:")
for i, expr in enumerate(results["expressions"]):
    print(f"y{i+1} = {expr}")
```

**Use Cases:**
- Systems of differential equations
- Multi-objective optimization
- Coupled physical phenomena
- Economic modeling with multiple indicators

## 🔧 **Core Regression APIs**

### **1. Sklearn-style API (Recommended for Single Output)**
```python
from dso import DeepSymbolicRegressor

# Simple interface for MISO problems
regressor = DeepSymbolicRegressor(
    function_set=["add", "sub", "mul", "div", "sin", "cos"],
    max_complexity=20,
    n_samples=1000000
)

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Access the discovered expression
print("Formula:", regressor.program_.pretty())
print("Complexity:", regressor.program_.complexity)
```

### **2. UnifiedDSO API (Recommended for Multi-Output)**
```python
from dso.unified_dso import UnifiedDSO

# Automatically handles any data variant
dso = UnifiedDSO(verbose=True)
results = dso.fit((X, y))  # Automatic variant detection

# Results include all discovered expressions
print("Data variant detected:", results["data_variant"])
print("Number of expressions:", len(results["expressions"]))
for i, expr in enumerate(results["expressions"]):
    print(f"Expression {i+1}: {expr}")
```

### **3. Direct DSO API (Maximum Control)**
```python
from dso import DeepSymbolicOptimizer
from dso.config import load_config

# Full control over configuration
config = load_config("dso_pkg/dso/config/config_regression.json")
config["task"]["dataset"] = (X, y)  # Your data
config["task"]["function_set"] = ["add", "mul", "sin", "exp"]
config["training"]["n_samples"] = 2000000

model = DeepSymbolicOptimizer(config)
result = model.train()

print("Best expression:", result["expression"])
print("Reward:", result["r"])
```

## 🎛️ **Configuration Options**

### **Function Sets for Different Domains**

```python
# Basic arithmetic
basic_functions = ["add", "sub", "mul", "div"]

# Scientific computing
scientific_functions = ["add", "sub", "mul", "div", "sin", "cos", "exp", "log", "sqrt"]

# Advanced mathematics
advanced_functions = ["add", "sub", "mul", "div", "sin", "cos", "tan", 
                     "exp", "log", "sqrt", "pow", "abs", "inv"]

# Physics/Engineering
physics_functions = ["add", "sub", "mul", "div", "sin", "cos", "exp", "log", 
                    "sqrt", "pow", "abs", "inv", "tanh"]
```

### **Metrics for Different Problems**

```python
# Regression metrics
metrics = {
    "inv_nrmse": "1/(1 + normalized_rmse)",    # Default, higher is better
    "neg_mse": "-mean_squared_error",          # Negative MSE
    "pearson": "pearson_correlation",          # Linear correlation  
    "spearman": "spearman_correlation",        # Rank correlation
    "r2": "r_squared_coefficient"              # Coefficient of determination
}
```

## 🚀 **Advanced Features**

### **MIMO Training Strategies**

DSO provides different approaches for multi-output problems:

#### **Independent Expressions (Current Default)**
```python
# Each output gets its own independent expression
# Advantage: Simple, parallelizable
# Use case: When outputs are not strongly related

config = {
    "task": {"task_type": "regression"},
    "mimo_strategy": "independent",
    "training": {"n_samples": 1000000}
}
```

#### **Shared Components (Future)**
```python
# Expressions can share sub-expressions
# Advantage: More efficient for related outputs
# Use case: When outputs share common patterns

config = {
    "mimo_strategy": "shared_components",
    "shared_complexity_weight": 0.1
}
```

### **Benchmarking & Evaluation**

DSO includes comprehensive benchmarking capabilities:

```python
# Built-in benchmark datasets
benchmarks = [
    "Nguyen-1",    # x^3 + x^2 + x
    "Nguyen-2",    # x^4 + x^3 + x^2 + x
    "Nguyen-5",    # sin(x^2)*cos(x) - 1
    "Nguyen-7",    # log(x + 1) + log(x^2 + 1)
    "Keijzer-6",   # sum of 1/i from 1 to x
    "Korns-12",    # 2 - 2.1*cos(9.8*x)*sin(1.3*w)
]

# Run benchmark
python -m dso.run dso_pkg/dso/config/examples/regression/Nguyen-2.json

# MIMO benchmarks
mimo_benchmarks = [
    "MIMO-simple",     # 2x2 system
    "MIMO-benchmark",  # 3x3 system
    "MIMO-physics",    # Physical system simulation
]

# Run MIMO benchmark
python tools/python/run.py bench-mimo dso_pkg/dso/config/examples/regression/MIMO-simple.json
```

### **Custom Dataset Integration**

#### **CSV Files**
```python
# Load from CSV
config = {
    "task": {
        "task_type": "regression",
        "dataset": "path/to/your/data.csv",
        # CSV should have columns: x1, x2, ..., y or y1, y2, ...
    }
}
```

#### **NumPy Arrays**
```python
# Use in-memory data
X = np.random.rand(1000, 3)
y = some_function(X)

# Single-output
model = DeepSymbolicRegressor()
model.fit(X, y)

# Multi-output
dso = UnifiedDSO()
results = dso.fit((X, y))
```

#### **Pandas DataFrames**
```python
import pandas as pd

# Load complex dataset
df = pd.read_csv("complex_data.csv")
X = df[["feature1", "feature2", "feature3"]].values
y = df[["target1", "target2"]].values  # Multi-output

dso = UnifiedDSO()
results = dso.fit((X, y))
```

## 📈 **Performance Optimization**

### **Training Configuration**

```python
# Basic configuration
config = {
    "training": {
        "n_samples": 2000000,      # Total expressions to evaluate
        "batch_size": 1000,        # Expressions per training step
        "epsilon": 0.05,           # Exploration probability
        "n_cores_batch": 4,        # Parallel evaluation cores
        "early_stopping": True     # Stop when solution found
    }
}

# MIMO-specific optimizations
mimo_config = {
    "training": {
        "n_samples": 5000000,      # More samples for multi-output
        "batch_size": 500,         # Smaller batches for MIMO
        "mimo_batch_allocation": "proportional"  # Allocate by output complexity
    }
}
```

### **Memory Management**

```python
# For large datasets
large_data_config = {
    "training": {
        "batch_evaluation": True,   # Evaluate in batches
        "max_memory_gb": 8,        # Limit memory usage
        "dataset_subsample": 0.8   # Use subset for training
    }
}
```

## 🔍 **Results Analysis**

### **Single-Output Results**
```python
# Access results
result = model.train()
best_program = result["program"]

print("Expression:", best_program.pretty())
print("Complexity:", best_program.complexity)
print("Training reward:", result["r"])

# Test on new data
y_pred = best_program.execute(X_test)
mse = np.mean((y_test - y_pred) ** 2)
print("Test MSE:", mse)
```

### **Multi-Output Results**
```python
# MIMO results analysis
results = dso.fit((X, y))

print(f"Data variant: {results['data_variant']}")
print(f"Input features: {results['n_inputs']}")
print(f"Output targets: {results['n_outputs']}")

# Individual expression analysis
for i, expr in enumerate(results["expressions"]):
    print(f"\nOutput {i+1}:")
    print(f"  Expression: {expr}")
    print(f"  Complexity: {results['complexities'][i]}")
    print(f"  Reward: {results['rewards'][i]}")

# Combined prediction
if "combined_program" in results:
    y_pred_all = results["combined_program"].execute(X_test)
    print("Combined prediction shape:", y_pred_all.shape)
```

### **Hall of Fame Analysis**
```python
# Load detailed results from CSV
import pandas as pd

# Training statistics
stats_df = pd.read_csv("log/experiment_name/dso_experiment_0.csv")
print("Training progress:")
print(stats_df[["step", "r_best", "complexity_best"]].tail())

# Hall of Fame (best expressions found)
hof_df = pd.read_csv("log/experiment_name/dso_experiment_0_hof.csv")
print("\nTop 5 expressions:")
for i in range(min(5, len(hof_df))):
    row = hof_df.iloc[i]
    print(f"{i+1}. {row['expression']} (reward: {row['r']:.6f})")
```

## 🎯 **Best Practices**

### **For Single-Output Problems**
1. Start with basic function set: `["add", "sub", "mul", "div"]`
2. Add complexity gradually: `["sin", "cos"]`, then `["exp", "log"]`
3. Use `DeepSymbolicRegressor` for simple sklearn-style interface
4. Monitor training progress and adjust `n_samples` if needed

### **For Multi-Output Problems**
1. Use `UnifiedDSO` for automatic configuration
2. Ensure sufficient training samples: `n_samples ≥ 2M * n_outputs`
3. Consider output relationships when interpreting results
4. Use smaller batch sizes for better gradient estimates

### **For Large Datasets**
1. Enable batch evaluation to manage memory
2. Consider dataset subsampling for initial exploration
3. Use multiple cores: `n_cores_batch = min(8, cpu_count())`
4. Monitor memory usage and adjust accordingly

### **For Complex Functions**
1. Increase maximum complexity limit gradually
2. Use domain-specific function sets
3. Enable early stopping to avoid overfitting
4. Validate results on held-out test data

This comprehensive regression system makes DSO suitable for a wide range of scientific and engineering applications, from simple curve fitting to complex multi-output system identification.
