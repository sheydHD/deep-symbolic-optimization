# Getting Started with DSO

> Version: 2.0 • Last updated: 2025-09-15

This guide walks you through setting up DSO and running your first symbolic regression experiment on the classic **Nguyen-2** benchmark. You'll experience DSO's powerful modular architecture with support for both single and multi-output regression (MIMO).

## 🎯 **Why DSO for Symbolic Regression?**

DSO discovers **exact mathematical formulas** from data instead of black-box models:

```python
# Traditional ML: y = neural_network.predict(x)  ← Black box
# DSO Result:     y = sin(x1) + x2*x3 - 0.5     ← Interpretable formula!
```

### **Key Advantages**
- 🔬 **Scientific Discovery**: Find interpretable mathematical relationships
- 🎯 **Reproducible Results**: Deterministic training with configurable seeding
- ⚡ **High Performance**: Optimized execution with Cython acceleration
- 📊 **Multi-Output**: Discover multiple related expressions simultaneously (MIMO)
- 🧩 **Modular Design**: Automatic data variant detection (SISO/MISO/SIMO/MIMO)
- 🧪 **Research-Ready**: Ideal for scientific computing and analysis

## Prerequisites

Before starting, ensure you have:
- **Python 3.10 or 3.11** (check with `python --version`)
- **Git** for cloning the repository
- **GCC/Clang** toolchain for compiling Cython extensions
- (Optional) **CUDA-enabled GPU** for faster training

## Step 1: Installation & Setup

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/your-org/dso.git && cd dso

# Run automated setup using the modern approach
./main.sh modern setup

# Activate the virtual environment
source .venv/bin/activate
```

### What This Does

The setup script will:
1. Create a Python virtual environment using `uv`
2. Install all required dependencies from `pyproject.toml`
3. Compile Cython extensions for fast execution
4. Install DSO in development mode
5. Configure the environment for optimal performance

### Verify Installation

Test that everything works correctly:

```bash
# Run the test suite (includes MIMO tests)
pytest -q dso_pkg/dso/test/

# Or use the tools helper
python tools/python/run.py test

# Expected output: All tests should pass
```

## Step 2: Your First Regression Experiment

Let's run symbolic regression on the **Nguyen-2** benchmark, which tries to discover the formula `x^4 + x^3 + x^2 + x`. This demonstrates DSO's power to find exact mathematical relationships.

### Run the Benchmark

There are multiple ways to run DSO benchmarks:

#### Option 1: Direct command line
```bash
python -m dso.run dso_pkg/dso/config/examples/regression/Nguyen-2.json
```

#### Option 2: Interactive menu (Recommended)
```bash
python tools/python/run.py
# Select option 3 for MISO benchmark
```

#### Option 3: Tools helper
```bash
python tools/python/run.py bench-miso dso_pkg/dso/config/examples/regression/Nguyen-2.json
```

### What Happens During Training

1. **🚀 Environment Setup**: DSO configures the environment and loads the dataset
2. **🧠 Neural Policy Setup**: RNN policy network initialized for expression generation
3. **📊 Expression Generation**: Neural network generates candidate mathematical expressions
4. **⚡ Fast Evaluation**: Each expression is tested on training data using optimized Cython operations
5. **🎯 Policy Learning**: Network learns using REINFORCE algorithm with reward feedback
6. **🔄 Iteration**: Process repeats until convergence or completion criteria

### Monitor Progress

During training, you'll see output like:

```
Generation 0: Best reward = -2.45, Best complexity = 12, Expressions: 50
Generation 100: Best reward = -0.85, Best complexity = 8, Expressions: 45  
Generation 500: Best reward = -0.01, Best complexity = 7, Expressions: 20
...
🎉 Found exact solution: x1^4 + x1^3 + x1^2 + x1 (Reward: 1.0000)
```

**Key Benefits You'll Notice:**
- ⚡ **Efficient Training**: Optimized Cython execution speeds up evaluation
- 🎯 **Reproducible Results**: Consistent results with proper seeding
- 💾 **Memory Efficient**: Smart memory management for large expression spaces
- 📱 **Progress Monitoring**: Live training statistics and best expressions

Training typically takes 5-20 minutes depending on your hardware and configuration.

### Performance Example

```python
# Example of what DSO achieves with proper configuration:
# Run 1 (seed=0): Discovered x1^4 + x1^3 + x1^2 + x1, Reward: 1.0000, Time: 8.3min
# Run 2 (seed=0): Discovered x1^4 + x1^3 + x1^2 + x1, Reward: 1.0000, Time: 8.3min
# Run 3 (seed=0): Discovered x1^4 + x1^3 + x1^2 + x1, Reward: 1.0000, Time: 8.3min
# ← Consistent results with proper seeding! Perfect for scientific research
```

## Step 3: Multi-Output Regression (MIMO)

Try DSO's advanced **MIMO** capability to discover multiple related expressions simultaneously:

### Run MIMO Example

There are several ways to run MIMO (Multiple Input, Multiple Output) experiments:

#### Option 1: Interactive menu
```bash
python tools/python/run.py
# Select option 4 for MIMO benchmark
```

#### Option 2: Direct MIMO command
```bash
python tools/python/run.py bench-mimo
# Uses default MIMO config: MIMO-benchmark.json
```

#### Option 3: Custom MIMO config
```bash
python tools/python/run.py bench-mimo dso_pkg/dso/config/examples/regression/MIMO-simple.json
```

### What MIMO Discovers

```python
# Instead of finding one expression:
# y = f(x1, x2, x3)

# MIMO finds multiple related expressions:
# y1 = sin(x1) + x2        ← Output 1
# y2 = x1^2 - cos(x3)      ← Output 2  
# y3 = exp(x1 * x2)        ← Output 3

# Perfect for:
# 🔬 Systems of equations
# 📊 Multi-target prediction
# 🧬 Scientific modeling
```

### MIMO Training Benefits

- **🔗 Related Discovery**: Find multiple related expressions simultaneously
- **⚡ Efficiency**: Discover multiple outputs in one training session
- **🎯 Consistency**: Expressions with appropriate complexity for each output
- **🔬 System Analysis**: Understand multi-dimensional relationships in data
- **🧩 Automatic Detection**: DSO automatically detects MIMO data and configures accordingly

## Step 4: Analyze Results

### Locate Output Files

Results are saved in a timestamped directory:

```
log/Nguyen-2_YYYY-MM-DD-HHMMSS/
├── config.json              # Experiment configuration
├── dso_Nguyen-2_0.csv       # Training statistics
├── dso_Nguyen-2_0_hof.csv   # Hall of Fame (best expressions)
├── dso_Nguyen-2_plot_hof.png # Visualization of results
└── summary.csv              # Final summary with metrics
```

### Load and Inspect Results

```python
import pandas as pd
import numpy as np
from dso import Program

# Load the hall of fame (best expressions found)
hof_df = pd.read_csv("log/Nguyen-2_YYYY-MM-DD-HHMMSS/dso_Nguyen-2_0_hof.csv")

# Display the best expressions
print("Top 5 expressions found:")
for i in range(min(5, len(hof_df))):
    expr = hof_df.iloc[i]
    print(f"{i+1}. {expr['expression']} (reward: {expr['r']:.6f})")

# Load and examine the best program
best_tokens = hof_df.iloc[0]['traversal']  # Token sequence
best_program = Program.from_tokens(eval(best_tokens))

print(f"\nBest expression: {best_program.pretty()}")
print(f"Complexity: {best_program.complexity}")
print(f"Reward: {hof_df.iloc[0]['r']:.8f}")
```

### Test the Discovered Expression

```python
# Test the discovered expression on new data
X_test = np.random.uniform(-1, 1, (100, 1))  # Random test inputs
y_true = X_test**4 + X_test**3 + X_test**2 + X_test  # True function
y_pred = best_program.execute(X_test)  # Predicted values

# Calculate accuracy
mse = np.mean((y_true.flatten() - y_pred) ** 2)
print(f"Test MSE: {mse:.8f}")

if mse < 1e-10:
    print("🎉 Perfect match! DSO found the exact formula.")
else:
    print(f"Close approximation with MSE = {mse:.2e}")
```

## Step 4: Advanced Testing

### Run Multiple Benchmarks

Try other classic benchmarks to see DSO's capabilities:

```bash
# Easy benchmark (polynomial): x^3 + x^2 + x
python -m dso.run dso_pkg/dso/config/examples/regression/Nguyen-1.json

# Medium difficulty (trigonometric): e^(-x^2) * sin(3*pi*x)
python -m dso.run dso_pkg/dso/config/examples/regression/Nguyen-5.json

# Hard benchmark (complex composition)
python -m dso.run dso_pkg/dso/config/examples/regression/Nguyen-12.json

# Or use the interactive menu for guided selection
python tools/python/run.py
```

### Run System Tests

Verify the complete installation:

```bash
# Run all unit tests (including MIMO tests)
pytest -q dso_pkg/dso/test/

# Or use the tools helper
python tools/python/run.py test

# Expected output: All tests pass
```

## Step 5: Understanding Configuration

DSO uses JSON configuration files to specify experiments. Let's examine the Nguyen-2 config:

```json
{
  "task": {
    "task_type": "regression",
    "dataset": "Nguyen-2",
    "function_set": ["add", "sub", "mul", "div", "sin", "cos", "exp", "log"],
    "metric": "neg_nmse"
  },
  "training": {
    "n_samples": 2000000,
    "batch_size": 1000,
    "epsilon": 0.05
  },
  "policy": {
    "policy_type": "rnn",
    "cell": "lstm", 
    "num_layers": 1,
    "hidden_size": 32
  }
}
```

**Key Parameters:**
- **`function_set`**: Available mathematical operators
- **`n_samples`**: Number of expressions to evaluate during training
- **`batch_size`**: Expressions evaluated per training step
- **`epsilon`**: Exploration probability for the policy

## Next Steps

Now that you've successfully run your first DSO experiment, explore these topics:

### 🧠 **Understand the Fundamentals**
- **[Core Concepts](concept.md)** - Learn about symbolic regression and reinforcement learning
- **[System Architecture](architecture.md)** - Understand how DSO components work together

### 🔧 **Customize Your Experiments**
- **[Token System](tokens.md)** - Add custom mathematical operators
- **[Training Process](training.md)** - Tune training parameters for better results
- **[Constraints](constraints.md)** - Incorporate domain knowledge and physical laws

### 🚀 **Advanced Applications**
- **[MIMO Extensions](mimo.md)** - Multi-output regression for complex systems
- **[Advanced Features](advanced.md)** - Multi-task learning and ensemble methods

### 📊 **Real Data Applications**

Try DSO on your own datasets:

#### Single-Output Regression
```python
# Load your data
import pandas as pd
data = pd.read_csv("your_dataset.csv")
X = data[["input1", "input2", "input3"]].values
y = data["target"].values

# Run DSO with sklearn-style API
from dso import DeepSymbolicRegressor
model = DeepSymbolicRegressor()
model.fit(X, y)

print("Discovered formula:", model.program_.pretty())
```

#### Multi-Output Regression (MIMO)
```python
# Multi-output data
X = data[["input1", "input2"]].values
y = data[["target1", "target2", "target3"]].values

# Use UnifiedDSO for automatic MIMO detection
from dso.unified_dso import UnifiedDSO
dso = UnifiedDSO()
results = dso.fit((X, y))

print("MIMO expressions found:")
for i, expr in enumerate(results["expressions"]):
    print(f"Output {i+1}: {expr}")
```

## Troubleshooting

### Common Issues

**Issue**: "ModuleNotFoundError: No module named 'dso'"
- **Solution**: Make sure you activated the virtual environment: `source .venv/bin/activate`

**Issue**: Cython compilation errors
- **Solution**: Install development tools: `sudo apt-get install build-essential` (Ubuntu) or `xcode-select --install` (macOS)

**Issue**: Slow training on CPU
- **Solution**: Consider using GPU acceleration or reducing `n_samples` in the config

**Issue**: Training doesn't converge
- **Solution**: Try increasing `n_samples` or adjusting the `function_set` for your problem

### Get Help

- **GitHub Issues**: Report bugs and get support
- **Documentation**: Browse the comprehensive guides in this documentation
- **Community**: Join discussions with other DSO users

---

**Congratulations!** 🎉 You've successfully set up DSO and run your first symbolic regression experiment. You're now ready to explore the fascinating world of automated mathematical discovery.
