# Getting Started with DSO

> Version: 1.0 • Last updated: 2025-09-01

This guide walks you through setting up DSO and running your first symbolic regression experiment on the classic **Nguyen-2** benchmark. You'll learn the complete workflow from installation to result analysis.

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

# Run automated setup
./main.sh  
# Press '1' when prompted to setup environment

# Activate the virtual environment
source .venv/bin/activate
```

### What This Does

The setup script will:
1. Create a Python virtual environment using `uv`
2. Install all required dependencies 
3. Compile Cython extensions for fast execution
4. Install DSO in development mode

### Verify Installation

Test that everything works correctly:

```bash
# Run the test suite
pytest -q dso/dso/test/

# Expected output: All tests should pass
```

## Step 2: Your First Experiment

Let's run symbolic regression on the **Nguyen-2** benchmark, which tries to discover the formula `x^4 + x^3 + x^2 + x`.

### Run the Benchmark

```bash
python tools/python/benchmark/benchmark.py dso/dso/config/examples/regression/Nguyen-2.json
```

### What Happens During Training

1. **Initialization**: DSO loads the Nguyen-2 dataset and configures the search space
2. **Expression Generation**: Neural network generates candidate mathematical expressions
3. **Evaluation**: Each expression is tested on training data and assigned a fitness score
4. **Learning**: Network learns to generate better expressions using reinforcement learning
5. **Iteration**: Process repeats for multiple generations until convergence

### Monitor Progress

During training, you'll see output like:

```
Generation 0: Best reward = -2.45, Best complexity = 12
Generation 100: Best reward = -0.85, Best complexity = 8  
Generation 500: Best reward = -0.01, Best complexity = 7
...
Found exact solution: x1^4 + x1^3 + x1^2 + x1
```

Training typically takes 5-20 minutes depending on your hardware.

## Step 3: Analyze Results

### Locate Output Files

Results are saved in a timestamped directory:

```
log/Nguyen-2_YYYY-MM-DD-HHMMSS/
├── config.json              # Experiment configuration
├── dso_Nguyen-2_0.csv       # Training statistics
├── dso_Nguyen-2_0_hof.csv   # Hall of Fame (best expressions)
├── dso_Nguyen-2_plot_hof.png # Visualization of results
└── summary.csv              # Final summary
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
# Easy benchmark (polynomial)
python tools/python/benchmark/benchmark.py dso/dso/config/examples/regression/Nguyen-1.json

# Medium difficulty (trigonometric)  
python tools/python/benchmark/benchmark.py dso/dso/config/examples/regression/Nguyen-5.json

# Hard benchmark (complex composition)
python tools/python/benchmark/benchmark.py dso/dso/config/examples/regression/Nguyen-12.json
```

### Run System Tests

Verify the complete installation:

```bash
# Run all unit tests
pytest -q dso/dso/test/

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

```python
# Load your data
import pandas as pd
data = pd.read_csv("your_dataset.csv")
X = data[["input1", "input2", "input3"]].values
y = data["target"].values

# Run DSO
from dso import DeepSymbolicRegressor
model = DeepSymbolicRegressor()
model.fit(X, y)

print("Discovered formula:", model.program_.pretty())
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
