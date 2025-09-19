# Quick Start Guide

## Basic Usage

This guide demonstrates DSO's core functionality with a simple symbolic regression example.

### Import DSO

```python
import numpy as np
from dso import DeepSymbolicOptimizer
```

### Prepare Data

Create or load your dataset. For this example, we'll use a synthetic function:

```python
# Generate synthetic data: y = x^2 + 2*x + 1
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = X[:, 0]**2 + 2*X[:, 0] + 1 + np.random.normal(0, 0.1, 100)
```

### Configure DSO

```python
# Create DSO instance with default configuration
dso = DeepSymbolicOptimizer(
    task="regression",
    function_set=["add", "sub", "mul", "div", "pow", "sin", "cos", "exp", "log"],
    max_length=20,
    n_samples=1000000
)
```

### Train Model

```python
# Fit the model to discover mathematical expressions
dso.fit(X, y)
```

### Retrieve Results

```python
# Get the best discovered expression
best_expression = dso.program_
print(f"Discovered expression: {best_expression}")

# Evaluate the expression
predictions = dso.predict(X)
mse = np.mean((y - predictions)**2)
print(f"Mean Squared Error: {mse:.6f}")
```

### Complete Example

```python
import numpy as np
from dso import DeepSymbolicOptimizer

# Generate data
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = X[:, 0]**2 + 2*X[:, 0] + 1 + np.random.normal(0, 0.1, 100)

# Configure and train DSO
dso = DeepSymbolicOptimizer(
    task="regression",
    function_set=["add", "sub", "mul", "pow"],
    max_length=15,
    n_samples=500000
)

# Train and get results
dso.fit(X, y)
print(f"Best expression: {dso.program_}")
print(f"Training MSE: {dso.r_best_:.6f}")
```

## Next Steps

- Learn about [regression capabilities](/regression/overview)
- Explore [configuration options](/regression/configuration)
- Review [practical examples](/examples/basic-regression)