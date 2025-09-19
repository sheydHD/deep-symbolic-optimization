# Multi-Output Regression (MIMO)

Multi-output regression extends DSO's capabilities to simultaneously discover mathematical expressions for multiple target variables. This is particularly valuable for modeling systems with interconnected outputs or discovering systems of equations.

## Problem Formulation

For a dataset with input features `X ∈ ℝ^(n×d)` and multiple target variables `Y ∈ ℝ^(n×m)`, DSO discovers a system of expressions:

```
y₁ = f₁(x₁, x₂, ..., xₑ)
y₂ = f₂(x₁, x₂, ..., xₑ)
...
yₘ = fₘ(x₁, x₂, ..., xₑ)
```

Where each `fᵢ` is an independent mathematical expression optimized for the corresponding output variable.

## Configuration

### Basic MIMO Setup

```python
from dso import DeepSymbolicOptimizer

# Configure for multi-output regression
dso = DeepSymbolicOptimizer(
    task="regression",
    function_set=["add", "sub", "mul", "div", "sin", "cos", "exp"],
    max_length=20,
    n_samples=1000000,
    multi_output=True
)
```

### Data Format

```python
import numpy as np

# Input features (same as single output)
X = np.array([[x11, x12, ..., x1d],
              [x21, x22, ..., x2d],
              ...])  # Shape: (n_samples, n_features)

# Multiple target variables
Y = np.array([[y11, y12, ..., y1m],
              [y21, y22, ..., y2m],
              ...])  # Shape: (n_samples, n_outputs)
```

## Training Process

### Model Training

```python
# Train MIMO model
dso.fit(X, Y)

# Access results for each output
expressions = dso.programs_  # List of expressions for each output
scores = dso.r_best_list_   # Performance scores for each output
```

### Individual Output Analysis

```python
# Analyze each discovered expression
for i, (expr, score) in enumerate(zip(expressions, scores)):
    print(f"Output {i+1}:")
    print(f"  Expression: {expr}")
    print(f"  Training Score: {score:.6f}")
    print(f"  Length: {len(expr.tokens)}")
    print()
```

### Prediction

```python
# Predict all outputs simultaneously
Y_pred = dso.predict(X)  # Shape: (n_samples, n_outputs)

# Access individual output predictions
y1_pred = Y_pred[:, 0]
y2_pred = Y_pred[:, 1]
# ... etc
```

## MIMO-Specific Features

### Automatic Variant Detection

DSO automatically detects the data variant:

```python
# DSO automatically determines:
# SISO: Single input, single output
# MISO: Multiple inputs, single output  
# SIMO: Single input, multiple outputs
# MIMO: Multiple inputs, multiple outputs

print(f"Detected variant: {dso.data_variant_}")
```

### Shared Variable Optimization

All output expressions share the same input variables, enabling discovery of relationships between outputs:

```python
# Example discovered system:
# y1 = sin(x1) + x2²
# y2 = cos(x1) - x2
# y3 = x1 * x2 + 0.5
```

### Independent Expression Evolution

Each output variable has its own expression evolution process, allowing for different:
- Expression complexity per output
- Convergence rates per output
- Optimal function sets per output

## Advanced Configuration

### Output-Specific Parameters

```python
# Configure different parameters per output
dso = DeepSymbolicOptimizer(
    task="regression",
    function_set=["add", "sub", "mul", "div", "sin", "cos"],
    max_length=[15, 20, 25],  # Different max lengths per output
    multi_output=True,
    output_weights=[1.0, 2.0, 1.5]  # Weight importance of each output
)
```

### Coupled Training

```python
# Enable coupling between output expressions
dso = DeepSymbolicOptimizer(
    task="regression",
    multi_output=True,
    coupled_training=True,  # Expressions can reference other outputs
    coupling_strength=0.1   # Strength of coupling regularization
)
```

## Performance Analysis

### Individual Output Metrics

```python
from sklearn.metrics import mean_squared_error, r2_score

# Calculate metrics for each output
for i in range(Y.shape[1]):
    y_true = Y[:, i]
    y_pred = Y_pred[:, i]
    
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"Output {i+1}: MSE={mse:.6f}, R²={r2:.6f}")
```

### Overall System Performance

```python
# Overall system metrics
overall_mse = np.mean([mean_squared_error(Y[:, i], Y_pred[:, i]) 
                       for i in range(Y.shape[1])])
overall_r2 = np.mean([r2_score(Y[:, i], Y_pred[:, i]) 
                      for i in range(Y.shape[1])])

print(f"System MSE: {overall_mse:.6f}")
print(f"System R²: {overall_r2:.6f}")
```

## Applications

### Physical Systems
Model systems with multiple observable quantities:
```python
# Example: Projectile motion
# Inputs: [initial_velocity, angle, time]
# Outputs: [x_position, y_position, velocity_x, velocity_y]
```

### Engineering Applications
```python
# Example: Heat exchanger
# Inputs: [flow_rate, inlet_temp, pressure]
# Outputs: [outlet_temp, heat_transfer, pressure_drop]
```

### Economic Modeling
```python
# Example: Market analysis
# Inputs: [interest_rate, inflation, gdp_growth]
# Outputs: [stock_index, bond_yield, currency_value]
```

## Best Practices

### Data Preparation
1. **Scaling**: Normalize all outputs to similar ranges
2. **Correlation Analysis**: Identify relationships between outputs
3. **Temporal Alignment**: Ensure time-series outputs are synchronized

### Training Strategy
1. **Balanced Complexity**: Avoid overly complex expressions for any single output
2. **Convergence Monitoring**: Track convergence for each output separately
3. **Multiple Runs**: Perform several independent training runs
4. **Validation**: Use holdout data to validate discovered system

### Result Interpretation
1. **Cross-Output Validation**: Check consistency between related outputs
2. **Physical Constraints**: Verify outputs satisfy known physical laws
3. **Sensitivity Analysis**: Test system response to input variations