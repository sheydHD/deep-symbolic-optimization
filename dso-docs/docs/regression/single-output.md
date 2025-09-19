# Single Output Regression

Single output regression is DSO's fundamental capability, discovering mathematical expressions that map input variables to a single target variable.

## Problem Formulation

For a dataset with input features `X ∈ ℝ^(n×d)` and target values `y ∈ ℝ^n`, DSO seeks to find a function `f` such that:

```
y_i ≈ f(x_i1, x_i2, ..., x_id)
```

Where `f` is a mathematical expression composed of basic functions, operators, and constants.

## Configuration

### Basic Configuration

```python
from dso import DeepSymbolicOptimizer

dso = DeepSymbolicOptimizer(
    task="regression",
    function_set=["add", "sub", "mul", "div", "sin", "cos", "exp", "log"],
    max_length=20,
    n_samples=1000000
)
```

### Key Parameters

- **function_set**: Available mathematical functions
- **max_length**: Maximum expression length (complexity control)
- **n_samples**: Training budget (number of expressions to evaluate)
- **batch_size**: Number of expressions evaluated per iteration

## Training Process

### Data Preparation

```python
import numpy as np

# Ensure proper data format
X = np.array(X)  # Shape: (n_samples, n_features)
y = np.array(y)  # Shape: (n_samples,)

# Optional: data normalization
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
```

### Model Training

```python
# Train DSO
dso.fit(X_scaled, y_scaled)

# Access results
best_expression = dso.program_
training_score = dso.r_best_
```

### Prediction

```python
# Make predictions
y_pred_scaled = dso.predict(X_scaled)

# Inverse transform if data was scaled
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
```

## Expression Analysis

### Expression Complexity
DSO provides several metrics for expression analysis:

```python
# Expression length (number of tokens)
expression_length = len(dso.program_.tokens)

# Expression depth (tree depth)
expression_depth = dso.program_.depth

# Expression representation
print(f"Expression: {dso.program_}")
print(f"Infix notation: {dso.program_.pretty()}")
```

### Performance Metrics

```python
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Calculate performance metrics
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print(f"MSE: {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"R²: {r2:.6f}")
```

## Advanced Configuration

### Custom Function Sets

```python
# Physics-oriented function set
physics_functions = ["add", "sub", "mul", "div", "pow", "sin", "cos", "exp"]

# Engineering function set
engineering_functions = ["add", "sub", "mul", "div", "sqrt", "log", "abs"]

# Basic arithmetic only
arithmetic_functions = ["add", "sub", "mul", "div"]
```

### Training Control

```python
dso = DeepSymbolicOptimizer(
    task="regression",
    function_set=["add", "sub", "mul", "div", "sin", "cos"],
    max_length=15,
    n_samples=500000,
    batch_size=1000,
    early_stopping=True,
    epsilon=0.001,  # Early stopping threshold
    n_epochs_convergence=20  # Convergence patience
)
```

## Best Practices

### Data Preprocessing
1. **Normalization**: Scale features to similar ranges
2. **Outlier Handling**: Remove or cap extreme values
3. **Missing Data**: Handle NaN values appropriately
4. **Feature Selection**: Remove irrelevant or highly correlated features

### Training Configuration
1. **Function Set Selection**: Choose functions relevant to problem domain
2. **Expression Length**: Balance complexity and interpretability
3. **Training Budget**: Sufficient samples for convergence
4. **Multiple Runs**: Perform several independent runs for robustness

### Result Validation
1. **Cross-Validation**: Validate on unseen data
2. **Expression Verification**: Manually verify discovered formulas
3. **Sensitivity Analysis**: Test expression robustness
4. **Domain Validation**: Ensure expressions make physical/logical sense