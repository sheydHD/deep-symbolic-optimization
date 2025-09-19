# Basic Regression Examples

## Linear Regression

### Simple Linear Function

```python
import numpy as np
from dso import DeepSymbolicOptimizer

# Generate linear data: y = 2*x + 3 + noise
np.random.seed(42)
X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = 2 * X[:, 0] + 3 + np.random.normal(0, 0.1, 100)

# Configure DSO for simple arithmetic
dso = DeepSymbolicOptimizer(
    task="regression",
    function_set=["add", "sub", "mul", "div"],
    max_length=10,
    n_samples=100000
)

# Train and evaluate
dso.fit(X, y)
print(f"Discovered expression: {dso.program_}")
print(f"Expected: 2*x + 3")
print(f"Training R²: {dso.r_best_:.6f}")
```

### Multiple Linear Regression

```python
# Generate multi-variable linear data: y = 2*x1 + 3*x2 - x3 + 1
X = np.random.randn(200, 3)
y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + 1 + np.random.normal(0, 0.05, 200)

dso = DeepSymbolicOptimizer(
    function_set=["add", "sub", "mul"],
    max_length=15,
    n_samples=200000
)

dso.fit(X, y)
print(f"Discovered: {dso.program_}")

# Evaluate on test data
X_test = np.random.randn(50, 3)
y_test = 2*X_test[:, 0] + 3*X_test[:, 1] - X_test[:, 2] + 1
y_pred = dso.predict(X_test)

from sklearn.metrics import r2_score
test_r2 = r2_score(y_test, y_pred)
print(f"Test R²: {test_r2:.6f}")
```

## Non-Linear Regression

### Polynomial Functions

```python
# Quadratic function: y = x² + 2*x + 1
X = np.linspace(-3, 3, 150).reshape(-1, 1)
y = X[:, 0]**2 + 2*X[:, 0] + 1 + np.random.normal(0, 0.1, 150)

dso = DeepSymbolicOptimizer(
    function_set=["add", "sub", "mul", "pow"],
    max_length=12,
    n_samples=300000
)

dso.fit(X, y)
print(f"Discovered: {dso.program_}")

# Visualize results
import matplotlib.pyplot as plt

X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)
y_plot = dso.predict(X_plot)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.6, label='Data')
plt.plot(X_plot, y_plot, 'r-', linewidth=2, label=f'DSO: {dso.program_}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression with DSO')
plt.legend()
plt.grid(True)
plt.show()
```

### Trigonometric Functions

```python
# Sine wave with offset: y = 2*sin(x) + 0.5*x
X = np.linspace(0, 4*np.pi, 200).reshape(-1, 1)
y = 2*np.sin(X[:, 0]) + 0.5*X[:, 0] + np.random.normal(0, 0.1, 200)

dso = DeepSymbolicOptimizer(
    function_set=["add", "sub", "mul", "sin", "cos"],
    max_length=15,
    n_samples=500000
)

dso.fit(X, y)
print(f"Discovered: {dso.program_}")
print(f"Expected: 2*sin(x) + 0.5*x")
```

## Multi-Output Examples

### Coupled System

```python
# System of equations:
# y1 = x1² + x2
# y2 = sin(x1) + cos(x2)
X = np.random.uniform(-2, 2, (300, 2))
Y = np.column_stack([
    X[:, 0]**2 + X[:, 1],
    np.sin(X[:, 0]) + np.cos(X[:, 1])
])

# Add noise
Y += np.random.normal(0, 0.05, Y.shape)

dso = DeepSymbolicOptimizer(
    function_set=["add", "sub", "mul", "pow", "sin", "cos"],
    max_length=15,
    n_samples=1000000,
    multi_output=True
)

dso.fit(X, Y)

# Analyze results for each output
for i, (expr, score) in enumerate(zip(dso.programs_, dso.r_best_list_)):
    print(f"Output {i+1}: {expr} (R² = {score:.6f})")
```

## Real-World Example

### Boston Housing Dataset

```python
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and prepare data
data = load_boston()
X, y = data.data, data.target

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

# Configure DSO for real-world data
dso = DeepSymbolicOptimizer(
    function_set=["add", "sub", "mul", "div", "sqrt", "log", "abs"],
    max_length=25,
    n_samples=2000000,
    early_stopping=True,
    epsilon=1e-8
)

# Train model
dso.fit(X_train_scaled, y_train_scaled)

# Evaluate performance
y_pred_scaled = dso.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Discovered expression: {dso.program_}")
print(f"Test MSE: {mse:.4f}")
print(f"Test R²: {r2:.4f}")

# Compare with linear regression
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train_scaled, y_train_scaled)
y_pred_lr_scaled = lr.predict(X_test_scaled)
y_pred_lr = scaler_y.inverse_transform(y_pred_lr_scaled.reshape(-1, 1)).ravel()

lr_mse = mean_squared_error(y_test, y_pred_lr)
lr_r2 = r2_score(y_test, y_pred_lr)

print(f"\nLinear Regression Comparison:")
print(f"Linear MSE: {lr_mse:.4f}")
print(f"Linear R²: {lr_r2:.4f}")
print(f"DSO Improvement: {((lr_mse - mse) / lr_mse * 100):.1f}% MSE reduction")
```

## Cross-Validation Example

```python
from sklearn.model_selection import cross_val_score, KFold

def dso_cross_validate(X, y, config, cv=5):
    """Perform cross-validation with DSO."""
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    expressions = []
    
    for train_idx, val_idx in kfold.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        dso = DeepSymbolicOptimizer(**config)
        dso.fit(X_train, y_train)
        
        y_pred = dso.predict(X_val)
        score = r2_score(y_val, y_pred)
        
        scores.append(score)
        expressions.append(str(dso.program_))
    
    return scores, expressions

# Example usage
config = {
    "function_set": ["add", "sub", "mul", "div"],
    "max_length": 15,
    "n_samples": 500000
}

scores, expressions = dso_cross_validate(X, y, config)

print("Cross-Validation Results:")
print(f"Mean R²: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
print(f"Individual scores: {scores}")
print("\nDiscovered expressions:")
for i, expr in enumerate(expressions):
    print(f"Fold {i+1}: {expr}")
```