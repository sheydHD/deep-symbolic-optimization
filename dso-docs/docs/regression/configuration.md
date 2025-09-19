# Configuration

## Overview

DSO provides extensive configuration options to customize behavior for specific regression problems. Configuration can be specified through Python parameters, configuration files, or command-line arguments.

## Core Configuration Parameters

### Task Configuration
```python
dso = DeepSymbolicOptimizer(
    task="regression",           # Task type (currently only "regression")
    dataset="your_data",         # Dataset identifier or path
    logdir="./logs"             # Directory for logging and output
)
```

### Function Set Configuration
```python
# Standard function sets
function_set = [
    "add", "sub", "mul", "div",     # Arithmetic operations
    "sin", "cos", "tan",            # Trigonometric functions
    "exp", "log", "sqrt",           # Exponential/logarithmic
    "abs", "sigmoid", "tanh"        # Special functions
]

dso = DeepSymbolicOptimizer(
    function_set=function_set,
    protected=True              # Enable protected operations (e.g., division by zero)
)
```

### Expression Constraints
```python
dso = DeepSymbolicOptimizer(
    max_length=20,              # Maximum expression length
    min_length=4,               # Minimum expression length
    max_depth=10,               # Maximum tree depth
    const_range=[-1.0, 1.0],    # Range for constant optimization
    const_optimizer="scipy"      # Constant optimization method
)
```

### Training Parameters
```python
dso = DeepSymbolicOptimizer(
    n_samples=1000000,          # Total training budget
    batch_size=1000,            # Expressions per iteration
    n_epochs=None,              # Maximum epochs (auto if None)
    early_stopping=True,        # Enable early stopping
    epsilon=1e-12,              # Early stopping threshold
    n_epochs_convergence=20     # Epochs for convergence detection
)
```

## Advanced Configuration

### Multi-Output Settings
```python
dso = DeepSymbolicOptimizer(
    multi_output=True,          # Enable multi-output regression
    output_weights=None,        # Relative weights for each output
    shared_training=True,       # Share training across outputs
    independent_early_stop=False  # Independent early stopping per output
)
```

### Neural Network Configuration
```python
dso = DeepSymbolicOptimizer(
    cell="lstm",                # RNN cell type ("lstm", "rnn", "gru")
    num_layers=1,               # Number of RNN layers
    num_units=32,               # Hidden units per layer
    learning_rate=0.0005,       # Learning rate for policy network
    entropy_weight=0.005,       # Entropy regularization weight
    entropy_gamma=0.7           # Entropy decay factor
)
```

### Performance Optimization
```python
dso = DeepSymbolicOptimizer(
    n_cores_batch=1,            # CPU cores for batch evaluation
    parallel_eval=True,         # Enable parallel expression evaluation
    use_memory=True,            # Cache expression evaluations
    memory_capacity=1e6,        # Memory cache size
    optimize_constants=True,    # Enable constant optimization
)
```

## Configuration Files

### JSON Configuration
```json
{
    "task": "regression",
    "function_set": ["add", "sub", "mul", "div", "sin", "cos"],
    "max_length": 20,
    "n_samples": 1000000,
    "batch_size": 1000,
    "early_stopping": true,
    "epsilon": 1e-12,
    "learning_rate": 0.0005,
    "entropy_weight": 0.005
}
```

### Loading Configuration
```python
import json
from dso import DeepSymbolicOptimizer

# Load from file
with open('config.json', 'r') as f:
    config = json.load(f)

dso = DeepSymbolicOptimizer(**config)
```

## Domain-Specific Configurations

### Physics Applications
```python
physics_config = {
    "function_set": ["add", "sub", "mul", "div", "pow", "sin", "cos", "exp", "log"],
    "max_length": 25,
    "const_range": [-10.0, 10.0],
    "protected": True,
    "optimize_constants": True
}
```

### Engineering Applications  
```python
engineering_config = {
    "function_set": ["add", "sub", "mul", "div", "sqrt", "abs", "pow"],
    "max_length": 15,
    "const_range": [-5.0, 5.0],
    "early_stopping": True,
    "epsilon": 1e-8
}
```

### Financial Modeling
```python
finance_config = {
    "function_set": ["add", "sub", "mul", "div", "log", "exp", "abs"],
    "max_length": 20,
    "const_range": [-2.0, 2.0],
    "batch_size": 2000,
    "n_samples": 2000000
}
```

## Performance Tuning

### Memory Management
```python
# For large datasets
large_data_config = {
    "batch_size": 500,          # Smaller batches for memory efficiency
    "use_memory": False,        # Disable caching to save memory
    "n_cores_batch": 4,         # Parallel processing
    "parallel_eval": True
}
```

### Speed Optimization
```python
# For faster training
speed_config = {
    "batch_size": 2000,         # Larger batches for speed
    "use_memory": True,         # Enable caching
    "n_cores_batch": 8,         # Maximum parallel cores
    "optimize_constants": False, # Skip constant optimization for speed
    "early_stopping": True      # Stop early when converged
}
```

### Accuracy Optimization
```python
# For maximum accuracy
accuracy_config = {
    "n_samples": 5000000,       # Large training budget
    "batch_size": 1000,         # Moderate batch size
    "max_length": 30,           # Allow complex expressions
    "optimize_constants": True, # Fine-tune constants
    "epsilon": 1e-15,           # Strict convergence threshold
    "n_epochs_convergence": 50  # Patient convergence detection
}
```

## Validation and Testing

### Cross-Validation Configuration
```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in kfold.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    dso = DeepSymbolicOptimizer(**config)
    dso.fit(X_train, y_train)
    
    # Validate on holdout set
    val_score = dso.score(X_val, y_val)
```

### Reproducibility Settings
```python
# Ensure reproducible results
reproducible_config = {
    "random_state": 42,         # Fixed random seed
    "deterministic": True,      # Deterministic training
    "n_samples": 1000000,       # Fixed training budget
    "batch_size": 1000          # Fixed batch size
}
```