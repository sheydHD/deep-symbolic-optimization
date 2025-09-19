# Core API Reference

## DeepSymbolicOptimizer Class

The main interface for DSO functionality.

### Constructor

```python
DeepSymbolicOptimizer(
    task="regression",
    function_set=None,
    max_length=20,
    n_samples=1000000,
    batch_size=1000,
    **kwargs
)
```

#### Parameters

- **task** (`str`): Task type. Currently only `"regression"` is supported.
- **function_set** (`list`): List of function names to include in expressions.
- **max_length** (`int`): Maximum number of tokens in generated expressions.
- **n_samples** (`int`): Total number of expressions to evaluate during training.
- **batch_size** (`int`): Number of expressions to evaluate per iteration.

### Methods

#### fit(X, y)
Train the DSO model on provided data.

```python
dso.fit(X, y)
```

**Parameters:**
- **X** (`array-like`): Input features, shape `(n_samples, n_features)`
- **y** (`array-like`): Target values, shape `(n_samples,)` or `(n_samples, n_outputs)`

**Returns:** `self`

#### predict(X)
Generate predictions using the discovered expression.

```python
y_pred = dso.predict(X)
```

**Parameters:**
- **X** (`array-like`): Input features for prediction

**Returns:** `array-like` - Predicted values

#### score(X, y)
Evaluate the model performance using R² score.

```python
r2_score = dso.score(X, y)
```

**Parameters:**
- **X** (`array-like`): Input features
- **y** (`array-like`): True target values

**Returns:** `float` - R² score

### Properties

#### program_
The best discovered expression.

```python
best_expression = dso.program_
print(f"Expression: {best_expression}")
print(f"Infix notation: {best_expression.pretty()}")
```

#### r_best_
Training performance of the best expression.

```python
training_score = dso.r_best_
print(f"Training R²: {training_score:.6f}")
```

#### programs_ (Multi-output only)
List of best expressions for each output variable.

```python
for i, expr in enumerate(dso.programs_):
    print(f"Output {i+1}: {expr}")
```

#### r_best_list_ (Multi-output only)
List of training scores for each output variable.

```python
for i, score in enumerate(dso.r_best_list_):
    print(f"Output {i+1} R²: {score:.6f}")
```

## Expression Objects

### Program Class

Represents a mathematical expression discovered by DSO.

#### Properties

- **tokens** (`list`): Token sequence representing the expression
- **depth** (`int`): Tree depth of the expression
- **length** (`int`): Number of tokens in the expression
- **complexity** (`float`): Complexity measure of the expression

#### Methods

##### pretty()
Return human-readable infix notation of the expression.

```python
infix_str = program.pretty()
```

##### execute(X)
Evaluate the expression on input data.

```python
result = program.execute(X)
```

##### optimize_constants(X, y)
Optimize constants in the expression for given data.

```python
optimized_program = program.optimize_constants(X, y)
```

## Configuration Classes

### FunctionSet

Manages the set of available mathematical functions.

```python
from dso.functions import FunctionSet

# Create custom function set
function_set = FunctionSet([
    "add", "sub", "mul", "div",
    "sin", "cos", "exp", "log"
])

# Use in DSO
dso = DeepSymbolicOptimizer(function_set=function_set)
```

### TaskConfig

Configuration object for regression tasks.

```python
from dso.task import TaskConfig

config = TaskConfig(
    task_type="regression",
    metric="neg_mse",
    reward_function="exp_neg_mse",
    extra_metric_test=["r2", "mae"]
)
```

## Utility Functions

### load_dataset()
Load benchmark datasets for testing.

```python
from dso.utils import load_dataset

X, y = load_dataset("nguyen-1")
```

### save_model()
Save trained DSO model to disk.

```python
from dso.utils import save_model

save_model(dso, "my_model.pkl")
```

### load_model()
Load trained DSO model from disk.

```python
from dso.utils import load_model

dso = load_model("my_model.pkl")
```

## Error Handling

### Common Exceptions

#### DSORuntimeError
Raised when DSO encounters a runtime error during training.

```python
from dso.exceptions import DSORuntimeError

try:
    dso.fit(X, y)
except DSORuntimeError as e:
    print(f"DSO training failed: {e}")
```

#### InvalidConfigError
Raised when configuration parameters are invalid.

```python
from dso.exceptions import InvalidConfigError

try:
    dso = DeepSymbolicOptimizer(max_length=-1)  # Invalid
except InvalidConfigError as e:
    print(f"Invalid configuration: {e}")
```

#### NumericalInstabilityError
Raised when expressions produce numerical instabilities.

```python
from dso.exceptions import NumericalInstabilityError

try:
    result = program.execute(X)
except NumericalInstabilityError as e:
    print(f"Numerical instability: {e}")
```

## Advanced Usage

### Custom Reward Functions

```python
def custom_reward_function(r):
    """Custom reward function that heavily penalizes complexity."""
    return r["inv_nrmse"] - 0.1 * r["complexity"]

dso = DeepSymbolicOptimizer(
    reward_function=custom_reward_function
)
```

### Custom Function Implementation

```python
from dso.functions import create_tokens

# Define custom function
def my_function(x):
    return np.tanh(x) + 1

# Register with DSO
custom_token = create_tokens(
    n_input_var=1,
    function=my_function,
    name="my_function",
    arity=1,
    complexity=1.0
)

dso = DeepSymbolicOptimizer(
    function_set=["add", "sub", "mul"] + [custom_token]
)
```

### Training Callbacks

```python
class TrainingCallback:
    def __init__(self):
        self.epoch_rewards = []
    
    def on_epoch_end(self, epoch, reward_history):
        self.epoch_rewards.append(max(reward_history))
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Best reward = {max(reward_history):.6f}")

callback = TrainingCallback()
dso = DeepSymbolicOptimizer(callbacks=[callback])
```