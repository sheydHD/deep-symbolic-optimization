# Symbolic Regression

## Methodology

Symbolic regression in DSO employs a sophisticated approach combining neural networks with evolutionary computation principles. The framework automatically discovers mathematical expressions that accurately model data relationships while maintaining interpretability.

## Algorithm Components

### Policy Network
The core of DSO is a recurrent neural network (RNN) that learns to generate mathematical expressions. The policy network:
- Generates expressions as sequences of tokens
- Learns from expression performance feedback
- Adapts generation strategy over iterations
- Maintains population diversity

### Expression Evaluation
Each generated expression undergoes comprehensive evaluation:
- **Execution**: Expression computed on training dataset
- **Error Calculation**: Mean squared error between predictions and targets
- **Complexity Assessment**: Length and structural complexity measurement
- **Stability Check**: Numerical robustness validation

### Reward Function
DSO optimizes expressions using a multi-objective reward function:

```
R = -log(1 + MSE) - λ * complexity_penalty
```

Where:
- `MSE` is the mean squared error
- `λ` controls complexity-accuracy trade-off
- `complexity_penalty` prevents overfitting

## Function Library

DSO supports a comprehensive set of mathematical functions:

### Arithmetic Operations
- `add`: Addition (x + y)
- `sub`: Subtraction (x - y)
- `mul`: Multiplication (x * y)
- `div`: Protected division (x / y, with zero protection)
- `pow`: Power function (x^y, with overflow protection)

### Trigonometric Functions
- `sin`: Sine function
- `cos`: Cosine function
- `tan`: Tangent function (with asymptote protection)

### Exponential and Logarithmic
- `exp`: Exponential function (with overflow protection)
- `log`: Natural logarithm (with domain protection)
- `sqrt`: Square root (with domain protection)

### Advanced Functions
- `abs`: Absolute value
- `sigmoid`: Sigmoid activation function
- `tanh`: Hyperbolic tangent

## Variable and Constant Handling

### Input Variables
- Automatically detected from input data structure
- Normalized to prevent numerical issues
- Accessible as x1, x2, ..., xn in expressions

### Constants
- Optimizable numeric values embedded in expressions
- Initialized randomly within specified ranges
- Fine-tuned during expression evaluation

## Expression Constraints

### Length Constraints
- Minimum expression length: Prevents trivial solutions
- Maximum expression length: Controls computational complexity
- Adaptive length limits based on problem complexity

### Structural Constraints
- Maximum tree depth limitations
- Function arity restrictions
- Recursive structure prevention

## Optimization Features

### Multi-Start Training
DSO can perform multiple independent training runs to improve solution quality and assess result consistency.

### Early Stopping
Automatic termination when:
- Target accuracy achieved
- Training convergence detected
- Maximum iterations reached
- Computational budget exhausted

### Population Management
- Maintains diverse expression populations
- Prevents premature convergence
- Balances exploration and exploitation

## Numerical Stability

DSO incorporates robust numerical handling:
- Protected arithmetic operations
- Overflow and underflow prevention
- Invalid operation detection
- Graceful degradation for unstable expressions

## Performance Optimization

### Vectorized Evaluation
- Batch processing of expression populations
- Efficient NumPy-based computations
- Memory-optimized data handling

### Cython Acceleration
- Critical path functions implemented in Cython
- Significant performance improvements for large datasets
- Maintains Python interface convenience