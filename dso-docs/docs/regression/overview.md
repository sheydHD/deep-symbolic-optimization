# Regression Overview

DSO's core functionality centers on symbolic regression - the automated discovery of mathematical expressions that model relationships in data. Unlike traditional regression methods that fit predetermined functional forms, DSO searches through the space of possible mathematical expressions to find optimal formulas.

## Symbolic Regression Fundamentals

Symbolic regression seeks to find a mathematical expression `f(x)` such that `y = f(x) + ε`, where `ε` represents noise or model error. DSO accomplishes this through reinforcement learning, where a neural network policy learns to generate increasingly effective mathematical expressions.

## Problem Types

DSO supports multiple input-output configurations:

### Single Input Single Output (SISO)
- **Input**: One variable (scalar)
- **Output**: One target variable
- **Example**: `y = sin(x) + x^2`

### Multiple Input Single Output (MISO)
- **Input**: Multiple variables (vector)
- **Output**: One target variable
- **Example**: `y = x1^2 + 2*x2*x3 - x4`

### Single Input Multiple Output (SIMO)
- **Input**: One variable (scalar)
- **Output**: Multiple target variables
- **Example**: `y1 = sin(x), y2 = cos(x)`

### Multiple Input Multiple Output (MIMO)
- **Input**: Multiple variables (vector)
- **Output**: Multiple target variables
- **Example**: System of equations with shared variables

## Expression Representation

DSO represents mathematical expressions as trees composed of:

- **Variables**: Input features (x1, x2, ..., xn)
- **Constants**: Optimizable numeric values
- **Functions**: Mathematical operations (add, mul, sin, exp, etc.)

## Training Process

1. **Expression Generation**: RNN policy generates candidate expressions
2. **Evaluation**: Each expression is executed on training data
3. **Fitness Scoring**: Expressions scored based on accuracy and complexity
4. **Policy Update**: RNN learns from successful expressions via REINFORCE
5. **Iteration**: Process repeats until convergence or termination criteria

## Key Features

### Automatic Complexity Control
DSO balances model accuracy against complexity to prevent overfitting and ensure interpretable results.

### Robust Evaluation
Built-in protection against numerical instabilities such as division by zero, overflow, and invalid operations.

### Parallel Processing
Efficient evaluation of expression populations using vectorized operations and parallel execution.

### Convergence Detection
Automatic detection of training convergence to optimize computational resources.

## Performance Metrics

DSO optimizes expressions based on:
- **Mean Squared Error (MSE)**: Primary accuracy metric
- **Complexity Penalty**: Based on expression length and structure
- **Numerical Stability**: Penalizes expressions prone to numerical issues

## Output Interpretation

DSO provides:
- **Best Expression**: Human-readable mathematical formula
- **Performance Metrics**: Accuracy and complexity scores
- **Training Statistics**: Convergence behavior and population diversity
- **Expression Trees**: Hierarchical representation of discovered formulas