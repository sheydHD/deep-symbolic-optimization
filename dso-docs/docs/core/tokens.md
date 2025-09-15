# Tokens & Expression System

> Version: 1.0 • Last updated: 2025-09-01

This guide explains DSO's token system and how mathematical expressions are constructed, represented, and executed.

## 🧩 **Understanding Tokens**

Tokens are the fundamental building blocks of mathematical expressions in DSO. Think of them as mathematical LEGO pieces that can be combined to create complex formulas.

### **Token Properties**

Every token has these key attributes:

```python
Token(
    function=np.add,        # The mathematical operation
    name="add",             # Human-readable name
    arity=2,               # Number of inputs required
    complexity=1,          # Complexity cost for this operation
    input_var=None,        # For input variables (x1, x2, etc.)
    value=None            # For constants
)
```

## 📚 **Token Categories**

### **1. Binary Operators (arity=2)**

Mathematical operations requiring two inputs:

```python
# Arithmetic
add → x + y          # Addition
sub → x - y          # Subtraction  
mul → x * y          # Multiplication
div → x / y          # Division

# Advanced
pow → x^y            # Exponentiation
max → max(x, y)      # Maximum
min → min(x, y)      # Minimum
```

### **2. Unary Operators (arity=1)**

Mathematical functions taking one input:

```python
# Trigonometric
sin → sin(x)         # Sine
cos → cos(x)         # Cosine
tan → tan(x)         # Tangent

# Exponential/Logarithmic
exp → e^x            # Natural exponential
log → ln(x)          # Natural logarithm
sqrt → √x            # Square root

# Other
neg → -x             # Negation
abs → |x|            # Absolute value
inv → 1/x            # Reciprocal
n2 → x²              # Square
n3 → x³              # Cube
```

### **3. Input Variables (arity=0)**

Represent features from your dataset:

```python
x1 → First feature column
x2 → Second feature column
x3 → Third feature column
...
```

### **4. Constants (arity=0)**

Numerical values in expressions:

```python
# Learnable constants (optimized during training)
const → Placeholder for any real number

# Hard-coded constants
1.0 → Fixed value of 1.0
2.5 → Fixed value of 2.5
π → Mathematical constant pi
```

## 🌳 **Expression Trees**

### **Tree Representation**

Mathematical expressions are represented as binary trees in prefix notation:

```
Mathematical Expression: (x₁ + x₂) * sin(x₃)

Token Sequence: [mul, add, x1, x2, sin, x3]

Tree Structure:
        mul
       /   \
     add   sin
    / \     |
   x1  x2   x3
```

### **Prefix Notation Rules**

DSO uses prefix notation where operators come before their operands:

```python
# Infix: x + y
# Prefix: add x y

# Infix: sin(x) + cos(y) 
# Prefix: add sin(x) cos(y)
# Tokens: [add, sin, x, cos, y]
```

### **Tree Construction Process**

1. **Sequential Processing**: Tokens processed left-to-right
2. **Stack-Based**: Use stack to manage partial expressions
3. **Arity Matching**: Operators wait for required number of inputs
4. **Bottom-Up**: Leaf nodes (variables/constants) evaluated first

## ⚙️ **Expression Execution**

### **Stack-Based Execution Algorithm**

```python
def execute_expression(tokens, X):
    apply_stack = []
    
    for token in tokens:
        apply_stack.append([token])
        
        # Check if current operator has all inputs
        while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
            operator = apply_stack[-1][0]
            operands = apply_stack[-1][1:]
            
            # Execute operation
            if operator.input_var is not None:
                result = X[:, operator.input_var]  # Get input column
            else:
                result = operator.function(*operands)  # Apply function
            
            # Update stack
            apply_stack.pop()
            if apply_stack:
                apply_stack[-1].append(result)
            else:
                return result  # Final result
```

### **Vectorized Operations**

All operations work on entire arrays simultaneously:

```python
# Input data shape: [n_samples, n_features]
X = [[0.5, 0.3, 0.8],    # Sample 1
     [0.1, 0.9, 0.2],    # Sample 2  
     [0.7, 0.4, 0.6]]    # Sample 3

# Expression: add x1 x2
# Executes: X[:, 0] + X[:, 1]
# Result: [0.8, 1.0, 1.1]  # For all samples at once
```

## 🔧 **Token Library System**

### **Library Construction**

The Library class manages available tokens:

```python
from dso.library import Library
from dso.functions import create_tokens

# Create tokens for 2 input variables
tokens = create_tokens(
    n_input_var=2,
    function_set=["add", "sub", "mul", "sin", "cos"],
    protected=False
)

# Build library
library = Library(tokens)
```

### **Custom Token Creation**

Add your own mathematical operations:

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def custom_polynomial(x, y):
    return x**2 + 2*x*y + y**2

# Create custom tokens
custom_tokens = [
    Token(sigmoid, "sigmoid", arity=1, complexity=4),
    Token(custom_polynomial, "poly2", arity=2, complexity=6)
]
```

### **Protected vs Unprotected Operations**

#### **Unprotected (Default)**
Raw mathematical operations that may produce errors:

```python
div(1, 0)      → inf (division by zero)
log(-1)        → nan (log of negative)
sqrt(-4)       → nan (sqrt of negative)
```

#### **Protected Mode**
Safe versions that handle edge cases:

```python
protected_div(1, 0)     → 1.0 (returns 1 when |denominator| < 0.001)
protected_log(-1)       → 0.0 (returns 0 for non-positive inputs)
protected_sqrt(-4)      → 2.0 (uses absolute value)
```

## 🎯 **Expression Generation Process**

### **Neural Policy Generation**

RNN policy generates token sequences:

```python
# Policy network state
hidden_state = initial_state

# Generate tokens one by one
tokens = []
for step in range(max_length):
    # Get probability distribution over tokens
    probs = policy_network(hidden_state, current_context)
    
    # Sample next token
    token = sample_from_distribution(probs)
    tokens.append(token)
    
    # Update state
    hidden_state = update_state(hidden_state, token)
    
    # Check if expression is complete
    if is_complete_expression(tokens):
        break

return tokens
```

### **Expression Validation**

Generated token sequences are validated:

```python
def validate_expression(tokens):
    stack_depth = 0
    
    for token in tokens:
        if token.arity == 0:  # Terminal (variable/constant)
            stack_depth += 1
        else:  # Operator
            if stack_depth < token.arity:
                return False  # Not enough operands
            stack_depth = stack_depth - token.arity + 1
    
    return stack_depth == 1  # Should have exactly one result
```

## 📊 **Expression Complexity**

### **Complexity Scoring**

Each token contributes to overall expression complexity:

```python
# Token complexities (typical values)
add, sub, mul → 1 point
div, n2       → 2 points  
sin, cos, n3  → 3 points
exp, log      → 4 points

# Example expression: sin(x1) + x2^2
# Complexity = 3 (sin) + 1 (add) + 2 (n2) = 6 points
```

### **Length vs Complexity**

- **Length**: Number of tokens in expression
- **Complexity**: Weighted sum based on operation difficulty
- **Trade-off**: Balance between accuracy and simplicity

## 🔄 **Constant Optimization**

### **Placeholder Constants**

Special tokens that get optimized:

```python
# Expression with constant: mul const x1
# Initial: const = 1.0, so expression = 1.0 * x1
# After optimization: const = 2.5, so expression = 2.5 * x1
```

### **Optimization Process**

```python
def optimize_constants(program):
    # Extract constant positions
    const_positions = program.const_pos
    
    # Define objective function
    def objective(const_values):
        program.set_constants(const_values)
        predictions = program.execute(X_train)
        return -metric(y_train, predictions)  # Minimize negative reward
    
    # Optimize using gradient descent
    initial_values = [1.0] * len(const_positions)
    optimized_values = optimizer.minimize(objective, initial_values)
    
    # Update program with optimized constants
    program.set_constants(optimized_values)
```

## 🚀 **Advanced Features**

### **State Checkers (Decision Trees)**

Special tokens for conditional logic:

```python
# StateChecker token: if x1 < 0.5 then branch_A else branch_B
state_checker = StateChecker(
    state_index=0,      # Check x1 
    threshold=0.5,      # Threshold value
    complexity=5        # Complexity cost
)
```

### **Polynomial Tokens**

Fit polynomial relationships:

```python
# Polynomial token automatically fits: ax² + bx + c
poly_token = Polynomial(
    degree=2,           # Quadratic polynomial
    variables=[0, 1],   # Use x1 and x2
    complexity=8        # High complexity
)
```

This token system provides the foundation for DSO's powerful symbolic expression discovery capabilities, enabling the automatic construction of interpretable mathematical models from data.
