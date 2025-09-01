# Core Concepts

> Version: 1.0 • Last updated: 2025-09-01

Deep Symbolic Optimization (DSO) is a cutting-edge framework that automatically discovers human-readable mathematical expressions from data using the power of deep reinforcement learning. This guide introduces the fundamental concepts that make DSO uniquely effective.

## What is Symbolic Regression?

**Symbolic Regression** is the task of finding mathematical expressions that best describe the relationship between input variables and output targets. Unlike traditional regression that fits predefined model forms (linear, polynomial, etc.), symbolic regression discovers the mathematical structure itself.

### Traditional vs. Symbolic Regression

| Traditional Regression | Symbolic Regression |
|----------------------|-------------------|
| `y = ax + b` (fixed form) | `y = ?` (discover form) |
| Fit coefficients | Discover structure + coefficients |
| Limited expressiveness | Unlimited mathematical creativity |
| Black-box prediction | Interpretable equations |

### Examples of Symbolic Regression

**Physics Discovery:**
- **Data**: Pendulum period measurements at different lengths
- **Discovery**: `T = 2π√(L/g)` (pendulum equation)

**Engineering Applications:**
- **Data**: Material stress-strain measurements  
- **Discovery**: `σ = E·ε` (Hooke's law)

**Economic Modeling:**
- **Data**: Supply, demand, and price data
- **Discovery**: `P = a·S^α / D^β` (economic equilibrium)

## The DSO Innovation

DSO revolutionizes symbolic regression by combining **Genetic Programming** with **Deep Reinforcement Learning**:

### Traditional Genetic Programming Limitations
- **Random search**: Inefficient exploration of expression space
- **No learning**: Each generation starts from scratch
- **Slow convergence**: Requires many evaluations to find good solutions

### DSO's Deep Learning Solution
- **Intelligent search**: Neural network learns to generate promising expressions
- **Cumulative learning**: Knowledge accumulates across generations
- **Fast convergence**: Efficient exploration guided by learned policy

## Core Components

### 1. Expression Representation

DSO represents mathematical expressions as **token sequences**:

```python
# Mathematical expression: sin(x₁) + x₂²
# Token sequence: ["sin", "x1", "add", "x2", "x2", "mul"]
```

**Benefits:**
- **Flexible**: Can represent any mathematical expression
- **Neural-friendly**: Sequences work well with RNN/Transformer architectures
- **Executable**: Direct conversion to computational code

### 2. Policy Network (RNN)

The **policy network** is a neural network that learns to generate mathematical expressions:

```python
class RNNPolicy:
    def __init__(self):
        self.rnn = LSTM(hidden_size=32, num_layers=1)
        self.output_layer = Linear(hidden_size, num_tokens)
    
    def sample_expression(self):
        """Generate a mathematical expression as token sequence"""
        tokens = []
        hidden_state = self.initial_state()
        
        for step in range(max_length):
            # Predict next token probability
            logits = self.forward(hidden_state)
            probs = softmax(logits)
            
            # Sample token
            token = categorical_sample(probs)
            tokens.append(token)
            
            # Update hidden state
            hidden_state = self.rnn(token, hidden_state)
        
        return tokens
```

### 3. Expression Evaluation

Generated expressions are evaluated for **fitness**:

```python
def evaluate_expression(expression, X_train, y_train):
    """Compute fitness of mathematical expression"""
    try:
        # Execute expression on training data
        y_pred = expression.execute(X_train)
        
        # Compute accuracy (negative mean squared error)
        mse = np.mean((y_train - y_pred) ** 2)
        accuracy_reward = -mse
        
        # Add complexity penalty
        complexity_penalty = -0.01 * expression.complexity
        
        # Total fitness
        fitness = accuracy_reward + complexity_penalty
        return fitness
        
    except:
        # Invalid expressions get very low fitness
        return -1e6
```

### 4. Reinforcement Learning Loop

DSO uses the **REINFORCE algorithm** to train the policy:

```python
def training_step(policy, batch_size=1000):
    """Single training step using REINFORCE"""
    
    # 1. Sample expressions from current policy
    expressions = []
    log_probs = []
    
    for _ in range(batch_size):
        expr, log_prob = policy.sample_with_log_prob()
        expressions.append(expr)
        log_probs.append(log_prob)
    
    # 2. Evaluate expressions
    rewards = []
    for expr in expressions:
        reward = evaluate_expression(expr, X_train, y_train)
        rewards.append(reward)
    
    # 3. Compute policy gradient
    rewards = torch.tensor(rewards)
    log_probs = torch.stack(log_probs)
    
    # REINFORCE gradient: ∇θ J = E[∇θ log π(a|s) * R]
    policy_loss = -(log_probs * rewards).mean()
    
    # 4. Update policy parameters
    policy_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    return rewards.mean().item()
```
```
Traditional GP: Random mutations + Selection
DSO: Neural network learns to generate better expressions
Result: Much faster convergence to optimal solutions
```

## 🧠 **How DSO Works (Detailed)**

### **Step 1: Expression Generation**
```python
# RNN Policy generates token sequence
tokens = policy.sample()  # [add, mul, x1, x2, sin, x3]

# Tokens form mathematical expression
expression = Program(tokens)  # (x1 * x2) + sin(x3)
```

### **Step 2: Expression Evaluation**
```python
# Execute on training data
predictions = expression.execute(X_train)

# Compute fitness/reward
reward = metric(y_train, predictions)  # Higher = better fit
```

### **Step 3: Policy Learning**
```python
# Update neural network to favor good expressions
loss = compute_policy_gradient_loss(expressions, rewards)
optimizer.step(loss)
```

### **Step 4: Iteration**
Repeat until discovering optimal mathematical formula.

## 🔧 **Key Technical Features**

### **Modular Architecture**
- **Task Interface** (`dso.task`) - pluggable problem definitions
- **Policy Networks** - RNN, Transformer, or custom architectures
- **Token Libraries** - customizable mathematical operation sets
- **Constraint System** - domain knowledge incorporation

### **Performance Optimizations**
- **GPU Acceleration** - Cython/CUDA kernels for fast evaluation
- **Vectorized Operations** - batch processing of expressions
- **Memory Efficiency** - optimized data structures
- **Parallel Evaluation** - concurrent expression testing

### **Extensibility**
- **JSON Configuration** - easy experiment setup
- **Custom Operators** - add domain-specific functions
- **Plugin Architecture** - extend with new components
- **Multi-Backend** - support for different execution engines

## 👥 **Target Audience**

### **Researchers**
- Symbolic AI and program synthesis
- Reinforcement learning applications
- Automated scientific discovery
- Interpretable machine learning

### **Engineers** 
- Surrogate model development
- Control system design
- Mathematical modeling
- Performance optimization

### **Educators**
- Teaching symbolic computation
- Demonstrating AI techniques
- Research methodology
- Mathematical discovery

### **Data Scientists**
- Feature engineering
- Model interpretability
- Pattern discovery
- Mathematical relationship identification

## 🚀 **Why Choose DSO?**

### **Interpretability First**
Unlike black-box neural networks, DSO produces mathematical equations that humans can:
- Understand and verify
- Use in theoretical analysis
- Implement in other systems
- Publish in scientific papers

### **State-of-the-Art Performance**
- **1st place** in 2022 SRBench Symbolic Regression Competition
- Superior accuracy on benchmark problems
- Faster convergence than traditional methods

### **Production Ready**
- Robust, tested codebase
- Comprehensive documentation
- Active community support
- Industrial applications

## 📊 **Current Capabilities & Limitations**

### **Current Strengths (MISO)**
- **Multiple Input Single Output** regression
- Rich mathematical operator library
- Efficient neural policy networks
- Advanced constraint systems
- GPU-accelerated evaluation

### **Current Limitations**
- Single output variable per expression
- No multi-task learning
- Limited vector operations
- No shared sub-expression optimization

### **Future Extensions (MIMO)**
- **Multiple Input Multiple Output** support
- Simultaneous multi-expression discovery
- Vector-valued mathematical operations
- Shared knowledge between related tasks

## 🔗 **Quick Navigation**

- **Architecture Details**: [`core/architecture.md`](architecture.md)
- **Getting Started**: [`core/getting_started.md`](getting_started.md)
- **Setup Guide**: [`core/setup.md`](setup.md)
- **Project Structure**: See the main project repository for detailed structure information
