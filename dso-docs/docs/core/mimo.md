# MIMO Theory: Multiple Output Regression

> Version: 2.0 • Last updated: 2025-09-02 • **For Implementation Details**: See [MIMO Implementation Guide](./mimo_implementation.md)

This guide explains the theoretical foundations and design concepts for MIMO (Multiple Input Multiple Output) symbolic regression in DSO.

## 🎯 **MIMO Motivation**

### **Limitations of Traditional Approaches**

Current DSO discovers single mathematical expressions:

```mermaid
graph LR
    A[Input: X] --> B[Single Expression: f] --> C[Output: y]
    
    style A fill:#e1f5fe
    style B fill:#fff3e0
    style C fill:#c8e6c9
```

```
Example: [x1, x2, x3] → sin(x1) + x2*x3 → scalar output
```

### **MIMO Advantage**

MIMO extends DSO to discover multiple related expressions simultaneously:

```mermaid
graph LR
    A[Input: X] --> B[Multiple Expressions]
    B --> C[y1 = f1(X)]
    B --> D[y2 = f2(X)]
    B --> E[y3 = f3(X)]
    
    style A fill:#e1f5fe
    style B fill:#fff3e0
    style C fill:#c8e6c9
    style D fill:#c8e6c9
    style E fill:#c8e6c9
```

```
Example: [x1, x2, x3] → {
    y1 = sin(x1) + x2,
    y2 = x1² + cos(x3), 
    y3 = exp(x2) * x1
}
```

## 🏗️ **MIMO Architecture Strategies**

### **Option A: Independent Expressions (Recommended)**

Each output has its own independent symbolic expression:

```python
class MIMOProgram:
    """Container for multiple independent programs"""
    def __init__(self, programs_list):
        self.programs = programs_list  # List[Program]
        self.n_outputs = len(programs_list)
    
    def execute(self, X):
        """Execute all programs and stack outputs"""
        outputs = []
        for program in self.programs:
            y_i = program.execute(X)  # Shape: [n_samples]
            outputs.append(y_i)
        
        return np.column_stack(outputs)  # Shape: [n_samples, n_outputs]
    
    @property 
    def invalid(self):
        """Check if any component program is invalid"""
        return any(program.invalid for program in self.programs)
```

### **Option B: Shared Sub-expressions**

Programs share common computational components:

```python
class SharedMIMOProgram:
    """Programs with shared sub-expressions"""
    def __init__(self, shared_programs, output_programs):
        self.shared = shared_programs      # Common computations
        self.outputs = output_programs     # Output-specific computations
    
    def execute(self, X):
        # Compute shared components
        shared_results = {}
        for name, program in self.shared.items():
            shared_results[name] = program.execute(X)
        
        # Compute outputs using shared components
        outputs = []
        for program in self.outputs:
            y_i = program.execute(X, shared_context=shared_results)
            outputs.append(y_i)
        
        return np.column_stack(outputs)
```

### **Option C: Vector-Valued Functions**

Single expressions that output multiple values:

```python
class VectorProgram:
    """Single program outputting multiple values"""
    def __init__(self, tokens, output_dim):
        self.tokens = tokens
        self.output_dim = output_dim
    
    def execute(self, X):
        # Execute expression that returns multiple outputs
        return self.vector_execute(X)  # Shape: [n_samples, n_outputs]

# Requires new vector-valued operators
vector_tokens = [
    Token(lambda x: np.stack([np.sin(x), np.cos(x)], axis=1), 
          "sin_cos", arity=1, output_dim=2),
    Token(lambda x1, x2: np.stack([x1+x2, x1*x2, x1-x2], axis=1),
          "arithmetic_triplet", arity=2, output_dim=3)
]
```

## 🎯 **MIMO Task Implementation**

### **Data Format**

```python
# MIMO dataset format
# CSV: x1, x2, ..., xn, y1, y2, ..., ym

# Example: 2 inputs, 3 outputs  
mimo_data = [
    [0.5, 0.3, 1.2, 0.8, 2.1],  # x1, x2, y1, y2, y3
    [0.1, 0.9, 0.5, 1.4, 0.7],
    [0.7, 0.4, 1.1, 1.2, 1.8]
]
```

### **MIMO Regression Task**

```python
class MIMORegressionTask(RegressionTask):
    def __init__(self, function_set, dataset, n_outputs, **kwargs):
        super().__init__(function_set, dataset, **kwargs)
        self.n_outputs = n_outputs
        
        # Validate data dimensions
        if self.y_train.ndim == 1 and n_outputs > 1:
            raise ValueError("Single output data provided for MIMO task")
        elif self.y_train.ndim == 2:
            assert self.y_train.shape[1] == n_outputs
    
    def reward_function(self, mimo_program, optimizing=False):
        """Compute reward for MIMO program"""
        Y_hat = mimo_program.execute(self.X_train)  # [n_samples, n_outputs]
        
        if mimo_program.invalid:
            return self.invalid_reward
        
        # Strategy 1: Average reward across outputs
        total_reward = 0
        for i in range(self.n_outputs):
            y_i = self.y_train[:, i]
            y_hat_i = Y_hat[:, i] 
            reward_i = self.metric(y_i, y_hat_i)
            total_reward += reward_i
        
        return total_reward / self.n_outputs
    
    def evaluate(self, mimo_program):
        """Evaluate MIMO program on test data"""
        Y_hat = mimo_program.execute(self.X_test)
        
        if mimo_program.invalid:
            return {f"nmse_test_output_{i}": None for i in range(self.n_outputs)}
        
        info = {}
        nmse_values = []
        
        for i in range(self.n_outputs):
            y_test_i = self.y_test[:, i]
            y_hat_i = Y_hat[:, i]
            
            # Individual output metrics
            nmse_i = np.mean((y_test_i - y_hat_i) ** 2) / np.var(y_test_i)
            success_i = nmse_i < self.threshold
            
            info[f"nmse_test_output_{i}"] = nmse_i
            info[f"success_output_{i}"] = success_i
            nmse_values.append(nmse_i)
        
        # Overall MIMO metrics
        info["nmse_test_mean"] = np.mean(nmse_values)
        info["nmse_test_max"] = np.max(nmse_values)
        info["success_all"] = all(info[f"success_output_{i}"] for i in range(self.n_outputs))
        info["success_any"] = any(info[f"success_output_{i}"] for i in range(self.n_outputs))
        
        return info
```

## 🧠 **MIMO Policy Networks**

### **Independent Sampling Strategy**

```python
class IndependentMIMOPolicy(RNNPolicy):
    def __init__(self, n_outputs, **kwargs):
        super().__init__(**kwargs)
        self.n_outputs = n_outputs
    
    def sample(self, batch_size):
        """Sample n_outputs expressions independently"""
        all_programs = []
        
        for batch_idx in range(batch_size):
            programs_for_sample = []
            
            for output_idx in range(self.n_outputs):
                # Sample independent expression for each output
                actions, log_probs = super().sample(1)
                program = Program(actions[0])
                programs_for_sample.append(program)
            
            mimo_program = MIMOProgram(programs_for_sample)
            all_programs.append(mimo_program)
        
        return all_programs
```

### **Shared Context Sampling**

```python
class SharedContextMIMOPolicy(RNNPolicy):
    def __init__(self, n_outputs, context_size=64, **kwargs):
        super().__init__(**kwargs)
        self.n_outputs = n_outputs
        self.context_size = context_size
        
        # Shared context network
        self.context_encoder = nn.Linear(self.hidden_size, context_size)
        self.context_decoder = nn.Linear(context_size, self.hidden_size)
    
    def sample_with_shared_context(self, batch_size):
        """Sample expressions with shared context between outputs"""
        all_programs = []
        
        for batch_idx in range(batch_size):
            # Generate shared context
            shared_context = self.generate_shared_context()
            
            programs_for_sample = []
            for output_idx in range(self.n_outputs):
                # Condition each expression on shared context
                initial_state = self.condition_on_context(shared_context)
                actions, log_probs = self.sample_conditioned(initial_state)
                program = Program(actions)
                programs_for_sample.append(program)
            
            mimo_program = MIMOProgram(programs_for_sample)
            all_programs.append(mimo_program)
        
        return all_programs
```

## 🎛️ **MIMO Training Strategies**

### **Strategy 1: Joint Training**

Train all outputs simultaneously with shared gradients:

```python
class JointMIMOTrainer:
    def train_step(self):
        # Sample MIMO programs
        mimo_programs = self.policy.sample(self.batch_size)
        
        # Evaluate each MIMO program  
        rewards = []
        for mimo_program in mimo_programs:
            reward = self.task.reward_function(mimo_program)
            rewards.append(reward)
        
        # Single policy update using combined rewards
        loss = self.compute_mimo_loss(mimo_programs, rewards)
        self.optimizer.step(loss)
```

### **Strategy 2: Alternating Training**

Train one output at a time:

```python
class AlternatingMIMOTrainer:
    def __init__(self, n_outputs):
        self.n_outputs = n_outputs
        self.current_output = 0
    
    def train_step(self):
        # Focus on one output this iteration
        output_idx = self.current_output
        
        # Sample and train for specific output
        programs = self.policy.sample_for_output(output_idx, self.batch_size)
        rewards = [self.task.reward_function_single(prog, output_idx) 
                  for prog in programs]
        
        # Update policy for this output
        loss = self.compute_single_output_loss(programs, rewards, output_idx)
        self.optimizer.step(loss)
        
        # Rotate to next output
        self.current_output = (self.current_output + 1) % self.n_outputs
```

### **Strategy 3: Hierarchical Training**

Learn outputs in order of difficulty:

```python
class HierarchicalMIMOTrainer:
    def __init__(self, output_difficulties):
        self.output_difficulties = output_difficulties  # Easy to hard
        self.current_level = 0
    
    def train_step(self):
        # Train outputs up to current difficulty level
        active_outputs = self.output_difficulties[:self.current_level + 1]
        
        # Sample and train active outputs
        mimo_programs = self.policy.sample_partial(active_outputs, self.batch_size)
        rewards = [self.task.reward_function_partial(prog, active_outputs) 
                  for prog in mimo_programs]
        
        # Update policy
        loss = self.compute_partial_loss(mimo_programs, rewards, active_outputs)
        self.optimizer.step(loss)
        
        # Advance level if current outputs are solved
        if self.check_convergence(active_outputs):
            self.current_level = min(self.current_level + 1, len(self.output_difficulties) - 1)
```

## ⚙️ **MIMO Configuration**

### **JSON Configuration**

```json
{
  "task": {
    "task_type": "mimo_regression",
    "n_outputs": 3,
    "dataset": "path/to/mimo_dataset.csv",
    
    // MIMO-specific settings
    "mimo_strategy": "independent",     // "independent", "shared", "vector"
    "mimo_reward": "mean",              // "mean", "weighted", "min", "max" 
    "mimo_training": "joint",           // "joint", "alternating", "hierarchical"
    
    // Per-output weights (optional)
    "output_weights": [1.0, 1.5, 0.8],
    
    // Per-output thresholds
    "output_thresholds": [1e-12, 1e-10, 1e-12]
  },
  
  "policy": {
    "policy_type": "mimo_rnn",
    "shared_context": true,
    "context_size": 64
  }
}
```

### **Programmatic Setup**

```python
# Create MIMO dataset
X = np.random.randn(1000, 3)  # 3 input features
Y = np.column_stack([
    np.sin(X[:, 0]) + X[:, 1],              # Output 1
    X[:, 0]**2 + np.cos(X[:, 2]),           # Output 2  
    np.exp(X[:, 1]) * X[:, 0]               # Output 3
])

# Setup MIMO task
mimo_task = MIMORegressionTask(
    function_set=["add", "sub", "mul", "sin", "cos", "exp"],
    dataset=(X, Y),
    n_outputs=3,
    mimo_strategy="independent"
)

# Setup MIMO policy
mimo_policy = IndependentMIMOPolicy(
    n_outputs=3,
    library=mimo_task.library
)

# Train MIMO model
mimo_model = DeepSymbolicOptimizer(
    task=mimo_task,
    policy=mimo_policy,
    trainer="joint"
)

mimo_model.train()
```

## 📊 **MIMO Evaluation Metrics**

### **Per-Output Metrics**
```python
mimo_metrics = {
    "output_0_nmse": 0.0001,     # Individual output performance
    "output_1_nmse": 0.0005,
    "output_2_nmse": 0.0002,
    
    "output_0_success": True,     # Success flags
    "output_1_success": True,
    "output_2_success": True
}
```

### **Overall MIMO Metrics**
```python
overall_metrics = {
    "nmse_mean": 0.00027,        # Average across outputs
    "nmse_max": 0.0005,          # Worst output performance
    "nmse_weighted": 0.00035,    # Weighted by importance
    
    "success_all": True,         # All outputs successful
    "success_any": True,         # Any output successful  
    "success_rate": 1.0,         # Fraction successful
    
    "total_complexity": 45,      # Sum of all expression complexities
    "mean_complexity": 15        # Average expression complexity
}
```

## 🚀 **Implementation Roadmap**

### **Phase 1: Basic MIMO (Weeks 1-2)**
- [ ] Implement `MIMOProgram` wrapper class
- [ ] Create `MIMORegressionTask` with multi-output data loading
- [ ] Implement independent sampling strategy
- [ ] Add basic MIMO evaluation metrics
- [ ] Test with synthetic MIMO datasets

### **Phase 2: Advanced Policies (Weeks 3-4)**  
- [ ] Implement shared context sampling
- [ ] Add alternating training strategy
- [ ] Create hierarchical training approach
- [ ] Develop MIMO-specific constraints
- [ ] Add weighted reward functions

### **Phase 3: Optimization (Weeks 5-6)**
- [ ] Parallel execution of multiple expressions
- [ ] Memory optimization for large MIMO problems
- [ ] GPU acceleration for MIMO evaluation
- [ ] Advanced sampling strategies (correlated outputs)
- [ ] Shared sub-expression optimization

### **Phase 4: Extensions (Weeks 7-8)**
- [ ] Vector-valued operators and expressions
- [ ] Multi-task transfer learning
- [ ] MIMO visualization tools
- [ ] Integration with existing benchmark suite
- [ ] Documentation and tutorials

## 💡 **Research Opportunities**

### **Novel Research Directions**
- **Cross-Output Learning**: How outputs can inform each other's discovery
- **MIMO Constraints**: Constraints that operate across multiple outputs
- **Hierarchical Decomposition**: Discovering output relationships and dependencies
- **Transfer Learning**: Using MISO models to bootstrap MIMO discovery
- **Multi-Objective Optimization**: Balancing accuracy vs complexity across outputs

### **Benchmark Development**
- Create MIMO benchmark datasets for evaluation
- Develop MIMO-specific success criteria
- Compare against multi-output machine learning methods
- Establish baseline performance metrics

This MIMO extension will significantly expand DSO's capabilities while maintaining its core strength of discovering interpretable mathematical relationships.
