# Constraints & Priors

> Version: 1.0 • Last updated: 2025-09-01

This guide explains DSO's constraint system for incorporating domain knowledge and guiding the search toward meaningful mathematical expressions.

## 🎯 **Why Use Constraints?**

Constraints serve multiple critical purposes in symbolic regression:

- **Guide Search**: Direct exploration toward promising areas of expression space
- **Incorporate Knowledge**: Use domain expertise to improve discovery
- **Prevent Invalid**: Avoid mathematical expressions that don't make sense
- **Improve Efficiency**: Reduce search space by eliminating poor solutions
- **Ensure Quality**: Maintain expression interpretability and usability

## 🏗️ **Constraint System Architecture**

### **Joint Prior System**

DSO uses a `JointPrior` that combines multiple individual constraints:

```python
class JointPrior:
    def __init__(self, constraints_list):
        self.constraints = constraints_list
    
    def __call__(self, actions, parent, sibling, dangling):
        """
        Evaluate all constraints and return combined penalty
        
        Returns:
            log_penalty: Log probability penalty (0 = allowed, -inf = forbidden)
        """
        total_penalty = 0
        for constraint in self.constraints:
            penalty = constraint(actions, parent, sibling, dangling)
            total_penalty += penalty
        
        return total_penalty
```

### **Constraint Application**

Constraints are applied during expression generation:

```python
def sample_next_token(policy, current_state, priors):
    # Get token probabilities from policy
    logits = policy.forward(current_state)
    
    # Apply constraint penalties
    for i, token in enumerate(library.tokens):
        penalty = priors(token, current_state)
        logits[i] += penalty  # Add log penalty
    
    # Sample from constrained distribution  
    probabilities = softmax(logits)
    next_token = sample(probabilities)
    
    return next_token
```

## 🔧 **Core Constraint Types**

### **1. Length Constraints**

Control expression complexity by limiting size:

```python
class LengthConstraint:
    def __init__(self, min_length=4, max_length=30):
        self.min_length = min_length
        self.max_length = max_length
    
    def __call__(self, actions, parent, sibling, dangling):
        current_length = len(actions)
        
        # Forbid early termination
        if current_length < self.min_length and dangling == 0:
            return -np.inf
        
        # Forbid exceeding max length
        if current_length >= self.max_length:
            return -np.inf
            
        return 0.0  # No penalty
```

### **2. Repeat Constraints**

Limit repeated use of the same operator:

```python
class RepeatConstraint:
    """
    Prevents excessive repetition of specific tokens
    
    Example: Limit 'sin' to appear at most 2 times
    """
    def __init__(self, tokens, max_repeats=2):
        self.restricted_tokens = tokens
        self.max_repeats = max_repeats
    
    def __call__(self, actions, parent, sibling, dangling):
        for token_name in self.restricted_tokens:
            count = actions.count(token_name)
            if count >= self.max_repeats:
                # Forbid adding another instance
                if current_token == token_name:
                    return -np.inf
        
        return 0.0
```

**Configuration Example:**
```json
{
  "prior": {
    "repeat": {
      "tokens": ["sin", "cos", "exp"],
      "max_": 2,
      "on": true
    }
  }
}
```

### **3. Relational Constraints**

Control relationships between different types of tokens:

#### **Child Relationship**
Restrict which tokens can be children of specific parents:

```python
class ChildConstraint:
    """
    Example: Prevent 'exp' from being a child of 'log'
    (avoids expressions like log(exp(x)) which simplify to x)
    """
    def __init__(self, parents, forbidden_children):
        self.parents = parents
        self.forbidden_children = forbidden_children
    
    def __call__(self, actions, parent, sibling, dangling):
        if parent in self.parents and current_token in self.forbidden_children:
            return -np.inf
        return 0.0
```

#### **Sibling Relationship** 
Control which tokens can appear as siblings:

```python
class SiblingConstraint:
    """
    Example: Prevent both 'sin' and 'cos' from being siblings of same parent
    (encourages diversity in trigonometric usage)
    """
    def __init__(self, forbidden_pairs):
        self.forbidden_pairs = forbidden_pairs
    
    def __call__(self, actions, parent, sibling, dangling):
        for token1, token2 in self.forbidden_pairs:
            if sibling == token1 and current_token == token2:
                return -np.inf
            if sibling == token2 and current_token == token1:
                return -np.inf
        return 0.0
```

**Configuration Example:**
```json
{
  "prior": {
    "relational": {
      "targets": "sin,cos",
      "effectors": "x1", 
      "relationship": "sibling",
      "on": true
    }
  }
}
```

### **4. Trigonometric Constraints**

Special handling for trigonometric functions:

```python
class TrigConstraint:
    """
    Prevents nested trigonometric functions
    Example: Forbids sin(cos(x)) or cos(sin(x))
    """
    def __init__(self):
        self.trig_functions = ["sin", "cos", "tan"]
    
    def __call__(self, actions, parent, sibling, dangling):
        # If parent is trigonometric
        if parent in self.trig_functions:
            # Forbid trigonometric children
            if current_token in self.trig_functions:
                return -np.inf
        
        return 0.0
```

### **5. Inverse Function Constraints**

Prevent expressions with obvious simplifications:

```python
class InverseConstraint:
    """
    Prevents inverse operations that cancel out
    Examples: 
    - exp(log(x)) → x
    - log(exp(x)) → x  
    - sin(asin(x)) → x
    """
    def __init__(self):
        self.inverse_pairs = {
            "exp": "log",
            "log": "exp", 
            "sin": "asin",
            "asin": "sin"
        }
    
    def __call__(self, actions, parent, sibling, dangling):
        if parent in self.inverse_pairs:
            if current_token == self.inverse_pairs[parent]:
                return -np.inf
        
        return 0.0
```

### **6. Unique Child Constraints**

Prevent identical operands in commutative operations:

```python
class UniqueChildConstraint:
    """
    Prevents expressions like x + x or x * x
    (can be better represented as 2*x or x^2)
    """
    def __init__(self, operators=["add", "mul"]):
        self.operators = operators
    
    def __call__(self, actions, parent, sibling, dangling):
        if parent in self.operators:
            if sibling == current_token:
                return -np.inf  # Forbid identical siblings
        
        return 0.0
```

## 🧠 **Advanced Constraint Features**

### **State Checker Constraints**

For decision tree components:

```python
class StateCheckerConstraint:
    """
    Ensures proper ordering of state checker thresholds
    Example: If checking x1 < 0.2, don't allow x1 < 0.1 as sibling
    """
    def __init__(self):
        self.state_checkers = []
    
    def __call__(self, actions, parent, sibling, dangling):
        # Complex logic for state checker ordering
        # Ensures meaningful decision tree structures
        pass
```

### **Constant Constraints**

Control placement and usage of constants:

```python
class ConstConstraint:
    """
    Prevents overuse of constants
    Example: Limit to 3 constants per expression
    """
    def __init__(self, max_constants=3):
        self.max_constants = max_constants
    
    def __call__(self, actions, parent, sibling, dangling):
        const_count = actions.count("const")
        if const_count >= self.max_constants and current_token == "const":
            return -np.inf
        
        return 0.0
```

### **No Inputs Constraint**

Ensure expressions use input variables:

```python
class NoInputsConstraint:
    """
    Prevents expressions that don't use any input variables
    Example: Forbids constant-only expressions like "2.5 + 1.3"
    """
    def __call__(self, actions, parent, sibling, dangling):
        # Check if expression contains any input variables
        has_inputs = any(token.startswith('x') for token in actions)
        
        # If completing expression without inputs, forbid
        if dangling == 0 and not has_inputs:
            return -np.inf
        
        return 0.0
```

## ⚙️ **Constraint Configuration**

### **JSON Configuration Format**

```json
{
  "prior": {
    // Length constraints
    "length": {
      "min_": 4,
      "max_": 30,
      "on": true
    },
    
    // Repetition constraints  
    "repeat": {
      "tokens": ["sin", "cos", "exp", "log"],
      "max_": 2,
      "on": true
    },
    
    // Relational constraints
    "relational": [
      {
        "targets": "exp,log",
        "effectors": "log,exp", 
        "relationship": "child",
        "on": true
      },
      {
        "targets": "sin,cos",
        "effectors": "x1",
        "relationship": "sibling", 
        "on": true
      }
    ],
    
    // Trigonometric constraints
    "trig": {
      "on": true
    },
    
    // Inverse function constraints
    "inverse": {
      "on": true
    },
    
    // Unique children constraints
    "uchild": {
      "targets": "x1",
      "effectors": "add,mul",
      "on": true
    },
    
    // Constant constraints
    "const": {
      "on": true
    },
    
    // No inputs constraint
    "no_inputs": {
      "on": true
    }
  }
}
```

### **Programmatic Configuration**

```python
# Create individual constraints
length_constraint = LengthConstraint(min_length=4, max_length=30)
repeat_constraint = RepeatConstraint(["sin", "cos"], max_repeats=2)
trig_constraint = TrigConstraint()

# Combine into joint prior
joint_prior = JointPrior([
    length_constraint,
    repeat_constraint, 
    trig_constraint
])

# Apply to model
model.set_prior(joint_prior)
```

## 🎯 **Best Practices**

### **Constraint Selection Guidelines**

```python
# Start with basic constraints
essential_constraints = [
    "length",      # Prevent too long/short expressions
    "no_inputs",   # Ensure input variables are used
    "const"        # Limit constant usage
]

# Add domain-specific constraints
if trigonometric_domain:
    constraints.append("trig")      # Prevent nested trig
    constraints.append("inverse")   # Prevent exp(log(x))

if polynomial_fitting:
    constraints.append("repeat")    # Limit repeated operators
    constraints.append("uchild")    # Prevent x + x
```

### **Tuning Constraint Strength**

```python
# Constraints can have adjustable strength
class SoftConstraint:
    def __init__(self, base_constraint, strength=1.0):
        self.base_constraint = base_constraint
        self.strength = strength
    
    def __call__(self, *args):
        penalty = self.base_constraint(*args)
        return penalty * self.strength  # Scale penalty
```

### **Problem-Specific Customization**

```python
# Physics problems: Encourage dimensional consistency
physics_constraints = [
    DimensionalConstraint(),     # Check unit compatibility
    ConservationConstraint(),    # Enforce conservation laws
    SymmetryConstraint()         # Respect physical symmetries
]

# Financial modeling: Prevent unrealistic operations
finance_constraints = [
    PositivityConstraint(),      # Prices must be positive
    MonotonicityConstraint(),    # Some relationships must be monotonic
    BoundednessConstraint()      # Prevent explosive growth
]
```

## 🔍 **Debugging Constraints**

### **Constraint Violation Analysis**

```python
def analyze_constraint_violations(expressions, constraints):
    """
    Analyze which constraints are most frequently violated
    """
    violation_counts = {constraint.name: 0 for constraint in constraints}
    
    for expression in expressions:
        for constraint in constraints:
            if constraint.is_violated(expression):
                violation_counts[constraint.name] += 1
    
    return violation_counts
```

### **Constraint Effectiveness Metrics**

```python
def measure_constraint_effectiveness(with_constraints, without_constraints):
    """
    Compare performance with and without constraints
    """
    metrics = {
        "convergence_speed": measure_convergence_time(with_constraints, without_constraints),
        "solution_quality": measure_solution_quality(with_constraints, without_constraints), 
        "search_efficiency": measure_search_efficiency(with_constraints, without_constraints)
    }
    
    return metrics
```

## 🚀 **Advanced Applications**

### **Adaptive Constraints**

Constraints that change during training:

```python
class AdaptiveRepeatConstraint:
    def __init__(self, initial_max=3, min_max=1):
        self.max_repeats = initial_max
        self.min_max = min_max
    
    def update(self, training_progress):
        # Tighten constraints as training progresses
        if training_progress > 0.5:
            self.max_repeats = max(self.min_max, self.max_repeats - 1)
```

### **Learned Constraints**

Use ML to learn effective constraints:

```python
class LearnedConstraint:
    def __init__(self):
        self.classifier = train_constraint_classifier()
    
    def __call__(self, actions, parent, sibling, dangling):
        features = extract_features(actions, parent, sibling, dangling)
        violation_prob = self.classifier.predict_proba(features)
        
        # Convert probability to log penalty
        penalty = np.log(violation_prob)
        return penalty
```

This constraint system provides powerful tools for incorporating domain knowledge and guiding DSO toward discovering meaningful, interpretable mathematical expressions.
