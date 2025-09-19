---
sidebar_position: 1
---

# Deep Symbolic Optimization (DSO)

Deep Symbolic Optimization (DSO) is a framework for automated discovery of interpretable mathematical expressions from data using deep reinforcement learning. DSO generates human-readable mathematical formulas that provide transparency and interpretability compared to traditional black-box machine learning models.

## Overview

DSO employs neural networks to intelligently search the space of mathematical expressions, learning to generate formulas that accurately model input data. The framework is particularly valuable in scientific computing, engineering, and research applications where understanding the underlying mathematical relationships is critical.

## Key Capabilities

- **Symbolic Regression**: Automated discovery of mathematical expressions from numerical data
- **Multi-Output Support**: Comprehensive MIMO (Multiple Input Multiple Output) regression capabilities
- **Interpretable Results**: Generation of explicit mathematical equations rather than black-box models
- **High Performance**: GPU acceleration and parallel processing support
- **Production Ready**: Robust APIs with comprehensive testing and validation

## Performance

DSO has demonstrated state-of-the-art performance in symbolic regression:
- First place in the 2022 SRBench Symbolic Regression Competition
- Published research in leading AI conferences (ICLR, ICML, NeurIPS)
- Superior accuracy on standard benchmark datasets

## Algorithm

DSO uses a reinforcement learning approach where a recurrent neural network (RNN) policy generates candidate mathematical expressions as sequences of tokens. Each expression is evaluated on training data to compute a fitness score based on accuracy and complexity. The policy network learns through the REINFORCE algorithm, iteratively improving its ability to generate optimal mathematical formulas.

## Applications

### Scientific Computing
- Physics: Derive governing equations from experimental data
- Chemistry: Discover reaction kinetics and molecular relationships
- Biology: Model population dynamics and biological processes

### Engineering
- Control Systems: Design and optimization of control algorithms
- Signal Processing: Feature extraction and system identification
- Materials Science: Property prediction and design optimization

### Financial Modeling
- Risk Assessment: Quantitative models for risk analysis
- Algorithmic Trading: Discovery of market relationships
- Economic Modeling: Macroeconomic relationship identification

## Getting Started

To begin using DSO for symbolic regression, proceed to the [Installation](/installation/requirements) section for setup instructions, then review the [Regression Overview](/regression/overview) for core functionality.
Create interpretable control policies for autonomous systems:
- **Robotics**: Generate readable control laws instead of black-box neural networks
- **Aerospace**: Develop flight control algorithms with mathematical guarantees
- **Manufacturing**: Design process control equations for industrial systems

**Example**: For a robotic arm, DSO might find: `τ = Kₚ(θ_target - θ) + Kᵈ(ω_target - ω)`

### 🧪 **Scientific Discovery**
Accelerate research by automatically finding mathematical laws:
- **Chemistry**: Discover reaction rate equations from experimental data
- **Biology**: Find population dynamics equations from observation data
- **Materials Science**: Derive property-structure relationships

## Why Choose DSO?

### ✅ **Transparency & Trust**
Unlike black-box models, DSO provides mathematical expressions that can be:
- **Verified** by domain experts
- **Analyzed** for physical consistency  
- **Validated** against known laws
- **Interpreted** with confidence

### ⚡ **High Performance**
- **GPU acceleration** for fast training and evaluation
- **Parallel processing** for handling large datasets
- **Optimized algorithms** for efficient expression search
- **Scalable architecture** for complex problems

### 🔧 **Flexibility & Extensibility**
- **Custom operators**: Add domain-specific mathematical functions
- **Configurable constraints**: Incorporate prior knowledge and physical laws
- **Multiple architectures**: RNN, Transformer, and custom policy networks
- **Extensible framework**: Easy integration with existing workflows

### 🎯 **Research-Grade Quality**
- **Rigorous testing**: Comprehensive test suite ensuring reliability
- **Peer-reviewed**: Published methods in top-tier venues
- **Active development**: Continuous improvements and new features
- **Community support**: Growing ecosystem of users and contributors

## Quick Start

Get started with DSO in just three commands:

```bash
# 1. Clone and setup
git clone https://github.com/your-org/dso.git && cd dso
./main.sh modern setup  # Automated setup

# 2. Activate environment  
source .venv/bin/activate

# 3. Run your first experiment
python -m dso.run dso_pkg/dso/config/examples/regression/Nguyen-2.json

# Or use interactive menu
python tools/python/run.py
```

## Simple Example

Here's how to use DSO to discover mathematical formulas:

### Basic Sklearn-style API
```python
from dso import DeepSymbolicRegressor
import numpy as np

# Generate training data from a known function
X = np.random.random((100, 2))
y = np.sin(X[:,0]) + X[:,1] ** 2  # True formula: sin(x₀) + x₁²

# Train DSO to discover the formula
model = DeepSymbolicRegressor()
model.fit(X, y)

# View the discovered expression
print("Discovered formula:", model.program_.pretty())
# Output: "sin(x0) + x1^2"

# Use for prediction
y_pred = model.predict(X_test)
```

### Direct DSO API
```python
from dso import DeepSymbolicOptimizer
from dso.config import load_config

# Load configuration
config = load_config("dso_pkg/dso/config/config_regression.json")
config["task"]["dataset"] = "Nguyen-2"  # x^4 + x^3 + x^2 + x

# Create and train model
model = DeepSymbolicOptimizer(config)
result = model.train()

print("Best expression:", result["expression"])
print("Reward:", result["r"])
```

### MIMO (Multi-Output) Example
```python
from dso.unified_dso import UnifiedDSO
import numpy as np

# Multi-output data: 2 inputs → 3 outputs
X = np.random.random((100, 2))
y = np.column_stack([
    X[:,0] * X[:,1],        # Output 1: x₀ * x₁
    np.sin(X[:,0]),         # Output 2: sin(x₀)
    X[:,0] + X[:,1]         # Output 3: x₀ + x₁
])

# Unified DSO automatically detects MIMO
dso = UnifiedDSO()
results = dso.fit((X, y))

print("MIMO expressions found:")
for i, expr in enumerate(results["expressions"]):
    print(f"Output {i+1}: {expr}")
```

## Documentation Structure

This documentation is organized into four main sections:

### 📚 **Getting Started**
- **[Quick Start Guide](core/getting_started)** - Run your first DSO experiment
- **[Installation & Setup](core/setup)** - Detailed setup instructions and troubleshooting

### 🧠 **Core Concepts** 
- **[Fundamental Concepts](core/concept)** - Understanding symbolic regression and reinforcement learning
- **[System Architecture](core/architecture)** - How DSO components work together
- **[Regression Features](core/regression_features)** - Comprehensive regression capabilities and MIMO support
- **[Token System](core/tokens)** - Mathematical building blocks and operators
- **[Training Process](core/training)** - Neural network training and optimization
- **[Constraints & Priors](core/constraints)** - Incorporating domain knowledge

### 🚀 **Advanced Topics**
- **[MIMO Extensions](core/mimo)** - Multiple output regression capabilities  
- **[Advanced Features](core/advanced)** - Multi-task learning, custom operators, and ensemble methods

### 📋 **Development Guidelines**
- **Rules** - Coding standards, git workflow, and contribution guidelines
- **Structure** - Project organization and testing protocols

## Next Steps

Ready to get started? We recommend following this learning path:

1. **Begin with** [Getting Started](core/getting_started) to run your first experiment
2. **Understand** [Core Concepts](core/concept) to grasp the fundamentals  
3. **Explore** [System Architecture](core/architecture) for deeper technical understanding
4. **Experiment** with [Advanced Features](core/advanced) for specialized applications

---

*DSO is developed by researchers at Lawrence Livermore National Laboratory and has been published in top-tier AI conferences including ICLR, ICML, and NeurIPS.*
