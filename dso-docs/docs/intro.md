---
sidebar_position: 1
---

# 🏠 Deep Symbolic Optimization (DSO)

<p align="center">
<img src="/img/banner.png" width="750" />
</p>

## What is DSO?

**Deep Symbolic Optimization (DSO)** is a powerful framework that combines deep learning with symbolic optimization to discover mathematical expressions from data. Think of it as an AI that can find the hidden mathematical patterns in your data and express them as human-readable formulas.

## What can DSO do?

### 🔍 **Symbolic Regression**

DSO can take your data and find the mathematical formula that best explains it. For example:

- Given data points from a sine wave, it might discover: `y = sin(x)`
- Given complex relationships, it might find: `y = 2.5 * x^2 + 3.1 * sin(x) + 1.2`

### 🎮 **Control Systems**

DSO can create interpretable control policies for robots and simulations. Instead of black-box neural networks, you get readable rules like:

- "If the car is going too slow, accelerate by 0.5"
- "If the pendulum angle > 0.1, apply force = -2.3 \* angle"

### 🏆 **State-of-the-Art Performance**

DSO has achieved remarkable results:

- **1st place** in the 2022 SRBench Symbolic Regression Competition
- **Best performance** on both symbolic solution rate and accuracy
- **Published** in top AI conferences (ICLR, ICML, NeurIPS)

## Why use DSO?

### ✅ **Interpretable Results**

Unlike black-box neural networks, DSO gives you mathematical expressions you can understand, analyze, and trust.

### ⚡ **Fast & Efficient**

- GPU-accelerated evaluation
- Parallel processing support
- Optimized for both speed and accuracy

### 🔧 **Flexible & Extensible**

- Easy to add new mathematical functions
- Support for custom datasets and environments
- Pluggable policy representations (RNN, Transformer)

### 🎯 **Production Ready**

- Comprehensive testing and validation
- Well-documented APIs
- Active development and community support

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/your-org/dso.git && cd dso
./main.sh  # Press '1' when prompted

# 2. Activate environment
source .venv/bin/activate

# 3. Run your first experiment
python -m dso.run dso/config/config_regression.json --b Nguyen-7
```

## Simple Example

```python
from dso import DeepSymbolicRegressor
import numpy as np

# Generate some data
X = np.random.random((100, 2))
y = np.sin(X[:,0]) + X[:,1] ** 2

# Create and train the model
model = DeepSymbolicRegressor()
model.fit(X, y)

# Get the discovered formula
print(model.program_.pretty())
# Output might be: "sin(x0) + x1^2"
```

## What's Next?

- **[Getting Started](core/getting_started)** - Learn how to run your first DSO experiment
- **[Setup Guide](core/setup)** - Detailed installation and configuration instructions
- **[Core Concepts](core/concept)** - Understand the fundamental ideas behind DSO

---

_DSO is developed by researchers at Lawrence Livermore National Laboratory and has been published in top-tier AI conferences including ICLR, ICML, and NeurIPS._
