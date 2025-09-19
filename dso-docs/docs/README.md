# DSO Documentation

Welcome to the documentation for **Deep Symbolic Optimization (DSO)** - a framework for discovering interpretable mathematical expressions using deep reinforcement learning.

## 📖 Documentation Overview

This documentation is organized to guide you from initial setup to advanced applications:

### 🚀 **Installation**
Complete setup guide for getting DSO running:
- **[System Requirements](installation/requirements.md)** - Hardware and software prerequisites
- **[Installation Guide](installation/setup.md)** - Step-by-step installation instructions
- **[Quick Start](installation/quickstart.md)** - Run your first experiment

### 🧠 **Regression**  
Core functionality and symbolic regression capabilities:
- **[Overview](regression/overview.md)** - Symbolic regression fundamentals
- **[Single Output](regression/single-output.md)** - Standard regression problems
- **[Multi-Output](regression/multi-output.md)** - MIMO capabilities and systems
- **[Configuration](regression/configuration.md)** - Parameters and settings
- **[Optimization](regression/optimization.md)** - Performance tuning

### � **API Reference**
Technical documentation for developers:
- **[Core API](api/core-api.md)** - Main classes and methods
- **[Tokens](core/tokens.md)** - Mathematical operators and functions
- **[Constraints](core/constraints.md)** - Domain knowledge integration

### � **Examples**
Practical applications and use cases:
- **[Basic Regression](examples/basic-regression.md)** - Fundamental examples and tutorials

### ⚙️ **Advanced**
Advanced features and system internals:
- **[Architecture](core/architecture.md)** - System design and components
- **[Training](core/training.md)** - Neural network training process
- **[Advanced Features](core/advanced.md)** - Extended capabilities

## 🎯 Learning Paths

### **For New Users**
1. Review **[System Requirements](installation/requirements.md)** and complete **[Installation](installation/setup.md)**
2. Follow the **[Quick Start Guide](installation/quickstart.md)** to run your first experiment
3. Study **[Regression Overview](regression/overview.md)** to understand core concepts

### **For Researchers**
1. Master the **[Regression](regression/overview.md)** section completely
2. Explore **[Multi-Output](regression/multi-output.md)** capabilities for novel applications
3. Study **[Configuration](regression/configuration.md)** and **[Optimization](regression/optimization.md)**
4. Review **[Advanced Features](core/advanced.md)** for cutting-edge capabilities

### **For Developers**  
1. Complete the **Installation** section
2. Study **[Core API](api/core-api.md)** for programmatic integration
3. Examine **[Architecture](core/architecture.md)** for implementation details
4. Review **[Advanced Features](core/advanced.md)** for extension opportunities

## 🔍 Quick Reference

### **Key Concepts**
- **Symbolic Regression**: Discovering mathematical formulas from data
- **Policy Network**: Neural network that generates mathematical expressions
- **MIMO**: Multiple Input Multiple Output regression capabilities
- **REINFORCE**: Reinforcement learning algorithm for training
- **Configuration**: Parameter settings for problem-specific optimization

### **Main Components**
- **DeepSymbolicOptimizer**: Main interface for DSO functionality
- **Program**: Mathematical expression representation
- **Function Set**: Available mathematical operators
- **Training Process**: Reinforcement learning workflow
- **Evaluation**: Expression fitness and performance assessment

### **Typical Workflow**
1. **Setup**: Install DSO and prepare your dataset
2. **Configure**: Define function set and training parameters
3. **Train**: Run reinforcement learning to discover expressions
4. **Evaluate**: Analyze discovered mathematical formulas
5. **Apply**: Use expressions for prediction or scientific insight

## 💡 Tips for Success

### **Getting the Most from DSO**
- **Start Simple**: Begin with basic examples to understand the workflow
- **Understand Your Data**: DSO works best with clean, structured datasets
- **Choose Functions Wisely**: Include operators relevant to your domain
- **Monitor Training**: Watch convergence and adjust parameters as needed
- **Validate Results**: Use holdout data to verify discovered expressions

### **Common Pitfalls to Avoid**
- **Excessive Function Sets**: Large function sets slow down training
- **Insufficient Training**: DSO needs adequate iterations for good solutions
- **Ignoring Complexity**: Balance accuracy with expression interpretability
- **Poor Data Quality**: Noisy or insufficient data leads to poor discoveries

## 🆘 Getting Help

### **Documentation Issues**
- Check relevant troubleshooting sections in each guide
- Review **[Configuration](regression/configuration.md)** for parameter guidance
- Search existing **[GitHub Issues](https://github.com/sheydHD/deep-symbolic-optimization/issues)**

### **Technical Support**
- **Community**: Join discussions in GitHub Issues and Discussions
- **Academic**: Contact authors for research-related questions
- **Professional**: Contact for enterprise support and consulting

## 📚 External Resources

### **Related Tools**
- **[PySR](https://github.com/MilesCranmer/PySR)** - Alternative symbolic regression framework
- **[SymPy](https://www.sympy.org/)** - Symbolic mathematics library
- **[scikit-learn](https://scikit-learn.org/)** - Machine learning toolkit

### **Datasets & Benchmarks**
- **[SRBench](https://github.com/cavalab/srbench)** - Comprehensive benchmark suite
- **[Feynman Equations](https://space.mit.edu/home/tegmark/aifeynman.html)** - Physics equation datasets
- **[PMLB](https://github.com/EpistasisLab/pmlb)** - Penn Machine Learning Benchmarks

---

**Ready to discover mathematical relationships with AI?** Start with our **[Installation Guide](installation/requirements.md)** and explore the power of symbolic regression! 🚀