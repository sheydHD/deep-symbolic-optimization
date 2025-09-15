# DSO Documentation

Welcome to the documentation for **Deep Symbolic Optimization (DSO)** - a state-of-the-art framework for discovering interpretable mathematical expressions using deep reinforcement learning.

## 📖 Documentation Overview

This documentation is organized into four main sections to guide you from initial setup to advanced applications:

### 🚀 **Getting Started**
Perfect for newcomers who want to quickly understand and use DSO:
- **[Installation & Setup](core/setup.md)** - Complete setup guide with troubleshooting
- **[Quick Start Tutorial](core/getting_started.md)** - Run your first DSO experiment in minutes

### 🧠 **Core Concepts**  
Deep dive into the technical foundations of DSO:
- **[Fundamental Concepts](core/concept.md)** - Symbolic regression and reinforcement learning basics
- **[System Architecture](core/architecture.md)** - How DSO components work together
- **[Token System](core/tokens.md)** - Mathematical building blocks and operators
- **[Training Process](core/training.md)** - Neural network training and optimization
- **[Constraints & Priors](core/constraints.md)** - Incorporating domain knowledge

### 🚀 **Advanced Topics**
Specialized features for research and complex applications:
- **[MIMO Extensions](core/mimo.md)** - Multiple output regression capabilities
- **[Advanced Features](core/advanced.md)** - Multi-task learning, custom operators, ensemble methods

### 📋 **Development**
Guidelines for contributing and extending DSO:
- **[Rules](rules/)** - Coding standards, git workflow, and contribution guidelines  
- **[Project Structure](https://github.com/sheydHD/deep-symbolic-optimization)** - See the main repository for codebase organization

## 🎯 Learning Paths

### **For New Users**
1. Start with **[Quick Start](core/getting_started.md)** to run your first experiment
2. Read **[Core Concepts](core/concept.md)** to understand the fundamentals
3. Explore **[System Architecture](core/architecture.md)** for deeper understanding

### **For Researchers**
1. Master the **[Core Concepts](core/concept.md)** section completely
2. Study **[Training Process](core/training.md)** and **[Constraints](core/constraints.md)**
3. Explore **[Advanced Features](core/advanced.md)** for cutting-edge capabilities
4. Consider **[MIMO Extensions](core/mimo.md)** for novel applications

### **For Developers**  
1. Complete the **Getting Started** section
2. Review **[Development Rules](rules/)** and the main repository structure
3. Study **[System Architecture](core/architecture.md)** for implementation details
4. Examine **[Advanced Features](core/advanced.md)** for extension opportunities

## 🔍 Quick Reference

### **Key Concepts**
- **Symbolic Regression**: Discovering mathematical formulas from data
- **Policy Network**: Neural network that generates mathematical expressions
- **Token System**: Mathematical operators as building blocks
- **REINFORCE**: Reinforcement learning algorithm for training
- **Constraints**: Domain knowledge integration mechanism

### **Main Components**
- **RNN Policy**: Expression generation neural network
- **Program**: Mathematical expression representation
- **Task**: Problem definition and evaluation
- **Library**: Available mathematical operators
- **Prior**: Constraint and domain knowledge system

### **Typical Workflow**
1. **Setup**: Install DSO and prepare your dataset
2. **Configure**: Define function set and training parameters
3. **Train**: Run reinforcement learning to discover expressions
4. **Evaluate**: Analyze discovered mathematical formulas
5. **Apply**: Use expressions for prediction or scientific insight

## 💡 Tips for Success

### **Getting the Most from DSO**
- **Start Simple**: Begin with small problems to understand the workflow
- **Understand Your Data**: DSO works best with clean, well-structured datasets
- **Choose Functions Wisely**: Include operators relevant to your domain
- **Use Constraints**: Incorporate domain knowledge for better results
- **Monitor Training**: Watch convergence and adjust parameters as needed

### **Common Pitfalls to Avoid**
- **Too Many Functions**: Large function sets slow down training
- **Insufficient Training**: DSO needs adequate iterations to find good solutions
- **Ignoring Complexity**: Balance accuracy with expression simplicity
- **Poor Data Quality**: Noisy or insufficient data leads to poor discoveries

## 🆘 Getting Help

### **Documentation Issues**
- Check the **[Troubleshooting](core/setup.md#troubleshooting)** section
- Search through existing **[GitHub Issues](https://github.com/your-org/dso/issues)**
- Review **[FAQ sections](core/setup.md#getting-help)** in relevant guides

### **Technical Support**
- **Community**: Join discussions in GitHub Issues and Discussions
- **Academic**: Cite relevant papers and contact authors for research questions
- **Enterprise**: Contact for professional support and consulting

### **Contributing**
- Read **[Contribution Guidelines](rules/)** before submitting changes
- Follow **[Code Style](rules/code_style.md)** and **[Testing Rules](rules/testing_rules.md)**
- Check the main repository for codebase organization

## 📚 External Resources

### **Academic Papers**
- **[Original DSO Paper](https://arxiv.org/abs/xxxx.xxxxx)** - Foundational methodology
- **[SRBench Competition](https://arxiv.org/abs/xxxx.xxxxx)** - Benchmark results
- **[Recent Advances](https://arxiv.org/abs/xxxx.xxxxx)** - Latest improvements

### **Related Tools**
- **[PySR](https://github.com/MilesCranmer/PySR)** - Alternative symbolic regression tool
- **[SymPy](https://www.sympy.org/)** - Symbolic mathematics library
- **[GPlearn](https://gplearn.readthedocs.io/)** - Genetic programming for symbolic regression

### **Datasets & Benchmarks**
- **[SRBench](https://github.com/cavalab/srbench)** - Comprehensive benchmark suite
- **[Feynman Equations](https://space.mit.edu/home/tegmark/aifeynman.html)** - Physics equation datasets
- **[PMLB](https://github.com/EpistasisLab/pmlb)** - Penn Machine Learning Benchmarks

---

**Ready to discover mathematical laws with AI?** Start with our **[Quick Start Guide](core/getting_started.md)** and join the growing community of researchers using DSO for scientific discovery! 🚀