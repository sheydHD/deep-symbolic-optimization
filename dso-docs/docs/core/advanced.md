# Advanced DSO Features

> Version: 1.0 • Last updated: 2025-09-01

This guide covers advanced features and capabilities of Deep Symbolic Optimization beyond basic regression tasks.

## 🔬 **Multi-Task Learning**

### **Task Hierarchy**
DSO supports hierarchical task structures where simpler expressions inform more complex ones:

```python
# Define task hierarchy
task_hierarchy = {
    "level_1": ["basic_arithmetic", "simple_trigonometry"],
    "level_2": ["polynomial_combinations", "exponential_functions"], 
    "level_3": ["complex_compositions", "multi_variable_interactions"]
}

# Hierarchical training
hierarchical_optimizer = DeepSymbolicOptimizer(
    task_hierarchy=task_hierarchy,
    transfer_knowledge=True,
    progressive_complexity=True
)
```

### **Transfer Learning**
Pre-trained policies can be fine-tuned for new but related tasks:

```python
# Load pre-trained policy
base_policy = load_policy("pretrained_physics_expressions.pkl")

# Fine-tune for new domain
chemistry_task = RegressionTask(
    dataset="chemical_reactions.csv",
    function_set=["add", "mul", "exp", "log", "pow"]
)

transferred_optimizer = DeepSymbolicOptimizer(
    task=chemistry_task,
    policy=base_policy,
    fine_tune=True,
    freeze_layers=["embedding", "encoder"]  # Keep some layers frozen
)
```

## 🧮 **Custom Function Sets**

### **Domain-Specific Functions**
Create specialized operators for specific domains:

```python
# Physics operators
physics_functions = [
    # Quantum mechanics
    Token(lambda psi: np.abs(psi)**2, "probability_density", arity=1),
    Token(lambda E, h: E / h, "frequency", arity=2),
    
    # Thermodynamics  
    Token(lambda n, R, T: n * R * T, "ideal_gas", arity=3),
    Token(lambda T, T0: T / T0, "reduced_temperature", arity=2),
    
    # Electromagnetism
    Token(lambda q1, q2, r: q1 * q2 / (r**2), "coulomb_force", arity=3),
    Token(lambda B, v, q: q * np.cross(B, v), "lorentz_force", arity=3)
]

# Chemistry operators
chemistry_functions = [
    # Reaction kinetics
    Token(lambda k, A: k * A, "first_order", arity=2),
    Token(lambda k, A, B: k * A * B, "second_order", arity=3),
    
    # Thermochemistry
    Token(lambda H, T, S: H - T * S, "gibbs_energy", arity=3),
    Token(lambda Ea, R, T: np.exp(-Ea / (R * T)), "arrhenius", arity=3)
]
```

### **Protected Operations**
Functions that handle edge cases gracefully:

```python
def protected_log(x, epsilon=1e-10):
    """Logarithm that doesn't crash on invalid inputs"""
    x_safe = np.abs(x) + epsilon
    return np.log(x_safe)

def protected_divide(x, y, epsilon=1e-10):
    """Division that avoids divide-by-zero"""
    y_safe = np.where(np.abs(y) < epsilon, epsilon, y)
    return x / y_safe

def protected_sqrt(x):
    """Square root of absolute value"""
    return np.sqrt(np.abs(x))

# Register protected functions
protected_set = [
    Token(protected_log, "plog", arity=1),
    Token(protected_divide, "pdiv", arity=2), 
    Token(protected_sqrt, "psqrt", arity=1)
]
```

## 🎯 **Advanced Constraints**

### **Mathematical Constraints**
Enforce mathematical properties in discovered expressions:

```python
class MathematicalConstraints:
    def __init__(self):
        self.constraints = []
    
    def add_symmetry_constraint(self, variables):
        """Enforce symmetry w.r.t. specified variables"""
        def symmetry_check(program):
            # Check if f(x,y) = f(y,x)
            return self.check_symmetry(program, variables)
        
        self.constraints.append(symmetry_check)
    
    def add_monotonicity_constraint(self, variable, direction="increasing"):
        """Enforce monotonic behavior"""
        def monotonic_check(program):
            test_values = np.linspace(-10, 10, 100)
            outputs = program.execute(test_values) 
            
            if direction == "increasing":
                return np.all(np.diff(outputs) >= 0)
            else:
                return np.all(np.diff(outputs) <= 0)
        
        self.constraints.append(monotonic_check)
    
    def add_boundary_constraint(self, bounds):
        """Enforce output bounds"""
        def boundary_check(program):
            test_inputs = np.random.randn(1000, program.n_inputs)
            outputs = program.execute(test_inputs)
            return np.all((outputs >= bounds[0]) & (outputs <= bounds[1]))
        
        self.constraints.append(boundary_check)
```

### **Physical Constraints**
Enforce physical laws and principles:

```python
class PhysicsConstraints:
    def add_dimensional_analysis(self, input_dimensions, output_dimension):
        """Ensure dimensional consistency"""
        def dimensional_check(program):
            # Track dimensions through expression
            program_dimensions = self.analyze_dimensions(program, input_dimensions)
            return program_dimensions == output_dimension
        
        return dimensional_check
    
    def add_conservation_law(self, conservation_type="energy"):
        """Enforce conservation laws"""
        def conservation_check(program):
            # Check if expression conserves specified quantity
            if conservation_type == "energy":
                return self.check_energy_conservation(program)
            elif conservation_type == "momentum":
                return self.check_momentum_conservation(program)
        
        return conservation_check
    
    def add_causality_constraint(self, time_variable):
        """Ensure causal relationships"""
        def causality_check(program):
            # Output at time t should only depend on inputs at t' <= t
            return self.check_temporal_causality(program, time_variable)
        
        return causality_check
```

## 📊 **Advanced Evaluation Metrics**

### **Multi-Objective Metrics**
Balance multiple criteria simultaneously:

```python
class MultiObjectiveEvaluator:
    def __init__(self, weights=None):
        self.weights = weights or {"accuracy": 0.6, "complexity": 0.3, "interpretability": 0.1}
    
    def evaluate(self, program, X_test, y_test):
        # Accuracy component
        y_pred = program.execute(X_test)
        mse = np.mean((y_test - y_pred) ** 2)
        accuracy_score = 1 / (1 + mse)
        
        # Complexity component (lower is better)
        complexity_score = 1 / (1 + program.complexity)
        
        # Interpretability component
        interpretability_score = self.compute_interpretability(program)
        
        # Weighted combination
        total_score = (
            self.weights["accuracy"] * accuracy_score +
            self.weights["complexity"] * complexity_score +
            self.weights["interpretability"] * interpretability_score
        )
        
        return {
            "accuracy": accuracy_score,
            "complexity": complexity_score, 
            "interpretability": interpretability_score,
            "total_score": total_score
        }
    
    def compute_interpretability(self, program):
        """Assess how interpretable an expression is"""
        factors = []
        
        # Penalize deep nesting
        max_depth = program.max_depth
        depth_penalty = 1 / (1 + max_depth / 5)
        factors.append(depth_penalty)
        
        # Reward common functions
        common_functions = {"add", "mul", "sin", "cos", "exp", "log"}
        used_functions = set(token.name for token in program.tokens)
        common_ratio = len(used_functions & common_functions) / len(used_functions)
        factors.append(common_ratio)
        
        # Penalize many variables
        n_variables = len(program.input_variables)
        variable_penalty = 1 / (1 + n_variables / 3)
        factors.append(variable_penalty)
        
        return np.mean(factors)
```

### **Robustness Testing**
Evaluate expression stability across different conditions:

```python
class RobustnessEvaluator:
    def evaluate_noise_robustness(self, program, X_test, y_test, noise_levels=[0.01, 0.05, 0.1]):
        """Test performance under input noise"""
        results = {}
        
        for noise_level in noise_levels:
            # Add noise to inputs
            X_noisy = X_test + np.random.normal(0, noise_level, X_test.shape)
            
            try:
                y_pred = program.execute(X_noisy)
                mse = np.mean((y_test - y_pred) ** 2)
                results[f"mse_noise_{noise_level}"] = mse
            except:
                results[f"mse_noise_{noise_level}"] = float('inf')
        
        return results
    
    def evaluate_extrapolation(self, program, X_train, X_test, y_test):
        """Test extrapolation beyond training range"""
        
        # Define extrapolation regions
        train_mins = np.min(X_train, axis=0)
        train_maxs = np.max(X_train, axis=0)
        
        # Find test points outside training range
        extrapolation_mask = np.any(
            (X_test < train_mins) | (X_test > train_maxs), axis=1
        )
        
        if np.any(extrapolation_mask):
            X_extrap = X_test[extrapolation_mask]
            y_extrap = y_test[extrapolation_mask]
            
            try:
                y_pred_extrap = program.execute(X_extrap)
                extrap_mse = np.mean((y_extrap - y_pred_extrap) ** 2)
                return {"extrapolation_mse": extrap_mse}
            except:
                return {"extrapolation_mse": float('inf')}
        
        return {"extrapolation_mse": None}
```

## 🔄 **Advanced Training Strategies**

### **Curriculum Learning**
Gradually increase task difficulty:

```python
class CurriculumTrainer:
    def __init__(self, base_optimizer):
        self.base_optimizer = base_optimizer
        self.curriculum_stage = 0
        self.stage_thresholds = [0.8, 0.9, 0.95]  # Success rates to advance
    
    def get_current_task(self):
        """Return task appropriate for current curriculum stage"""
        if self.curriculum_stage == 0:
            # Start with simple functions
            return self.create_simple_task()
        elif self.curriculum_stage == 1:
            # Add more complexity
            return self.create_medium_task()
        else:
            # Full complexity
            return self.create_complex_task()
    
    def train_step(self):
        current_task = self.get_current_task()
        
        # Train on current task
        self.base_optimizer.task = current_task
        results = self.base_optimizer.train_step()
        
        # Check if we should advance curriculum
        success_rate = results.get("success_rate", 0)
        if success_rate > self.stage_thresholds[self.curriculum_stage]:
            self.advance_curriculum()
        
        return results
    
    def advance_curriculum(self):
        """Move to next curriculum stage"""
        if self.curriculum_stage < len(self.stage_thresholds) - 1:
            self.curriculum_stage += 1
            print(f"Advanced to curriculum stage {self.curriculum_stage}")
```

### **Ensemble Methods**
Combine multiple DSO runs for better results:

```python
class DSO_Ensemble:
    def __init__(self, n_models=5):
        self.models = []
        self.n_models = n_models
    
    def train_ensemble(self, task):
        """Train multiple independent DSO models"""
        for i in range(self.n_models):
            # Each model gets different random seed
            model = DeepSymbolicOptimizer(
                task=task,
                config_path=f"config_ensemble_{i}.json",
                random_seed=i * 42
            )
            
            model.train()
            self.models.append(model)
    
    def predict_ensemble(self, X):
        """Combine predictions from all models"""
        predictions = []
        
        for model in self.models:
            best_program = model.get_best_program()
            y_pred = best_program.execute(X)
            predictions.append(y_pred)
        
        # Ensemble strategies
        pred_mean = np.mean(predictions, axis=0)
        pred_median = np.median(predictions, axis=0)
        pred_std = np.std(predictions, axis=0)
        
        return {
            "prediction_mean": pred_mean,
            "prediction_median": pred_median,
            "prediction_uncertainty": pred_std
        }
    
    def select_best_model(self, validation_X, validation_y):
        """Select best performing model from ensemble"""
        best_score = float('inf')
        best_model = None
        
        for model in self.models:
            program = model.get_best_program()
            y_pred = program.execute(validation_X)
            mse = np.mean((validation_y - y_pred) ** 2)
            
            if mse < best_score:
                best_score = mse
                best_model = model
        
        return best_model
```

## 🎨 **Visualization and Analysis**

### **Expression Trees**
Visualize discovered expressions as trees:

```python
import matplotlib.pyplot as plt
import networkx as nx

class ExpressionVisualizer:
    def plot_expression_tree(self, program, save_path=None):
        """Create tree visualization of expression"""
        G = nx.DiGraph()
        
        # Build graph from program tokens
        self.build_tree_graph(G, program.tokens)
        
        # Create layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Plot
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, 
               with_labels=True,
               node_color='lightblue',
               node_size=1000,
               font_size=10,
               arrows=True)
        
        plt.title(f"Expression Tree: {program.pretty()}")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_complexity_evolution(self, training_stats):
        """Plot how expression complexity evolves during training"""
        generations = training_stats["generation"]
        complexities = training_stats["best_complexity"]
        rewards = training_stats["best_reward"]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Complexity over time
        ax1.plot(generations, complexities, 'b-', linewidth=2)
        ax1.set_ylabel("Expression Complexity")
        ax1.set_title("Complexity Evolution")
        ax1.grid(True)
        
        # Reward over time
        ax2.plot(generations, rewards, 'r-', linewidth=2)
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Best Reward")
        ax2.set_title("Reward Evolution")
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
```

### **Performance Analysis**
Analyze training dynamics and convergence:

```python
class PerformanceAnalyzer:
    def analyze_convergence(self, training_logs):
        """Analyze convergence characteristics"""
        rewards = training_logs["rewards"]
        
        # Detect convergence
        convergence_window = 100
        convergence_threshold = 0.01
        
        if len(rewards) < convergence_window:
            return {"converged": False}
        
        recent_rewards = rewards[-convergence_window:]
        reward_std = np.std(recent_rewards)
        
        converged = reward_std < convergence_threshold
        
        return {
            "converged": converged,
            "convergence_generation": len(rewards) - convergence_window if converged else None,
            "final_reward_std": reward_std,
            "reward_trend": np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
        }
    
    def identify_plateaus(self, training_logs, plateau_threshold=0.001):
        """Identify training plateaus"""
        rewards = training_logs["rewards"]
        plateaus = []
        
        plateau_start = None
        for i in range(1, len(rewards)):
            improvement = rewards[i] - rewards[i-1]
            
            if abs(improvement) < plateau_threshold:
                if plateau_start is None:
                    plateau_start = i-1
            else:
                if plateau_start is not None:
                    plateaus.append((plateau_start, i-1))
                    plateau_start = None
        
        return plateaus
    
    def analyze_exploration_exploitation(self, training_logs):
        """Analyze exploration vs exploitation balance"""
        entropies = training_logs.get("policy_entropy", [])
        
        if not entropies:
            return {"exploration_metric": None}
        
        # High entropy = more exploration
        avg_entropy = np.mean(entropies)
        entropy_trend = np.polyfit(range(len(entropies)), entropies, 1)[0]
        
        return {
            "average_entropy": avg_entropy,
            "entropy_trend": entropy_trend,  # Negative = decreasing exploration
            "exploration_level": "high" if avg_entropy > 1.0 else "medium" if avg_entropy > 0.5 else "low"
        }
```

This advanced feature set significantly extends DSO's capabilities for specialized applications and research scenarios.
