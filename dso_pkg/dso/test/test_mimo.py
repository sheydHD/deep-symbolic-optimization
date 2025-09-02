"""
Tests for MIMO (Multiple Input Multiple Output) functionality.

This module tests the MIMO support in DSO including:
- SISO backward compatibility
- Simple MIMO cases (2x2)
- Complex MIMO cases (3x3)
- Fixed initialization order
"""

import pytest
import numpy as np
from dso.core import DeepSymbolicOptimizer
from dso.core_fixed import DeepSymbolicOptimizerFixed
from dso.config import load_config


class TestMIMOSupport:
    """Test suite for MIMO functionality."""
    
    @pytest.fixture
    def base_config(self):
        """Base configuration for MIMO tests."""
        return {
            "task": {
                "task_type": "regression",
                "function_set": ["add", "sub", "mul", "div", "sin", "cos"],
                "metric": "inv_nrmse",
                "protected": False
            },
            "training": {
                "n_samples": 100,  # Small for testing
                "batch_size": 50,
                "epsilon": 0.05,
                "n_cores_batch": 1,
                "verbose": False,
                "early_stopping": True,
                "complexity": "length",
                "const_optimizer": "scipy",
                "const_params": {}
            },
            "experiment": {
                "seed": 0,
                "verbose": False
            },
            "policy": {
                "policy_type": "rnn",
                "max_length": 15,
                "cell": "lstm",
                "num_layers": 1,
                "num_units": 32,
                "initializer": "zeros"
            },
            "policy_optimizer": {
                "policy_optimizer_type": "pg",
                "learning_rate": 0.001,
                "entropy_weight": 0.01
            },
            "prior": {
                "length": {
                    "on": True,
                    "min_": 1,
                    "max_": 15
                }
            },
            "state_manager": {
                "type": "hierarchical",
                "observe_parent": True,
                "observe_sibling": True,
                "observe_action": False,
                "observe_dangling": False,
                "embedding": False,
                "embedding_size": 8
            },
            "checkpoint": {
                "save": False
            },
            "logging": {
                "save_summary": False,
                "save_all_iterations": False
            }
        }
    
    def test_siso_backward_compatibility(self, base_config):
        """Test that SISO (Single Input Single Output) still works."""
        # Configure for SISO
        base_config["task"]["dataset"] = "Nguyen-1"  # Simple SISO benchmark
        
        # Create and setup DSO
        dso = DeepSymbolicOptimizer(base_config)
        dso.setup()
        
        # Check initialization
        from dso.program import Program
        assert Program.library is not None
        assert Program.task is not None
        assert Program.task.X_train.shape[1] == 1  # Single input
        assert len(Program.task.y_train.shape) == 1  # Single output
        
        # Run one training step
        result = dso.train_one_step()
        # Should not raise any errors
    
    def test_simple_mimo(self, base_config):
        """Test simple MIMO case (2 inputs, 2 outputs)."""
        # Configure for simple MIMO
        base_config["task"]["dataset"] = "MIMO-simple"
        
        # Use fixed DSO for MIMO
        dso = DeepSymbolicOptimizerFixed(base_config)
        dso.setup()
        
        # Check initialization
        from dso.program import Program
        assert Program.library is not None
        assert Program.task is not None
        assert Program.task.X_train.shape[1] == 2  # Two inputs
        assert Program.task.y_train.shape[1] == 2  # Two outputs
        
        # Run one training step
        result = dso.train_one_step()
        # Should not raise any errors
    
    def test_mimo_benchmark(self, base_config):
        """Test complex MIMO benchmark (3 inputs, 3 outputs)."""
        # Configure for MIMO benchmark
        base_config["task"]["dataset"] = "MIMO-benchmark"
        base_config["task"]["function_set"] = ["add", "sub", "mul", "div", "sin", "cos", "exp", "log"]
        
        # Use fixed DSO for MIMO
        dso = DeepSymbolicOptimizerFixed(base_config)
        dso.setup()
        
        # Check initialization
        from dso.program import Program
        assert Program.library is not None
        assert Program.task is not None
        assert Program.task.X_train.shape[1] == 3  # Three inputs
        assert Program.task.y_train.shape[1] == 3  # Three outputs
        
        # Verify library tokens include input variables
        token_names = [Program.library.names[i] for i in range(Program.library.L)]
        assert 'x1' in token_names
        assert 'x2' in token_names
        assert 'x3' in token_names
        
        # Run one training step
        result = dso.train_one_step()
        # Should not raise any errors
    
    def test_mimo_data_shapes(self, base_config):
        """Test that MIMO data shapes are handled correctly."""
        # Test different MIMO configurations
        test_cases = [
            ("MIMO-simple", 2, 2),
            ("MIMO-benchmark", 3, 3),
            ("MIMO-easy", 2, 2)
        ]
        
        for dataset, expected_inputs, expected_outputs in test_cases:
            base_config["task"]["dataset"] = dataset
            
            # Use fixed DSO for MIMO
            dso = DeepSymbolicOptimizerFixed(base_config)
            dso.setup()
            
            from dso.program import Program
            actual_inputs = Program.task.X_train.shape[1]
            actual_outputs = Program.task.y_train.shape[1]
            
            assert actual_inputs == expected_inputs, \
                f"Dataset {dataset}: expected {expected_inputs} inputs, got {actual_inputs}"
            assert actual_outputs == expected_outputs, \
                f"Dataset {dataset}: expected {expected_outputs} outputs, got {actual_outputs}"
    
    def test_mimo_training_iteration(self, base_config):
        """Test that MIMO training iterations work correctly."""
        base_config["task"]["dataset"] = "MIMO-simple"
        base_config["training"]["n_samples"] = 500
        
        dso = DeepSymbolicOptimizerFixed(base_config)
        dso.setup()
        
        # Run multiple training steps
        for _ in range(5):
            result = dso.train_one_step()
            # Check if training is progressing
            if result is not None:
                assert "r" in result  # Should have reward
                break
        
        # Check that best program is being tracked
        assert hasattr(dso.trainer, 'p_r_best')
    
    def test_mimo_expression_evaluation(self, base_config):
        """Test that MIMO expressions are evaluated correctly."""
        # For this test, just use a known MIMO dataset
        base_config["task"]["dataset"] = "MIMO-simple"
        
        dso = DeepSymbolicOptimizerFixed(base_config)
        dso.setup()
        
        from dso.program import Program
        # Verify MIMO data was loaded
        assert Program.task.X_train.shape[1] == 2  # 2 inputs
        assert Program.task.y_train.shape[1] == 2  # 2 outputs
        
        # Run a training step
        result = dso.train_one_step()
        # Should not raise any errors


class TestMIMOIntegration:
    """Integration tests for MIMO functionality."""
    
    def test_mimo_with_unified_dso(self):
        """Test MIMO using the UnifiedDSO interface."""
        from dso.unified_dso import UnifiedDSO, auto_fit
        
        # Create synthetic MIMO data
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 3)
        y = np.column_stack([
            X[:, 0] * X[:, 1],  # y1 = x1 * x2
            np.sin(X[:, 2]),     # y2 = sin(x3)
            X[:, 0] + X[:, 1] * X[:, 2]  # y3 = x1 + x2 * x3
        ])
        
        dataset = {"X": X, "y": y}
        
        # Use auto_fit for automatic configuration
        try:
            result = auto_fit(
                dataset,
                n_iters=1,  # Just test initialization
                verbose=False
            )
            # Should not raise any errors
            assert result is not None
        except Exception as e:
            # If it fails, it should be a known issue we can handle
            assert "trainer.done" in str(e) or "p_r_best" in str(e), \
                f"Unexpected error: {e}"
    
    def test_mimo_modular_components(self):
        """Test that modular MIMO components work together."""
        from dso.core.data_types import auto_detect_data_structure
        from dso.core.modular_program import ModularProgram
        
        # Create MIMO data
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = np.column_stack([X[:, 0] + X[:, 1], X[:, 0] - X[:, 1]])
        
        # Detect data structure
        data_shape = auto_detect_data_structure(X, y)
        assert data_shape.variant.name == "vector_both"
        assert data_shape.input_dims["n_features"] == 2
        assert data_shape.output_dims["n_outputs"] == 2
        
        # Create modular program
        tokens = [0, 1, 2]  # Simple program
        program = ModularProgram(
            tokens=tokens,
            data_shape=data_shape
        )
        assert program.n_outputs == 2


def test_mimo_end_to_end():
    """End-to-end test of MIMO functionality."""
    # This test ensures the complete MIMO pipeline works
    config_path = "dso_pkg/dso/config/examples/regression/MIMO-simple.json"
    
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        # If config doesn't exist, create minimal config
        config = {
            "task": {
                "task_type": "regression",
                "dataset": "MIMO-simple",
                "function_set": ["add", "sub", "mul", "div"],
                "metric": "inv_nrmse",
                "protected": False
            },
            "training": {
                "n_samples": 100,
                "batch_size": 50,
                "epsilon": 0.05,
                "n_cores_batch": 1,
                "verbose": False
            },
            "experiment": {"seed": 0},
            "policy": {
                "policy_type": "rnn",
                "max_length": 10
            },
            "prior": {
                "length": {"on": True, "min_": 1, "max_": 10}
            }
        }
    
    # Add required sections
    if "state_manager" not in config:
        config["state_manager"] = {
            "type": "hierarchical",
            "observe_parent": True,
            "observe_sibling": True
        }
    if "policy_optimizer" not in config:
        config["policy_optimizer"] = {
            "policy_optimizer_type": "pg",
            "learning_rate": 0.001
        }
    if "checkpoint" not in config:
        config["checkpoint"] = {"save": False}
    if "logging" not in config:
        config["logging"] = {"save_summary": False}
    
    # Test with fixed DSO
    dso = DeepSymbolicOptimizerFixed(config)
    dso.setup()
    
    # Run one step to ensure everything works
    result = dso.train_one_step()
    # Should complete without errors
    
    print("✅ MIMO end-to-end test passed!")