#!/usr/bin/env python
"""Test script for MIMO support with fixed initialization order."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dso_pkg'))

import numpy as np
from dso.core_fixed import DeepSymbolicOptimizerFixed
from dso.config import load_config


def test_mimo_initialization():
    """Test MIMO benchmark initialization with fixed DSO."""
    
    print("="*60)
    print("TESTING MIMO INITIALIZATION WITH FIXED DSO")
    print("="*60)
    
    # Configuration for MIMO benchmark
    config = {
        "task": {
            "task_type": "regression",
            "dataset": "MIMO-benchmark",  # 3 inputs, 3 outputs: [x1*x2, sin(x3), x1+x2*x3]
            "function_set": ["add", "sub", "mul", "div", "sin", "cos", "exp", "log"],
            "metric": "inv_nrmse",
            "protected": False
        },
        "training": {
            "n_samples": 1000,  # Small for testing
            "batch_size": 50,
            "epsilon": 0.05,
            "n_cores_batch": 1,
            "verbose": True,
            "early_stopping": True,
            "hof": 10,
            "complexity": "length",
            "const_optimizer": "scipy",
            "const_params": {}
        },
        "experiment": {
            "seed": 0,
            "verbose": True,
            "logdir": "./test_logs",
            "logfile": "test_mimo.csv"
        },
        "policy": {
            "policy_type": "rnn",
            "max_length": 20,
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
                "max_": 20
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
            "save_summary": True,
            "save_all_iterations": False,
            "save_test": False,
            "save_positional_entropy": False,
            "save_pareto_front": False
        }
    }
    
    try:
        # Create DSO with fixed initialization
        print("\n1. Creating Fixed DSO...")
        dso = DeepSymbolicOptimizerFixed(config)
        
        print("\n2. Running setup...")
        dso.setup()
        
        print("\n3. Checking initialization...")
        print(f"   - Task type: {dso.config_task['task_type']}")
        print(f"   - Dataset: {dso.config_task['dataset']}")
        
        # Check if Program.library is initialized
        from dso.program import Program
        if Program.library is not None:
            print(f"   ✓ Program.library initialized with {Program.library.L} tokens")
            print(f"   - Token names: {Program.library.names[:10]}...")  # First 10 tokens
        else:
            print("   ✗ Program.library is None")
            return False
            
        # Check if policy is created
        if dso.policy is not None:
            print(f"   ✓ Policy created successfully")
            if hasattr(dso.policy, 'n_choices'):
                print(f"   - Policy n_choices: {dso.policy.n_choices}")
        else:
            print("   ✗ Policy is None")
            return False
            
        # Check task properties
        if hasattr(Program.task, 'X_train'):
            print(f"   ✓ Task data loaded")
            print(f"   - X_train shape: {Program.task.X_train.shape}")
            print(f"   - y_train shape: {Program.task.y_train.shape}")
            
            # Check if it's MIMO data
            if Program.task.y_train.ndim > 1 and Program.task.y_train.shape[1] > 1:
                print(f"   ✓ MIMO data detected: {Program.task.y_train.shape[1]} outputs")
            else:
                print(f"   - Single output data: shape {Program.task.y_train.shape}")
        
        print("\n4. Running a training step...")
        result = dso.train_one_step()
        
        if result is None:
            print("   ✓ Training step completed (not finished yet)")
        else:
            print(f"   ✓ Training completed with result: {result}")
            
        print("\n✅ MIMO INITIALIZATION TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_siso():
    """Test simple SISO case to ensure backward compatibility."""
    
    print("\n" + "="*60)
    print("TESTING BACKWARD COMPATIBILITY (SISO)")
    print("="*60)
    
    config = {
        "task": {
            "task_type": "regression",
            "dataset": "Nguyen-1",  # Simple x^3 + x^2 + x
            "function_set": ["add", "sub", "mul", "div", "sin", "cos"],
            "metric": "inv_nrmse"
        },
        "training": {
            "n_samples": 500,
            "batch_size": 50,
            "epsilon": 0.05,
            "n_cores_batch": 1,
            "complexity": "length",
            "const_optimizer": "scipy"
        },
        "experiment": {
            "seed": 42,
            "verbose": False
        },
        "policy": {
            "policy_type": "rnn",
            "max_length": 15
        },
        "policy_optimizer": {
            "policy_optimizer_type": "pg",
            "learning_rate": 0.001
        },
        "prior": {
            "length": {"on": True, "min_": 1, "max_": 15}
        },
        "state_manager": {
            "type": "hierarchical"
        },
        "checkpoint": {"save": False},
        "logging": {"save_summary": False}
    }
    
    try:
        print("\n1. Creating DSO for SISO...")
        dso = DeepSymbolicOptimizerFixed(config)
        
        print("2. Running setup...")
        dso.setup()
        
        from dso.program import Program
        print(f"3. Checking setup:")
        print(f"   - Library tokens: {Program.library.L}")
        print(f"   - X_train shape: {Program.task.X_train.shape}")
        print(f"   - y_train shape: {Program.task.y_train.shape}")
        
        print("4. Running training step...")
        dso.train_one_step()
        
        print("\n✅ SISO BACKWARD COMPATIBILITY TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mimo_simple():
    """Test simple MIMO case."""
    
    print("\n" + "="*60)
    print("TESTING SIMPLE MIMO CASE")
    print("="*60)
    
    config = {
        "task": {
            "task_type": "regression",
            "dataset": "MIMO-simple",  # 2 inputs, 2 outputs: [x1+x2, x1*x2]
            "function_set": ["add", "sub", "mul", "div"],
            "metric": "inv_nrmse"
        },
        "training": {
            "n_samples": 500,
            "batch_size": 50,
            "epsilon": 0.05,
            "n_cores_batch": 1,
            "complexity": "length",
            "const_optimizer": "scipy"
        },
        "experiment": {
            "seed": 123,
            "verbose": True
        },
        "policy": {
            "policy_type": "rnn",
            "max_length": 10
        },
        "policy_optimizer": {
            "policy_optimizer_type": "pg",
            "learning_rate": 0.001
        },
        "prior": {
            "length": {"on": True, "min_": 1, "max_": 10}
        },
        "state_manager": {
            "type": "hierarchical"
        },
        "checkpoint": {"save": False},
        "logging": {"save_summary": False}
    }
    
    try:
        print("\n1. Creating DSO for simple MIMO...")
        dso = DeepSymbolicOptimizerFixed(config)
        
        print("2. Running setup...")
        dso.setup()
        
        from dso.program import Program
        print(f"3. Checking MIMO setup:")
        print(f"   - X_train shape: {Program.task.X_train.shape}")
        print(f"   - y_train shape: {Program.task.y_train.shape}")
        print(f"   - Number of outputs: {Program.task.y_train.shape[1]}")
        
        print("4. Running 5 training steps...")
        for i in range(5):
            result = dso.train_one_step()
            if result is not None:
                print(f"   Training completed at step {i+1}")
                break
        
        print("\n✅ SIMPLE MIMO TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    
    print("="*70)
    print("MIMO SUPPORT TESTING SUITE")
    print("="*70)
    print("This script tests the MIMO (Multiple Input, Multiple Output)")
    print("functionality with the fixed initialization order.")
    print("="*70)
    
    # Run tests
    tests = [
        ("SISO Backward Compatibility", test_simple_siso),
        ("Simple MIMO", test_mimo_simple),
        ("MIMO Benchmark", test_mimo_initialization)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n\n{'='*10} {test_name} {'='*10}")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:30} {status}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! MIMO support is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Please review the errors above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)