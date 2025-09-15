"""
Unified Deep Symbolic Optimization interface with automatic variant detection.

This module provides a single entry point for DSO that automatically detects
the data structure and configures the appropriate components for scalar, vector,
matrix, and tensor symbolic optimization.
"""

import numpy as np
import os
from typing import Union, Optional, Dict, Any, Tuple
import warnings

from dso.core.data_types import auto_detect_data_structure, DataShape, DataVariant
from dso.task.regression.modular_regression import ModularRegressionTask
from dso.core.modular_policy import create_modular_policy
from dso.core.modular_program import create_program_for_variant
from dso.library import Library
from dso.functions import create_tokens
from dso.train import Trainer
from dso.policy_optimizer import PolicyOptimizer
from dso.gp import gp_controller
from dso.config import load_config
from dso.train_stats import StatsLogger


class UnifiedDSO:
    """
    Unified Deep Symbolic Optimization interface.
    
    This class provides a single entry point for symbolic optimization that
    automatically detects the data structure and configures all components
    appropriately for different variants (scalar, vector, tensor).
    """
    
    def __init__(self, config_template: Optional[str] = None, verbose: bool = True):
        """
        Initialize unified DSO interface.
        
        Parameters
        ----------
        config_template : str, optional
            Path to configuration template or None for default
        verbose : bool, optional
            Whether to print detailed information
        """
        
        self.config_template = config_template
        self.verbose = verbose
        self.data_shape = None
        self.data_handler = None
        self.task = None
        self.policy = None
        self.trainer = None
        self.library = None
        self.config = None
        
        # Load base configuration
        self.config = load_config(config_template)
        
        if verbose:
            print("="*60)
            print("UNIFIED DEEP SYMBOLIC OPTIMIZATION")
            print("="*60)
            print("Automatic variant detection and configuration enabled")
            print("="*60)
            
    def fit(self, dataset, **kwargs) -> Dict[str, Any]:
        """
        Fit DSO to the provided dataset with automatic configuration.
        
        Parameters
        ----------
        dataset : various
            Dataset in various formats:
            - String: benchmark name or CSV file path
            - Tuple: (X, y) or (X_train, y_train, X_test, y_test)
            - Object: with X_train, y_train attributes
        **kwargs
            Additional configuration parameters that override config
            
        Returns
        -------
        Dict[str, Any]
            Training results and best expressions
        """
        
        # Update config with kwargs
        self._update_config(kwargs)
        
        # Step 1: Create modular task (handles data loading and variant detection)
        if self.verbose:
            print("\n1. LOADING AND ANALYZING DATA")
            print("-" * 40)
            
        # Prepare task config (avoid parameter conflicts)
        task_config = self.config.get('task', {}).copy()
        # Remove parameters that we're passing explicitly to avoid conflicts
        for param in ['dataset', 'function_set', 'metric']:
            task_config.pop(param, None)
            
        self.task = ModularRegressionTask(
            dataset=dataset,
            function_set=self.config.get('task', {}).get('function_set', 
                                        ["add", "sub", "mul", "div", "sin", "cos", "exp", "log"]),
            metric=self.config.get('task', {}).get('metric', 'inv_nrmse'),
            **task_config
        )
        
        # Extract data shape information
        self.data_shape = self.task.data_shape
        self.data_handler = self.task.data_handler
        
        # Set the task for the Program class (required for global access)
        from dso.program import Program
        # First set the execute method (protected or not)
        protected = self.config.get('task', {}).get('protected', False)
        Program.set_execute(protected)
        # Then set the task
        Program.set_task(self.task)
        
        # Step 2: Create modular library
        if self.verbose:
            print("\n2. CONFIGURING FUNCTION LIBRARY")
            print("-" * 40)
            
        self._create_library()
        
        # Step 3: Create modular policy
        if self.verbose:
            print("\n3. CONFIGURING POLICY")
            print("-" * 40)
            
        self._create_policy()
        
        # Step 4: Create trainer and other components
        if self.verbose:
            print("\n4. CONFIGURING TRAINING")
            print("-" * 40)
            
        self._create_trainer()
        
        # Step 6: Run training
        if self.verbose:
            print("\n5. STARTING TRAINING")
            print("-" * 40)
            
        results = self._run_training()
        
        # Step 7: Post-process results
        if self.verbose:
            print("\n6. POST-PROCESSING RESULTS")
            print("-" * 40)
            
        processed_results = self._process_results(results)
        
        if self.verbose:
            print("\n" + "="*60)
            print("TRAINING COMPLETED")
            print("="*60)
            self._print_summary(processed_results)
            
        return processed_results
        
    def predict(self, X: np.ndarray, program=None) -> np.ndarray:
        """
        Make predictions using the best found program.
        
        Parameters
        ----------
        X : np.ndarray
            Input data for prediction
        program : Program, optional
            Specific program to use, or None for best program
            
        Returns
        -------
        np.ndarray
            Predictions
        """
        
        if program is None:
            if not hasattr(self, 'best_program') or self.best_program is None:
                raise ValueError("No program available. Run fit() first.")
            program = self.best_program
            
        # Use data handler for consistent execution
        if self.data_handler is not None:
            return self.data_handler.execute_program(program, X)
        else:
            return program.execute(X)
            
    def evaluate(self, X: np.ndarray, y: np.ndarray, program=None) -> Dict[str, float]:
        """
        Evaluate program performance on given data.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        y : np.ndarray
            True output data
        program : Program, optional
            Program to evaluate, or None for best program
            
        Returns
        -------
        Dict[str, float]
            Evaluation metrics
        """
        
        y_pred = self.predict(X, program)
        
        # Compute metrics using data handler
        if self.data_handler is not None:
            reward = self.data_handler.compute_reward(y, y_pred)
        else:
            # Fallback computation
            mse = np.mean((y - y_pred) ** 2)
            var = np.var(y)
            reward = 1.0 / (1.0 + mse / (var + 1e-12))
            
        # Compute NMSE
        if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
            nmse = np.mean((y.flatten() - y_pred.flatten()) ** 2) / (np.var(y.flatten()) + 1e-12)
        else:
            # Multi-output NMSE
            mse_per_output = np.mean((y - y_pred) ** 2, axis=0)
            var_per_output = np.var(y, axis=0)
            nmse_per_output = mse_per_output / (var_per_output + 1e-12)
            nmse = np.mean(nmse_per_output)
            
        return {
            'reward': reward,
            'nmse': nmse,
            'mse': np.mean((y.flatten() - y_pred.flatten()) ** 2),
            'r2': 1 - nmse
        }
        
    def get_variant_info(self) -> Dict[str, Any]:
        """Get information about detected data variant."""
        
        if self.data_shape is None:
            return {'status': 'No data loaded'}
            
        info = {
            'variant': self.data_shape.variant.value,
            'n_inputs': self.data_shape.n_inputs,
            'n_outputs': self.data_shape.n_outputs,
            'input_shape': self.data_shape.X_shape,
            'output_shape': self.data_shape.y_shape,
            'is_scalar': self.data_shape.is_scalar,
            'is_vector': self.data_shape.is_vector,
            'is_tensor': self.data_shape.is_tensor,
            'is_mimo': self.data_shape.is_mimo
        }
        
        if self.task is not None:
            info.update(self.task.get_variant_info())
            
        return info
        
    def _update_config(self, kwargs):
        """Update configuration with provided kwargs."""
        
        for key, value in kwargs.items():
            if '.' in key:
                # Handle nested keys like 'task.metric'
                parts = key.split('.')
                config_section = self.config
                for part in parts[:-1]:
                    if part not in config_section:
                        config_section[part] = {}
                    config_section = config_section[part]
                config_section[parts[-1]] = value
            else:
                # Handle top-level keys
                if key not in self.config:
                    self.config[key] = {}
                if isinstance(value, dict):
                    self.config[key].update(value)
                else:
                    self.config[key] = value
                    
    def _create_library(self):
        """Create function library appropriate for the data variant."""
        
        function_set = self.config.get('task', {}).get('function_set', 
                                                     ["add", "sub", "mul", "div", "sin", "cos", "exp", "log"])
        
        # Adjust function set based on variant
        if self.data_shape.variant in [DataVariant.MATRIX, DataVariant.TENSOR_3D, DataVariant.TENSOR_ND]:
            # Add tensor-specific functions if needed
            tensor_functions = ["matmul", "transpose", "reshape"]
            function_set = list(set(function_set + tensor_functions))
            
        tokens = create_tokens(
            n_input_var=self.data_shape.n_inputs,
            function_set=function_set,
            protected=self.config.get('task', {}).get('protected', False)
        )
        
        self.library = Library(tokens)
        
        if self.verbose:
            print(f"Library created with {len(self.library.tokens)} tokens")
            print(f"Function set: {function_set}")
            
    def _create_policy(self):
        """Create modular policy appropriate for the data variant."""
        
        policy_config = self.config.get('policy', {})
        strategy = policy_config.get('multi_output_strategy', 'auto')
        
        # Filter out parameters that RNNPolicy doesn't accept
        valid_policy_params = {
            'debug', 'max_length', 'action_prob_lowerbound', 'max_attempts_at_novel_batch',
            'sample_novel_batch', 'cell', 'num_layers', 'num_units', 'initializer'
        }
        filtered_config = {k: v for k, v in policy_config.items() if k in valid_policy_params}
        
        self.policy = create_modular_policy(
            library=self.library,
            data_shape=self.data_shape,
            strategy=strategy,
            **filtered_config
        )
        
    def _create_trainer(self):
        """Create trainer and associated components."""
        
        # Create policy optimizer
        optimizer_config = self.config.get('policy_optimizer', {})
        from dso.policy_optimizer import make_policy_optimizer
        
        # Extract policy_optimizer_type from config
        policy_optimizer_type = optimizer_config.pop("policy_optimizer_type", "pg")
        self.policy_optimizer = make_policy_optimizer(
            self.policy,
            policy_optimizer_type,
            **optimizer_config
        )
        
        # Create GP controller if specified
        gp_config = self.config.get('gp_meld', {})
        if gp_config.get('run_gp_meld', False):
            self.gp_controller = gp_controller.GPController(
                **gp_config
            )
        else:
            self.gp_controller = None
            
        # Create logger
        logging_config = self.config.get('logging', {})
        output_file = logging_config.pop('output_file', 'dso_log.txt')
        self.logger = StatsLogger(output_file, **logging_config)
        
        # Create trainer
        training_config = self.config.get('training', {})
        # Remove logdir as it's not a Trainer parameter
        training_config.pop('logdir', None)
        self.trainer = Trainer(
            policy=self.policy,
            policy_optimizer=self.policy_optimizer,
            gp_controller=self.gp_controller,
            logger=self.logger,
            pool=None,  # Will be created if needed
            **training_config
        )
        
    def _run_training(self):
        """Run the training loop."""
        
        # Set the task for evaluation
        self.trainer.task = self.task
        
        # Run training loop
        n_iters = self.config.get('training', {}).get('n_iters', 100)
        
        for iteration in range(n_iters):
            # Run one training step
            self.trainer.run_one_step()
            
            # Check if training is done
            if self.trainer.done:
                break
                
            # Print progress periodically
            if iteration % 10 == 0 and hasattr(self.trainer, 'p_r_best') and self.trainer.p_r_best is not None:
                print(f"[Iteration {iteration}] Best reward: {self.trainer.p_r_best.r:.4f}")
        
        # Return the best program found
        if hasattr(self.trainer, 'p_r_best') and self.trainer.p_r_best is not None:
            p = self.trainer.p_r_best
            result = {
                "reward": p.r,
                "expression": repr(p.sympy_expr) if hasattr(p, 'sympy_expr') else str(p),
                "program": p,
                "traversal": repr(p)
            }
            return result
        else:
            return {'reward': 0, 'program': None, 'expression': None}
        
    def _process_results(self, results):
        """Process and enhance training results."""
        
        processed = results.copy()
        
        # Extract best program
        if 'program' in results:
            self.best_program = results['program']
        elif hasattr(self.trainer, 'hall_of_fame') and len(self.trainer.hall_of_fame) > 0:
            self.best_program = self.trainer.hall_of_fame[0]['program']
        else:
            self.best_program = None
            
        # Add variant information
        processed['variant_info'] = self.get_variant_info()
        
        # Add evaluation on test data
        if self.best_program is not None and hasattr(self.task, 'X_test'):
            test_eval = self.evaluate(self.task.X_test, self.task.y_test, self.best_program)
            processed['test_evaluation'] = test_eval
            
        return processed
        
    def _print_summary(self, results):
        """Print training summary."""
        
        print(f"Variant: {self.data_shape.variant.value}")
        print(f"Best reward: {results.get('r', 'N/A')}")
        
        if 'test_evaluation' in results:
            test_eval = results['test_evaluation']
            print(f"Test NMSE: {test_eval['nmse']:.6f}")
            print(f"Test R²: {test_eval['r2']:.6f}")
            
        if self.best_program is not None:
            print(f"Best expression: {self.best_program.str}")
            
        print(f"Training time: {results.get('t', 'N/A')} seconds")


# Convenience functions for different use cases
def fit_scalar(dataset, **kwargs) -> Dict[str, Any]:
    """Convenience function for scalar (SISO) symbolic regression."""
    dso = UnifiedDSO(**kwargs)
    return dso.fit(dataset, **kwargs)


def fit_vector(dataset, **kwargs) -> Dict[str, Any]:
    """Convenience function for vector (MISO) symbolic regression."""
    dso = UnifiedDSO(**kwargs)
    return dso.fit(dataset, **kwargs)


def fit_mimo(dataset, strategy: str = "independent", **kwargs) -> Dict[str, Any]:
    """Convenience function for MIMO symbolic regression."""
    kwargs['policy.multi_output_strategy'] = strategy
    dso = UnifiedDSO(**kwargs)
    return dso.fit(dataset, **kwargs)


def fit_tensor(dataset, **kwargs) -> Dict[str, Any]:
    """Convenience function for tensor symbolic regression."""
    kwargs['policy.multi_output_strategy'] = 'independent'
    dso = UnifiedDSO(**kwargs)
    return dso.fit(dataset, **kwargs)


def auto_fit(dataset, **kwargs) -> Dict[str, Any]:
    """
    Automatic fitting with complete variant detection and configuration.
    
    This is the most convenient function that handles everything automatically.
    """
    dso = UnifiedDSO(verbose=kwargs.get('verbose', True))
    results = dso.fit(dataset, **kwargs)
    
    return {
        'results': results,
        'dso': dso,  # Return DSO instance for further use
        'best_program': dso.best_program,
        'variant_info': dso.get_variant_info()
    }


# Example usage and testing
def create_test_datasets():
    """Create test datasets for different variants."""
    
    np.random.seed(42)
    
    datasets = {}
    
    # Scalar (SISO)
    X_scalar = np.random.randn(100, 1)
    y_scalar = X_scalar.flatten() ** 2 + 0.1 * np.random.randn(100)
    datasets['scalar'] = (X_scalar, y_scalar)
    
    # Vector input (MISO)
    X_miso = np.random.randn(100, 3)
    y_miso = X_miso[:, 0] * X_miso[:, 1] + np.sin(X_miso[:, 2]) + 0.1 * np.random.randn(100)
    datasets['miso'] = (X_miso, y_miso)
    
    # Vector output (SIMO)
    X_simo = np.random.randn(100, 1)
    y_simo = np.column_stack([
        X_simo.flatten() ** 2,
        np.sin(X_simo.flatten()),
        np.cos(X_simo.flatten())
    ]) + 0.1 * np.random.randn(100, 3)
    datasets['simo'] = (X_simo, y_simo)
    
    # MIMO
    X_mimo = np.random.randn(100, 3)
    y_mimo = np.column_stack([
        X_mimo[:, 0] * X_mimo[:, 1],
        np.sin(X_mimo[:, 2]),
        X_mimo[:, 0] + X_mimo[:, 1] * X_mimo[:, 2]
    ]) + 0.1 * np.random.randn(100, 3)
    datasets['mimo'] = (X_mimo, y_mimo)
    
    return datasets


if __name__ == "__main__":
    # Test the unified interface
    datasets = create_test_datasets()
    
    for name, dataset in datasets.items():
        print(f"\n{'='*60}")
        print(f"TESTING {name.upper()} DATASET")
        print('='*60)
        
        try:
            result = auto_fit(dataset, training={'n_samples': 1000, 'batch_size': 100})
            print(f"✓ {name} test completed successfully")
        except Exception as e:
            print(f"✗ {name} test failed: {e}")
