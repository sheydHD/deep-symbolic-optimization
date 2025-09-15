"""
Modular regression task that automatically adapts to different data variants.

This module extends the original regression task to automatically detect and handle:
- Scalar (SISO): Single input, single output
- Vector Input (MISO): Multiple inputs, single output  
- Vector Output (SIMO): Single input, multiple outputs
- Vector Both (MIMO): Multiple inputs, multiple outputs
- Matrix/Tensor: Higher-dimensional data structures

The task automatically configures itself based on the input data structure.
"""

import numpy as np
import os
from typing import Union, Tuple, Optional, Dict, Any

from dso.task.regression.regression import RegressionTask as BaseRegressionTask
from dso.core.data_types import auto_detect_data_structure, DataShape, DataHandler, DataVariant
from dso.library import Library
from dso.functions import create_tokens
from dso.task.regression.dataset import BenchmarkDataset


class ModularRegressionTask(BaseRegressionTask):
    """
    Modular regression task with automatic data variant detection.
    
    This class extends the base regression task to automatically detect the data
    structure (scalar, vector, tensor) and configure appropriate execution paths.
    """
    
    def __init__(self, dataset=None, function_set=None, metric="inv_nrmse", 
                 metric_params=(1.0,), extra_metric_test=None, extra_metric_test_params=(),
                 reward_noise=0.0, reward_noise_type="r", threshold=1e-12,
                 normalize_variance=False, protected=False, 
                 decision_tree_threshold_set=None, poly_optimizer_params=None, **kwargs):
        """
        Initialize modular regression task with automatic data variant detection.
        
        Parameters are the same as base RegressionTask, but the task will automatically
        configure itself based on the detected data structure.
        """
        
        # Step 1: Detect data structure and create handler
        self.data_shape, self.data_handler = auto_detect_data_structure(dataset, verbose=True)
        
        # Step 2: Initialize base class
        super().__init__(
            dataset=dataset,
            function_set=function_set,
            metric=metric,
            metric_params=metric_params,
            extra_metric_test=extra_metric_test,
            extra_metric_test_params=extra_metric_test_params,
            reward_noise=reward_noise,
            reward_noise_type=reward_noise_type,
            threshold=threshold,
            normalize_variance=normalize_variance,
            protected=protected,
            decision_tree_threshold_set=decision_tree_threshold_set,
            poly_optimizer_params=poly_optimizer_params
        )
        
        # Override variance calculations for multi-output cases
        self._setup_variance_calculations()
        
    def _load_and_prepare_data(self, dataset):
        """Load and prepare data from various sources."""
        
        if isinstance(dataset, str):
            # Load from benchmark dataset
            if os.path.exists(dataset):
                # Load from CSV file
                self._load_from_csv(dataset)
            else:
                # Load from benchmark name
                benchmark_dataset = BenchmarkDataset(dataset)
                self.X_train = benchmark_dataset.X_train
                self.y_train = benchmark_dataset.y_train
                self.X_test = benchmark_dataset.X_test
                self.y_test = benchmark_dataset.y_test
                self.name = dataset
                
        elif isinstance(dataset, tuple):
            # Sklearn-like (X, y) or (X_train, y_train, X_test, y_test)
            if len(dataset) == 2:
                self.X_train, self.y_train = dataset
                self.X_test, self.y_test = dataset  # Use same data for test
            elif len(dataset) == 4:
                self.X_train, self.y_train, self.X_test, self.y_test = dataset
            else:
                raise ValueError("Dataset tuple must have 2 or 4 elements")
            self.name = "custom_dataset"
            
        elif hasattr(dataset, 'X_train'):
            # Dataset object with attributes
            self.X_train = dataset.X_train
            self.y_train = dataset.y_train
            self.X_test = getattr(dataset, 'X_test', dataset.X_train)
            self.y_test = getattr(dataset, 'y_test', dataset.y_train)
            self.name = getattr(dataset, 'name', 'dataset_object')
            
        else:
            raise ValueError(f"Unsupported dataset type: {type(dataset)}")
            
        # Ensure numpy arrays
        self.X_train = np.asarray(self.X_train)
        self.y_train = np.asarray(self.y_train)
        self.X_test = np.asarray(self.X_test)
        self.y_test = np.asarray(self.y_test)
        
        # Store noiseless versions
        self.y_train_noiseless = self.y_train.copy()
        self.y_test_noiseless = self.y_test.copy()
        
    def _load_from_csv(self, filepath):
        """Load data from CSV file with automatic input/output detection."""
        import pandas as pd
        
        data = pd.read_csv(filepath)
        
        # Auto-detect input/output columns
        # Convention: columns starting with 'x' or 'X' are inputs, 'y' or 'Y' are outputs
        input_cols = [col for col in data.columns if col.lower().startswith('x')]
        output_cols = [col for col in data.columns if col.lower().startswith('y')]
        
        if not input_cols or not output_cols:
            # Fallback: assume all but last columns are inputs, last is output
            input_cols = data.columns[:-1].tolist()
            output_cols = [data.columns[-1]]
            
        X = data[input_cols].values
        y = data[output_cols].values
        
        # Squeeze y if single output
        if y.shape[1] == 1:
            y = y.squeeze()
            
        # Split into train/test (80/20 split)
        n_train = int(0.8 * len(X))
        self.X_train = X[:n_train]
        self.y_train = y[:n_train]
        self.X_test = X[n_train:]
        self.y_test = y[n_train:]
        self.name = os.path.basename(filepath).split('.')[0]
        
    def _configure_for_variant(self):
        """Configure task parameters based on detected data variant."""
        
        # Prepare data using appropriate handler
        self.X_train, self.y_train = self.data_handler.prepare_data(self.X_train, self.y_train)
        self.X_test, self.y_test = self.data_handler.prepare_data(self.X_test, self.y_test)
        
        # Set variant-specific attributes
        self.variant = self.data_shape.variant
        self.n_input_var = self.data_shape.input_dims['n_features']
        self.n_output_var = self.data_shape.output_dims['n_outputs']
        
        # Compatibility attributes
        self.is_scalar = self.data_shape.is_scalar
        self.is_vector = self.data_shape.is_vector
        self.is_tensor = self.data_shape.is_tensor
        self.is_mimo = self.data_shape.is_mimo
        
        print(f"Task configured for {self.variant.value} with {self.n_input_var} inputs, {self.n_output_var} outputs")
        
    def _setup_variance_calculations(self):
        """Setup variance calculations for different output dimensions."""
        
        if self.n_output_var == 1:
            # Single output - use existing calculations
            self.var_y_test = np.var(self.y_test)
            self.var_y_test_noiseless = np.var(self.y_test_noiseless)
        else:
            # Multiple outputs - compute per-output variances
            if self.y_test.ndim == 1:
                # This shouldn't happen, but handle gracefully
                self.var_y_test = np.var(self.y_test)
                self.var_y_test_noiseless = np.var(self.y_test_noiseless)
            else:
                self.var_y_test = np.var(self.y_test, axis=0)
                self.var_y_test_noiseless = np.var(self.y_test_noiseless, axis=0)
                
    def reward_function(self, p):
        """
        Compute reward using the appropriate data handler.
        
        This method delegates to the data handler to compute variant-specific rewards.
        """
        
        # Execute program
        try:
            if self.n_output_var == 1:
                # Single output case
                y_pred = p.execute(self.X_train)
                if hasattr(y_pred, 'flatten'):
                    y_pred = y_pred.flatten()
            else:
                # Multi-output case
                y_pred = self.data_handler.execute_program(p, self.X_train)
                
            # Handle invalid predictions
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                return self.invalid_reward
                
            # Apply reward noise if specified
            if self.reward_noise > 0:
                y_pred = self._add_reward_noise(y_pred)
                
            # Compute reward using data handler
            reward = self.data_handler.compute_reward(self.y_train, y_pred)
            
            # Apply threshold check
            if reward > (1.0 - self.threshold):
                reward = self.max_reward
                
            return reward
            
        except Exception as e:
            # Return minimum reward for any execution errors
            return self.invalid_reward
            
    def evaluate(self, p, eval_test=True):
        """
        Evaluate program on train and optionally test data.
        
        Returns variant-appropriate evaluation metrics.
        """
        
        results = {}
        
        # Training evaluation
        try:
            if self.n_output_var == 1:
                y_pred_train = p.execute(self.X_train).flatten()
            else:
                y_pred_train = self.data_handler.execute_program(p, self.X_train)
                
            results['nmse_train'] = self._compute_nmse(self.y_train, y_pred_train)
            results['r_train'] = self.data_handler.compute_reward(self.y_train, y_pred_train)
            
        except Exception:
            results['nmse_train'] = np.inf
            results['r_train'] = self.invalid_reward
            
        # Test evaluation
        if eval_test:
            try:
                if self.n_output_var == 1:
                    y_pred_test = p.execute(self.X_test).flatten()
                else:
                    y_pred_test = self.data_handler.execute_program(p, self.X_test)
                    
                results['nmse_test'] = self._compute_nmse(self.y_test, y_pred_test)
                results['r_test'] = self.data_handler.compute_reward(self.y_test, y_pred_test)
                
                # Noiseless test evaluation
                results['nmse_test_noiseless'] = self._compute_nmse(self.y_test_noiseless, y_pred_test)
                
            except Exception:
                results['nmse_test'] = np.inf
                results['r_test'] = self.invalid_reward
                results['nmse_test_noiseless'] = np.inf
                
        return results
        
    def _compute_nmse(self, y_true, y_pred):
        """Compute NMSE appropriate for the data variant."""
        
        if self.n_output_var == 1:
            # Single output NMSE
            mse = np.mean((y_true - y_pred) ** 2)
            var = np.var(y_true)
            return mse / (var + 1e-12)
        else:
            # Multi-output NMSE - average across outputs
            if y_true.ndim == 1 or y_pred.ndim == 1:
                # Handle edge case
                mse = np.mean((y_true.flatten() - y_pred.flatten()) ** 2)
                var = np.var(y_true.flatten())
                return mse / (var + 1e-12)
            else:
                mse_per_output = np.mean((y_true - y_pred) ** 2, axis=0)
                var_per_output = np.var(y_true, axis=0)
                nmse_per_output = mse_per_output / (var_per_output + 1e-12)
                return np.mean(nmse_per_output)
                
    def _add_reward_noise(self, y_pred):
        """Add reward noise appropriate for the data variant."""
        
        if self.reward_noise_type == "y_hat":
            if self.n_output_var == 1:
                y_rms = np.sqrt(np.mean(self.y_train ** 2))
            else:
                y_rms = np.sqrt(np.mean(self.y_train ** 2, axis=0))
            scale = self.reward_noise * y_rms
        else:  # "r" type
            scale = self.reward_noise
            
        noise_shape = y_pred.shape
        noise = self.rng.normal(loc=0, scale=scale, size=noise_shape)
        return y_pred + noise
        
    def get_variant_info(self) -> Dict[str, Any]:
        """Get information about the detected data variant."""
        
        return {
            'variant': self.variant.value,
            'data_shape': self.data_shape.summary(),
            'n_input_var': self.n_input_var,
            'n_output_var': self.n_output_var,
            'input_dims': self.data_shape.input_dims,
            'output_dims': self.data_shape.output_dims,
            'is_scalar': self.is_scalar,
            'is_vector': self.is_vector,
            'is_tensor': self.is_tensor,
            'is_mimo': self.is_mimo,
            'handler_class': self.data_handler.__class__.__name__
        }
        
    def print_variant_summary(self):
        """Print a summary of the detected data variant."""
        
        info = self.get_variant_info()
        print("\n" + "="*60)
        print("MODULAR REGRESSION TASK SUMMARY")
        print("="*60)
        print(f"Dataset: {self.name}")
        print(f"Variant: {info['variant']}")
        print(f"Handler: {info['handler_class']}")
        print(f"Input variables: {info['n_input_var']}")
        print(f"Output variables: {info['n_output_var']}")
        print(f"Training shape: X={self.X_train.shape}, y={self.y_train.shape}")
        print(f"Test shape: X={self.X_test.shape}, y={self.y_test.shape}")
        print("="*60)


# Factory function for easy task creation
def create_modular_regression_task(dataset, **kwargs) -> ModularRegressionTask:
    """
    Factory function to create a modular regression task.
    
    Parameters
    ----------
    dataset : str, tuple, or object
        Dataset specification
    **kwargs
        Additional parameters for task configuration
        
    Returns
    -------
    ModularRegressionTask
        Configured regression task
    """
    
    task = ModularRegressionTask(dataset=dataset, **kwargs)
    task.print_variant_summary()
    return task


# Convenience functions for specific variants
def create_scalar_task(dataset, **kwargs):
    """Create a task optimized for scalar (SISO) problems."""
    task = create_modular_regression_task(dataset, **kwargs)
    if not task.is_scalar:
        print(f"Warning: Dataset is {task.variant.value}, not scalar")
    return task


def create_mimo_task(dataset, **kwargs):
    """Create a task optimized for MIMO problems."""
    task = create_modular_regression_task(dataset, **kwargs)
    if not task.is_mimo:
        print(f"Warning: Dataset is {task.variant.value}, not MIMO")
    return task


def create_tensor_task(dataset, **kwargs):
    """Create a task optimized for tensor problems.""" 
    task = create_modular_regression_task(dataset, **kwargs)
    if not task.is_tensor:
        print(f"Warning: Dataset is {task.variant.value}, not tensor")
    return task
