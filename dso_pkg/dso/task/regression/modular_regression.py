"""
Modular regression task with automatic data variant detection.

This task automatically detects the structure of the input data (scalar, vector, tensor)
and configures appropriate execution paths.

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
        
        # Provide default function_set if None
        if function_set is None:
            function_set = ["add", "sub", "mul", "div", "sin", "cos", "exp", "log"]
        
        # Step 1: Validate dataset input before calling parent
        if dataset is not None and not isinstance(dataset, (dict, str, tuple, np.ndarray)):
            raise ValueError(f"Invalid dataset type: {type(dataset)}. Expected dict, str, tuple, or numpy array.")
        
        # Step 2: Call parent constructor with validated dataset
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
        
        # Step 3: After loading, detect data structure and create handler
        self.data_shape, self.data_handler = auto_detect_data_structure(self.X_train, self.y_train, verbose=True)
        
        # Step 4: Add missing attributes for compatibility
        self.variant = self.data_shape.variant  # Direct access to variant
        self.n_input_var = self.data_shape.n_inputs
        self.n_output_var = self.data_shape.n_outputs
        
        # Add convenience properties
        self.is_scalar = self.data_shape.is_scalar
        self.is_miso = self.data_shape.is_miso
        self.is_simo = self.data_shape.is_simo
        self.is_mimo = self.data_shape.is_mimo
        self.is_tensor = self.data_shape.is_tensor
        
        # Override variance calculation for multi-output data
        self._calculate_output_variances()
    
    def _calculate_output_variances(self):
        """Calculate output variances correctly for multi-output data."""
        if self.data_shape.is_scalar or self.data_shape.is_miso:
            # For scalar and MISO (single output), use the parent calculation
            self.var_y_test = np.var(self.y_test)
            self.var_y_test_noiseless = np.var(self.y_test_noiseless)
        else:
            # For multi-output data, calculate variance per output
            self.var_y_test = np.var(self.y_test, axis=0)
            self.var_y_test_noiseless = np.var(self.y_test_noiseless, axis=0)
        
        print(f"Task configured for {self.data_shape.variant.value} with {self.data_shape.input_dims['n_features']} inputs, {self.data_shape.output_dims['n_outputs']} outputs")
    
    def reward_function(self, program, optimizing=False):
        """
        Reward function that handles both single-output and multi-output cases.
        """
        # For MIMO cases, we need special handling
        if self.is_mimo or self.is_simo:
            return self._mimo_reward_function(program, optimizing)
        else:
            # For scalar/MISO cases, use parent implementation
            return super().reward_function(program, optimizing)
    
    def _mimo_reward_function(self, program, optimizing=False):
        """
        Reward function specifically for MIMO cases.
        
        For MIMO, we expect to receive multiple programs (one per output)
        or a single program that can generate multiple outputs.
        """
        # Check if program has multiple sub-programs (MultiProgram case)
        if hasattr(program, 'programs') and len(program.programs) > 0:
            # MultiProgram case - evaluate each sub-program
            rewards = []
            for i, sub_program in enumerate(program.programs):
                # Get the i-th output column as target
                y_target = self.y_train[:, i:i+1] if self.y_train.ndim > 1 else self.y_train
                reward = self._single_output_reward(sub_program, y_target, optimizing)
                rewards.append(reward)
            # Return average reward across all outputs
            return np.mean(rewards)
        else:
            # Single program case - it should generate multi-output
            y_hat = program.execute(self.X_train)
            
            # Handle the case where program returns single output but we expect multiple
            if y_hat.ndim == 1 and self.y_train.ndim > 1:
                # Program is single-output but task expects multi-output
                # Replicate the single output across all expected outputs
                y_hat = np.tile(y_hat[:, np.newaxis], (1, self.y_train.shape[1]))
            
            # For invalid expressions, return invalid_reward
            if program.invalid:
                return -1.0 if optimizing else self.invalid_reward
            
            # Compute multi-output metric
            r = self._multi_output_metric(self.y_train, y_hat)
            
            # Add noise if specified
            if self.reward_noise and self.reward_noise_type == "r":
                r += self.rng.normal(loc=0, scale=self.scale)
                if self.normalize_variance:
                    r /= np.sqrt(1 + 12 * self.scale ** 2)
            
            return r
    
    def _single_output_reward(self, program, y_target, optimizing=False):
        """Helper function to compute reward for a single output."""
        # Execute the program
        y_hat = program.execute(self.X_train)
        
        # For invalid expressions, return invalid_reward
        if program.invalid:
            return -1.0 if optimizing else self.invalid_reward
        
        # Compute single-output metric
        var_y = np.var(y_target) if len(y_target) > 1 else 1.0
        metric_func = lambda y, y_hat: 1/(1 + np.sqrt(np.mean((y - y_hat)**2)/var_y))
        
        return metric_func(y_target.flatten(), y_hat.flatten())
    
    def evaluate(self, program):
        """
        Evaluation function that handles both single-output and multi-output cases.
        """
        # For MIMO cases, we need special handling
        if self.is_mimo or self.is_simo:
            return self._mimo_evaluate(program)
        else:
            # For scalar/MISO cases, use parent implementation
            return super().evaluate(program)
    
    def _mimo_evaluate(self, program):
        """
        Evaluation function specifically for MIMO cases.
        """
        # Execute the program on test data
        y_hat = program.execute(self.X_test)
        
        # Handle single-output program with multi-output target
        if y_hat.ndim == 1 and self.y_test.ndim > 1:
            y_hat = np.tile(y_hat[:, np.newaxis], (1, self.y_test.shape[1]))
        
        if program.invalid:
            nmse_test = np.inf
            nmse_test_noiseless = np.inf
            success = False
        else:
            # For multi-output, compute average NMSE across outputs
            if self.y_test.ndim > 1 and y_hat.ndim > 1:
                # Multi-output case
                nmse_per_output = []
                nmse_noiseless_per_output = []
                
                for i in range(self.y_test.shape[1]):
                    y_true_i = self.y_test[:, i]
                    y_pred_i = y_hat[:, i] if y_hat.shape[1] > i else y_hat[:, 0]
                    
                    # Use per-output variance if available
                    var_test_i = self.var_y_test[i] if hasattr(self.var_y_test, '__len__') else self.var_y_test
                    var_test_noiseless_i = self.var_y_test_noiseless[i] if hasattr(self.var_y_test_noiseless, '__len__') else self.var_y_test_noiseless
                    
                    nmse_i = np.mean((y_true_i - y_pred_i) ** 2) / var_test_i
                    nmse_per_output.append(nmse_i)
                    
                    y_true_noiseless_i = self.y_test_noiseless[:, i]
                    nmse_noiseless_i = np.mean((y_true_noiseless_i - y_pred_i) ** 2) / var_test_noiseless_i
                    nmse_noiseless_per_output.append(nmse_noiseless_i)
                
                # Average across outputs
                nmse_test = np.mean(nmse_per_output)
                nmse_test_noiseless = np.mean(nmse_noiseless_per_output)
                
            else:
                # Single output case (fallback)
                nmse_test = np.mean((self.y_test.flatten() - y_hat.flatten()) ** 2) / np.var(self.y_test)
                nmse_test_noiseless = np.mean((self.y_test_noiseless.flatten() - y_hat.flatten()) ** 2) / np.var(self.y_test_noiseless)
            
            # Success is defined by NMSE on noiseless test data below threshold
            success = nmse_test_noiseless < self.threshold
        
        info = {
            "nmse_test": nmse_test,
            "nmse_test_noiseless": nmse_test_noiseless,
            "success": success
        }
        
        # Add extra metric if specified
        if hasattr(self, 'metric_test') and self.metric_test is not None:
            if program.invalid:
                info["r_test"] = self.invalid_reward
            else:
                info["r_test"] = self._multi_output_metric(self.y_test, y_hat)
        
        return info
    
    def _multi_output_metric(self, y_true, y_pred):
        """Compute metric for multi-output case."""
        # Compute normalized RMSE for each output and average
        if y_true.ndim == 1 or y_pred.ndim == 1:
            # Handle 1D case
            var_y = np.var(y_true) if len(y_true) > 1 else 1.0
            return 1/(1 + np.sqrt(np.mean((y_true - y_pred)**2)/var_y))
        else:
            # Multi-output case: compute average inverse NRMSE across outputs
            rewards = []
            for i in range(y_true.shape[1]):
                y_col = y_true[:, i]
                y_hat_col = y_pred[:, i] if y_pred.shape[1] > i else y_pred[:, 0]
                var_y = np.var(y_col) if len(y_col) > 1 else 1.0
                reward = 1/(1 + np.sqrt(np.mean((y_col - y_hat_col)**2)/var_y))
                rewards.append(reward)
            return np.mean(rewards)
        """Compute metric for multi-output case."""
        # Compute normalized RMSE for each output and average
        if y_true.ndim == 1 or y_pred.ndim == 1:
            # Handle 1D case
            var_y = np.var(y_true) if len(y_true) > 1 else 1.0
            return 1/(1 + np.sqrt(np.mean((y_true - y_pred)**2)/var_y))
        else:
            # Multi-output case: compute average inverse NRMSE across outputs
            rewards = []
            for i in range(y_true.shape[1]):
                y_col = y_true[:, i]
                y_hat_col = y_pred[:, i] if y_pred.shape[1] > i else y_pred[:, 0]
                var_y = np.var(y_col) if len(y_col) > 1 else 1.0
                reward = 1/(1 + np.sqrt(np.mean((y_col - y_hat_col)**2)/var_y))
                rewards.append(reward)
            return np.mean(rewards)


def create_modular_regression_task(dataset, config):
    """
    Factory function to create a modular regression task from config.
    
    Parameters
    ----------
    dataset : tuple, str, or dict
        Dataset specification
    config : dict
        Configuration dictionary
        
    Returns
    -------
    ModularRegressionTask
        Configured modular regression task
    """
    task_config = config.get('task', {})
    
    return ModularRegressionTask(
        dataset=dataset,
        function_set=task_config.get('function_set', None),
        metric=task_config.get('metric', 'inv_nrmse'),
        metric_params=task_config.get('metric_params', (1.0,)),
        extra_metric_test=task_config.get('extra_metric_test', None),
        extra_metric_test_params=task_config.get('extra_metric_test_params', ()),
        reward_noise=task_config.get('reward_noise', 0.0),
        reward_noise_type=task_config.get('reward_noise_type', 'r'),
        threshold=task_config.get('threshold', 1e-12),
        normalize_variance=task_config.get('normalize_variance', False),
        protected=task_config.get('protected', False),
        decision_tree_threshold_set=task_config.get('decision_tree_threshold_set', None),
        poly_optimizer_params=task_config.get('poly_optimizer_params', None)
    )
