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
    
    def reward_function(self, program):
        """
        Reward function that uses the modular data handler for evaluation.
        """
        return super().reward_function(program)


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
