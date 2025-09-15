"""
Modular data type detection and handling system for DSO.

This module provides automatic detection and unified handling of different data variants:
- Scalar: Single input, single output (SISO)
- Vector: Multiple inputs, single output (MISO) or single input, multiple outputs (SIMO)
- Tensor: Multiple inputs, multiple outputs (MIMO) with arbitrary dimensions
- Matrix: 2D tensor operations
- Higher-order tensors: 3D+ tensors for specialized applications

The system automatically detects the data structure and configures the appropriate
execution path while maintaining backward compatibility.
"""

import numpy as np
from typing import Union, Tuple, List, Optional, Any, Dict
from enum import Enum
from abc import ABC, abstractmethod
import warnings


class DataVariant(Enum):
    """Enumeration of supported data variants."""
    SCALAR = "scalar"           # Single input, single output (SISO)
    VECTOR_INPUT = "vector_in"  # Multiple inputs, single output (MISO)
    VECTOR_OUTPUT = "vector_out" # Single input, multiple outputs (SIMO)
    VECTOR_BOTH = "vector_both" # Multiple inputs, multiple outputs (MIMO - 1D)
    MATRIX = "matrix"           # 2D tensor operations
    TENSOR_3D = "tensor_3d"     # 3D tensor operations
    TENSOR_ND = "tensor_nd"     # N-dimensional tensor operations


class DataShape:
    """Container for data shape information with automatic variant detection."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize data shape container with automatic variant detection.
        
        Parameters
        ----------
        X : np.ndarray
            Input data array
        y : np.ndarray
            Output data array
        """
        self.X = X
        self.y = y
        self.X_shape = X.shape
        self.y_shape = y.shape
        
        # Detect variant
        self.variant = self._detect_variant()
        
        # Extract dimensions
        self.n_samples = X.shape[0]
        self.input_dims = self._extract_input_dims()
        self.output_dims = self._extract_output_dims()
        
        # Compatibility flags
        self.is_scalar = self.variant == DataVariant.SCALAR
        self.is_vector = self.variant in [DataVariant.VECTOR_INPUT, DataVariant.VECTOR_OUTPUT, DataVariant.VECTOR_BOTH]
        self.is_tensor = self.variant in [DataVariant.MATRIX, DataVariant.TENSOR_3D, DataVariant.TENSOR_ND]
        
        # Legacy compatibility
        self.is_miso = self.variant == DataVariant.VECTOR_INPUT
        self.is_simo = self.variant == DataVariant.VECTOR_OUTPUT
        self.is_mimo = self.variant == DataVariant.VECTOR_BOTH
        
        # Convenience attributes for backward compatibility
        self.n_inputs = self.input_dims['n_features']
        self.n_outputs = self.output_dims['n_outputs']
        
    def _detect_variant(self) -> DataVariant:
        """Automatically detect data variant from array shapes."""
        X_ndim = self.X.ndim
        y_ndim = self.y.ndim
        
        # Handle edge cases
        if self.X.size == 0 or self.y.size == 0:
            raise ValueError("Empty arrays not supported")
            
        # SCALAR: X is (n, 1), y is (n,) or (n, 1)
        if X_ndim == 2 and self.X.shape[1] == 1 and (y_ndim == 1 or (y_ndim == 2 and self.y.shape[1] == 1)):
            return DataVariant.SCALAR
            
        # VECTOR_INPUT (MISO): X is (n, m) where m > 1, y is (n,) or (n, 1)
        if X_ndim == 2 and self.X.shape[1] > 1 and (y_ndim == 1 or (y_ndim == 2 and self.y.shape[1] == 1)):
            return DataVariant.VECTOR_INPUT
            
        # VECTOR_OUTPUT (SIMO): X is (n, 1), y is (n, m) where m > 1
        if X_ndim == 2 and self.X.shape[1] == 1 and y_ndim == 2 and self.y.shape[1] > 1:
            return DataVariant.VECTOR_OUTPUT
            
        # VECTOR_BOTH (MIMO - 1D): X is (n, m) where m > 1, y is (n, k) where k > 1
        if X_ndim == 2 and self.X.shape[1] > 1 and y_ndim == 2 and self.y.shape[1] > 1:
            return DataVariant.VECTOR_BOTH
            
        # MATRIX: Either X or y has 3D structure representing 2D matrices
        if X_ndim == 3 or y_ndim == 3:
            return DataVariant.MATRIX
            
        # TENSOR_3D: Either X or y has 4D structure
        if X_ndim == 4 or y_ndim == 4:
            return DataVariant.TENSOR_3D
            
        # TENSOR_ND: Higher dimensional tensors
        if X_ndim > 4 or y_ndim > 4:
            return DataVariant.TENSOR_ND
            
        # Fallback - treat as vector input if unclear
        warnings.warn(f"Could not determine data variant for X shape {self.X_shape}, y shape {self.y_shape}. "
                     f"Defaulting to VECTOR_INPUT.")
        return DataVariant.VECTOR_INPUT
        
    def _extract_input_dims(self) -> Dict[str, int]:
        """Extract input dimension information."""
        dims = {}
        
        if self.variant == DataVariant.SCALAR:
            dims['n_features'] = 1
            dims['feature_shape'] = ()
        elif self.variant in [DataVariant.VECTOR_INPUT, DataVariant.VECTOR_BOTH]:
            dims['n_features'] = self.X_shape[1]
            dims['feature_shape'] = ()
        elif self.variant == DataVariant.VECTOR_OUTPUT:
            dims['n_features'] = 1
            dims['feature_shape'] = ()
        elif self.variant == DataVariant.MATRIX:
            if self.X.ndim == 3:
                dims['n_features'] = self.X_shape[1]
                dims['feature_shape'] = (self.X_shape[2],)
            else:
                dims['n_features'] = self.X_shape[1]
                dims['feature_shape'] = ()
        else:  # Higher-dimensional tensors
            dims['n_features'] = self.X_shape[1]
            dims['feature_shape'] = self.X_shape[2:]
            
        return dims
        
    def _extract_output_dims(self) -> Dict[str, int]:
        """Extract output dimension information."""
        dims = {}
        
        if self.variant in [DataVariant.SCALAR, DataVariant.VECTOR_INPUT]:
            dims['n_outputs'] = 1
            dims['output_shape'] = ()
        elif self.variant in [DataVariant.VECTOR_OUTPUT, DataVariant.VECTOR_BOTH]:
            dims['n_outputs'] = self.y_shape[1]
            dims['output_shape'] = ()
        elif self.variant == DataVariant.MATRIX:
            if self.y.ndim == 3:
                dims['n_outputs'] = self.y_shape[1]
                dims['output_shape'] = (self.y_shape[2],)
            else:
                dims['n_outputs'] = self.y_shape[1] if self.y.ndim == 2 else 1
                dims['output_shape'] = ()
        else:  # Higher-dimensional tensors
            dims['n_outputs'] = self.y_shape[1] if self.y.ndim > 1 else 1
            dims['output_shape'] = self.y_shape[2:] if self.y.ndim > 2 else ()
            
        return dims
        
    def summary(self) -> str:
        """Return a human-readable summary of the data structure."""
        return (f"DataShape Summary:\n"
                f"  Variant: {self.variant.value}\n"
                f"  Input shape: {self.X_shape}\n"
                f"  Output shape: {self.y_shape}\n"
                f"  Samples: {self.n_samples}\n"
                f"  Input dims: {self.input_dims}\n"
                f"  Output dims: {self.output_dims}")


class DataHandler(ABC):
    """Abstract base class for variant-specific data handlers."""
    
    @abstractmethod
    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for processing."""
        pass
        
    @abstractmethod
    def execute_program(self, program, X: np.ndarray) -> np.ndarray:
        """Execute program on input data."""
        pass
        
    @abstractmethod
    def compute_reward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute reward/fitness for predictions."""
        pass
        
    @abstractmethod
    def validate_shapes(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Validate that data shapes are compatible with this handler."""
        pass


class ScalarHandler(DataHandler):
    """Handler for scalar (SISO) data."""
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ensure data is in correct scalar format."""
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        y = y.flatten() if y.ndim > 1 else y
        return X, y
        
    def execute_program(self, program, X: np.ndarray) -> np.ndarray:
        """Execute program for scalar output."""
        return program.execute(X).flatten()
        
    def compute_reward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute NMSE for scalar predictions."""
        mse = np.mean((y_true - y_pred) ** 2)
        var = np.var(y_true)
        return 1.0 / (1.0 + mse / (var + 1e-12))
        
    def validate_shapes(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Validate scalar data shapes."""
        return (X.ndim <= 2 and (X.shape[1] == 1 if X.ndim == 2 else True) and
                y.ndim <= 2 and (y.shape[1] == 1 if y.ndim == 2 else True))


class VectorInputHandler(DataHandler):
    """Handler for vector input (MISO) data."""
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ensure data is in correct MISO format."""
        X = X.reshape(X.shape[0], -1) if X.ndim != 2 else X
        y = y.flatten() if y.ndim > 1 else y
        return X, y
        
    def execute_program(self, program, X: np.ndarray) -> np.ndarray:
        """Execute program for MISO case."""
        return program.execute(X).flatten()
        
    def compute_reward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute NMSE for MISO predictions."""
        mse = np.mean((y_true - y_pred) ** 2)
        var = np.var(y_true)
        return 1.0 / (1.0 + mse / (var + 1e-12))
        
    def validate_shapes(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Validate MISO data shapes."""
        return (X.ndim == 2 and X.shape[1] > 1 and
                (y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1)))


class VectorOutputHandler(DataHandler):
    """Handler for vector output (SIMO) data."""
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ensure data is in correct SIMO format."""
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        y = y.reshape(y.shape[0], -1) if y.ndim != 2 else y
        return X, y
        
    def execute_program(self, program, X: np.ndarray) -> np.ndarray:
        """Execute program for SIMO case - multiple programs or vector-valued program."""
        if hasattr(program, 'execute_multi_output'):
            return program.execute_multi_output(X)
        else:
            # Fallback: execute single program and replicate/transform output
            single_output = program.execute(X).flatten()
            n_outputs = self.n_outputs
            return np.tile(single_output[:, np.newaxis], (1, n_outputs))
        
    def compute_reward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute average NMSE across outputs for SIMO predictions."""
        mse_per_output = np.mean((y_true - y_pred) ** 2, axis=0)
        var_per_output = np.var(y_true, axis=0)
        nmse_per_output = mse_per_output / (var_per_output + 1e-12)
        avg_nmse = np.mean(nmse_per_output)
        return 1.0 / (1.0 + avg_nmse)
        
    def validate_shapes(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Validate SIMO data shapes."""
        return (X.ndim == 2 and X.shape[1] == 1 and
                y.ndim == 2 and y.shape[1] > 1)


class VectorBothHandler(DataHandler):
    """Handler for vector both (MIMO) data."""
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ensure data is in correct MIMO format."""
        X = X.reshape(X.shape[0], -1) if X.ndim != 2 else X
        y = y.reshape(y.shape[0], -1) if y.ndim != 2 else y
        return X, y
        
    def execute_program(self, program, X: np.ndarray) -> np.ndarray:
        """Execute program for MIMO case - multiple programs."""
        if hasattr(program, 'execute_multi_output'):
            return program.execute_multi_output(X)
        else:
            # Fallback: execute single program for first output
            single_output = program.execute(X).flatten()
            n_outputs = self.n_outputs
            return np.tile(single_output[:, np.newaxis], (1, n_outputs))
        
    def compute_reward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute average NMSE across outputs for MIMO predictions."""
        mse_per_output = np.mean((y_true - y_pred) ** 2, axis=0)
        var_per_output = np.var(y_true, axis=0)
        nmse_per_output = mse_per_output / (var_per_output + 1e-12)
        avg_nmse = np.mean(nmse_per_output)
        return 1.0 / (1.0 + avg_nmse)
        
    def validate_shapes(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Validate MIMO data shapes."""
        return (X.ndim == 2 and X.shape[1] > 1 and
                y.ndim == 2 and y.shape[1] > 1)


class TensorHandler(DataHandler):
    """Handler for tensor data (matrices and higher-dimensional tensors)."""
    
    def __init__(self, tensor_variant: DataVariant):
        self.tensor_variant = tensor_variant
        
    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare tensor data - preserve original shapes."""
        return X, y
        
    def execute_program(self, program, X: np.ndarray) -> np.ndarray:
        """Execute program for tensor case."""
        if hasattr(program, 'execute_tensor'):
            return program.execute_tensor(X)
        else:
            # Fallback: flatten, execute, reshape
            original_shape = X.shape
            X_flat = X.reshape(X.shape[0], -1)
            output_flat = program.execute(X_flat)
            
            # Determine output shape based on y structure
            if hasattr(self, 'output_shape_template'):
                output_shape = (X.shape[0],) + self.output_shape_template
                return output_flat.reshape(output_shape)
            else:
                return output_flat
        
    def compute_reward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute tensor-based reward."""
        # Flatten tensors for MSE computation
        y_true_flat = y_true.reshape(y_true.shape[0], -1)
        y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)
        
        mse = np.mean((y_true_flat - y_pred_flat) ** 2)
        var = np.var(y_true_flat)
        return 1.0 / (1.0 + mse / (var + 1e-12))
        
    def validate_shapes(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Validate tensor data shapes."""
        return X.ndim >= 3 or y.ndim >= 3


class DataTypeDetector:
    """Factory class for automatic data type detection and handler creation."""
    
    _handlers = {
        DataVariant.SCALAR: ScalarHandler,
        DataVariant.VECTOR_INPUT: VectorInputHandler,
        DataVariant.VECTOR_OUTPUT: VectorOutputHandler,
        DataVariant.VECTOR_BOTH: VectorBothHandler,
        DataVariant.MATRIX: lambda: TensorHandler(DataVariant.MATRIX),
        DataVariant.TENSOR_3D: lambda: TensorHandler(DataVariant.TENSOR_3D),
        DataVariant.TENSOR_ND: lambda: TensorHandler(DataVariant.TENSOR_ND),
    }
    
    @classmethod
    def detect_and_create_handler(cls, X: np.ndarray, y: np.ndarray) -> Tuple[DataShape, DataHandler]:
        """
        Automatically detect data variant and create appropriate handler.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        y : np.ndarray
            Output data
            
        Returns
        -------
        Tuple[DataShape, DataHandler]
            Data shape information and appropriate handler
        """
        data_shape = DataShape(X, y)
        
        # Create handler
        handler_class = cls._handlers[data_shape.variant]
        if callable(handler_class) and not isinstance(handler_class, type):
            handler = handler_class()  # For lambda functions
        else:
            handler = handler_class()
            
        # Store necessary information in handler
        if hasattr(handler, 'n_outputs'):
            handler.n_outputs = data_shape.n_outputs
        if hasattr(handler, 'output_shape_template'):
            handler.output_shape_template = data_shape.output_dims['output_shape']
            
        return data_shape, handler
    
    @classmethod
    def register_handler(cls, variant: DataVariant, handler_class: type):
        """Register a custom handler for a data variant."""
        cls._handlers[variant] = handler_class
        

def auto_detect_data_structure(X: np.ndarray, y: np.ndarray, verbose: bool = True) -> Tuple[DataShape, DataHandler]:
    """
    Convenience function for automatic data structure detection.
    
    Parameters
    ----------
    X : np.ndarray
        Input data array
    y : np.ndarray  
        Output data array
    verbose : bool, optional
        Whether to print detection results
        
    Returns
    -------
    Tuple[DataShape, DataHandler]
        Data shape information and appropriate handler
    """
    data_shape, handler = DataTypeDetector.detect_and_create_handler(X, y)
    
    if verbose:
        print("="*60)
        print("AUTOMATIC DATA STRUCTURE DETECTION")
        print("="*60)
        print(data_shape.summary())
        print(f"Handler: {handler.__class__.__name__}")
        print("="*60)
    
    return data_shape, handler


# Backward compatibility aliases
def detect_mimo_variant(X: np.ndarray, y: np.ndarray) -> str:
    """Legacy function for MIMO variant detection."""
    data_shape = DataShape(X, y)
    
    if data_shape.is_scalar:
        return "SISO"
    elif data_shape.is_miso:
        return "MISO"
    elif data_shape.is_simo:
        return "SIMO"
    elif data_shape.is_mimo:
        return "MIMO"
    else:
        return "TENSOR"
