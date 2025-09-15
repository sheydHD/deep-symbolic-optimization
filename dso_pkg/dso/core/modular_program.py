"""
Modular program execution system for different data variants.

This module extends the base Program class to handle execution across different
data structures: scalar, vector, matrix, and tensor operations.
"""

import numpy as np
from typing import Union, List, Tuple, Optional, Any
from abc import ABC, abstractmethod

from dso.program import Program
from dso.core.data_types import DataVariant, DataShape


class ProgramExecutor(ABC):
    """Abstract base class for variant-specific program execution."""
    
    @abstractmethod
    def execute(self, program: Program, X: np.ndarray) -> np.ndarray:
        """Execute program on input data."""
        pass
        
    @abstractmethod
    def can_handle(self, variant: DataVariant) -> bool:
        """Check if this executor can handle the given variant."""
        pass


class ScalarExecutor(ProgramExecutor):
    """Executor for scalar (SISO) programs."""
    
    def execute(self, program: Program, X: np.ndarray) -> np.ndarray:
        """Execute program for scalar output."""
        result = program.execute(X)
        return result.flatten() if hasattr(result, 'flatten') else np.array([result])
        
    def can_handle(self, variant: DataVariant) -> bool:
        return variant == DataVariant.SCALAR


class VectorExecutor(ProgramExecutor):
    """Executor for vector input (MISO) programs."""
    
    def execute(self, program: Program, X: np.ndarray) -> np.ndarray:
        """Execute program for vector input, scalar output."""
        result = program.execute(X)
        return result.flatten() if hasattr(result, 'flatten') else np.array([result])
        
    def can_handle(self, variant: DataVariant) -> bool:
        return variant == DataVariant.VECTOR_INPUT


class MultiOutputExecutor(ProgramExecutor):
    """Executor for multi-output programs (SIMO/MIMO)."""
    
    def __init__(self, n_outputs: int):
        self.n_outputs = n_outputs
        
    def execute(self, program: Program, X: np.ndarray) -> np.ndarray:
        """Execute program for multiple outputs."""
        
        if hasattr(program, 'execute_multi_output'):
            # Program has native multi-output support
            return program.execute_multi_output(X)
        elif hasattr(program, 'programs') and len(program.programs) == self.n_outputs:
            # Program is a collection of sub-programs
            outputs = []
            for sub_program in program.programs:
                output = sub_program.execute(X)
                outputs.append(output.flatten() if hasattr(output, 'flatten') else output)
            return np.column_stack(outputs)
        else:
            # Fallback: replicate single output
            single_output = program.execute(X)
            if hasattr(single_output, 'flatten'):
                single_output = single_output.flatten()
            return np.tile(single_output[:, np.newaxis], (1, self.n_outputs))
        
    def can_handle(self, variant: DataVariant) -> bool:
        return variant in [DataVariant.VECTOR_OUTPUT, DataVariant.VECTOR_BOTH]


class TensorExecutor(ProgramExecutor):
    """Executor for tensor programs."""
    
    def __init__(self, input_shape: Tuple, output_shape: Tuple):
        self.input_shape = input_shape
        self.output_shape = output_shape
        
    def execute(self, program: Program, X: np.ndarray) -> np.ndarray:
        """Execute program for tensor data."""
        
        if hasattr(program, 'execute_tensor'):
            # Program has native tensor support
            return program.execute_tensor(X)
        else:
            # Fallback: flatten, execute, reshape
            original_shape = X.shape
            X_flat = X.reshape(X.shape[0], -1)
            
            # Execute on flattened data
            result_flat = program.execute(X_flat)
            
            # Reshape to expected output
            if len(self.output_shape) > 0:
                output_shape = (X.shape[0],) + self.output_shape
                try:
                    return result_flat.reshape(output_shape)
                except ValueError:
                    # If reshape fails, return flattened result
                    return result_flat
            else:
                return result_flat
        
    def can_handle(self, variant: DataVariant) -> bool:
        return variant in [DataVariant.MATRIX, DataVariant.TENSOR_3D, DataVariant.TENSOR_ND]


class ModularProgram(Program):
    """
    Extended Program class with modular execution capabilities.
    
    This class automatically detects the appropriate execution method based on
    the data variant and delegates to specialized executors.
    """
    
    def __init__(self, tokens=None, data_shape: Optional[DataShape] = None):
        """
        Initialize modular program.
        
        Parameters
        ----------
        tokens : list, optional
            Token sequence for the program
        data_shape : DataShape, optional
            Data shape information for execution configuration
        """
        # Handle case when library is not initialized (for testing)
        if tokens is not None and Program.library is None:
            # Skip program initialization for testing scenarios
            self.tokens = tokens
            self.traversal = []
            self.invalid = False
            self.str = None
        else:
            super().__init__(tokens)
            
        self.data_shape = data_shape
        self.executor = None
        
        if data_shape is not None:
            self._configure_executor()
            
    def _configure_executor(self):
        """Configure the appropriate executor based on data shape."""
        
        variant = self.data_shape.variant
        
        if variant == DataVariant.SCALAR:
            self.executor = ScalarExecutor()
        elif variant == DataVariant.VECTOR_INPUT:
            self.executor = VectorExecutor()
        elif variant in [DataVariant.VECTOR_OUTPUT, DataVariant.VECTOR_BOTH]:
            n_outputs = self.data_shape.n_outputs
            self.executor = MultiOutputExecutor(n_outputs)
        elif variant in [DataVariant.MATRIX, DataVariant.TENSOR_3D, DataVariant.TENSOR_ND]:
            input_shape = self.data_shape.input_dims['feature_shape']
            output_shape = self.data_shape.output_dims['output_shape']
            self.executor = TensorExecutor(input_shape, output_shape)
        else:
            # Fallback to standard execution
            self.executor = VectorExecutor()
            
    def execute(self, X: np.ndarray) -> np.ndarray:
        """
        Execute program with automatic variant detection.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
            
        Returns
        -------
        np.ndarray
            Program output
        """
        
        if self.executor is not None:
            return self.executor.execute(self, X)
        else:
            # Fallback to base class execution
            return super().execute(X)
            
    def execute_multi_output(self, X: np.ndarray) -> np.ndarray:
        """
        Execute program for multi-output case.
        
        This method should be overridden by subclasses that support native
        multi-output execution.
        """
        
        if hasattr(self, 'programs') and isinstance(self.programs, list):
            # Execute multiple sub-programs
            outputs = []
            for program in self.programs:
                output = program.execute(X)
                outputs.append(output.flatten() if hasattr(output, 'flatten') else output)
            return np.column_stack(outputs)
        else:
            # Single program - replicate output
            single_output = super().execute(X)
            if hasattr(single_output, 'flatten'):
                single_output = single_output.flatten()
            n_outputs = self.data_shape.n_outputs if self.data_shape else 1
            return np.tile(single_output[:, np.newaxis], (1, n_outputs))
            
    def execute_tensor(self, X: np.ndarray) -> np.ndarray:
        """
        Execute program for tensor data.
        
        This method should be overridden by subclasses that support native
        tensor execution.
        """
        
        # Default implementation: flatten, execute, reshape
        original_shape = X.shape
        X_flat = X.reshape(X.shape[0], -1)
        result = super().execute(X_flat)
        
        # Try to reshape back to meaningful tensor shape
        if self.data_shape and len(self.data_shape.output_dims['output_shape']) > 0:
            output_shape = (X.shape[0],) + self.data_shape.output_dims['output_shape']
            try:
                return result.reshape(output_shape)
            except ValueError:
                return result
        else:
            return result


class MultiProgram:
    """
    Container for multiple programs to handle multi-output cases.
    
    This class manages multiple Program instances, one for each output dimension.
    """
    
    def __init__(self, programs: List[Program], data_shape: Optional[DataShape] = None):
        """
        Initialize multi-program container.
        
        Parameters
        ----------
        programs : List[Program]
            List of programs, one for each output
        data_shape : DataShape, optional
            Data shape information
        """
        self.programs = programs
        self.data_shape = data_shape
        self.n_programs = len(programs)
        
        # Propagate data shape to individual programs
        if data_shape is not None:
            for program in programs:
                if hasattr(program, 'data_shape'):
                    program.data_shape = data_shape
                    
    def execute(self, X: np.ndarray) -> np.ndarray:
        """Execute all programs and concatenate outputs."""
        
        outputs = []
        for program in self.programs:
            output = program.execute(X)
            outputs.append(output.flatten() if hasattr(output, 'flatten') else output)
            
        return np.column_stack(outputs)
        
    def execute_multi_output(self, X: np.ndarray) -> np.ndarray:
        """Execute for multi-output (same as execute for this class)."""
        return self.execute(X)
        
    @property
    def str(self) -> str:
        """String representation of all programs."""
        return " | ".join([p.str for p in self.programs])
        
    @property
    def tokens(self) -> List:
        """Combined tokens from all programs."""
        combined = []
        for i, program in enumerate(self.programs):
            if i > 0:
                combined.append("|")  # Separator token
            combined.extend(program.tokens)
        return combined
        
    @property
    def length(self) -> int:
        """Total length of all programs."""
        return sum(len(p.tokens) for p in self.programs)
        
    @property
    def complexity(self) -> float:
        """Combined complexity of all programs."""
        return sum(getattr(p, 'complexity', len(p.tokens)) for p in self.programs)


def create_program_for_variant(tokens=None, data_shape: Optional[DataShape] = None, 
                              multi_program_tokens: Optional[List[List]] = None) -> Union[ModularProgram, MultiProgram]:
    """
    Factory function to create appropriate program type for data variant.
    
    Parameters
    ----------
    tokens : list, optional
        Single program tokens
    data_shape : DataShape, optional
        Data shape information
    multi_program_tokens : List[List], optional
        Multiple program token sequences for multi-output cases
        
    Returns
    -------
    Union[ModularProgram, MultiProgram]
        Appropriate program instance
    """
    
    if multi_program_tokens is not None:
        # Create multi-program for multi-output cases
        programs = [ModularProgram(token_seq, data_shape) for token_seq in multi_program_tokens]
        return MultiProgram(programs, data_shape)
    else:
        # Create single modular program
        return ModularProgram(tokens, data_shape)


def adapt_existing_program(program: Program, data_shape: DataShape) -> Union[ModularProgram, MultiProgram]:
    """
    Adapt an existing program to work with the specified data variant.
    
    Parameters
    ----------
    program : Program
        Existing program to adapt
    data_shape : DataShape
        Target data shape
        
    Returns
    -------
    Union[ModularProgram, MultiProgram]
        Adapted program
    """
    
    if data_shape.output_dims['n_outputs'] == 1:
        # Single output - create modular program
        modular_program = ModularProgram(program.tokens, data_shape)
        return modular_program
    else:
        # Multi-output - replicate program for each output
        programs = []
        for _ in range(data_shape.output_dims['n_outputs']):
            programs.append(ModularProgram(program.tokens.copy(), data_shape))
        return MultiProgram(programs, data_shape)


# Utility functions for program creation
def create_scalar_program(tokens) -> ModularProgram:
    """Create a program optimized for scalar operations."""
    scalar_shape = DataShape(np.array([[1.0]]), np.array([1.0]))
    return ModularProgram(tokens, scalar_shape)


def create_vector_program(tokens, n_inputs: int) -> ModularProgram:
    """Create a program optimized for vector input operations."""
    X = np.random.randn(10, n_inputs)
    y = np.random.randn(10)
    vector_shape = DataShape(X, y)
    return ModularProgram(tokens, vector_shape)


def create_mimo_program(token_sequences: List[List], n_inputs: int, n_outputs: int) -> MultiProgram:
    """Create a multi-program for MIMO operations."""
    X = np.random.randn(10, n_inputs)
    y = np.random.randn(10, n_outputs)
    mimo_shape = DataShape(X, y)
    
    programs = [ModularProgram(tokens, mimo_shape) for tokens in token_sequences]
    return MultiProgram(programs, mimo_shape)
