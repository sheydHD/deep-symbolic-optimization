"""
Modular policy system for different data variants.

This module extends the RNN policy to automatically adapt to different data structures
and generate appropriate program sequences for scalar, vector, matrix, and tensor cases.
"""

import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional, Union, Dict, Any

from dso.policy.rnn_policy import RNNPolicy
from dso.core.data_types import DataVariant, DataShape
from dso.core.modular_program import create_program_for_variant, MultiProgram
from dso.program import from_tokens
from dso.tf_state_manager import make_state_manager
from dso.prior import make_prior


class ModularRNNPolicy(RNNPolicy):
    """
    Modular RNN policy that adapts to different data variants.
    
    This policy automatically configures its sampling behavior based on the
    detected data structure (scalar, vector, tensor) and generates appropriate
    program sequences.
    """
    
    def __init__(self, library, data_shape: Optional[DataShape] = None, 
                 multi_output_strategy: str = "replicate", **kwargs):
        """
        Initialize modular RNN policy.
        
        Parameters
        ----------
        library : Library
            Function library
        data_shape : DataShape, optional
            Data shape information for configuration
        multi_output_strategy : str, optional
            Strategy for multi-output generation:
            - "replicate": Generate one program, replicate for all outputs
            - "independent": Generate independent programs for each output
            - "shared": Generate programs with shared components
        **kwargs
            Additional arguments for base RNNPolicy
        """
        
        # Create a default state manager if not provided
        if 'state_manager' not in kwargs:
            # Create a simple state manager configuration
            state_manager_config = {
                'type': 'hierarchical',
                'observe_parent': True,
                'observe_sibling': True,
                'observe_action': False,
                'observe_dangling': False,
                'embedding': False,
                'embedding_size': 8
            }
            kwargs['state_manager'] = make_state_manager(state_manager_config)
        
        # Create a default prior if not provided
        if 'prior' not in kwargs:
            # Create a simple prior configuration - use length constraint
            prior_config = {
                'length': {'on': True, 'min_': 1, 'max_': 20}
            }
            kwargs['prior'] = make_prior(library, prior_config)
        
        super().__init__(kwargs['prior'], kwargs['state_manager'], **{k: v for k, v in kwargs.items() if k not in ['prior', 'state_manager']})
        
        self.data_shape = data_shape
        self.multi_output_strategy = multi_output_strategy
        
        # Configure for data variant
        if data_shape is not None:
            self._configure_for_variant()
        else:
            # Default configuration
            self.variant = DataVariant.VECTOR_INPUT
            self.n_outputs = 1
            self.n_inputs = 1
            
        # Strategy-specific parameters
        self.output_sharing_ratio = 0.3  # For shared strategy
        self.max_programs_per_output = 1  # For independent strategy
        
    def _configure_for_variant(self):
        """Configure policy parameters based on data variant."""
        
        self.variant = self.data_shape.variant
        self.n_outputs = self.data_shape.n_outputs
        self.n_inputs = self.data_shape.n_inputs
        
        # Adjust policy parameters based on variant
        if self.variant == DataVariant.SCALAR:
            # Scalar: simple single program
            self.n_outputs = 1
            self.complexity_penalty = 0.01
            
        elif self.variant == DataVariant.VECTOR_INPUT:
            # MISO: single program with multiple inputs
            self.n_outputs = 1
            self.complexity_penalty = 0.005  # Allow slightly more complex programs
            
        elif self.variant == DataVariant.VECTOR_OUTPUT:
            # SIMO: multiple programs or vector-valued program
            self.complexity_penalty = 0.005 * self.n_outputs
            
        elif self.variant == DataVariant.VECTOR_BOTH:
            # MIMO: multiple programs with multiple inputs
            self.complexity_penalty = 0.002 * self.n_outputs
            
        elif self.variant in [DataVariant.MATRIX, DataVariant.TENSOR_3D, DataVariant.TENSOR_ND]:
            # Tensor: complex multi-dimensional operations
            self.complexity_penalty = 0.001 * self.n_outputs
            
        print(f"Policy configured for {self.variant.value}: {self.n_inputs} inputs, {self.n_outputs} outputs")
        
    def sample(self, n_expressions: int = None, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample expressions appropriate for the data variant.
        
        Parameters
        ---------- 
        n_expressions : int, optional
            Number of expressions to sample
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            actions, observations, priors
        """
        
        if n_expressions is None:
            n_expressions = self.batch_size
            
        if self.n_outputs == 1:
            # Single output: use standard sampling
            return super().sample(n_expressions, **kwargs)
        else:
            # Multi-output: use strategy-specific sampling
            return self._sample_multi_output(n_expressions, **kwargs)
            
    def _sample_multi_output(self, n_expressions: int, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample expressions for multi-output cases."""
        
        if self.multi_output_strategy == "replicate":
            return self._sample_replicate_strategy(n_expressions, **kwargs)
        elif self.multi_output_strategy == "independent":
            return self._sample_independent_strategy(n_expressions, **kwargs)
        elif self.multi_output_strategy == "shared":
            return self._sample_shared_strategy(n_expressions, **kwargs)
        else:
            raise ValueError(f"Unknown multi-output strategy: {self.multi_output_strategy}")
            
    def _sample_replicate_strategy(self, n_expressions: int, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample single programs and replicate for all outputs.
        
        This is the simplest strategy where one program is used for all outputs.
        """
        
        # Sample single programs
        actions, obs, priors = super().sample(n_expressions, **kwargs)
        
        # Programs will be replicated during execution
        return actions, obs, priors
        
    def _sample_independent_strategy(self, n_expressions: int, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample independent programs for each output.
        
        This generates separate programs for each output dimension.
        """
        
        all_actions = []
        all_obs = []
        all_priors = []
        
        # Sample programs for each output
        for output_idx in range(self.n_outputs):
            # Modify library context for output-specific sampling
            output_actions, output_obs, output_priors = super().sample(n_expressions, **kwargs)
            
            all_actions.append(output_actions)
            all_obs.append(output_obs)
            all_priors.append(output_priors)
            
        # Combine all programs
        # For now, concatenate along batch dimension
        combined_actions = np.concatenate(all_actions, axis=0)
        combined_obs = np.concatenate(all_obs, axis=0)
        combined_priors = np.concatenate(all_priors, axis=0)
        
        return combined_actions, combined_obs, combined_priors
        
    def _sample_shared_strategy(self, n_expressions: int, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample programs with shared components across outputs.
        
        This generates programs that share some components but have output-specific parts.
        """
        
        # For now, implement as hybrid of replicate and independent
        n_shared = int(n_expressions * self.output_sharing_ratio)
        n_independent = n_expressions - n_shared
        
        if n_shared > 0:
            # Sample shared programs
            shared_actions, shared_obs, shared_priors = self._sample_replicate_strategy(n_shared, **kwargs)
        else:
            shared_actions = shared_obs = shared_priors = None
            
        if n_independent > 0:
            # Sample independent programs
            indep_actions, indep_obs, indep_priors = self._sample_independent_strategy(
                n_independent // self.n_outputs, **kwargs)
        else:
            indep_actions = indep_obs = indep_priors = None
            
        # Combine shared and independent
        if shared_actions is not None and indep_actions is not None:
            combined_actions = np.concatenate([shared_actions, indep_actions], axis=0)
            combined_obs = np.concatenate([shared_obs, indep_obs], axis=0)
            combined_priors = np.concatenate([shared_priors, indep_priors], axis=0)
        elif shared_actions is not None:
            combined_actions, combined_obs, combined_priors = shared_actions, shared_obs, shared_priors
        else:
            combined_actions, combined_obs, combined_priors = indep_actions, indep_obs, indep_priors
            
        return combined_actions, combined_obs, combined_priors
        
    def create_programs_from_actions(self, actions: np.ndarray) -> List[Union['ModularProgram', 'MultiProgram']]:
        """
        Create appropriate program instances from action sequences.
        
        Parameters
        ----------
        actions : np.ndarray
            Action sequences from policy sampling
            
        Returns
        -------
        List[Union[ModularProgram, MultiProgram]]
            List of program instances
        """
        
        programs = []
        
        if self.n_outputs == 1:
            # Single output: create standard programs
            for action_seq in actions:
                tokens = [self.library.tokens[a] for a in action_seq if a < len(self.library.tokens)]
                program = create_program_for_variant(tokens, self.data_shape)
                programs.append(program)
                
        else:
            # Multi-output: create programs based on strategy
            if self.multi_output_strategy == "replicate":
                # Single program replicated for all outputs
                for action_seq in actions:
                    tokens = [self.library.tokens[a] for a in action_seq if a < len(self.library.tokens)]
                    program = create_program_for_variant(tokens, self.data_shape)
                    programs.append(program)
                    
            elif self.multi_output_strategy == "independent":
                # Group actions by output
                n_programs_per_batch = len(actions) // self.n_outputs
                
                for i in range(n_programs_per_batch):
                    # Collect actions for all outputs for this batch item
                    multi_program_tokens = []
                    for output_idx in range(self.n_outputs):
                        action_idx = i + output_idx * n_programs_per_batch
                        if action_idx < len(actions):
                            action_seq = actions[action_idx]
                            tokens = [self.library.tokens[a] for a in action_seq if a < len(self.library.tokens)]
                            multi_program_tokens.append(tokens)
                            
                    if len(multi_program_tokens) == self.n_outputs:
                        program = create_program_for_variant(None, self.data_shape, multi_program_tokens)
                        programs.append(program)
                        
            else:  # shared strategy
                # Similar to independent but with some shared components
                programs = self._create_shared_programs(actions)
                
        return programs
        
    def _create_shared_programs(self, actions: np.ndarray) -> List['MultiProgram']:
        """Create programs with shared components."""
        
        # Simplified implementation: treat as independent for now
        return self.create_programs_from_actions(actions)
        
    def set_strategy(self, strategy: str):
        """Change the multi-output strategy."""
        
        valid_strategies = ["replicate", "independent", "shared"]
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}")
            
        self.multi_output_strategy = strategy
        print(f"Multi-output strategy set to: {strategy}")
        
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about current strategy configuration."""
        
        return {
            'variant': self.variant.value if hasattr(self, 'variant') else 'unknown',
            'n_inputs': self.n_inputs,
            'n_outputs': self.n_outputs,
            'strategy': self.multi_output_strategy,
            'complexity_penalty': self.complexity_penalty,
            'output_sharing_ratio': self.output_sharing_ratio,
            'max_programs_per_output': self.max_programs_per_output
        }
        
    def print_strategy_summary(self):
        """Print summary of current strategy configuration."""
        
        info = self.get_strategy_info()
        print("\n" + "="*50)
        print("MODULAR POLICY CONFIGURATION")
        print("="*50)
        print(f"Data variant: {info['variant']}")
        print(f"Inputs: {info['n_inputs']}, Outputs: {info['n_outputs']}")
        print(f"Multi-output strategy: {info['strategy']}")
        print(f"Complexity penalty: {info['complexity_penalty']}")
        if info['strategy'] == 'shared':
            print(f"Output sharing ratio: {info['output_sharing_ratio']}")
        print("="*50)


def create_modular_policy(library, data_shape: DataShape, strategy: str = "auto", **kwargs) -> ModularRNNPolicy:
    """
    Factory function to create a modular policy with appropriate strategy.
    
    Parameters
    ----------
    library : Library
        Function library
    data_shape : DataShape
        Data shape information
    strategy : str, optional
        Multi-output strategy ("auto", "replicate", "independent", "shared")
    **kwargs
        Additional arguments for policy
        
    Returns
    -------
    ModularRNNPolicy
        Configured modular policy
    """
    
    # Auto-select strategy based on data variant
    if strategy == "auto":
        if data_shape.output_dims['n_outputs'] == 1:
            strategy = "replicate"  # Not relevant for single output
        elif data_shape.output_dims['n_outputs'] <= 3:
            strategy = "independent"  # Small number of outputs
        else:
            strategy = "shared"  # Large number of outputs
            
    policy = ModularRNNPolicy(
        library=library,
        data_shape=data_shape,
        multi_output_strategy=strategy,
        **kwargs
    )
    
    policy.print_strategy_summary()
    return policy


# Convenience functions for specific variants
def create_scalar_policy(library, **kwargs) -> ModularRNNPolicy:
    """Create a policy optimized for scalar problems."""
    scalar_shape = DataShape(np.array([[1.0]]), np.array([1.0]))
    return create_modular_policy(library, scalar_shape, strategy="replicate", **kwargs)


def create_mimo_policy(library, n_inputs: int, n_outputs: int, strategy: str = "auto", **kwargs) -> ModularRNNPolicy:
    """Create a policy optimized for MIMO problems."""
    X = np.random.randn(10, n_inputs)
    y = np.random.randn(10, n_outputs)
    mimo_shape = DataShape(X, y)
    return create_modular_policy(library, mimo_shape, strategy=strategy, **kwargs)


def create_tensor_policy(library, input_shape: Tuple, output_shape: Tuple, **kwargs) -> ModularRNNPolicy:
    """Create a policy optimized for tensor problems."""
    # Create dummy tensor data
    full_input_shape = (10,) + input_shape
    full_output_shape = (10,) + output_shape
    X = np.random.randn(*full_input_shape)
    y = np.random.randn(*full_output_shape)
    tensor_shape = DataShape(X, y)
    return create_modular_policy(library, tensor_shape, strategy="independent", **kwargs)
