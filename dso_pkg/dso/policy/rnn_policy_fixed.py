"""Fixed RNN policy with proper library initialization handling."""

from dso.policy.rnn_policy import RNNPolicy as BaseRNNPolicy
from dso.program import Program


class RNNPolicyFixed(BaseRNNPolicy):
    """
    Fixed RNN policy that handles library initialization properly.
    
    This version checks if Program.library is initialized before accessing it,
    and provides a fallback mechanism for MIMO support.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with library check."""
        
        # Store original kwargs for later initialization if needed
        self._init_args = args
        self._init_kwargs = kwargs
        self._initialized = False
        
        # Try to initialize normally if library is available
        if Program.library is not None:
            super().__init__(*args, **kwargs)
            self._initialized = True
        else:
            # Defer initialization - store parameters
            self.prior = args[0] if len(args) > 0 else kwargs.get('prior')
            self.state_manager = args[1] if len(args) > 1 else kwargs.get('state_manager')
            
            # Set defaults to avoid errors
            self.n_choices = None
            self.max_length = kwargs.get('max_length', 30)
            self.batch_size = kwargs.get('batch_size', 100)
            self.debug = kwargs.get('debug', 0)
            
    def _complete_initialization(self):
        """Complete initialization once library is available."""
        
        if not self._initialized and Program.library is not None:
            # Now we can properly initialize
            super().__init__(*self._init_args, **self._init_kwargs)
            self._initialized = True
            
    def sample(self, *args, **kwargs):
        """Sample with initialization check."""
        
        # Ensure we're fully initialized before sampling
        self._complete_initialization()
        
        if not self._initialized:
            raise RuntimeError("Cannot sample: Program.library not initialized")
            
        return super().sample(*args, **kwargs)
        
    def compute_neg_log_likelihood(self, *args, **kwargs):
        """Compute NLL with initialization check."""
        
        # Ensure we're fully initialized
        self._complete_initialization()
        
        if not self._initialized:
            raise RuntimeError("Cannot compute NLL: Program.library not initialized")
            
        return super().compute_neg_log_likelihood(*args, **kwargs)


# Monkey patch to replace the original RNNPolicy
def patch_rnn_policy():
    """Patch the RNN policy to use the fixed version."""
    import dso.policy.policy as policy_module
    
    # Replace in the make_policy function
    original_make_policy = policy_module.make_policy
    
    def make_policy_patched(prior, state_manager, policy_type, **config_policy):
        """Patched factory function for Policy object."""
        
        if policy_type == "rnn":
            # Use the fixed version
            policy_class = RNNPolicyFixed
        elif policy_type == "modular":
            from dso.core.modular_policy import ModularRNNPolicy
            policy_class = ModularRNNPolicy
        else:
            # Custom policy import
            from dso.utils import import_custom_source
            policy_class = import_custom_source(policy_type)
            assert issubclass(policy_class, policy_module.Policy), \
                    f"Custom policy {policy_class} must subclass dso.policy.Policy."
            
        policy = policy_class(prior, state_manager, **config_policy)
        return policy
    
    # Replace the function
    policy_module.make_policy = make_policy_patched
    
    return True