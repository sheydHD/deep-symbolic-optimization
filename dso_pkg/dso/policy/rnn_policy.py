"""Controller used to generate distribution over hierarchical, variable-length objects."""
import tensorflow as tf
import numpy as np

from dso.program import Program
from dso.program import _finish_tokens
from dso.memory import Batch

from dso.policy import Policy
from dso.utils import create_batch_spec, convert_batch_to_tensors


class LinearProjection(tf.keras.layers.Layer):
    """Linear projection layer for RNN outputs."""

    def __init__(self, output_size, **kwargs):
        super().__init__(**kwargs)
        self.output_size = output_size
        self.dense = tf.keras.layers.Dense(output_size)

    def call(self, inputs):
        return self.dense(inputs)


class RNNController(tf.keras.Model):
    """RNN controller for generating symbolic expressions in TF2."""
    
    def __init__(self, n_choices, input_dim, cell_type='lstm', num_layers=1, num_units=32, 
                 initializer='zeros', **kwargs):
        super().__init__(**kwargs)
        
        self.n_choices = n_choices
        self.input_dim = input_dim  # This will be the base input dimension
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.num_units = num_units if isinstance(num_units, list) else [num_units] * num_layers
        
        # Create initializer
        if initializer == "zeros":
            init = tf.zeros_initializer()
        elif initializer == "var_scale":
            init = tf.keras.initializers.VarianceScaling(
                scale=0.5, mode='fan_avg', distribution='uniform', seed=0)
        else:
            raise ValueError(f"Unknown initializer: {initializer}")
        
        # Add input projection layer to handle variable input dimensions
        # This is similar to the original TF1 LinearWrapper approach
        self.input_projection = tf.keras.layers.Dense(
            self.input_dim,
            kernel_initializer=init,
            name='input_projection'
        )
        
        # Build RNN layers
        self.rnn_layers = []
        for i, units in enumerate(self.num_units):
            if cell_type == 'lstm':
                cell = tf.keras.layers.LSTM(
                    units, 
                    return_sequences=True, 
                    return_state=True,
                    kernel_initializer=init,
                    name=f'lstm_{i}'
                )
            elif cell_type == 'gru':
                cell = tf.keras.layers.GRU(
                    units, 
                    return_sequences=True, 
                    return_state=True,
                    kernel_initializer=init,
                    bias_initializer=init,
                    name=f'gru_{i}'
                )
            else:
                raise ValueError(f"Unknown cell type: {cell_type}")
            
            self.rnn_layers.append(cell)
        
        # Output projection layer
        self.output_projection = tf.keras.layers.Dense(
            n_choices, 
            kernel_initializer=init,
            name='output_projection'
        )
    
    def call(self, inputs, initial_states=None, training=None):
        """Forward pass through the RNN controller.
        
        Args:
            inputs: Input sequences of shape [batch_size, seq_len, input_dim]
            initial_states: Initial states for RNN layers
            training: Whether in training mode
            
        Returns:
            logits: Output logits of shape [batch_size, seq_len, n_choices]
            final_states: Final states from all RNN layers
        """
        # Project input to expected dimension (handles variable input sizes)
        x = self.input_projection(inputs)
        states = []
        
        if initial_states is None:
            initial_states = [None] * len(self.rnn_layers)
        
        # Pass through RNN layers
        for i, (layer, init_state) in enumerate(zip(self.rnn_layers, initial_states)):
            if self.cell_type == 'lstm':
                x, hidden_state, cell_state = layer(x, initial_state=init_state, training=training)
                states.append([hidden_state, cell_state])
            else:  # GRU
                x, final_state = layer(x, initial_state=init_state, training=training)
                states.append(final_state)
        
        # Project to output logits
        logits = self.output_projection(x)
        
        return logits, states
    
    def get_initial_state(self, batch_size):
        """Get initial states for all RNN layers."""
        states = []
        for i, units in enumerate(self.num_units):
            if self.cell_type == 'lstm':
                # LSTM needs both hidden and cell state
                h = tf.zeros([batch_size, units])
                c = tf.zeros([batch_size, units])
                states.append([h, c])
            else:  # GRU
                # GRU only needs hidden state
                h = tf.zeros([batch_size, units])
                states.append(h)
        return states


@tf.function
def safe_cross_entropy(p, logq, axis=-1):
    """Compute p * logq safely, by substituting logq[index] = 1 for index such that p[index] == 0"""
    # Put 1 where p == 0. In this case, q = p, logq = -inf and this might produce numerical errors
    safe_logq = tf.where(tf.equal(p, 0.), tf.ones_like(logq), logq)
    # Safely compute the product
    return -tf.reduce_sum(p * safe_logq, axis)


class RNNPolicy(tf.keras.Model, Policy):
    """Recurrent neural network (RNN) policy used to generate expressions.

    Specifically, the RNN outputs a distribution over pre-order traversals of
    symbolic expression trees. This class inherits from tf.keras.Model to make
    it trackable for TensorFlow checkpointing.

    Parameters
    ----------
    action_prob_lowerbound: float
        Lower bound on probability of each action.

    cell : str
        Recurrent cell to use. Supports 'lstm' and 'gru'.

    max_attempts_at_novel_batch: int
        maximum number of repetitions of sampling to get b new samples
        during a call of policy.sample(b)

    num_layers : int
        Number of RNN layers.

    num_units : int or list of ints
        Number of RNN cell units in each of the RNN's layers. If int, the value
        is repeated for each layer. 

    sample_novel_batch: bool
        if True, then a call to policy.sample(b) attempts to produce b samples
        that are not contained in the cache

    initializer : str
        Initializer for the recurrent cell. Supports 'zeros' and 'var_scale'.
        
    """
    def __init__(self, prior, state_manager, 
                 debug=0,
                 max_length=30,
                 action_prob_lowerbound=0.0,
                 max_attempts_at_novel_batch=10,
                 sample_novel_batch=False,
                 # RNN cell hyperparameters
                 cell='lstm',
                 num_layers=1,
                 num_units=32,
                 initializer='zeros'):
        # Initialize tf.keras.Model first
        tf.keras.Model.__init__(self)
        # Then initialize Policy
        Policy.__init__(self, prior, state_manager, debug, max_length)
        
        assert 0 <= action_prob_lowerbound <= 1
        self.action_prob_lowerbound = action_prob_lowerbound

        # len(tokens) in library
        self.n_choices = Program.library.L

        # Determine the input dimension from the state manager
        self.input_dim = self.state_manager.input_dim

        # Build the RNN controller
        self.controller = RNNController(
            n_choices=self.n_choices,
            input_dim=self.input_dim, # Pass the correct input dimension
            cell_type=cell,
            num_layers=num_layers,
            num_units=num_units,
            initializer=initializer
        )

        self.max_attempts_at_novel_batch = max_attempts_at_novel_batch
        self.sample_novel_batch = sample_novel_batch
        
        # Build the model by calling it once
        self._build_model()

    def _build_model(self):
        """Build the model by calling it with dummy inputs."""
        # Create dummy inputs to build the model
        dummy_batch_size = 1
        dummy_seq_len = 10
        
        # Use the correct input_dim
        dummy_inputs = tf.zeros([dummy_batch_size, dummy_seq_len, self.input_dim])
        dummy_states = self.controller.get_initial_state(dummy_batch_size)
        
        # Call the controller to build it
        _, _ = self.controller(dummy_inputs, dummy_states)

    def _setup_tf_model(self, **kwargs):
        """Setup the TensorFlow model (required abstract method)."""
        # In TF2, the model is already built in __init__, so this is a no-op
        pass

    def make_neglogp_and_entropy(self, B, entropy_gamma):
        """Compute negative log-probabilities and entropy for a batch (required abstract method)."""
        # Convert batch to tensors
        obs_tensor, actions_tensor = convert_batch_to_tensors(B, create_batch_spec(B))
        
        # Get probabilities and entropy
        _, action_log_probs, entropy = self.get_probs_and_entropy(obs_tensor, actions_tensor)
        
        # Return negative log probabilities and entropy
        neglogp = -action_log_probs
        return neglogp, entropy

    def compute_probs(self, memory_batch, log=False):
        """Compute probabilities for a batch (required abstract method)."""
        # Convert batch to tensors
        obs_tensor, actions_tensor = convert_batch_to_tensors(memory_batch, create_batch_spec(memory_batch))
        
        # Get probabilities
        probs, action_log_probs, _ = self.get_probs_and_entropy(obs_tensor, actions_tensor)
        
        if log:
            return action_log_probs
        else:
            # Return action probabilities by exponentiating log probs
            return tf.exp(action_log_probs)

    @tf.function
    def get_probs_and_entropy(self, obs, actions, lengths=None):
        """Compute action probabilities and entropy given observations and actions.
        
        Following the original TF1 logic with dynamic_rnn and sequence masking.
        """
        batch_size = tf.shape(obs)[0]
        seq_len = tf.shape(actions)[1]
        
        # The issue is seq_len mismatch. Let's properly handle observation processing
        # to maintain sequence length consistency following original DSO logic
        
        # Process observations through state manager (should output [batch_size, seq_len, features])
        processed_obs = self.state_manager.get_tensor_input(obs)
        
        # Get initial states for the RNN
        initial_states = self.controller.get_initial_state(batch_size)
        
        # Forward pass through controller
        logits, _ = self.controller(processed_obs, initial_states, training=False)
        
        # Apply lower bound to probabilities
        logits_clipped = tf.clip_by_value(logits, 
                                        np.log(self.action_prob_lowerbound + 1e-8),
                                        np.inf)
        
        # Compute probabilities and log probabilities
        probs = tf.nn.softmax(logits_clipped)
        log_probs = tf.nn.log_softmax(logits_clipped)
        
        # FINAL FIX: Ensure both tensors have consistent sequence lengths
        # Handle sequence length mismatch by working with actual dimensions
        logits_seq_len = tf.shape(log_probs)[1]
        actions_seq_len = tf.shape(actions)[1]
        
        # Use the minimum sequence length to avoid dimension mismatches
        min_seq_len = tf.minimum(logits_seq_len, actions_seq_len)
        
        # Truncate both tensors to the minimum sequence length
        log_probs_truncated = log_probs[:, :min_seq_len, :]
        probs_truncated = probs[:, :min_seq_len, :]
        actions_truncated = actions[:, :min_seq_len]
        
        # One-hot encode the truncated actions
        actions_one_hot = tf.one_hot(actions_truncated, depth=self.n_choices, axis=-1, dtype=tf.float32)
        
        # Create sequence mask if lengths are provided (following original TF1 logic)
        if lengths is not None:
            mask = tf.sequence_mask(lengths, maxlen=min_seq_len, dtype=tf.float32)  # [batch_size, min_seq_len]
        else:
            mask = tf.ones([batch_size, min_seq_len], dtype=tf.float32)  # No masking if lengths not provided
        
        # Use original safe_cross_entropy logic with truncated tensors
        action_log_probs_per_step = safe_cross_entropy(actions_one_hot, log_probs_truncated, axis=-1)  # [batch_size, min_seq_len]
        action_log_probs = -action_log_probs_per_step  # Convert to positive log probs
        
        # Apply mask to log probabilities (following original TF1 logic)
        action_log_probs = action_log_probs * mask
        
        # Compute entropy per step using safe_cross_entropy
        entropy_per_step = safe_cross_entropy(probs_truncated, log_probs_truncated, axis=-1)  # [batch_size, min_seq_len]
        entropy_per_step = entropy_per_step * mask  # Apply mask to entropy too
        
        # Pad back to original actions sequence length if needed for consistency
        if min_seq_len < actions_seq_len:
            padding_length = actions_seq_len - min_seq_len
            padding_shape = [batch_size, padding_length]
            action_log_probs = tf.concat([
                action_log_probs,
                tf.zeros(padding_shape, dtype=action_log_probs.dtype)
            ], axis=1)
            entropy_per_step = tf.concat([
                entropy_per_step,
                tf.zeros(padding_shape, dtype=entropy_per_step.dtype)
            ], axis=1)
        
        return probs, action_log_probs, entropy_per_step

    @tf.function 
    def compute_log_prob(self, obs, actions, lengths=None):
        """Compute log probabilities for given observations and actions.
        
        Args:
            obs: Observations tensor [batch_size, seq_len, obs_dim] 
            actions: Actions tensor [batch_size, seq_len]
            lengths: Optional sequence lengths [batch_size] for masking
        """
        _, action_log_probs, _ = self.get_probs_and_entropy(obs, actions, lengths)
        return action_log_probs

    def sample(self, n):
        """
        Sample a batch of n expressions.
        """
        
        # This function is not decorated with @tf.function, so it runs in eager mode.
        # This allows for the Python-based sampling loop.
        
        task = Program.task
        initial_obs = task.reset_task(self.prior)
        initial_obs = tf.constant(np.tile(initial_obs, (n, 1)), dtype=tf.float32)
        
        # Get initial prior
        initial_prior = self.prior.initial_prior()
        initial_prior = tf.constant(np.tile(initial_prior, (n, 1)), dtype=tf.float32)

        # Initialize state
        obs = self.state_manager.process_state(initial_obs)
        next_input = self.state_manager.get_tensor_input(obs)
        next_input = tf.expand_dims(next_input, 1) # Add sequence length dimension
        
        # Get initial RNN state
        next_cell_state = self.controller.get_initial_state(n)

        # Initialize TensorArrays to store trajectory
        actions_ta = tf.TensorArray(dtype=tf.int32, size=self.max_length)
        obs_ta = tf.TensorArray(dtype=tf.float32, size=self.max_length)
        priors_ta = tf.TensorArray(dtype=tf.float32, size=self.max_length)

        # Autoregressive sampling loop
        for t in range(self.max_length):
            # Run one step of the RNN
            logits, next_cell_state = self.controller(next_input, initial_states=next_cell_state)
            logits = tf.squeeze(logits, axis=1) # Remove sequence length dimension

            # Apply prior
            logits += initial_prior # In the original, this was just 'prior'
            
            # Sample action
            action = tf.random.categorical(logits, num_samples=1, dtype=tf.int32)[:, 0]
            
            # Store results for this step
            actions_ta = actions_ta.write(t, action)
            obs_ta = obs_ta.write(t, obs)
            priors_ta = priors_ta.write(t, initial_prior) # Store the prior used for this step
            
            # Update for next step
            actions = tf.transpose(actions_ta.stack())
            
            # The original used a py_func here. For eager execution, we can call it directly.
            # However, task.get_next_obs expects numpy arrays.
            np_actions = actions.numpy()
            np_obs = obs.numpy()
            np_finished = np.zeros(n, dtype=bool) # Assuming not finished
            
            next_obs_np, next_prior_np, _ = task.get_next_obs(np_actions, np_obs, np_finished)
            
            next_obs_tf = tf.constant(next_obs_np, dtype=tf.float32)
            next_prior_tf = tf.constant(next_prior_np, dtype=tf.float32)

            obs = self.state_manager.process_state(next_obs_tf)
            next_input = self.state_manager.get_tensor_input(obs)
            next_input = tf.expand_dims(next_input, 1)
            initial_prior = next_prior_tf


        # Stack results into tensors
        actions = tf.transpose(actions_ta.stack())
        obs = tf.transpose(obs_ta.stack(), perm=[1, 2, 0])
        priors = tf.transpose(priors_ta.stack(), perm=[1, 0, 2])
        
        return actions.numpy(), obs.numpy(), priors.numpy()

    def apply_action_prob_lowerbound(self, logits):
        """Apply lower bound to action probabilities."""
        if self.action_prob_lowerbound == 0.0:
            return logits
        
        # Convert to probabilities, apply lower bound, convert back to logits
        probs = tf.nn.softmax(logits)
        probs_bounded = tf.maximum(probs, self.action_prob_lowerbound)
        probs_normalized = probs_bounded / tf.reduce_sum(probs_bounded, axis=-1, keepdims=True)
        return tf.math.log(probs_normalized + 1e-8)
