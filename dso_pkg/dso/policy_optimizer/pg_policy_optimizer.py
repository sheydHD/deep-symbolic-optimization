import tensorflow as tf
from ..policy_optimizer import PolicyOptimizer
from ..policy import Policy

class PGPolicyOptimizer(PolicyOptimizer):
    """Vanilla policy gradient policy optimizer.

    Parameters
    ----------
    policy : Policy
        The policy to optimize.
        
    debug : int
        Debug level.
        
    summary : bool
        Whether to write summaries.
        
    logdir : str
        Directory for logging.
        
    optimizer : str
        Optimizer type ('adam', 'rmsprop', 'sgd').
        
    learning_rate : float
        Learning rate for optimization.
        
    entropy_weight : float
        Weight for entropy regularization.
        
    entropy_gamma : float
        Gamma for entropy decay.
    """
    
    def __init__(self,
            policy : Policy,
            debug : int = 0,
            summary : bool = False,
            logdir : str = None,
            # Optimizer hyperparameters
            optimizer : str = 'adam',
            learning_rate : float = 0.001,
            # Loss hyperparameters
            entropy_weight : float = 0.005,
            entropy_gamma : float = 1.0) -> None:
        super().__init__(policy, debug, summary, logdir, optimizer, learning_rate, entropy_weight, entropy_gamma)
        
        # Create optimizer
        if optimizer == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    def _set_loss(self):
        """Set the loss function for policy gradient (required abstract method)."""
        # For PG, loss is computed dynamically in _compute_loss
        # This method is required by the abstract base class but is a no-op for PG
        pass

    def _compute_loss(self, batch):
        """Compute loss from batch (required abstract method)."""
        # For PG, we use the batch-based approach in train_step
        # This method is required by the abstract base class but delegates to _compute_batch_loss
        baseline = tf.reduce_mean(tf.cast(batch.rewards, tf.float32))
        return self._compute_batch_loss(baseline, batch)







    def train_step(self, baseline, sampled_batch):
        """Modern TF2 approach with exact hybrid logic."""
        # Increment iteration counter (like hybrid)
        if not hasattr(self, 'iterations'):
            self.iterations = tf.Variable(0, dtype=tf.int64, name="iterations")
        self.iterations.assign_add(1)
        
        with tf.GradientTape() as tape:
            # Extract rewards (already filtered by trainer quantile logic)
            rewards = tf.cast(sampled_batch.rewards, tf.float32)
            baseline_tf = tf.cast(baseline, tf.float32)
            
            # Compute advantages: (R - b) exactly like hybrid
            advantages = rewards - baseline_tf
            
            # Compute log probabilities using the TF2 method with proper sequence lengths
            probs, log_probs, entropy = self.policy.get_probs_and_entropy(sampled_batch.obs, sampled_batch.actions, sampled_batch.lengths)
            
            # Sum log probs over sequence dimension to get neglogp
            neglogp = -tf.reduce_sum(log_probs, axis=1)  # Negative log prob like hybrid
            
            # Policy gradient loss: E[(R - b) * neglogp] (exactly like hybrid)
            pg_loss = tf.reduce_mean(advantages * neglogp)
            
            # Entropy loss (like hybrid): first sum over sequence, then average over batch
            entropy_per_sequence = tf.reduce_sum(entropy, axis=1)  # Sum over masked sequence
            entropy_loss = -self.entropy_weight * tf.reduce_mean(entropy_per_sequence)
            
            # Total loss (like hybrid: loss = entropy_loss + pg_loss)  
            total_loss = entropy_loss + pg_loss
        
        # Apply gradients using TF2 (modernized from sess.run(train_op))
        trainable_vars = self.policy.controller.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        filtered_gradients = [(g, v) for g, v in zip(gradients, trainable_vars) if g is not None]
        if filtered_gradients:
            self.optimizer.apply_gradients(filtered_gradients)
            # Debug: Check if learning is happening by monitoring gradient norms
            if hasattr(self, 'debug') and self.debug >= 2:
                grad_norms = [tf.norm(g).numpy() for g, _ in filtered_gradients]
                print(f"DEBUG: Gradient norms: {grad_norms[:3]}...")  # Show first 3
        
        return None  # Hybrid returns None
