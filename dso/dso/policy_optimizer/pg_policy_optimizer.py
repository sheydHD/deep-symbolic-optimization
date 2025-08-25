import tensorflow as tf
from dso.policy_optimizer import PolicyOptimizer
from dso.policy import Policy

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

    @tf.function
    def _compute_loss(self, batch):
        """Compute policy gradient loss using TF2."""
        # Get rewards from batch
        r = tf.cast(batch.rewards, tf.float32)  # Ensure float32 type
        
        # Compute log probabilities and entropy
        log_probs = self.policy.compute_log_prob(batch.obs, batch.actions)
        
        # Baseline is the mean of current rewards
        baseline = tf.reduce_mean(r)
        
        # Compute policy gradient loss
        advantages = r - baseline
        pg_loss = -tf.reduce_mean(advantages * log_probs)
        
        # Add entropy regularization
        _, _, entropy = self.policy.get_probs_and_entropy(batch.obs, batch.actions)
        entropy_loss = -self.entropy_weight * tf.reduce_mean(entropy)
        
        total_loss = pg_loss + entropy_loss
        
        return total_loss, pg_loss, entropy_loss

    @tf.function
    def train_step(self, baseline, sampled_batch):
        """Perform one training step using TF2 GradientTape."""
        with tf.GradientTape() as tape:
            total_loss, pg_loss, entropy_loss = self._compute_loss(sampled_batch)
        
        # Compute gradients
        gradients = tape.gradient(total_loss, self.policy.controller.trainable_variables)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.policy.controller.trainable_variables))
        
        return {
            'total_loss': total_loss,
            'pg_loss': pg_loss, 
            'entropy_loss': entropy_loss
        }

    def train(self, baseline, sampled_batch):
        """Train the policy with the given batch."""
        # Convert batch to tensors if needed
        if not isinstance(sampled_batch.actions, tf.Tensor):
            from dso.utils import convert_batch_to_tensors
            sampled_batch = convert_batch_to_tensors(sampled_batch)
        
        # Perform training step
        metrics = self.train_step(sampled_batch)
        
        # Log metrics if summary is enabled
        if self.summary and self.logdir:
            with tf.summary.create_file_writer(self.logdir).as_default():
                tf.summary.scalar("total_loss", metrics['total_loss'])
                tf.summary.scalar("pg_loss", metrics['pg_loss']) 
                tf.summary.scalar("entropy_loss", metrics['entropy_loss'])
        
        return metrics
