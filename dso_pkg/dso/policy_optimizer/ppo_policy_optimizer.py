import tensorflow as tf
import numpy as np
from dso.policy_optimizer import PolicyOptimizer
from dso.policy import Policy
from dso.memory import Batch

class PPOPolicyOptimizer(PolicyOptimizer):
    """Proximal policy optimization policy optimizer.

    Parameters
    ----------
    policy : Policy
        The policy to optimize.
        
    eps_clip : float
        Clip ratio to use for PPO.
        
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
            entropy_gamma : float = 1.0,
            # PPO hyperparameters
            eps_clip : float = 0.2) -> None:
        super().__init__(policy, debug, summary, logdir, optimizer, learning_rate, entropy_weight, entropy_gamma)
        
        # Parameters specific for the algorithm
        self.eps_clip = eps_clip
        
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
        """Set the loss function for PPO (required abstract method)."""
        # For PPO, loss is computed dynamically in _compute_loss
        # This method is required by the abstract base class but is a no-op for PPO
        pass

    @tf.function
    def _compute_loss(self, batch, old_log_probs):
        """Compute PPO loss using TF2."""
        # Get rewards from batch
        r = batch.rewards
        
        # Compute current log probabilities and entropy
        current_log_probs = self.policy.compute_log_prob(batch.obs, batch.actions)
        _, _, entropy = self.policy.get_probs_and_entropy(batch.obs, batch.actions)
        
        # Compute ratio for PPO
        ratio = tf.exp(current_log_probs - old_log_probs)
        
        # Baseline is the mean of current rewards
        baseline = tf.reduce_mean(r)
        advantages = r - baseline
        
        # PPO clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = tf.clip_by_value(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
        ppo_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
        
        # Add entropy regularization
        entropy_loss = -self.entropy_weight * tf.reduce_mean(entropy)
        
        total_loss = ppo_loss + entropy_loss
        
        return total_loss, ppo_loss, entropy_loss

    @tf.function
    def train_step(self, batch, old_log_probs):
        """Perform one training step using TF2 GradientTape."""
        with tf.GradientTape() as tape:
            total_loss, ppo_loss, entropy_loss = self._compute_loss(batch, old_log_probs)
        
        # Compute gradients
        gradients = tape.gradient(total_loss, self.policy.controller.trainable_variables)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.policy.controller.trainable_variables))
        
        return {
            'total_loss': total_loss,
            'ppo_loss': ppo_loss, 
            'entropy_loss': entropy_loss
        }

    def train(self, baseline, sampled_batch):
        """Train the policy with PPO algorithm."""
        # Convert batch to tensors if needed
        if not isinstance(sampled_batch.actions, tf.Tensor):
            from dso.utils import convert_batch_to_tensors
            sampled_batch = convert_batch_to_tensors(sampled_batch)
        
        # Get old log probabilities (for first iteration, use current)
        old_log_probs = self.policy.compute_log_prob(sampled_batch.obs, sampled_batch.actions)
        old_log_probs = tf.stop_gradient(old_log_probs)  # Don't compute gradients through old policy
        
        # Perform training step
        metrics = self.train_step(sampled_batch, old_log_probs)
        
        # Log metrics if summary is enabled
        if self.summary and self.logdir:
            with tf.summary.create_file_writer(self.logdir).as_default():
                tf.summary.scalar("total_loss", metrics['total_loss'])
                tf.summary.scalar("ppo_loss", metrics['ppo_loss']) 
                tf.summary.scalar("entropy_loss", metrics['entropy_loss'])
        
        return metrics
