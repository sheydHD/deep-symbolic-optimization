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
            # PPO hyperparameters (preserve original parameter names)
            ppo_clip_ratio : float = 0.2,
            ppo_n_iters : int = 10,
            ppo_n_mb : int = 4) -> None:
        super().__init__(policy, debug, summary, logdir, optimizer, learning_rate, entropy_weight, entropy_gamma)
        
        # Parameters specific for the algorithm (preserve original naming)
        self.ppo_clip_ratio = ppo_clip_ratio
        self.ppo_n_iters = ppo_n_iters
        self.ppo_n_mb = ppo_n_mb
        self.rng = np.random.RandomState(0)  # Used for PPO minibatch sampling
        
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
        _, _, entropy = self.policy.get_probs_and_entropy(batch.obs, batch.actions, batch.lengths)
        
        # Compute ratio for PPO
        ratio = tf.exp(current_log_probs - old_log_probs)
        
        # Baseline is the mean of current rewards
        baseline = tf.reduce_mean(r)
        advantages = r - baseline
        
        # PPO clipped surrogate loss (use original parameter name)
        surr1 = ratio * advantages
        surr2 = tf.clip_by_value(ratio, 1.0 - self.ppo_clip_ratio, 1.0 + self.ppo_clip_ratio) * advantages
        ppo_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
        
        # Add entropy regularization
        # Entropy loss: first sum over sequence, then average over batch
        entropy_per_sequence = tf.reduce_sum(entropy, axis=1)  # Sum over masked sequence  
        entropy_loss = -self.entropy_weight * tf.reduce_mean(entropy_per_sequence)
        
        total_loss = ppo_loss + entropy_loss
        
        return total_loss, ppo_loss, entropy_loss

    def train_step(self, baseline, sampled_batch):
        """Perform one training step using TF2 GradientTape.
        
        Maintains the original TF1.x interface: train_step(baseline, sampled_batch)
        Implements multiple PPO iterations and minibatching as per original.
        """
        # Convert batch to tensors if needed
        if not isinstance(sampled_batch.actions, tf.Tensor):
            from dso.utils import convert_batch_to_tensors
            sampled_batch = convert_batch_to_tensors(sampled_batch)
        
        n_samples = sampled_batch.rewards.shape[0]
        
        # Compute old log probabilities before updating policy
        old_log_probs = self.policy.compute_log_prob(sampled_batch.obs, sampled_batch.actions)
        old_log_probs = tf.stop_gradient(old_log_probs)  # Don't compute gradients through old policy
        
        # Perform multiple steps of minibatch training (preserve original PPO logic)
        indices = np.arange(n_samples)
        
        total_metrics = {'total_loss': 0.0, 'ppo_loss': 0.0, 'entropy_loss': 0.0}
        
        for ppo_iter in range(self.ppo_n_iters):
            self.rng.shuffle(indices)  # in-place
            # list of [ppo_n_mb] arrays
            minibatches = np.array_split(indices, self.ppo_n_mb)
            
            for i, mb in enumerate(minibatches):
                # Create minibatch
                from dso.memory import Batch
                sampled_batch_mb = Batch(
                    actions=tf.gather(sampled_batch.actions, mb),
                    obs=tf.gather(sampled_batch.obs, mb),
                    priors=tf.gather(sampled_batch.priors, mb),
                    lengths=tf.gather(sampled_batch.lengths, mb),
                    rewards=tf.gather(sampled_batch.rewards, mb),
                    on_policy=tf.gather(sampled_batch.on_policy, mb)
                )
                old_log_probs_mb = tf.gather(old_log_probs, mb)
                
                # Perform one gradient step on minibatch
                metrics = self._internal_train_step(sampled_batch_mb, old_log_probs_mb)
                
                # Accumulate metrics
                for key in total_metrics:
                    total_metrics[key] += metrics[key]
        
        # Average metrics
        num_steps = self.ppo_n_iters * self.ppo_n_mb
        for key in total_metrics:
            total_metrics[key] /= num_steps
        
        return total_metrics

    @tf.function
    def _internal_train_step(self, batch, old_log_probs):
        """Internal training step for one minibatch."""
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


