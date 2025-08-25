import tensorflow as tf
from dso.policy_optimizer import PolicyOptimizer
from dso.policy import Policy
from dso.utils import create_batch_spec, convert_batch_to_tensors

class PQTPolicyOptimizer(PolicyOptimizer):
    """Priority Queue Training policy gradient policy optimizer.

    Parameters
    ----------
    policy : Policy
        The policy to optimize.
        
    pqt_k : int
        Size of priority queue.

    pqt_batch_size : int
        Size of batch to sample (with replacement) from priority queue.

    pqt_weight : float
        Coefficient for PQT loss function.

    pqt_use_pg : bool
        Use policy gradient loss when using PQT?   
        
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
            # PQT hyperparameters
            pqt_k : int = 10,
            pqt_batch_size : int = 1,
            pqt_weight : float = 0.1,
            pqt_use_pg : bool = False,
            pqt_mix_with_top : bool = True) -> None:
        
        # PQT specific parameters
        self.pqt_k = pqt_k
        self.pqt_batch_size = pqt_batch_size
        self.pqt_weight = pqt_weight
        self.pqt_use_pg = pqt_use_pg
        self.pqt_mix_with_top = pqt_mix_with_top
        
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
        
        # Initialize priority queue (as Python list for simplicity)
        self.priority_queue = []

    def update_priority_queue(self, batch):
        """Update priority queue with new batch."""
        # Add new samples to queue
        for i in range(len(batch.rewards)):
            sample = {
                'actions': batch.actions[i],
                'obs': batch.obs[i], 
                'priors': batch.priors[i],
                'lengths': batch.lengths[i],
                'reward': batch.rewards[i],
                'on_policy': batch.on_policy[i]
            }
            self.priority_queue.append(sample)
        
        # Sort by reward (descending) and keep top k
        self.priority_queue.sort(key=lambda x: x['reward'], reverse=True)
        self.priority_queue = self.priority_queue[:self.pqt_k]

    def _set_loss(self):
        """Set the loss function for PQT (required abstract method)."""
        # For PQT, loss is computed dynamically in _compute_loss
        # This method is required by the abstract base class but is a no-op for PQT
        pass

    @tf.function
    def _compute_loss(self, batch, pqt_batch=None):
        """Compute PQT loss using TF2."""
        # Get rewards from batch
        r = batch.rewards
        
        # Compute log probabilities and entropy
        log_probs = self.policy.compute_log_prob(batch.obs, batch.actions)
        _, _, entropy = self.policy.get_probs_and_entropy(batch.obs, batch.actions)
        
        # Standard policy gradient loss
        baseline = tf.reduce_mean(r)
        advantages = r - baseline
        # Ensure data types match for multiplication
        advantages = tf.cast(advantages, tf.float32)
        log_probs = tf.cast(log_probs, tf.float32)
        pg_loss = -tf.reduce_mean(advantages * log_probs)
        
        # PQT loss (if we have priority queue samples)
        pqt_loss = 0.0
        if pqt_batch is not None:
            pqt_log_probs = self.policy.compute_log_prob(pqt_batch.obs, pqt_batch.actions)
            pqt_advantages = pqt_batch.rewards - tf.cast(baseline, pqt_batch.rewards.dtype)
            pqt_loss = -tf.reduce_mean(pqt_advantages * pqt_log_probs)
        
        # Combine losses
        if self.pqt_use_pg:
            total_loss = pg_loss + self.pqt_weight * pqt_loss
        else:
            total_loss = self.pqt_weight * pqt_loss
            
        # Add entropy regularization
        entropy_loss = -self.entropy_weight * tf.reduce_mean(entropy)
        total_loss += entropy_loss
        
        return total_loss, pg_loss, pqt_loss, entropy_loss

    @tf.function
    def train_step(self, baseline, sampled_batch, pqt_batch=None):
        """Perform one training step using TF2 GradientTape."""
        with tf.GradientTape() as tape:
            total_loss, pg_loss, pqt_loss, entropy_loss = self._compute_loss(sampled_batch, pqt_batch)
        
        # Compute gradients
        gradients = tape.gradient(total_loss, self.policy.controller.trainable_variables)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.policy.controller.trainable_variables))
        
        return {
            'total_loss': total_loss,
            'pg_loss': pg_loss,
            'pqt_loss': pqt_loss,
            'entropy_loss': entropy_loss
        }

    def train(self, baseline, sampled_batch):
        """Train the policy with PQT algorithm."""
        # Convert batch to tensors if needed
        if not isinstance(sampled_batch.actions, tf.Tensor):
            sampled_batch = convert_batch_to_tensors(sampled_batch)
        
        # Update priority queue
        self.update_priority_queue(sampled_batch)
        
        # Sample from priority queue if it has enough samples
        pqt_batch = None
        if len(self.priority_queue) >= self.pqt_batch_size:
            # Sample from priority queue
            import random
            pqt_samples = random.choices(self.priority_queue, k=self.pqt_batch_size)
            
            # Convert to batch format
            from dso.memory import Batch
            import numpy as np
            
            pqt_batch = Batch(
                actions=tf.stack([s['actions'] for s in pqt_samples]),
                obs=tf.stack([s['obs'] for s in pqt_samples]),
                priors=tf.stack([s['priors'] for s in pqt_samples]), 
                lengths=tf.stack([s['lengths'] for s in pqt_samples]),
                rewards=tf.stack([s['reward'] for s in pqt_samples]),
                on_policy=tf.stack([s['on_policy'] for s in pqt_samples])
            )
        
        # Perform training step
        metrics = self.train_step(sampled_batch, pqt_batch)
        
        # Log metrics if summary is enabled
        if self.summary and self.logdir:
            with tf.summary.create_file_writer(self.logdir).as_default():
                tf.summary.scalar("total_loss", metrics['total_loss'])
                tf.summary.scalar("pg_loss", metrics['pg_loss'])
                tf.summary.scalar("pqt_loss", metrics['pqt_loss'])
                tf.summary.scalar("entropy_loss", metrics['entropy_loss'])
        
        return metrics
