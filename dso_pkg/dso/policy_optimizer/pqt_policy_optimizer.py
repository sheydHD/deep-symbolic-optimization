import tensorflow as tf
import numpy as np
from dso.policy_optimizer import PolicyOptimizer
from dso.policy import Policy
from dso.memory import Batch
from dso.utils import create_batch_spec, convert_batch_to_tensors

class PQTPolicyOptimizer(PolicyOptimizer):
    """Priority Queue Training policy gradient policy optimizer.

    Parameters
    ----------
    pqt_k : int
        Size of priority queue.

    pqt_batch_size : int
        Size of batch to sample (with replacement) from priority queue.

    pqt_weight : float
        Coefficient for PQT loss function.

    pqt_use_pg : bool
        Use policy gradient loss when using PQT?   
        
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
            pqt_weight : float = 200.0,
            pqt_use_pg: bool = False) -> None:
        
        self.pqt_k = pqt_k
        self.pqt_batch_size = pqt_batch_size
        self.pqt_weight = pqt_weight
        self.pqt_use_pg = pqt_use_pg
        
        # Call parent constructor (TF2.x version)
        super().__init__(policy, debug, summary, logdir, optimizer, learning_rate, entropy_weight, entropy_gamma)
        
        # Create optimizer (TF2.x style)
        if optimizer == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    def _set_loss(self):
        """Set the loss function for PQT (required abstract method).
        
        In TF2.x, we don't pre-define loss computation with placeholders.
        Loss is computed dynamically in train_step.
        """
        # This method is required by the abstract base class but is a no-op for TF2.x
        pass

    def _compute_loss(self, obs, actions, rewards):
        """Compute the loss function (required abstract method).
        
        In TF2.x implementation, actual loss computation happens in train_step.
        This method exists to satisfy the abstract interface.
        """
        # This method is required by the abstract base class
        # For PQT, the actual loss computation is more complex and happens in train_step
        log_probs = self.policy.compute_log_prob(obs, actions)
        baseline = tf.reduce_mean(rewards)
        advantages = rewards - baseline
        
        # Handle tensor shapes
        advantages = tf.cast(advantages, tf.float32)
        log_probs = tf.cast(log_probs, tf.float32)
        
        if len(tf.shape(log_probs)) > 1:
            log_probs = tf.reduce_sum(log_probs, axis=1)
            
        loss = -tf.reduce_mean(advantages * log_probs)
        return loss

    def _preppend_to_summary(self):
        """Add additional fields for the tensorflow summary.
        
        In TF2.x, summaries are written directly in train_step.
        """
        # This method is required by the abstract base class but is a no-op for TF2.x
        pass

    def train_step(self, baseline, sampled_batch, pqt_batch):
        """Perform one training step using TF2 GradientTape.
        
        Maintains the original TF1.x interface: train_step(baseline, sampled_batch, pqt_batch)
        """
        # Convert batches to tensors if needed
        if not isinstance(sampled_batch.actions, tf.Tensor):
            sampled_batch = convert_batch_to_tensors(sampled_batch)
        if pqt_batch is not None and not isinstance(pqt_batch.actions, tf.Tensor):
            pqt_batch = convert_batch_to_tensors(pqt_batch)
        
        # Perform training step with GradientTape
        with tf.GradientTape() as tape:
            # Compute standard policy gradient loss
            r = sampled_batch.rewards
            log_probs = self.policy.compute_log_prob(sampled_batch.obs, sampled_batch.actions)
            _, _, entropy = self.policy.get_probs_and_entropy(sampled_batch.obs, sampled_batch.actions, sampled_batch.lengths)
            
            baseline_val = tf.reduce_mean(r)
            advantages = r - baseline_val
            
            # Cast to ensure compatible types
            advantages = tf.cast(advantages, tf.float32)
            log_probs = tf.cast(log_probs, tf.float32)
            
            # Handle tensor shape: sum log_probs across sequence if needed
            if len(tf.shape(log_probs)) > 1:
                log_probs_sum = tf.reduce_sum(log_probs, axis=1)
            else:
                log_probs_sum = log_probs
                
            pg_loss = -tf.reduce_mean(advantages * log_probs_sum)
            
            # Compute PQT loss (original functionality)
            if pqt_batch is not None:
                pqt_log_probs = self.policy.compute_log_prob(pqt_batch.obs, pqt_batch.actions)
                pqt_log_probs = tf.cast(pqt_log_probs, tf.float32)
                
                # Handle tensor shape for PQT loss too
                if len(tf.shape(pqt_log_probs)) > 1:
                    pqt_neglogp = -tf.reduce_sum(pqt_log_probs, axis=1)
                else:
                    pqt_neglogp = -pqt_log_probs
                    
                pqt_loss = self.pqt_weight * tf.reduce_mean(pqt_neglogp)
            else:
                pqt_loss = 0.0
            
            # Entropy loss (same as original)
            # Entropy loss: first sum over sequence, then average over batch  
            entropy_per_sequence = tf.reduce_sum(entropy, axis=1)  # Sum over masked sequence
            entropy_loss = -self.entropy_weight * tf.reduce_mean(entropy_per_sequence)
            
            # Total loss (preserve original logic)
            if self.pqt_use_pg:
                total_loss = pg_loss + pqt_loss + entropy_loss
            else:
                total_loss = pqt_loss + entropy_loss

        # Compute and apply gradients
        gradients = tape.gradient(total_loss, self.policy.controller.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy.controller.trainable_variables))
        
        # Write summaries (TF2.x style)
        if self.summary and self.logdir:
            with tf.summary.create_file_writer(self.logdir).as_default():
                tf.summary.scalar("pqt_loss", pqt_loss)
                tf.summary.scalar("pg_loss", pg_loss)
                tf.summary.scalar("entropy_loss", entropy_loss)
                tf.summary.scalar("total_loss", total_loss)
        
        # Return empty dict to maintain interface compatibility
        # (Original returned summaries, but we handle that internally now)
        return {}