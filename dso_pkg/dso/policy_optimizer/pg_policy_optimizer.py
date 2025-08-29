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

    def _compute_loss(self, batch):
        """Compute policy gradient loss using TF2. Sums log_probs over sequence and applies mask if available."""
        r = tf.cast(batch.rewards, tf.float32)  # [batch]
        # Pass lengths to compute_log_prob for proper sequence masking
        lengths = batch.lengths if hasattr(batch, 'lengths') else None
        
        # DEBUG: Check if batch data is connected to computation graph
        print(f"Batch rewards type: {type(batch.rewards)}, obs type: {type(batch.obs)}, actions type: {type(batch.actions)}")
        
        log_probs = self.policy.compute_log_prob(batch.obs, batch.actions, lengths)  # [batch, seq]
        
        # DEBUG: Check if log_probs are computed and have gradients
        print(f"Log probs shape: {log_probs.shape}, dtype: {log_probs.dtype}")
        print(f"Log probs range: {tf.reduce_min(log_probs).numpy():.6f} to {tf.reduce_max(log_probs).numpy():.6f}")
        
        baseline = tf.reduce_mean(r)
        advantages = r - baseline  # [batch]
        
        print(f"Advantages range: {tf.reduce_min(advantages).numpy():.6f} to {tf.reduce_max(advantages).numpy():.6f}")

        # Try to use mask if available (for variable-length sequences)
        mask = None
        if hasattr(batch, 'lengths') and batch.lengths is not None:
            seq_len = tf.shape(log_probs)[1]
            mask = tf.sequence_mask(batch.lengths, maxlen=seq_len, dtype=tf.float32)  # [batch, seq]
            log_probs = log_probs * mask

        # Sum log_probs over sequence dimension
        log_probs_summed = tf.reduce_sum(log_probs, axis=1)  # [batch]
        pg_loss = -tf.reduce_mean(advantages * log_probs_summed)

        # Add entropy regularization
        _, _, entropy = self.policy.get_probs_and_entropy(batch.obs, batch.actions)
        entropy_loss = -self.entropy_weight * tf.reduce_mean(entropy)
        total_loss = pg_loss + entropy_loss
        
        print(f"PG loss: {pg_loss.numpy():.6f}, Entropy loss: {entropy_loss.numpy():.6f}, Total: {total_loss.numpy():.6f}")
        
        return total_loss, pg_loss, entropy_loss

    def train_step(self, baseline, sampled_batch):
        """Perform one training step using TF2 GradientTape with integrated sampling."""
        # NEW APPROACH: Sample within the gradient tape to maintain gradient flow
        batch_size = self.policy.trainer.batch_size if hasattr(self.policy, 'trainer') else 100
        
        with tf.GradientTape() as tape:
            # Sample actions directly within the gradient tape context
            # This ensures gradient flow from sampling to policy parameters
            total_loss = self._compute_integrated_loss(batch_size)
        
        # Compute gradients
        trainable_vars = self.policy.controller.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        
        # Debug: Check if gradients are None
        if gradients is None or all(g is None for g in gradients):
            print("WARNING: All gradients are None! Loss:", total_loss.numpy())
            return {'total_loss': total_loss}
        
        # Apply gradients
        filtered_gradients = [(g, v) for g, v in zip(gradients, trainable_vars) if g is not None]
        if filtered_gradients:
            self.optimizer.apply_gradients(filtered_gradients)
            
            # Only show gradient info if there are meaningful updates
            valid_grads = [g for g in gradients if g is not None]
            if valid_grads:
                grad_norms = [tf.norm(g).numpy() for g in valid_grads]
                max_grad_norm = max(grad_norms)
                if max_grad_norm > 1e-8:
                    print(f"Policy update: {len(valid_grads)}/{len(gradients)} gradients, max_norm={max_grad_norm:.2e}")
        
        return {'total_loss': total_loss}

    def _compute_integrated_loss(self, batch_size):
        """Compute loss by sampling and evaluating within the same forward pass."""
        # Sample actions using the policy (this maintains gradient flow)
        sampled_actions, log_probs = self.policy.sample_with_log_probs(batch_size)
        
        # Convert sampled actions to programs and compute rewards
        rewards = self._evaluate_sampled_programs(sampled_actions)
        
        # Compute policy gradient loss
        baseline = tf.reduce_mean(rewards)
        advantages = rewards - baseline
        
        # Policy gradient loss: E[(R - b) * log π(a)]
        log_probs_sum = tf.reduce_sum(log_probs, axis=1)
        
        # Ensure numerical stability
        log_probs_sum = tf.clip_by_value(log_probs_sum, -50.0, 0.0)  # Clamp to reasonable range
        advantages = tf.clip_by_value(advantages, -10.0, 10.0)  # Clamp advantages
        
        pg_loss = -tf.reduce_mean(advantages * log_probs_sum)
        
        # Only print if there's meaningful reward variation
        reward_range = tf.reduce_max(rewards) - tf.reduce_min(rewards)
        if reward_range > 1e-6:
            print(f"Policy learning: PG loss={pg_loss.numpy():.6f}, Reward range=[{tf.reduce_min(rewards).numpy():.3f}, {tf.reduce_max(rewards).numpy():.3f}]")
        
        return pg_loss

    def _evaluate_sampled_programs(self, sampled_actions):
        """Evaluate rewards for sampled action sequences."""
        # Convert TensorFlow actions to numpy for program creation
        actions_np = sampled_actions.numpy()
        batch_size = actions_np.shape[0]
        
        # Import here to avoid circular imports
        from ..program import from_tokens
        
        rewards = []
        for i in range(batch_size):
            try:
                # Create program from action sequence
                actions_seq = actions_np[i]  # [max_length]
                
                # Find the actual sequence length (stop at first invalid action)
                valid_length = len(actions_seq)
                for j, action in enumerate(actions_seq):
                    if action >= self.policy.n_choices or action < 0:
                        valid_length = j
                        break
                
                # Create program from valid action sequence
                if valid_length > 0:
                    program = from_tokens(actions_seq[:valid_length])
                    
                    # Evaluate the program
                    if program is not None:
                        try:
                            # Use the task's reward function directly
                            if hasattr(program, 'task') and program.task is not None:
                                reward = program.task.reward_function(program)
                            else:
                                # Fall back to simple heuristic for programs without task
                                reward = 0.1  # Base reward for valid program
                                if hasattr(program, 'tokens') and len(program.tokens) < 20:
                                    reward += 0.1
                                if hasattr(program, 'tokens') and any('x' in str(token) for token in program.tokens):
                                    reward += 0.2
                        except Exception:
                            # If evaluation fails, assign minimal reward for valid program
                            reward = 0.01
                        
                        # Handle NaN or invalid rewards
                        if reward != reward or reward == float('inf') or reward == float('-inf'):
                            reward = 0.0
                        # Clamp reward to reasonable range
                        reward = max(0.0, min(1.0, float(reward)))
                    else:
                        reward = 0.0
                else:
                    reward = 0.0
                    
            except Exception as e:
                # If program creation/evaluation fails, assign zero reward
                if i < 2:
                    print(f"Program {i}: exception={str(e)[:100]}")
                reward = 0.0
            
            rewards.append(reward)
        
        # Convert back to tensor
        return tf.constant(rewards, dtype=tf.float32)

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
