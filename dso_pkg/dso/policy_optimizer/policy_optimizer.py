from abc import ABC, abstractmethod

from dso.tf_config import tf
import numpy as np

from dso.program import Program
from dso.memory import Batch
from dso.utils import import_custom_source
from dso.policy.policy import Policy
from dso.utils import create_batch_spec, convert_batch_to_tensors

# Note: In TF2, we don't need placeholders, we pass data directly to functions
# summaries = tf.TensorArray  # Keep for type annotations

def make_policy_optimizer(policy, policy_optimizer_type, **config_policy_optimizer):
    """Factory function for policy optimizer object."""

    if policy_optimizer_type == "pg":
        from dso.policy_optimizer.pg_policy_optimizer import PGPolicyOptimizer
        policy_optimizer_class = PGPolicyOptimizer
    elif policy_optimizer_type == "pqt":
        from dso.policy_optimizer.pqt_policy_optimizer import PQTPolicyOptimizer
        policy_optimizer_class = PQTPolicyOptimizer
    elif policy_optimizer_type == "ppo":
        from dso.policy_optimizer.ppo_policy_optimizer import PPOPolicyOptimizer
        policy_optimizer_class = PPOPolicyOptimizer
    else:
        # Custom policy import
        policy_optimizer_class = import_custom_source(policy_optimizer_type)
        # Note: Should check if it's a subclass of PolicyOptimizer, not Policy
        assert hasattr(policy_optimizer_class, '__init__'), f"Custom policy optimizer {policy_optimizer_class} must be a valid class."
        
    policy_optimizer = policy_optimizer_class(policy,
                                              **config_policy_optimizer)

    return policy_optimizer

class PolicyOptimizer(tf.Module, ABC):
    """Abstract class for a policy optimizer. A policy optimizer is an 
    algorithm for optimizing the parameters of a parametrized policy.
    
    Inherits from tf.Module to make it trackable for TensorFlow checkpointing.

    To define a new optimizer, inherit from this class and add the following
    methods (look in _setup_policy_optimizer below):

        _set_loss() : Define the \\propto \\log(p(\tau|\theta)) loss for the method
        _preppend_to_summary() : Add additional fields for the tensorflow summary

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
        '''Parameters
        ----------
        policy : dso.policy.Policy
            Parametrized probability distribution over discrete objects

        debug : int
            Debug level, also used in learn(). 0: No debug. 1: Print shapes and
            number of parameters for each variable.

        summary : bool
            Write tensorboard summaries?

        logdir : str or None
            Directory for TensorBoard summaries.

        optimizer : str
            Optimizer to use. Supports 'adam', 'rmsprop', and 'sgd'.

        learning_rate : float
            Learning rate for optimizer.

        entropy_weight : float
            Coefficient for entropy bonus.

        entropy_gamma : float or None
            Gamma in entropy decay. None (or
            equivalently, 1.0) turns off entropy decay.
        '''    
        # Initialize tf.Module first
        tf.Module.__init__(self)
        
        self.policy = policy

        # Needed in _setup_optimizer
        self.debug = debug
        self.optimizer = optimizer
        self.learning_rate = learning_rate

        # Need in self.summary
        self.summary = summary
        self.logdir = logdir 
        self.n_choices = Program.library.L

        # Need in _init_loss_with_entropy
        self.entropy_weight = entropy_weight
        self.entropy_gamma = entropy_gamma

        # Iteration counter for summaries
        self.iterations = tf.Variable(0, dtype=tf.int64, name="iterations")


    def _init_loss_with_entropy(self) -> None:
        # Add entropy contribution to loss. The entropy regularizer does not
        # depend on the particular policy optimizer
        with tf.name_scope("losses"):

            self.neglogp, entropy = self.policy.make_neglogp_and_entropy(self.sampled_batch_ph, self.entropy_gamma)

            # Entropy loss
            self.entropy_loss = -self.entropy_weight * tf.reduce_mean(entropy, name="entropy_loss")
            loss = self.entropy_loss

            # self.loss is modified in the child object
            self.loss = loss


    def _set_loss(self) -> None:
        """Define the \\propto \\log(p(\tau|\theta)) loss for the method

        Returns
        -------
            None
        """
        # Default implementation - can be overridden by subclasses
        pass


    def _setup_optimizer(self):
        """ Setup the optimizer using TF2 eager execution
        """    
        def make_optimizer(name, learning_rate):
            if name == "adam":
                return tf.keras.optimizers.Adam(learning_rate=learning_rate)
            if name == "rmsprop":
                return tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.99)
            if name == "sgd":
                return tf.keras.optimizers.SGD(learning_rate=learning_rate)
            raise ValueError(f"Did not recognize optimizer '{name}'")

        # Create optimizer
        self.optimizer_instance = make_optimizer(name=self.optimizer, learning_rate=self.learning_rate)
        
        if self.debug >= 1:
            total_parameters = 0
            print("")
            # Get all trainable variables from the policy controller
            trainable_vars = self.policy.controller.trainable_variables
            for variable in trainable_vars:
                shape = variable.shape
                n_parameters = tf.size(variable).numpy()
                total_parameters += n_parameters
                print("Variable:    ", variable.name)
                print("  Shape:     ", shape)
                print("  Parameters:", n_parameters)
            print("Total parameters:", total_parameters)

    @tf.function
    def _compute_gradients(self, batch):
        """Compute gradients using TF2 GradientTape."""
        with tf.GradientTape() as tape:
            # Convert batch to tensors
            obs = tf.convert_to_tensor(batch.obs, dtype=tf.float32)
            actions = tf.convert_to_tensor(batch.actions, dtype=tf.int32)
            rewards = tf.convert_to_tensor(batch.rewards, dtype=tf.float32)
            
            # Compute loss
            loss = self._compute_loss(obs, actions, rewards)
        
        # Compute gradients
        gradients = tape.gradient(loss, self.policy.controller.trainable_variables)
        
        return loss, gradients

    def _apply_gradients(self, gradients):
        """Apply gradients to policy parameters."""
        self.optimizer_instance.apply_gradients(
            zip(gradients, self.policy.controller.trainable_variables)
        )

    def _compute_loss(self, obs, actions, rewards):
        """Compute the loss function. Default implementation."""
        # Default implementation using policy gradient approach
        with tf.GradientTape() as tape:
            # Get policy outputs
            log_probs = self.policy(obs, actions)
            # Simple policy gradient loss
            loss = -tf.reduce_mean(tf.cast(rewards, tf.float32) * log_probs)
        return loss


    # abstractmethod (override if needed)
    def _preppend_to_summary(self, iteration) -> None:
        """Add particular fields to the summary log.
        Override if needed.
        """
        pass
        

    def _setup_summary(self, iteration, loss_dict) -> None:
        """ Setup tensor flow summary for TF2
        """    
        # Log scalar metrics
        for key, value in loss_dict.items():
            tf.summary.scalar(key, value, step=iteration)
        
        # Log histograms of trainable variables
        for variable in self.policy.controller.trainable_variables:
            tf.summary.histogram(variable.name, variable, step=iteration)
            tf.summary.scalar(variable.name + '_norm', tf.norm(variable), step=iteration)
        
        if hasattr(self, 'norms'):
            tf.summary.scalar('gradient norm', self.norms, step=iteration)


    def _setup_policy_optimizer(self, 
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
        """Setup of the policy optimizer.
        """ 
        self._init(policy, debug, summary, logdir, optimizer, learning_rate, entropy_weight, entropy_gamma)
        self._init_loss_with_entropy()
        self._set_loss() # Abstract method defined in derived class
        self._setup_optimizer()
        if self.summary:
            self.writer = tf.summary.create_file_writer(self.logdir)
            with self.writer.as_default():
                self._preppend_to_summary(self.iterations) # Pass iteration to subclasses
                self._setup_summary(self.iterations) # Pass iteration        


    def train_step(self, 
            baseline : np.ndarray, 
            sampled_batch : Batch) -> None:
        """Computes loss, trains model, and returns summaries.

        Returns
        -------
            None
        """
        # Default implementation using policy gradient approach
        with tf.GradientTape() as tape:
            # Extract rewards
            rewards = tf.cast(sampled_batch.rewards, tf.float32)
            baseline_tf = tf.cast(baseline, tf.float32)
            
            # Compute advantages
            advantages = rewards - baseline_tf
            
            # Get policy outputs
            log_probs = self.policy(sampled_batch.obs, sampled_batch.actions)
            
            # Policy gradient loss
            pg_loss = tf.reduce_mean(advantages * log_probs)
            
            # Add entropy regularization if available
            if hasattr(self.policy, 'entropy_weight'):
                entropy = self.policy.entropy(sampled_batch.obs, sampled_batch.actions)
                entropy_loss = -self.policy.entropy_weight * tf.reduce_mean(entropy)
                total_loss = pg_loss + entropy_loss
            else:
                total_loss = pg_loss
        
        # Compute gradients and apply
        gradients = tape.gradient(total_loss, self.policy.controller.trainable_variables)
        self.optimizer_instance.apply_gradients(
            zip(gradients, self.policy.controller.trainable_variables)
        )



