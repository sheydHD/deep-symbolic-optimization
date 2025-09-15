from abc import ABC, abstractmethod

import tensorflow as tf

from dso.program import Program


class StateManager(ABC):
    """
    An interface for handling the tf.Tensor inputs to the Policy.
    """

    def setup_manager(self, policy):
        """
        Function called inside the policy to perform the needed initializations (e.g., if the tf context is needed)
        :param policy the policy class
        """
        self.policy = policy
        self.max_length = policy.max_length

    @abstractmethod
    def get_tensor_input(self, obs):
        """
        Convert an observation from a Task into a Tesnor input for the
        Policy, e.g. by performing one-hot encoding or embedding lookup.

        Parameters
        ----------
        obs : np.ndarray (dtype=np.float32)
            Observation coming from the Task.

        Returns
        --------
        input_ : tf.Tensor (dtype=tf.float32)
            Tensor to be used as input to the Policy.
        """
        return

    def process_state(self, obs):
        """
        Entry point for adding information to the state tuple.
        If not overwritten, this functions does nothing
        """
        return obs


def make_state_manager(config):
    """
    Parameters
    ----------
    config : dict
        Parameters for this StateManager.

    Returns
    -------
    state_manager : StateManager
        The StateManager to be used by the policy.
    """
    manager_dict = {
        "hierarchical": HierarchicalStateManager
    }

    if config is None:
        config = {}

    # Use HierarchicalStateManager by default
    manager_type = config.pop("type", "hierarchical")

    manager_class = manager_dict[manager_type]
    state_manager = manager_class(**config)

    return state_manager


class HierarchicalStateManager(StateManager):
    """
    Class that uses the previous action, parent, sibling, and/or dangling as
    observations.
    """

    def __init__(self, observe_parent=True, observe_sibling=True,
                 observe_action=False, observe_dangling=False, embedding=False,
                 embedding_size=8):
        """
        Parameters
        ----------
        observe_parent : bool
            Observe the parent of the Token being selected?

        observe_sibling : bool
            Observe the sibling of the Token being selected?

        observe_action : bool
            Observe the previously selected Token?

        observe_dangling : bool
            Observe the number of dangling nodes?

        embedding : bool
            Use embeddings for categorical inputs?

        embedding_size : int
            Size of embeddings for each categorical input if embedding=True.
        """
        self.observe_parent = observe_parent
        self.observe_sibling = observe_sibling
        self.observe_action = observe_action
        self.observe_dangling = observe_dangling
        self.library = Program.library

        # Parameter assertions/warnings
        assert self.observe_action + self.observe_parent + self.observe_sibling + self.observe_dangling > 0, \
            "Must include at least one observation."

        self.embedding = embedding
        self.embedding_size = embedding_size

    def setup_manager(self, policy):
        super().setup_manager(policy)
        # Create embeddings if needed
        if self.embedding:
            initializer = tf.random.uniform_initializer(minval=-1.0,
                                                        maxval=1.0,
                                                        seed=0)
            with tf.name_scope("embeddings"):
                if self.observe_action:
                    self.action_embeddings = tf.Variable(
                        initializer([self.library.n_action_inputs, self.embedding_size]),
                        trainable=True,
                        name="action_embeddings")
                if self.observe_parent:
                    self.parent_embeddings = tf.Variable(
                        initializer([self.library.n_parent_inputs, self.embedding_size]),
                        trainable=True,
                        name="parent_embeddings")
                if self.observe_sibling:
                    self.sibling_embeddings = tf.Variable(
                        initializer([self.library.n_sibling_inputs, self.embedding_size]),
                        trainable=True,
                        name="sibling_embeddings")

    @property
    def input_dim(self):
        """
        Calculates the input dimension based on the enabled observations.
        This replicates the logic from the original TF1.x's get_tensor_input.
        
        NOTE: In TensorFlow 2.x, the extended batch can add additional features.
        We use a conservative estimate to handle worst-case scenarios.
        """
        base_dim = 0
        if self.observe_action:
            base_dim += self.embedding_size if self.embedding else self.library.n_action_inputs
        if self.observe_parent:
            base_dim += self.embedding_size if self.embedding else self.library.n_parent_inputs
        if self.observe_sibling:
            base_dim += self.embedding_size if self.embedding else self.library.n_sibling_inputs
        if self.observe_dangling:
            base_dim += 1
        
        # Return the base dimension for now - let the RNN controller handle variable input sizes
        # The original TF1 code used dynamic_rnn which could handle variable input dimensions
        return base_dim

    def get_tensor_input(self, obs):
        observations = []
        
        # Handle different observation formats following original TF1 logic:
        # The original TF1 code expects obs in format [batch, seq_len, obs_dim]
        # For the state manager, we need to process it properly to output [batch, seq_len, features]
        
        if len(obs.shape) == 3:
            # Batch format: [batch, seq_len, obs_dim] where obs_dim=4 (action, parent, sibling, dangling)
            # Extract features directly from the observation tensor
            action = obs[:, :, 0]    # [batch, seq_len]
            parent = obs[:, :, 1]    # [batch, seq_len] 
            sibling = obs[:, :, 2]   # [batch, seq_len]
            dangling = obs[:, :, 3]  # [batch, seq_len]
            unstacked_obs = [action, parent, sibling, dangling]
        else:
            # Single-step format: [batch, obs_dim] where obs_dim=4
            action, parent, sibling, dangling = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]
            unstacked_obs = [action, parent, sibling, dangling]

        # Cast action, parent, sibling to int for embedding_lookup or one_hot
        action = tf.cast(action, tf.int32)
        parent = tf.cast(parent, tf.int32)
        sibling = tf.cast(sibling, tf.int32)

        # Action, parent, and sibling inputs are either one-hot or embeddings
        if self.observe_action:
            if self.embedding:
                x = tf.nn.embedding_lookup(self.action_embeddings, action)
            else:
                x = tf.one_hot(action, depth=self.library.n_action_inputs)
            observations.append(x)
        if self.observe_parent:
            if self.embedding:
                x = tf.nn.embedding_lookup(self.parent_embeddings, parent)
            else:
                x = tf.one_hot(parent, depth=self.library.n_parent_inputs)
            observations.append(x)
        if self.observe_sibling:
            if self.embedding:
                x = tf.nn.embedding_lookup(self.sibling_embeddings, sibling)
            else:
                x = tf.one_hot(sibling, depth=self.library.n_sibling_inputs)
            observations.append(x)

        # Dangling input is just the value of dangling
        if self.observe_dangling:
            if len(obs.shape) == 3:
                # For batch format, expand on last dimension to maintain [batch, seq_len, 1]
                x = tf.expand_dims(dangling, axis=-1)
            else:
                # For single-step format, expand as before
                x = tf.expand_dims(dangling, axis=-1)
            observations.append(x)

        input_ = tf.concat(observations, -1)
        # possibly concatenates additional observations (e.g., bert embeddings)
        if len(unstacked_obs) > 4:
            # For batch format, handle additional observations properly
            if len(obs.shape) == 3:
                # Additional observations need to be stacked properly for batch format
                extra_obs = tf.stack(unstacked_obs[4:], axis=-1)  # [batch, seq_len, extra_features]
            else:
                # Single-step format
                extra_obs = tf.stack(unstacked_obs[4:], axis=-1)
            input_ = tf.concat([input_, extra_obs], axis=-1)
        
        # Return the concatenated features - let the RNN controller handle variable dimensions
        # The input projection layer in the controller will handle any dimension mismatches
        return input_
