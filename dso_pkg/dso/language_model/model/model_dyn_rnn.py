"""Model architecture for LanguageModel using TF2"""

# Import TensorFlow with optimized configuration
from dso.tf_config import tf

def sequence_loss(logits, targets, weights,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None):
    """Compute sequence loss using TF2."""
    with tf.name_scope(name or "sequence_loss"):
        num_classes = tf.shape(logits)[2]
        logits_flat = tf.reshape(logits, [-1, num_classes])
        targets = tf.reshape(targets, [-1])
        
        if softmax_loss_function is None:
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=targets, logits=logits_flat)
        else:
            crossent = softmax_loss_function(labels=targets, logits=logits_flat)
            
        crossent *= tf.reshape(weights, [-1])
        
        if average_across_timesteps and average_across_batch:
            crossent = tf.reduce_sum(crossent)
            total_size = tf.reduce_sum(weights)
            total_size += 1e-12 
            crossent /= total_size
        elif average_across_timesteps and not average_across_batch:
            crossent = tf.reduce_sum(crossent, axis=[1])
            total_size = tf.reduce_sum(weights, axis=[1])
            total_size += 1e-12
            crossent /= total_size
        elif not average_across_timesteps and average_across_batch:
            crossent = tf.reduce_sum(crossent, axis=[0])
            total_size = tf.reduce_sum(weights, axis=[0])
            total_size += 1e-12
            crossent /= total_size
        else:
            crossent = tf.reshape(crossent, tf.shape(targets))
    return crossent


class LanguageModel(tf.keras.Model):
    """Language model using TF2 and Keras."""
    
    def __init__(self, vocabulary_size, embedding_size, num_layers, num_hidden, mode='train', **kwargs):
        super().__init__(**kwargs)
        
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.mode = mode
        
        # Build layers
        self.embedding = tf.keras.layers.Embedding(vocabulary_size, embedding_size)
        
        # LSTM layers
        self.lstm_layers = []
        for i in range(num_layers):
            return_sequences = True if i < num_layers - 1 else True  # Always return sequences for language modeling
            self.lstm_layers.append(
                tf.keras.layers.LSTM(
                    num_hidden,
                    return_sequences=return_sequences,
                    return_state=True,
                    name=f'lstm_{i}'
                )
            )
        
        # Output projection
        self.output_projection = tf.keras.layers.Dense(vocabulary_size, name='output_projection')
        self.dropout = tf.keras.layers.Dropout(0.5)
    
    def call(self, x, seq_len=None, initial_state=None, training=None):
        """Forward pass through the language model. Optionally uses sequence lengths for masking (strict TF1 equivalence)."""
        # Embedding
        embedded = self.embedding(x)
        embedded = self.dropout(embedded, training=training)

        # If sequence lengths are provided, create a mask
        mask = None
        if seq_len is not None:
            # Assume x shape: (batch, time)
            maxlen = tf.shape(x)[1]
            mask = tf.sequence_mask(seq_len, maxlen)

        # LSTM layers
        lstm_output = embedded
        states = []

        for i, lstm_layer in enumerate(self.lstm_layers):
            init_state = initial_state[i] if initial_state else None
            # Pass mask only to the first LSTM layer (Keras propagates it)
            if i == 0 and mask is not None:
                lstm_output, h_state, c_state = lstm_layer(lstm_output, initial_state=init_state, training=training, mask=mask)
            else:
                lstm_output, h_state, c_state = lstm_layer(lstm_output, initial_state=init_state, training=training)
            states.append([h_state, c_state])

        # Output projection
        logits = self.output_projection(lstm_output)

        return logits, states, mask
    
    def get_initial_state(self, batch_size):
        """Get initial states for all LSTM layers."""
        states = []
        for _ in range(self.num_layers):
            h = tf.zeros([batch_size, self.num_hidden])
            c = tf.zeros([batch_size, self.num_hidden])
            states.append([h, c])
        return states
