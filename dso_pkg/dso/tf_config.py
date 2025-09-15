"""TensorFlow 2.x configuration and setup utilities."""

import tensorflow as tf
import numpy as np
import os
import warnings
import logging

# Configure environment for optimal TF2 performance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Show warnings and errors only
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable oneDNN optimizations for better performance

# Configure logging
logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.getLogger('absl').setLevel(logging.WARNING)

# Filter warnings for a cleaner output
warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')
warnings.filterwarnings('ignore', message='pkg_resources is deprecated', category=UserWarning)


def setup_tensorflow():
    """Configure TensorFlow 2.x with optimal settings for DSO."""
    
    # Set random seeds for reproducibility
    tf.random.set_seed(0)
    np.random.seed(0)
    
    # Set Python random seed for complete reproducibility
    import random
    random.seed(0)
    
    # Enable deterministic operations for numerical consistency
    tf.config.experimental.enable_op_determinism()
    
    # Force deterministic GPU operations
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    # Use graph mode for better performance and determinism
    tf.config.run_functions_eagerly(False)
    
    # Single-threaded execution for deterministic behavior
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    
    # Configure GPU if available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    # Set floating point precision
    tf.keras.backend.set_floatx('float32')
    
    # Optimize JIT compilation
    tf.config.optimizer.set_jit(True)


def enable_mixed_precision():
    """Enable mixed precision training for better performance on modern GPUs."""
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)


def set_deterministic_seeds(seed=0):
    """Set all random seeds for complete reproducibility."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    # Also set Python random seed
    import random
    random.seed(seed)
    
    # Set environment variables for deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)


# Initialize TensorFlow with optimal settings
setup_tensorflow()

# Modern TensorFlow 2.x API aliases
# Core tensor operations
convert_to_tensor = tf.convert_to_tensor
constant = tf.constant
Variable = tf.Variable

# Math operations  
reduce_sum = tf.reduce_sum
reduce_mean = tf.reduce_mean
reduce_max = tf.reduce_max
reduce_min = tf.reduce_min
matmul = tf.linalg.matmul
transpose = tf.transpose

# Neural network operations
dense = tf.keras.layers.Dense
dropout = tf.keras.layers.Dropout
batch_normalization = tf.keras.layers.BatchNormalization

# Optimizers
AdamOptimizer = tf.keras.optimizers.Adam
RMSPropOptimizer = tf.keras.optimizers.RMSprop  
GradientDescentOptimizer = tf.keras.optimizers.SGD

# Loss functions
softmax_cross_entropy_with_logits = tf.nn.softmax_cross_entropy_with_logits
sparse_softmax_cross_entropy_with_logits = tf.nn.sparse_softmax_cross_entropy_with_logits

# Activation functions
relu = tf.nn.relu
sigmoid = tf.nn.sigmoid
tanh = tf.nn.tanh
softmax = tf.nn.softmax

# Initializers
zeros_initializer = tf.zeros_initializer
ones_initializer = tf.ones_initializer
random_normal_initializer = tf.random_normal_initializer
random_uniform_initializer = tf.random_uniform_initializer
truncated_normal_initializer = tf.keras.initializers.TruncatedNormal
glorot_uniform_initializer = tf.keras.initializers.GlorotUniform
glorot_normal_initializer = tf.keras.initializers.GlorotNormal

# Variable scoping
name_scope = tf.name_scope

# Random operations
random_normal = tf.random.normal
random_uniform = tf.random.uniform
set_random_seed = tf.random.set_seed

# Initialize TensorFlow with deterministic settings
setup_tensorflow()

# Control flow
cond = tf.cond
while_loop = tf.while_loop

# Gradient computation
GradientTape = tf.GradientTape
