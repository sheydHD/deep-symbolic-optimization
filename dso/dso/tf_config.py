"""TensorFlow 2.x configuration for DSO with v1 compatibility mode.

This module configures TensorFlow 2.x to work with legacy v1 code patterns
while minimizing unnecessary warnings and ensuring proper compatibility.
"""

import os
import warnings
import logging

# Configure environment before importing TensorFlow
# Suppress all TF C++ level logs except errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Show all logs
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

# Configure Python logging for TensorFlow - suppress deprecation warnings from TF1 compat
logging.getLogger('tensorflow').setLevel(logging.ERROR)
# Also configure absl logging used by TensorFlow
logging.getLogger('absl').setLevel(logging.ERROR)

# Filter only truly irrelevant warnings
# Keep deprecation warnings visible for actual issues
warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')
# Specific warning about resource variables that we understand but cannot fix yet
warnings.filterwarnings('ignore', message='.*non-resource variables are not supported.*')

# Known TF1 compatibility warnings that will be fixed in future migration
warnings.filterwarnings('ignore', message='.*MultiRNNCell.*is deprecated.*', module='tensorflow')
warnings.filterwarnings('ignore', message='.*py_func.*is deprecated.*', module='tensorflow')
warnings.filterwarnings('ignore', message='.*dynamic_rnn.*is deprecated.*', module='tensorflow')

# Suppress unrelated library warnings
warnings.filterwarnings('ignore', message='pkg_resources is deprecated', category=UserWarning)

# Import TensorFlow
import tensorflow as tf

# Global flag to track if we're using v1 compatibility mode
USING_TF_V1_COMPAT = True

def configure_tensorflow_v1_compat():
    """
    Configure TensorFlow 2.x to work with v1 code patterns.
    
    This is a temporary compatibility layer that should be removed
    once the codebase is fully migrated to TF2.
    
    We use targeted compatibility functions instead of disable_v2_behavior()
    to avoid the deprecated disable_resource_variables() call.
    """
    
    # Instead of disable_v2_behavior(), use specific compatibility settings:
    # 1. Disable eager execution (main difference between TF1 and TF2)
    tf.compat.v1.disable_eager_execution()
    
    # 2. Enable control flow v2 (this is actually better than v1)
    tf.compat.v1.enable_control_flow_v2()
    
    # 3. Set TF1 logging verbosity
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    
    # Configure GPU memory growth for both v1 and v2 patterns
    try:
        # Try v2 style first
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        # Fallback to v1 style if needed
        try:
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
        except Exception:
            pass  # No GPU or configuration failed
    
    return True

# Apply v1 compatibility configuration
is_configured = configure_tensorflow_v1_compat()

# Export commonly used tf.compat.v1 aliases to reduce code changes needed
# These will be removed during full TF2 migration
Session = tf.compat.v1.Session
placeholder = tf.compat.v1.placeholder
get_variable = tf.compat.v1.get_variable
variable_scope = tf.compat.v1.variable_scope
ConfigProto = tf.compat.v1.ConfigProto
reset_default_graph = tf.compat.v1.reset_default_graph
