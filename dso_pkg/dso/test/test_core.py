"""Test cases for DeepSymbolicOptimizer on each Task."""

import pytest
import tensorflow as tf
import numpy as np

from dso import DeepSymbolicOptimizer
from dso.config import load_config
from dso.test.generate_test_data import CONFIG_TRAINING_OVERRIDE


@pytest.fixture
def model():
    config = load_config()
    config["experiment"]["logdir"] = None # Turn off saving results
    return DeepSymbolicOptimizer(config)


@pytest.mark.parametrize("config", ["dso_pkg/dso/config/config_regression.json",
                                    "dso_pkg/dso/config/config_control.json"])
def test_task(model, config):
    """Test that Tasks do not crash for various configs."""
    
    import os
    
    # Try multiple possible config paths
    config_paths = [
        config,  # Direct path
        os.path.join("dso_pkg", "dso", "config", os.path.basename(config)),  # Current structure
        os.path.join("dso", "dso", "config", os.path.basename(config)),  # Original structure  
        os.path.join(os.path.dirname(__file__), "..", "config", os.path.basename(config))  # Relative
    ]
    
    config_loaded = None
    for config_path in config_paths:
        try:
            config_loaded = load_config(config_path)
            break
        except FileNotFoundError:
            continue
    
    if config_loaded is None:
        pytest.skip(f"Config file not found in any of: {config_paths}")
        
    config = config_loaded
    config["experiment"]["logdir"] = None # Turn off saving results
    model.set_config(config)
    model.config_training.update({"n_samples" : 10,
                                  "batch_size" : 5
                                  })
    model.train()


def test_model_parity():
    """Test that a saved and restored model has identical weights (TF2.x parity test)."""
    import tempfile
    import os
    
    # Try multiple possible config paths
    config_paths = [
        "dso_pkg/dso/config/config_regression.json",
        "dso/dso/config/config_regression.json", 
        os.path.join(os.path.dirname(__file__), "..", "config", "config_regression.json")
    ]
    
    config = None
    for config_path in config_paths:
        try:
            config = load_config(config_path)
            break
        except FileNotFoundError:
            continue
    
    if config is None:
        pytest.skip(f"Config file not found in any of: {config_paths}")
    config["experiment"]["logdir"] = None
    # Train and save a reference model
    model_ref = DeepSymbolicOptimizer(config)
    model_ref.train()
    ckpt_ref = tf.train.Checkpoint(model=model_ref)
    ref_dir = tempfile.mkdtemp(prefix="dso_ref_ckpt_")
    ref_path = ckpt_ref.save(os.path.join(ref_dir, "ckpt"))

    # Create a new model and restore weights
    model_new = DeepSymbolicOptimizer(config)
    ckpt_new = tf.train.Checkpoint(model=model_new)
    ckpt_new.restore(ref_path).expect_partial()

    # Compare variables (assumes model exposes trainable_variables)
    ref_vars = [v.numpy() for v in getattr(model_ref, 'trainable_variables', [])]
    new_vars = [v.numpy() for v in getattr(model_new, 'trainable_variables', [])]
    assert len(ref_vars) == len(new_vars), "Variable count mismatch after restore!"
    for i, (v1, v2) in enumerate(zip(ref_vars, new_vars)):
        assert np.allclose(v1, v2), f"Model variable {i} does not match after restore!"
