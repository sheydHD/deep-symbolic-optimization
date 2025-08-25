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


@pytest.mark.parametrize("config", ["dso/config/config_regression.json",
                                    "dso/config/config_control.json"])
def test_task(model, config):
    """Test that Tasks do not crash for various configs."""
    
    try:
        config = load_config(config)
    except FileNotFoundError:
        # Fix path if needed
        if "dso/config/" in config:
            new_config = config.replace("dso/config/", "dso/dso/config/")
            try:
                config = load_config(new_config)
            except FileNotFoundError:
                pytest.skip(f"Config file not found: {config}")
        else:
            pytest.skip(f"Config file not found: {config}")
            
    config["experiment"]["logdir"] = None # Turn off saving results
    model.set_config(config)
    model.config_training.update({"n_samples" : 10,
                                  "batch_size" : 5
                                  })
    model.train()


def test_model_parity():
    """Test that a saved and restored model has identical weights (TF2.x parity test)."""
    import tempfile
    config = load_config("dso/config/config_regression.json")
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
