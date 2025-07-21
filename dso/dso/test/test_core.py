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
