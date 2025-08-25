"""Generate model parity test case data for DeepSymbolicOptimizer."""
from importlib import resources
import os

import tensorflow as tf
import click

from dso import DeepSymbolicOptimizer
from dso.config import load_config


# Shorter config run for parity test
CONFIG_TRAINING_OVERRIDE = {
    "n_samples" : 1000,
    "batch_size" : 100
}

@click.command()
@click.option('--stringency', '--t', default="strong", type=str, help="stringency of the test data to generate")
def main(stringency):
    # Load config
    config = load_config()

    # Train the model
    model = DeepSymbolicOptimizer(config)

    if stringency == "strong":
        n_samples = 1000
        suffix = "_" + stringency
    elif stringency == "weak":
        n_samples = 100
        suffix = "_" + stringency

    model.config_training.update({"n_samples" : n_samples,
                                  "batch_size" : 100})

    # Turn on GP meld
    model.config_gp_meld.update({"run_gp_meld" : True,
                                 "generations" : 3,
                                 "population_size" : 10})

    model.train()

    # Save the TF model (TF2.x: use tf.train.Checkpoint or model.save if needed)
    tf_save_path = os.fspath(resources.files('dso.test').joinpath('data', 'test_model' + suffix))
    # Example for TF2.x: if model is a tf.keras.Model, use model.save(tf_save_path)
    # If not needed, comment out or implement appropriate TF2.x saving logic
    # model.save(tf_save_path)


if __name__ == "__main__":
    main()
