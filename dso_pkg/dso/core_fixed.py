"""Fixed core module with proper initialization order for MIMO support."""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import os
import zlib
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import random
from time import time
from datetime import datetime

import numpy as np
# Import TensorFlow with optimized configuration
from dso.tf_config import tf
import json5 as json

from dso.task import set_task
from dso.train import Trainer
from dso.checkpoint import Checkpoint
from dso.train_stats import StatsLogger
from dso.prior import make_prior
from dso.program import Program
from dso.config import load_config
from dso.tf_state_manager import make_state_manager

from dso.policy.policy import make_policy
from dso.policy_optimizer import make_policy_optimizer


class DeepSymbolicOptimizerFixed():
    """
    Fixed Deep symbolic optimization model with proper initialization order.
    
    This version ensures that Program.library is initialized before creating
    the policy, which is crucial for MIMO support.
    """

    def __init__(self, config=None):
        self.set_config(config)
        self.save_path = None  # Initialize save_path attribute

    def setup(self):
        """Setup with proper initialization order."""
        
        # Clear the cache and reset the compute graph
        Program.clear_cache()
        
        # CRITICAL: Set the task FIRST before creating policy
        # This ensures Program.library is initialized
        self.pool = self.make_pool_and_set_task()
        
        # Verify library is initialized
        if Program.library is None:
            raise RuntimeError("Program.library not initialized after setting task")
            
        print(f"Library initialized with {Program.library.L} tokens")
        
        # Set seeds after task initialization
        self.set_seeds()

        # Setup logdirs and output files
        self.output_file = self.make_output_file()
        self.save_config()
        
        # Set save_path if not already set
        if self.save_path is None:
            self.save_path = self.config_experiment.get("save_path", None)

        # Now create training components (policy needs initialized library)
        self.prior = self.make_prior()
        self.state_manager = self.make_state_manager()
        
        # Policy creation - library should be available now
        self.policy = self.make_policy()
        
        self.policy_optimizer = self.make_policy_optimizer()
        self.gp_controller = self.make_gp_controller()
        self.logger = self.make_logger()
        self.trainer = self.make_trainer()
        self.checkpoint = self.make_checkpoint()

    def train_one_step(self, override=None):
        """Train one iteration."""
        
        # Setup the model
        if not hasattr(self, 'trainer') or self.trainer is None:
            self.setup()

        # Run one step
        assert not self.trainer.done, "Training has already completed!"
        self.trainer.run_one_step(override)
        
        # Maybe save next checkpoint
        self.checkpoint.update()

        # If complete, return summary
        if self.trainer.done:
            return self.finish()

    def train(self):
        """Train the model until completion."""
        
        # Setup the model
        if not hasattr(self, 'trainer') or self.trainer is None:
            self.setup()

        # Train until complete
        while not self.trainer.done:
            self.trainer.run_one_step()
            self.checkpoint.update()

        # Finish
        return self.finish()

    def finish(self):
        """Finalize training and return results."""
        
        result = {
            "seed" : self.config_experiment["seed"],
        }
        result.update(self.trainer.results)

        hof = self.trainer.train_summary()
        
        if self.config_experiment['verbose']:
            print("\n")
            print("="*60)
            print(f"Hall of Fame:")
            print("="*60)
            print("\t".join(["R", "Count", "Expression"]))
            for i, p in enumerate(hof):
                count = p.on_policy_count + p.off_policy_count
                print("{:.6f}\t{}\t{}".format(p.r, count, p))

        # Save all results available only after all iterations are finished
        results_add = self.logger.save_results(self.pool, self.trainer.nevals)
        result.update(results_add)

        # Close the pool
        if self.pool is not None:
            self.pool.close()

        return result

    def set_config(self, config):
        config = load_config(config)

        self.config = defaultdict(dict, config)
        self.config_task = self.config["task"]
        self.config_prior = self.config["prior"]
        self.config_logger = self.config["logging"]
        self.config_training = self.config["training"]
        self.config_state_manager = self.config["state_manager"]
        self.config_policy = self.config["policy"]
        self.config_policy_optimizer = self.config["policy_optimizer"]
        self.config_gp_meld = self.config["gp_meld"]
        self.config_experiment = self.config["experiment"]
        self.config_checkpoint = self.config["checkpoint"]

    def save_config(self):
        # Save the config file
        if self.output_file is not None:
            output_file = self.output_file.replace("csv", "json")

            with open(output_file, 'w') as f:
                json.dump(self.config, f, indent=3)

    def set_seeds(self):
        seed = self.config_experiment["seed"]
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)

    def make_prior(self):
        prior = make_prior(Program.library, self.config_prior)
        if hasattr(prior, 'clear_cache'):
            prior.clear_cache()
        return prior

    def make_state_manager(self):
        state_manager = make_state_manager(self.config_state_manager)
        return state_manager

    def make_output_file(self):
        """Makes a log file for training"""
        
        output_file = self.config_experiment.get("logfile", None)
        
        if output_file is not None:
            logdir = self.config_experiment["logdir"]
            os.makedirs(logdir, exist_ok=True)
            output_file = os.path.join(logdir, output_file)
            
            with open(output_file, 'w') as f:
                f.write(f"Generated log for run timestamp {datetime.now()}\n")

        return output_file

    def make_logger(self):
        """Initializes a StatsLogger to save statistics of training to file"""
        
        # Filter out parameters that StatsLogger doesn't accept
        # Based on StatsLogger.__init__ signature
        valid_params = [
            'save_summary', 'save_all_iterations', 'hof',
            'save_pareto_front', 'save_positional_entropy', 'save_top_samples_per_batch',
            'save_cache', 'save_cache_r_min', 'save_freq', 'save_token_count'
        ]
        
        logger_kwargs = {}
        logger_kwargs['output_file'] = self.output_file
        for param in valid_params:
            if param in self.config_logger:
                logger_kwargs[param] = self.config_logger[param]
        
        logger = StatsLogger(**logger_kwargs)
        return logger

    def make_checkpoint(self):
        # Filter checkpoint config to only pass valid parameters
        valid_checkpoint_params = ['load_path', 'save_freq', 'units', 'save_on_done']
        checkpoint_kwargs = {k: v for k, v in self.config_checkpoint.items() 
                           if k in valid_checkpoint_params}
        
        checkpoint = Checkpoint(self, **checkpoint_kwargs)
        return checkpoint

    def make_policy_optimizer(self):
        # Extract policy_optimizer_type from config
        policy_optimizer_type = self.config_policy_optimizer.pop("policy_optimizer_type", "pg")
        policy_optimizer = make_policy_optimizer(self.policy,
                                                 policy_optimizer_type,
                                                 **self.config_policy_optimizer)
        return policy_optimizer

    def make_policy(self):
        """Create policy with library verification."""
        
        # Verify library is available
        if Program.library is None:
            raise RuntimeError("Cannot create policy: Program.library not initialized. "
                             "Ensure task is set before creating policy.")
        
        policy = make_policy(self.prior,
                             self.state_manager,
                             **self.config_policy)
        return policy

    def make_gp_controller(self):
        if self.config_gp_meld.pop("run_gp_meld", False):
            from dso.gp.gp_controller import GPController
            gp_controller = GPController(self.prior,
                                         self.config_prior,
                                         **self.config_gp_meld)
        else:
            gp_controller = None
        return gp_controller

    def make_pool_and_set_task(self):
        """Create the pool and set the Task for each worker."""
        
        # Set complexity and const optimizer here so pool can access them
        # Set the complexity function
        complexity = self.config_training["complexity"]
        Program.set_complexity(complexity)

        # Set the constant optimizer
        const_optimizer = self.config_training["const_optimizer"]
        const_params = self.config_training["const_params"]
        const_params = const_params if const_params is not None else {}
        Program.set_const_optimizer(const_optimizer, **const_params)

        pool = None
        n_cores_batch = self.config_training.get("n_cores_batch")
        if n_cores_batch is not None:
            if n_cores_batch == -1:
                n_cores_batch = cpu_count()
            if n_cores_batch > 1:
                pool = Pool(n_cores_batch,
                            initializer=set_task,
                            initargs=(self.config_task,))

        # Set the Task for the parent process - CRITICAL for library initialization
        set_task(self.config_task)

        return pool

    def make_trainer(self):
        """Makes the Trainer object which controls the training"""
        
        # Valid Trainer parameters based on Trainer.__init__ signature
        valid_trainer_params = [
            'n_samples', 'batch_size', 'alpha', 'epsilon', 'verbose', 
            'baseline', 'b_jumpstart', 'early_stopping', 'debug',
            'use_memory', 'memory_capacity', 'warm_start', 'memory_threshold',
            'complexity', 'const_optimizer', 'const_params', 'n_cores_batch'
        ]
        
        # Prepare trainer arguments, filtering only valid parameters
        trainer_kwargs = {}
        for param in valid_trainer_params:
            if param in self.config_training:
                trainer_kwargs[param] = self.config_training[param]
        
        # Override with experiment config values where appropriate
        if 'verbose' in self.config_experiment:
            trainer_kwargs['verbose'] = self.config_experiment['verbose']
        if 'debug' in self.config_experiment:
            trainer_kwargs['debug'] = self.config_experiment['debug']
        if 'use_memory' in self.config_experiment:
            trainer_kwargs['use_memory'] = self.config_experiment['use_memory']
        if 'memory_capacity' in self.config_experiment:
            trainer_kwargs['memory_capacity'] = self.config_experiment['memory_capacity']
        
        trainer = Trainer(self.policy,
                         self.policy_optimizer,
                         self.gp_controller,
                         self.logger,
                         self.pool,
                         **trainer_kwargs)

        return trainer


# Patch the original DeepSymbolicOptimizer
def patch_dso_core():
    """Replace the original DSO with the fixed version."""
    import dso.core as core_module
    core_module.DeepSymbolicOptimizer = DeepSymbolicOptimizerFixed
    return True