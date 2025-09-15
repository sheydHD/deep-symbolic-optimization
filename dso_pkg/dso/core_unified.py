"""Unified core deep symbolic optimizer with MIMO support."""

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


class DeepSymbolicOptimizer():
    """
    Unified Deep symbolic optimization model with MIMO support.
    
    This version combines the best of both implementations:
    - Proper initialization order for MIMO support
    - Backward compatibility with existing code
    - Clean architecture without duplication
    
    Parameters
    ----------
    config : dict or str
        Config dictionary or path to JSON.
    mimo_mode : bool
        If True, uses MIMO-safe initialization order (default: auto-detect)
    
    Attributes
    ----------
    config : dict
        Configuration parameters for training.
    
    Methods
    -------
    setup : Initialize the model
    train : Train the model
    train_one_step : Train for one iteration
    """
    
    def __init__(self, config=None, mimo_mode=None):
        self.set_config(config)
        self.save_path = None
        
        # Auto-detect MIMO mode if not specified
        if mimo_mode is None:
            self.mimo_mode = self._detect_mimo_mode()
        else:
            self.mimo_mode = mimo_mode
    
    def _detect_mimo_mode(self):
        """Auto-detect if MIMO mode is needed based on dataset."""
        if "task" in self.config and "dataset" in self.config["task"]:
            dataset = self.config["task"]["dataset"]
            # Check if it's a known MIMO dataset
            mimo_datasets = ["MIMO-simple", "MIMO-benchmark", "MIMO-easy", "MIMO-modular"]
            if any(mimo in str(dataset) for mimo in mimo_datasets):
                return True
        return False
    
    def setup(self):
        """Setup with proper initialization order."""
        
        # Clear the cache and reset the compute graph
        Program.clear_cache()
        
        # Initialize in correct order based on mode
        if self.mimo_mode:
            # MIMO mode: Initialize task first to ensure library is ready
            self.pool = self.make_pool_and_set_task()
            
            # Verify library is initialized for MIMO
            if Program.library is None:
                raise RuntimeError("Program.library not initialized after setting task")
            
            if self.config_experiment.get("verbose", False):
                print(f"MIMO mode: Library initialized with {Program.library.L} tokens")
        else:
            # Standard mode: Can initialize pool later
            self.pool = None
        
        # Set seeds
        self.set_seeds()
        
        # Setup logdirs and output files
        self.output_file = self.make_output_file()
        self.save_config()
        
        # Set save_path if not already set
        if self.save_path is None:
            self.save_path = self.config_experiment.get("save_path", None)
        
        # Initialize pool if not in MIMO mode
        if not self.mimo_mode:
            self.pool = self.make_pool_and_set_task()
        
        # Create training components
        self.prior = self.make_prior()
        self.state_manager = self.make_state_manager()
        self.policy = self.make_policy()
        self.policy_optimizer = self.make_policy_optimizer()
        self.gp_controller = self.make_gp_controller()
        self.logger = self.make_logger()
        self.trainer = self.make_trainer()
        self.checkpoint = self.make_checkpoint()
    
    def train_one_step(self, override=None):
        """Train one iteration."""
        
        # Setup the model if needed
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
        self.setup()
        
        # Train the model until done
        while not self.trainer.done:
            result = self.train_one_step()
        
        return result
    
    def finish(self):
        """After training completes, finish up and return summary dict."""
        
        # Return statistics of best Program
        p = self.trainer.p_r_best
        result = {"seed": self.config_experiment["seed"]}  # Seed listed first
        result.update({"r": p.r})
        result.update(p.evaluate)
        result.update({
            "expression": repr(p.sympy_expr),
            "traversal": repr(p),
            "program": p
        })
        
        # Save all results available only after all iterations are finished
        if self.logger.save_pareto_front:
            # Check for complexity-vs-accuracy Pareto front
            pf = self.trainer.compute_pareto_front(self.trainer.hof)
            result.update({"pareto_front": pf})
        
        # Save results file
        if self.save_path is not None:
            save_path = os.path.join(self.save_path, "dso_result.json")
            with open(save_path, 'w') as f:
                # Filter result dict to get JSON-serializable components only
                json_result = {
                    k: v for k, v in result.items() 
                    if k not in ["program", "pareto_front"]
                }
                json.dump(json_result, f, indent=3)
            print("Training complete. Results saved to path:", save_path)
        
        return result
    
    def set_config(self, config):
        """Set the config, either from dict or path."""
        config = load_config(config) if isinstance(config, str) else config
        self.config = defaultdict(dict, config)
        self.config_task = self.config["task"]
        self.config_prior = self.config["prior"]
        self.config_state_manager = self.config["state_manager"]
        self.config_training = self.config["training"]
        self.config_policy = self.config["policy"]
        self.config_policy_optimizer = self.config["policy_optimizer"]
        self.config_experiment = self.config["experiment"]
        self.config_logging = self.config["logging"]
        self.config_checkpoint = self.config["checkpoint"]
        self.config_gp_meld = self.config["gp_meld"]
    
    def set_seeds(self):
        """Set seeds for reproducibility."""
        seed = self.config_experiment["seed"]
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
    
    def save_config(self):
        """Save the config file to the log directory."""
        if self.save_path is not None:
            config_path = os.path.join(self.save_path, "config.json")
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=3)
    
    def make_prior(self):
        prior = make_prior(Program.library, **self.config_prior)
        return prior
    
    def make_state_manager(self):
        state_manager = make_state_manager(Program.library, **self.config_state_manager)
        return state_manager
    
    def make_policy(self):
        policy = make_policy(Program.library,
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
        
        # Set the Task for the parent process
        set_task(self.config_task)
        
        return pool
    
    def make_output_file(self):
        """Generates an output filename."""
        
        # If logdir is not provided (e.g. for pytest), results are not saved
        if self.config_experiment.get("logdir") is None:
            self.save_path = None
            if self.config_experiment.get("verbose", False):
                print("WARNING: logdir not provided. Results will not be saved to file.")
            return None
        
        # When using run.py, timestamp is already generated
        timestamp = self.config_experiment.get("timestamp")
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
            self.config_experiment["timestamp"] = timestamp
        
        # Generate save path
        task_name = Program.task.name
        save_path = os.path.join(
            self.config_experiment["logdir"],
            f"{task_name}_{timestamp}"
        )
        os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path
        
        # Generate output filename
        output_file = os.path.join(save_path, "summary.csv")
        
        return output_file
    
    def make_logger(self):
        logger = StatsLogger(self.output_file, **self.config_logging)
        return logger
    
    def make_policy_optimizer(self):
        policy_optimizer = make_policy_optimizer(self.policy, **self.config_policy_optimizer)
        return policy_optimizer
    
    def make_trainer(self):
        trainer = Trainer(self.prior,
                        self.state_manager,
                        self.policy,
                        self.policy_optimizer,
                        self.pool,
                        self.gp_controller,
                        self.logger,
                        **self.config_training)
        return trainer
    
    def make_checkpoint(self):
        checkpoint = Checkpoint(self.policy, self.policy_optimizer, self.trainer, **self.config_checkpoint)
        return checkpoint
    
    def save(self, save_path):
        self.checkpoint.save(save_path)
    
    def load(self, load_path):
        self.checkpoint.load(load_path)


# For backward compatibility
DeepSymbolicOptimizerFixed = DeepSymbolicOptimizer  # Alias for compatibility