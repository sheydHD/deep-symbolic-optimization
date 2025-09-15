"""Deep Symbolic Optimization."""

__version__ = "2.0.0"

import importlib.util
import os
spec = importlib.util.spec_from_file_location("core", os.path.join(os.path.dirname(__file__), "core.py"))
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
DeepSymbolicOptimizer = core.DeepSymbolicOptimizer
from .task.regression.sklearn import DeepSymbolicRegressor

__all__ = [
    "DeepSymbolicOptimizer",
    "DeepSymbolicRegressor",
]