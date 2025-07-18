"""Deep Symbolic Optimization."""

__version__ = "2.0.0"

from .core import DeepSymbolicOptimizer
from .task.regression.sklearn import DeepSymbolicRegressor

__all__ = [
    "DeepSymbolicOptimizer",
    "DeepSymbolicRegressor",
]