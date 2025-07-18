"""Proxy so that `import dso` works when the repo root is on PYTHONPATH."""
import sys
from pathlib import Path

# Add the inner dso directory to the Python path
inner_dso_path = Path(__file__).parent / "dso"
if str(inner_dso_path) not in sys.path:
    sys.path.insert(0, str(inner_dso_path))

# Import the modules directly
from dso.core import DeepSymbolicOptimizer
from dso.task.regression.sklearn import DeepSymbolicRegressor

# Re-export the public names
__all__ = ["DeepSymbolicOptimizer", "DeepSymbolicRegressor"]
__version__ = "2.0.0"
