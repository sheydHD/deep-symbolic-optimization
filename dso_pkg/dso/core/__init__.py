"""
Core modules for Deep Symbolic Optimization.

This package contains core functionality including data type detection,
modular program execution, and policy management.
"""

from .data_types import (
    DataVariant,
    DataShape,
    DataHandler,
    auto_detect_data_structure
)

from .modular_program import (
    ModularProgram,
    ProgramExecutor,
    MultiProgram
)

from .modular_policy import (
    ModularRNNPolicy
)

# Import DeepSymbolicOptimizer from the parent core.py file for backward compatibility
import os
import importlib.util
spec = importlib.util.spec_from_file_location("_core", os.path.join(os.path.dirname(__file__), "..", "core.py"))
_core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_core)
DeepSymbolicOptimizer = _core.DeepSymbolicOptimizer

__all__ = [
    'DataVariant',
    'DataShape', 
    'DataHandler',
    'auto_detect_data_structure',
    'ModularProgram',
    'ProgramExecutor',
    'MultiProgram',
    'ModularRNNPolicy',
    'DeepSymbolicOptimizer'
]
