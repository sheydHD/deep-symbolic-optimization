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

__all__ = [
    'DataVariant',
    'DataShape', 
    'DataHandler',
    'auto_detect_data_structure',
    'ModularProgram',
    'ProgramExecutor',
    'MultiProgram',
    'ModularRNNPolicy'
]
