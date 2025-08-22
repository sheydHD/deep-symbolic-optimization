"""Pytest configuration for DSO test suite.

Keeps project root clean by localizing warning filters to the test package.
Suppresses a third‑party deprecation warning (pygame -> pkg_resources) that is
not actionable within this codebase.
Remove this once pygame eliminates the pkg_resources import or once the
warning is no longer emitted.
"""
from __future__ import annotations
import warnings

# Suppress noisy external deprecation warning only for tests needing gym/control.
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated",
    category=UserWarning,
)
