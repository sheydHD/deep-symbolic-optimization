
"""Pytest configuration for DSO test suite.

Ensures repo root is on sys.path for robust absolute imports.
Suppresses a third‑party deprecation warning (pygame -> pkg_resources) that is
not actionable within this codebase.
Remove this once pygame eliminates the pkg_resources import or once the
warning is no longer emitted.
"""
from __future__ import annotations
import warnings
import sys
from pathlib import Path

# Ensure the parent of the inner dso directory (repo root) is on sys.path for all test runs
import os
test_dir = Path(__file__).resolve().parent
dso_dir = test_dir.parent
repo_root = dso_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Suppress noisy external deprecation warning only for tests needing gym/control.
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated",
    category=UserWarning,
)
