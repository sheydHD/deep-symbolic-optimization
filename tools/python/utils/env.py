#!/usr/bin/env python3
"""Environment + process helper utilities for DSO tooling.

Centralised here so other action modules stay slim.
"""
from __future__ import annotations

from pathlib import Path
import subprocess, sys

THIS_FILE = Path(__file__).resolve()
PROJECT   = THIS_FILE.parents[3]          # repo root
TOOLS_DIR = PROJECT / "tools" / "python"

def python_exe() -> str:
    """Return path to preferred python (project .venv if present)."""
    if sys.platform == "win32":
        cand = PROJECT / ".venv" / "Scripts" / "python.exe"
    else:
        cand = PROJECT / ".venv" / "bin" / "python"
    return str(cand) if cand.exists() else sys.executable

def run(cmd: list[str], **kw) -> None:
    """Execute command with live output; raise on failure."""
    print("➜", *cmd, flush=True)
    subprocess.run(cmd, check=True, **kw)

__all__ = [
    "PROJECT",
    "TOOLS_DIR",
    "python_exe",
    "run",
]
