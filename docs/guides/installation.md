# Installation Guide

> Version: 1.0 • Last updated: 2025-07-07

## Prerequisites

- Python **3.10** or **3.11** (check with `python --version`).
- GCC / Clang toolchain for Cython extensions.
- (Optional) CUDA-enabled GPU for acceleration.
- **uv package manager** – install once via `pipx install uv` (or `pip install --upgrade uv`).

## Quick Install

```bash
# Clone repository
git clone https://github.com/your-org/dso.git && cd dso

# Create & activate virtual environment (uv) and install all dependencies
# This script compiles requirements from .in files and installs the DSO package in editable mode.
python tools/python/setup/setup.py

# Activate the environment
source .venv/bin/activate
```

## Test the Setup

```bash
# Run all tests
pytest -q dso/dso/test/
```

If all tests pass, you are ready to go!
