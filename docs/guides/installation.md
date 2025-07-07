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

# Create & activate virtual environment (uv)
uv venv .venv
source .venv/bin/activate

# Install core dependencies
uv pip install -r requirements.txt

# Install extras
uv pip install -r requirements-control.txt   # RL/control tasks
```

## Editable Install

```bash
uv pip install -e .  # Editable mode; reflects local changes immediately
```

## Test the Setup

```bash
pytest -q tests/
```

If all tests pass, you are ready to go!
