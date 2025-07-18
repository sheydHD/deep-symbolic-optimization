# Installation Guide

> Version: 1.0 • Last updated: 2025-07-07

## Prerequisites

- Python **3.10** or **3.11** (check with `python --version`).
- GCC / Clang toolchain for Cython extensions.
- (Optional) CUDA-enabled GPU for acceleration.
- **uv package manager** – install once via `pipx install uv` (or `pip install --upgrade uv`).

## Quick Install

1. Clone the repo:  
   `git clone https://github.com/your-org/dso.git && cd dso`

2. Run setup:  
   `./main.sh` (press `1` when prompted)

3. Activate environment:  
   `source .venv/bin/activate`

That's it! You're ready to use DSO.

## Test the Setup

```bash
# Run all tests
pytest -q dso/dso/test/
```

If all tests pass, you are ready to go!

## Understanding `main.sh`

The `main.sh` script is the primary entry point for interacting with the Deep Symbolic Optimization (DSO) project. It provides a convenient way to perform common development and operational tasks through a simple command-line interface.

The script offers the following options:

- **Option 1: Setup the Environment**
  This option (executed by default if no arguments are provided to `main.sh`, or by explicitly running `python tools/python/setup/setup.py`) sets up the necessary Python virtual environment and installs all project dependencies, including the DSO package itself in editable mode. It compiles requirements from the `.in` files located in `configs/requirements/in_files/`.

- **Option 2: Run Tests**
  This option (typically accessed via `main.sh test` or `pytest -q dso/dso/test/`) executes the project's test suite. It verifies that the environment is correctly set up and that all packages and code functionalities are working as expected.

- **Option 3: Run Benchmarks**
  This option (accessible via `main.sh benchmark` or `python tools/python/benchmark/benchmark.py`) allows you to run performance benchmarks on the code. This is crucial for assessing the efficiency and functionality of optimizations and changes within the project.
