# Getting Started

> Version: 1.0 • Last updated: 2025-07-07

This quick guide walks you through training a symbolic regression model on the classic _Nguyen-2_ benchmark.

## 1. Install & Activate Environment

First, ensure you have set up your environment as described in [`installation.md`](installation.md).

Then, activate your virtual environment:

```bash
source .venv/bin/activate
```

## 2. Run the Example Benchmark

To run a symbolic regression benchmark with the Nguyen-2 dataset, use the following command:

```bash
python tools/python/benchmark/benchmark.py dso/dso/config/examples/regression/Nguyen-2.json
```

This command will:

1. Parse the JSON config file for Nguyen-2.
2. Build the search space and launch the training process.

Logs and checkpoints will be stored in a timestamped directory (e.g., `log/regression_YYYY-MM-DD_HH-MM-SS/`).

## 3. Visualise the Result

After a successful run, you can inspect the best found program:

```python
from dso import DeepSymbolicOptimizer
import json
import os

# Replace with the actual path to your latest run directory
latest_run_dir = "log/regression_YYYY-MM-DD_HH-MM-SS" # Update this path

with open(os.path.join(latest_run_dir, 'best_program.json')) as f:
    prog = DeepSymbolicOptimizer.load(json.load(f)['program'])
print(prog.sympy_expr)
```

## 4. Run Tests

To verify your setup and the codebase integrity, run the unit tests:

```bash
pytest -q dso/dso/test/
```

## 5. Next Steps

- Explore other datasets in `dso/dso/task/regression/data/`.
- Tweak the search space via `dso/dso/scripts/search_space.py`.
- Read the [Architecture Overview](../architecture/overview.md) to understand the algorithm internals.
