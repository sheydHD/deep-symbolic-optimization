# Getting Started

> Version: 1.0 • Last updated: 2025-07-07

This quick guide walks you through training a symbolic regression model on the classic _Nguyen-2_ benchmark.

## 1. Install & Activate Environment

See [`installation.md`](installation.md) if you haven't set up DSO yet.

```bash
source .venv/bin/activate
```

## 2. Run the Example

```bash
python -m dso.run \
    --config dso/config/examples/regression/Nguyen-2.json \
    --output runs/nguyen2
```

The script will:

1. Parse the JSON config.
2. Build the search space.
3. Launch training for the configured number of iterations.

Logs and checkpoints are stored under `runs/nguyen2/`.

## 3. Visualise the Result

```python
from dso import DeepSymbolicOptimizer
import json

with open('runs/nguyen2/best_program.json') as f:
    prog = DeepSymbolicOptimizer.load(json.load(f)['program'])
print(prog.sympy_expr)
```

## 4. Next Steps

- Explore other datasets in `dso/task/regression/data/`.
- Tweak the search space via `dso/scripts/search_space.py`.
- Read the [Architecture Overview](../architecture/overview.md) to understand the algorithm internals.
