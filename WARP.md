# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Repository: Deep Symbolic Optimization (DSO)

Scope of this guide

- Practical commands you’ll commonly need (setup, build, lint, tests, running/analysing experiments)
- The big-picture architecture so you can navigate and reason about changes quickly
- Key paths and constraints specific to this repo

Do not repeat obvious or generic advice. Prefer the provided scripts and entry points when available.

Common commands

Environment setup (recommended)

- Interactive launcher (delegates to the modern Python toolchain):
  ./main.sh
- Non-interactive setup (creates/updates .venv, installs deps via the repo’s setup tooling):
  ./main.sh setup
- Activate the environment later:
  source .venv/bin/activate

Installing the package for development

- Editable install (root of this repo), including dev tools and optional extras:
  pip install -e ./dso[dev]

  ## for control/RL extras:

  pip install -e ./dso[control]

  ## or everything:

  pip install -e ./dso[all]

Build package artifacts (wheel/sdist)

- From the dso/ subproject (contains pyproject.toml):
  python -m build dso/

  ## Resulting dist/ under dso/

Run tests

- All tests (preferred):
  ./main.sh test
- Direct pytest invocation:
  pytest dso/dso/test -q -rs
- Run a single test file or node:
  pytest dso/dso/test/test_some_module.py::TestClass::test_case -q
- Filter by keyword:
  pytest dso/dso/test -k "keyword" -q

Linting and formatting (available via dev extras)

- Black (format):
  black dso/
- isort (imports):
  isort dso/
- Flake8 (lint):
  flake8 dso/
- mypy (types) – where stubs exist:
  mypy dso/

Run training/experiments

- CLI (single or batch runs) via the module entry point:
  python -m dso.run path/to/config.json
- Named benchmarks (overrides dataset/env in config):
  python -m dso.run dso/dso/config/config_regression.json --b Nguyen-7
  python -m dso.run dso/dso/config/config_control.json --b MountainCarContinuous-v0
- Batch runs across seeds and cores:
  python -m dso.run dso/dso/config/config_regression.json --b Nguyen-1 --runs 100 --n_cores_task 12 --seed 500
- Results directory: created under config["experiment"]["logdir"] (default ./log) with a timestamp, e.g. log/regression_YYYY-MM-DD-HHMMSS/

Analyze logs after runs

- Post-process and summarize (Hall of Fame, Pareto Front, plots):
  python -m dso.logeval ./log/regression_YYYY-MM-DD-HHMMSS --show_hof --show_pf --save_plots

Docs (optional, if you work on documentation)

- Live docs server (uses MkDocs; dev extras provide dependencies):
  uv run mkdocs serve

Big-picture architecture

What DSO is

- A framework to optimize symbolic programs for two first-class task families:
  - regression: recover expressions from data
  - control: learn compact symbolic policies for RL environments
- Configuration-driven: a single JSON specifies experiment/task/policy/optimizer/prior/logging; defaults are merged from dso/dso/config/\*.json.

Primary execution paths

- CLI: python -m dso.run
  - Parses options (Click) for batch execution across seeds and processes
  - Loads/merges configuration (dso/dso/config/**init**.py: load_config/get_base_config)
  - For each run: instantiates DeepSymbolicOptimizer, trains, aggregates per-seed summaries, then invokes post-processing via dso.logeval
- Python API: from dso import DeepSymbolicOptimizer, DeepSymbolicRegressor
  - DeepSymbolicRegressor wraps DeepSymbolicOptimizer with an sklearn-style fit/predict, running fully in-memory (no file logging)

Core orchestration (dso/dso/core.py)

- DeepSymbolicOptimizer is the central object that binds together:
  - Task (dso.task): domain-specific data generation/evaluation (regression or control) assigned via set_task and cached in Program
  - Prior (dso.prior): token priors and constraints for the search space
  - State manager (dso.tf_state_manager): handles TF graph/session state
  - Policy (dso.policy): generative model over token sequences (expressions)
  - Policy optimizer (dso.policy_optimizer): training objective/algorithm (risk-seeking PG, PPO, priority-queue training)
  - Optional GP meld (dso.gp): genetic programming inner-loop optimizer that refines candidates
  - Trainer (dso.train): drives sampling, evaluation, and updates until termination
  - Logger/Checkpoint (dso.train_stats, dso.checkpoint): periodic metrics, Hall of Fame (HoF), Pareto front, checkpoints, and CSV artifacts
- Program (dso.program): representation of candidate expressions, caches, complexity, and execution; holds the current Task and Library
- Multiprocessing
  - Two axes of parallelism are supported but not nested simultaneously:
    - n_cores_task: parallelize independent runs (seeds)
    - n_cores_batch: parallelize evaluation within a run (DeepSymbolicOptimizer guards against nested pools)

Configuration model (dso/dso/config/**init**.py)

- load_config(config) merges:
  - config_common.json
  - task-specific defaults (config_regression.json or config_control.json)
  - language prior defaults (config_language.json) if enabled
  - user overrides from the provided JSON path or dict
- Key top-level groups you’ll commonly adjust (see README for details):
  - experiment, task, training, policy, policy_optimizer, gp_meld, logging, checkpoint, prior, state_manager, postprocess

Logging and evaluation

- Train-time CSVs per seed under the experiment directory, plus HoF and Pareto Front CSVs if enabled; summary.csv aggregates per-seed outcomes
- dso.logeval can summarize and plot results across seeds from a given experiment path

Notable constraints and implementation details

- Python ≥ 3.11 (see dso/setup.py)
- TensorFlow 2.x in TF1 compatibility mode (core.py calls tf.compat.v1.disable_v2_behavior()); single-threaded TF session is used to avoid resource contention in parallel runs
- Cython extension dso.cyfunc is built as part of packaging; build requires numpy/Cython headers (declared in pyproject)
- The repo provides a modern launcher (./main.sh → tools/python/run.py) that wraps setup, tests, and a simple benchmark runner; prefer it for local workflows

Minimal repo map for orientation (not exhaustive)

- dso/ Python package project root (pyproject.toml, setup.py)
- dso/dso/ Package source (core orchestrator, tasks, policy, optimizer, config, logging, scripts)
- configs/ Pinned requirement sets (core/dev/extras) compiled via uv
- tools/ Cross-platform automation (./main.sh dispatches to tools/python/run.py)
- docs/ Documentation site scaffolding
- README.md User-facing overview, configs and task descriptions

Task-specific pointers

- Regression quickly: python -m dso.run dso/dso/config/config_regression.json --b Nguyen-7
- Control quickly: python -m dso.run dso/dso/config/config_control.json --b MountainCarContinuous-v0
- Sklearn interface (in-memory):
  from dso import DeepSymbolicRegressor
  model = DeepSymbolicRegressor() # or pass a config dict/path
  model.fit(X, y)
  yhat = model.predict(X2)

What this file intentionally omits

- Exhaustive file listings (discoverable via your editor/terminal)
- Generic software engineering practices not codified in this repo
