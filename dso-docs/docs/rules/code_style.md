# Code Style & Quality Guide

> Version: 1.0 • Last updated: 2025-07-07

This document defines mandatory coding conventions and quality gates for the Deep Symbolic Optimization (DSO) project.

Follow these rules to ensure readability, consistency, and long-term maintainability.

## 1. Toolchain

| Task               | Tool  | Invocation (local)         |
| ------------------ | ----- | -------------------------- |
| Formatting         | black | `uv pip run black .`       |
| Linting + Imports  | ruff  | `uv pip run ruff check .`  |
| Static typing      | mypy  | `uv pip run mypy .`        |
| Complexity metrics | radon | `uv pip run radon cc dso/` |

All of the above run automatically via **pre-commit** hooks.

## 2. Formatting (Black)

- Line length: **88** chars (Black default).
- Single quotes `'` preferred unless escaping is required.
- Never commit unformatted code – run `make format` or rely on pre-commit.

## 3. Linting (Ruff)

- Ruff ruleset `"all"` minus legacy Py2 checks.
- Important categories: `F/E/W` (flake8), `I` (isort), `D` (pydocstyle), `S` (bandit).
- CI treats **any** Ruff finding as an error.

## 4. Typing (Mypy)

- Run with `--strict`.
- 3rd-party stubs installed via `uv pip install types-<pkg>`.
- `# type: ignore` only with linked issue explaining why.

## 5. Naming Conventions

| Element   | Style            | Example             |
| --------- | ---------------- | ------------------- |
| Modules   | `snake_case.py`  | `memory_manager.py` |
| Packages  | lowercase        | `policy`            |
| Classes   | PascalCase       | `DeepOptimizer`     |
| Functions | snake_case       | `evaluate_fitness`  |
| Constants | UPPER_SNAKE_CASE | `MAX_DEPTH`         |

Avoid cryptic abbreviations unless industry-standard (`np`, `pd`, `tf`).

## 6. Imports

Ruff (isort) enforces three groups separated by a blank line:

1. Standard library
2. Third-party
3. Local (`dso`, `tests`)

## 7. Docstrings

- Use **Google style** (NumPy style acceptable).
- Required for every public module, class, function, and method.
- Sections: `Args`, `Returns`, `Raises`, `Examples`.

## 8. Error Handling & Logging

- Catch the **most specific** exception possible.
- Use the standard `logging` library (`logging.getLogger(__name__)`).
- Never use bare `except:`.

## 9. TODO / FIXME Comments

Reference an issue: `# TODO(#123): rationale`.

## 10. Testability

- Write deterministic functions when feasible.
- Minimise side-effects; inject dependencies.
