---
draft: true
---

# Deep Symbolic Optimization – Modernization Roadmap

> Purpose: Bring the repository up-to-date with the modern Python (3.11+) scientific stack, replace deprecated libraries, and improve overall code quality, testing, and maintainability.
>
> **Legend**  
> `[ ]` pending `[⟳]` in-progress `[✓]` done

---

## 1. Preparation & Baseline

1.1 [ ] Create a new long-lived branch `modernize-2025` off `main` to track all changes.

1.2 [ ] Ensure you are using **Python ≥ 3.10** (check with `python --version`).  
 If necessary, install via `pyenv` or Conda.

1.3 [ ] Create a reproducible dev environment using **`uv`** (project convention) or **Poetry**:

```bash
uv venv .venv
source .venv/bin/activate
```

1.4 [ ] Pin _current_ package versions using **`uv pip freeze`** (`uv pip freeze > requirements-legacy.txt`) so we can still run the historical code while migrating.

1.5 [ ] Enable **pre-commit** hooks with modern tools: **ruff** (lint + import-sort), **black** (format), and **mypy** (typing skeleton).

1.6 [ ] For **every** code or documentation change, ensure compliance with the internal standards described in [`git_rules`](../rules/git_rules.md) and [`docs_rules`](../rules/docs_rules.md).  
 Commit messages **must** follow _Conventional Commits_; docs must follow the formatting & structure rules. Add missing docs _in the same PR_ as the change.

---

## 2. Continuous Integration

2.1 [ ] Add GitHub Actions workflow `.github/workflows/ci.yml` running tests on **Python 3.10 & 3.11**.

2.2 [ ] Fail fast on **ruff** (lint/format) & **mypy** type errors.

---

## 3. Dependency Audit & Upgrade Strategy

3.1 [ ] Document all direct dependencies in a **`pyproject.toml`** / **`requirements.txt`** pair; group extras (`control`, `regression`).

3.2 [ ] Replace / upgrade _critical_ outdated libraries:

- TensorFlow 1.14 → **TensorFlow 2.x** or **PyTorch 2.x** (decision in step 4).
- Gym 0.15 → **Gymnasium** (or Gym 1.x).
- stable-baselines 2 → **stable-baselines3**.
- numpy ≤ 1.19 → **numpy ≥ 1.26** (compatible with TF2 / PyTorch).
- numba 0.53 → latest 0.58+.
- Cython pin removed – rely on latest.

  3.3 [ ] Remove obsolete packages (`progress`, `prettytable`?), prefer maintained alternatives (`rich`, `tabulate`).

  3.4 [ ] Add safety net: run **`uv pip audit`** (built-in vulnerability scanner) & **safety** in CI.

---

## 4. Framework Migration (TF1 → TF2 or PyTorch)

4.1 [ ] **Short-term compatibility layer**: introduce `dso/compat/tf.py` wrapping `tf.compat.v1` to avoid massive churn.

4.2 [ ] Refactor session-based code (`tf.Session`, `tf.reset_default_graph`, etc.) to _eager execution_ or **`tf.function`** where possible.

4.3 [ ] Evaluate feasibility of full PyTorch port (may simplify future maintenance). Create POC for `core.DeepSymbolicOptimizer` training loop in PyTorch Lightning.

4.4 [ ] Decide path, document in README, and update the rest of roadmap accordingly.

---

## 5. RL Components Update (control tasks)

5.1 [ ] Update custom environments (`task/control/envs`) to Gymnasium API (`reset(seed=…)`, `terminated`, `truncated`).

5.2 [ ] Migrate from stable-baselines 2 APIs to stable-baselines3 equivalents (`TD3`, `SAC`, etc.).

5.3 [ ] Replace MPI-based code with SB3 vectorized environments or Ray where needed.

---

## 6. Cython & Numba Extensions

6.1 [ ] Modernize `dso/cyfunc.pyx`:

- Use **typed memoryviews** instead of `np.ndarray` indexing.
- Replace deprecated `cdef inline` patterns.

  6.2 [ ] Ensure compilation succeeds with **Cython ≥ 3** & numpy ≥ 1.26 headers.

  6.3 [ ] Re-benchmark vs. pure-Python & Numba implementations; choose fastest/simple path.

---

## 7. Codebase Cleanup & Typing

7.1 [ ] Adopt **PEP 561** type hints across modules (`dso/*`).

7.2 [ ] Split large modules (>300 LOC) into smaller, focused files (e.g., `core.py`, `library.py`).

7.3 [ ] Remove unused imports & dead code (`Program.clear_cache`, etc.).

7.4 [ ] Add docstrings (Google style) and inline comments where algorithmic logic is non-obvious.

---

## 8. Testing Enhancement

8.1 [ ] Ensure all existing pytest suites pass under the _compatibility layer_ (step 4.1).

8.2 [ ] Introduce parametrized tests for multiple frameworks (TF2 vs. PyTorch) behind a `--framework` marker.

8.3 [ ] Add regression tests for RL environments & GP meld logic.

8.4 [ ] Target **≥ 80 %** coverage; run in CI using `pytest-cov`.

---

## 9. Documentation

9.1 [ ] Replace legacy README portions with up-to-date installation & usage guides (Markdown + diagrams).

9.2 [ ] Auto-generate API docs with **mkdocs-material** or **Sphinx** (with `numpydoc`).

9.3 [ ] Add example notebooks under `examples/` showcasing typical workflows (regression, control, hybrid).

---

## 10. Release & Maintenance

10.1 [ ] Bump version to `2.0.0` following **semver** once migration complete.

10.2 [ ] Publish wheels to PyPI (build via `cibuildwheel`).

10.3 [ ] Tag release, update Zenodo DOI, and archive datasets.

---

_Feel free to break down any step further; the checklist is intentionally granular but can still be split if a sub-task proves complex._
