# Testing Rules

> Version: 1.0 • Last updated: 2025-07-07

## 1. Framework & Tools

- **pytest** ≥ 8.0 for runner.
- **hypothesis** for property-based tests on mathematical functions.
- **pytest-cov** for coverage reporting.

## 2. Layout

- Tests under `tests/` mirroring package paths (`tests/dso/test_core.py`).
- Naming: `test_<module>.py`.
- Each public function/class must have at least one unit test.

## 3. Coverage Targets

- Minimum **80 % branch coverage**; CI fails otherwise.
- New code should not lower overall coverage.

## 4. Parallel & Deterministic

- Tests must be deterministic; seed randomness.
- Enable parallel run: `pytest -n auto`.

## 5. Fixtures & Factories

- Shared fixtures go in `tests/conftest.py`.
- Prefer factory helpers over large static data files.

## 6. Markers

- `@pytest.mark.slow` – >10 s runtime; excluded from default CI.
- `@pytest.mark.integration` – spans multiple components.

## 7. Arrange-Act-Assert Pattern

Structure each test into three clearly separated sections to improve readability.
