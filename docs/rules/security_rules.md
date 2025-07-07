# Security Rules

> Version: 1.0 • Last updated: 2025-07-07

## 1. Secrets Management

- **Never** commit secrets, API keys, or credentials.
- `.env` files are git-ignored; rely on environment variables.
- Pre-commit hook `git-secrets` scans staged changes.

## 2. Dependency Hygiene

- Install via `uv pip install` for full hash-pinned lockfiles.
- Run `uv pip audit` in CI; build fails on critical vulnerabilities.
- Dependabot or Renovate keeps dependencies current.

## 3. Secure Coding Practices

- Avoid `eval`, `exec`, or unsanitised inputs.
- Use parameterised queries for DB interactions.
- Validate external data, preferably with **pydantic** models.

## 4. Runtime Hardening

- Set `PYTHONHASHSEED=0` during tests for reproducibility.
- Implement structured logging and rotate logs securely.

## 5. Vulnerability Disclosure

- Follow `SECURITY.md`; contact maintainers via security@dso.example.com.
