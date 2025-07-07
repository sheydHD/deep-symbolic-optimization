# Release Process

> Version: 1.0 • Last updated: 2025-07-07

1. Ensure `main` is green on CI and docs are up-to-date.
2. Bump version in `pyproject.toml` (`poetry version <type>` or manual semver bump).
3. Update `CHANGELOG.md` – move Unreleased section under new version heading.
4. Create branch `release/vX.Y.Z` and open PR (type: `release`).
5. After approval, merge & tag: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`.
6. GitHub Actions will build wheels via **cibuildwheel** and publish to PyPI.
7. Verify artefacts; close related Milestone; announce on Slack/Twitter.
