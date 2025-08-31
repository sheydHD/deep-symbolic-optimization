# Project Structure Conventions

> Version: 1.0 • Last updated: 2025-07-07

Canonical directory layout for the DSO repository.

```text
.
├── dso/               # Source code package
│   └── ...
├── tests/             # Pytest suites mirroring dso/ structure
├── docs/              # Documentation (no executable code)
│   └── rules/         # Policy documents (this folder)
├── examples/          # Jupyter notebooks & usage demos
├── scripts/           # One-off helper scripts
├── data/              # Small reference datasets (<10 MB)
├── .github/workflows/ # CI pipelines
├── pyproject.toml     # Build & tooling configuration
├── CHANGELOG.md
├── LICENSE
└── README.md
```

## Rules

1. Every top-level directory has a `README.md` (except `tests`, explained in main README).
2. Update `CHANGELOG.md` in every release PR.
3. Do **not** commit generated artefacts (builds, virtualenvs); track via `.gitignore`.
4. Large data goes through DVC or external storage; commit pointer files only.
5. Integration tests live in `tests/integration/`, end-to-end flows in `tests/e2e/`.
6. Experimental notebooks reside in `examples/` and must not import from `tests/`.
