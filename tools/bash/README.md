# Bash Scripts

This sub-tree houses the Linux/macOS automation layer.

## Key files

| File / Dir              | Role                                                     |
| ----------------------- | -------------------------------------------------------- |
| `cli/cli_legacy.sh`     | Interactive menu (setup, tests, benchmark)               |
| `setup/setup_legacy.sh` | Non-interactive legacy installer (Python 3.6 + TF 1.14)  |
| `lib/utils.sh`          | Reusable helpers (coloured logging, prompts, path utils) |
| `run.sh`                | Thin wrapper – kept for parity with Windows `run.bat`.   |

All scripts are **POSIX-compliant** and self-contained. They resolve the repo root dynamically so they can be launched from any depth:

```bash
./main.sh            # root dispatcher → tools/bash/cli/cli_legacy.sh
```

The installer detects an existing `pyenv` installation, installs Python 3.6.15 on demand, creates `.venv_36`, and installs pinned dependencies from `configs/requirements/`.
