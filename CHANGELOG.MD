# Changelog

**Author**: Antoni Dudij
**Date**: 2025-07-11
**Status**: Unreleased

---

## \[Unreleased]

### Added

- **Initial Documentation System**

  - Established Docusaurus-based documentation hub with core configuration.
  - Added navigation structure and rules overview page (`dso-docs`).

- **Cross-Platform Legacy Setup Automation**

  - Introduced a comprehensive CLI system for setting up the legacy Python 3.6 + TensorFlow 1.14 environment.
  - Features interactive CLI menus for Bash (Unix/Linux/macOS) and Batch (Windows).
  - Automated installation of Python 3.6.15 via `pyenv`.
  - Modular directory structure for maintainability:

    - `tools/bash/cli/cli_legacy.sh`: Bash menu.
    - `tools/bash/setup/setup_legacy.sh`: Bash setup.
    - `tools/bash/test/test_legacy.sh`: Bash test runner.
    - `tools/bat/cli/cli_legacy.bat`: Batch menu.
    - `tools/bat/setup/setup_legacy.bat`: Batch setup.
    - `tools/bat/test/test_legacy.bat`: Batch test runner.

  - Unified entry points:

    - `main.sh` (Unix/Linux/macOS)
    - `main.bat` (Windows)

  - Integrated regression and benchmark testing (including quick Nguyen-5 benchmark).

- **Requirements Management**

  - Added `configs/requirements/legacy/base_legacy_extended.txt` for full legacy installation.
  - Added `configs/requirements/legacy/base_legacy_regression.txt` for minimal regression use case.

### Changed

- **Documentation Enhancements**

  - Updated `docs/guides/legacy_setup.md`:

    - Emphasized new CLI-based workflow.
    - Preserved manual setup steps for advanced users.

- **File Structure & Scripts**

  - Expanded `.gitignore` to support modern Python environments, build tools, and common IDEs.
  - Added `.cursorignore` for compatibility with Cursor analysis tools.
  - Migrated images to `docs/attachments/images/` and updated all corresponding references.

### Removed

- Deleted outdated `docs/index.md` (content merged into `README.md`).
- Removed deprecated `configs/requirements/requirements-legacy36.txt` (merged functionality into new base files).

---

### Notes & Next Steps

- CI integration (t8) pending: implement job executing full setup and test flow.
- Extend CLI with additional benchmark test options.
- Evaluate integration of modern Python environments (e.g., Python 3.10+) into the CLI framework.

---
