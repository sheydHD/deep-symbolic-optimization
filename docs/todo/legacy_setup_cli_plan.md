---
draft: true
---

# Legacy Setup CLI Automation Plan

> Purpose: Provide lightweight, modular shell scripts that automate installation, testing, and benchmarking of the **DSO legacy (Python 3.6 + TF 1.14)** stack. New users should be able to reproduce the workflow described in `docs/guides/legacy_setup.md` via an interactive menu.
>
> **Legend** `[ ]` pending `[⟳]` in-progress `[✓]` done

---

## 1 Repository Structure

| ID  | Task                                                                   | Status |
| --- | ---------------------------------------------------------------------- | ------ |
| 1.1 | Create top-level `tools/` directory (with `bash/` & `bat/` sub-trees). | ✓      |
| 1.2 | Add Bash entry-point `tools/bash/cli_legacy.sh` (empty skeleton).      | ✓      |
| 1.3 | Add repo-root dispatcher `main.sh` calling the CLI helper.             | ✓      |

## 2 CLI Framework (Bash)

- [ ] 2.1 Design interactive menu with options: **Setup • Test • Benchmark • Quit**.
- [ ] 2.2 Extract reusable helpers into `tools/bash/lib/utils.sh` (colours, logging, prompts).
- [ ] 2.3 Ensure **POSIX-compliant** syntax (works on macOS & Linux) and dynamic path resolution (`SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)`).

## 3 Setup Workflow (Legacy stack only)

| Sub-step | Description                                                                                                                                                         | Status |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| 3.1      | Detect host OS & confirm Bash ≥ 3.2.                                                                                                                                | [ ]    |
| 3.2      | Check for **pyenv**; install automatically if missing.                                                                                                              | [ ]    |
| 3.3      | Install **Python 3.6.15** via pyenv (skip if already present).                                                                                                      | [ ]    |
| 3.4      | Create **project-local venv** (`.venv_36`) inside repo root.                                                                                                        | [ ]    |
| 3.5      | Prompt user: **Regression-only** vs **Full legacy** install.                                                                                                        | [ ]    |
| 3.6      | Install dependencies using selected requirements file:<br/>• `configs/requirements/base_legacy_regression.txt`  or<br/>• `configs/requirements/base_legacy_all.txt` | [ ]    |
| 3.7      | Pin pip tooling (`pip`, `setuptools`, `wheel`) in venv.                                                                                                             | [ ]    |
| 3.8      | Verify installation by importing `tensorflow` & `dso`.                                                                                                              | [ ]    |
| 3.9      | Offer to rerun setup idempotently (skip steps already satisfied).                                                                                                   | [ ]    |

## 4 Testing & Benchmarking

- [ ] 4.1 **Test** option:
  - Regression-only → `pytest -q dso/task/regression/`
  - Full install → `pytest -q`
- [ ] 4.2 **Benchmark** option: run quick Nguyen-5 example as documented.
- [ ] 4.3 Prompt post-setup: _"Run unit tests now?"_ / _"Run benchmark now?"_.

## 5 Script Quality & UX Enhancements

- [ ] 5.1 Modularise: one function per high-level operation (`install_pyenv`, `create_venv`, `install_deps`, `run_tests`, ...).
- [ ] 5.2 Consistent coloured logging (`info`, `warn`, `error`).
- [ ] 5.3 Graceful error handling with exit codes.
- [ ] 5.4 Generate concise summary at the end (installed Python path, venv, chosen profile).

## 6 Cross-platform Notes

- [ ] 6.1 Mirror functionality in Windows Batch files under `tools/bat/` (tracked separately).

## 7 Documentation Updates

- [ ] 7.1 Revise `docs/guides/legacy_setup.md` to reference the new CLI.
- [ ] 7.2 Add command snippets (`./main.sh legacy`) & screenshots/gif (optional).

## 8 CI Integration

- [ ] 8.1 Add GitHub Actions job that spins a container, runs CLI **Setup → Test** to ensure scripts stay green.

## 9 Milestones & Ownership

| Milestone                         | Owner | Target Date |
| --------------------------------- | ----- | ----------- |
| Core Bash CLI completed           | _TBD_ | 2025-07-XX  |
| Regression workflow validated     | _TBD_ | 2025-07-XX  |
| Full legacy workflow validated    | _TBD_ | 2025-07-XX  |
| Docs & CI merged to `compat-py36` | _TBD_ | 2025-07-XX  |

---

**Next action**: Implement section 2 (CLI menu) and section 3 (automated setup) following the above checklist.
