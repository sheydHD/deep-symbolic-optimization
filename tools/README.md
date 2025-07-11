# Tools Directory

This folder contains convenience scripts that automate local workflows for Deep Symbolic Optimization.

| Sub-folder | Purpose                                                                                                                  |
| ---------- | ------------------------------------------------------------------------------------------------------------------------ |
| `bash/`    | POSIX-shell implementation (Linux/macOS). Entry-point: `main.sh` → menu driven utility for setup, testing, benchmarking. |
| `bat/`     | Windows CMD implementation. Entry-point: `main.bat` with identical menu structure.                                       |

Both variants share the same high-level UX:

1. **Setup** – Opens a submenu
   • Legacy stack (Python 3.6 + TF 1.14) – fully automated installer
   • Modern stack – _placeholder for future work_
2. **Run tests** – Executes `pytest` inside the detected virtualenv.
3. **Run benchmark** – Quick Nguyen-5 regression benchmark.

The Bash version centralises common helpers in `bash/lib/utils.sh`. Batch scripts are standalone to avoid `CALL` path complications on Windows.

> Note: Scripts never write outside the repository. Virtual environments live in `.venv_36` (legacy) to keep them self-contained and easy to wipe.
