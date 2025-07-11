# Windows Batch Scripts

Automation for Windows users mirrors the Bash workflow.

## Entry points

| File                     | Purpose                                       |
| ------------------------ | --------------------------------------------- |
| `cli/cli_legacy.bat`     | Interactive menu identical to Bash version    |
| `setup/setup_legacy.bat` | Legacy installer using **pyenv-win** + `venv` |
| `run.bat`                | Reserved wrapper (currently unused)           |

### Requirements

1. **pyenv-win** installed and on `%PATH%` (`pyenv --version` should work).
2. Developer Command Prompt / PowerShell with _ExecutionPolicy_ allowing script execution.

### Usage

```
main.bat                 :: opens menu
main.bat setup full      :: you can still call underlying installers directly
```

The installer creates `.venv_36\` in the repo root. To re-activate later:

```
call .venv_36\Scripts\activate
```

Note: Colourised output isn't portable across cmd.exe; messages are plain text.
