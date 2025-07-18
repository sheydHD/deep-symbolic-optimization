@echo off
:: Legacy test runner for Batch.

setlocal

:: SCRIPT_DIR is the directory of the current script
set SCRIPT_DIR=%~dp0

:: This script assumes it's called from tools/bat/run.bat or directly

:: Activate the legacy virtual environment if it exists
if exist ".venv_36\Scripts\activate.bat" (
    call ".venv_36\Scripts\activate.bat"
)

:: Check if pytest is available
where pytest >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: pytest not found - run setup first.
    goto :eof
)

echo Running tests...
pytest -q

if %errorlevel% neq 0 (
    echo ERROR: Tests failed.
) else (
    echo Tests completed.
)

endlocal
