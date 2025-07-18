@echo off
:: Legacy benchmark runner for Batch.

setlocal

:: SCRIPT_DIR is the directory of the current script
set SCRIPT_DIR=%~dp0

:: This script assumes it's called from tools/bat/run.bat or directly

:: Activate the legacy virtual environment if it exists
if exist ".venv_36\Scripts\activate.bat" (
    call ".venv_36\Scripts\activate.bat"
)

:: Check if dso package is available
python -c "import dso" >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: dso package not found - run setup first.
    goto :eof
)

echo Executing benchmark (Nguyen-5)
python -m dso.run dso/config/examples/regression/Nguyen-2.json --benchmark Nguyen-5

if %errorlevel% neq 0 (
    echo ERROR: Benchmark failed.
) else (
    echo Benchmark finished.
)

endlocal
