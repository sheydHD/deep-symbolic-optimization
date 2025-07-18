@echo off
:: Modern benchmark wrapper for Batch. Calls the Python benchmark script.

set SCRIPT_DIR=%~dp0

:: Call the Python modern benchmark script
python "%SCRIPT_DIR%\..\..\python\benchmark\benchmark.py" %*
