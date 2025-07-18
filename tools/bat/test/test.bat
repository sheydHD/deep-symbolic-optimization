@echo off
:: Modern test wrapper for Batch. Calls the Python test script.

set SCRIPT_DIR=%~dp0

:: Call the Python modern test script
python "%SCRIPT_DIR%\..\..\python\test\test.py" %*
