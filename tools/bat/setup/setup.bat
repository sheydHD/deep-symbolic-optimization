@echo off
:: Modern setup wrapper for Batch. Calls the Python setup script.

set SCRIPT_DIR=%~dp0

:: Call the Python modern setup script
python "%SCRIPT_DIR%\..\..\python\setup\setup.py" %* 