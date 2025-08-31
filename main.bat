@echo off
:: Root launcher for Deep Symbolic Optimization using the modern toolchain.

set SCRIPT_DIR=%~dp0
set MODERN="%SCRIPT_DIR%\tools\python\run.py"

:: Execute the modern toolchain (menu shown if no args)
python %MODERN% %*
