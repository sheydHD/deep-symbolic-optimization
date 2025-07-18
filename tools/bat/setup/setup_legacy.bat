@echo off
:: Legacy setup script for Batch.

setlocal

:: SCRIPT_DIR is the directory of the current script
set SCRIPT_DIR=%~dp0

:: This script assumes it's called from tools/bat/run.bat or directly
:: If run directly, you might need to adjust paths for utils or activate venv manually

:: Placeholder for actual setup logic, which would involve:
:: - Checking for pyenv-win
:: - Installing Python 3.6.x
:: - Creating .venv_36
:: - Installing dependencies from configs/requirements/legacy/

echo.
echo Choose installation profile:
echo   1) Regression-only (light)
echo   2) Full legacy (all extras)
set /p PROFILE_CHOICE="Profile [1/2]: "

if "%PROFILE_CHOICE%"=="2" (
    echo Running full legacy setup...
    :: Example: call the actual legacy setup logic, e.g., using python or a specific batch file
    :: For now, we just echo, as the actual setup_legacy.bat logic would be complex
) else (
    echo Running regression-only setup...
)

:: In a real scenario, you would call the actual setup logic here
:: For example, if you had a separate setup_core.bat or similar:
:: call "%SCRIPT_DIR%..\setup_core.bat" %*

:: This is a placeholder for the actual setup process.
echo Legacy setup complete (placeholder).

endlocal 