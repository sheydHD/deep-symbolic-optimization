@echo off
REM Automated installer for the DSO legacy environment (Windows, Python 3.6.15 + TF 1.14).
REM Usage: setup_legacy.bat [regression|full]

setlocal ENABLEDELAYEDEXPANSION

set "PY_VERSION=3.6.15"
set "VENV_NAME=.venv_36"
set "PROFILE=%~1"
if "%PROFILE%"=="" set "PROFILE=regression"

REM Check for pyenv-win
where pyenv >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] pyenv-win not found. Install from https://github.com/pyenv-win/pyenv-win first.
    pause
    exit /b 1
)

REM Ensure Python version installed
pyenv versions | find "%PY_VERSION%" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] Installing Python %PY_VERSION% via pyenv...
    pyenv install %PY_VERSION%
) else (
    echo [INFO] Python %PY_VERSION% already installed.
)

REM Create virtualenv if needed
if exist "%VENV_NAME%" (
    echo [INFO] Virtualenv %VENV_NAME% already exists.
) else (
    echo [INFO] Creating virtualenv %VENV_NAME%.
    pyenv shell %PY_VERSION%
    python -m venv "%VENV_NAME%"
)

REM Activate venv
call "%VENV_NAME%\Scripts\activate.bat"

REM Upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel >nul

REM Install deps
if "%PROFILE%"=="full" (
    set "REQ_FILE=configs\requirements\base_legacy_all.txt"
    set "EXTRA=[all]"
) else (
    set "REQ_FILE=configs\requirements\base_legacy_regression.txt"
    set "EXTRA=[regression]"
)

echo [INFO] Installing dependencies from %REQ_FILE%
python -m pip install -r "%REQ_FILE%"
python -m pip install -e .\dso%EXTRA% --no-deps

REM Basic sanity import test
python - <<PY
import importlib, sys
for pkg in ("tensorflow", "dso"):
    try:
        importlib.import_module(pkg)
        print(f"{pkg} import OK")
    except ImportError as e:
        print(f"Import failed: {e}")
        sys.exit(1)
PY

if "%ERRORLEVEL%" NEQ "0" (
    echo [ERROR] Import test failed.
    pause
    exit /b 1
)

REM Ask to run tests
set /p RUN_TESTS=Run unit tests now? [y/N]:
if /I "%RUN_TESTS%"=="Y" (
    if "%PROFILE%"=="full" (
        python -m pytest -q
    ) else (
        python -m pytest -q dso\dso\task\regression\
    )
)

REM Ask to run benchmark
set /p RUN_BENCH=Run benchmark now? (Nguyen-5) [y/N]:
if /I "%RUN_BENCH%"=="Y" (
    python -m dso.run dso\dso\config\examples\regression\Nguyen-2.json --benchmark Nguyen-5
)

echo [OK] Legacy setup complete. To activate later run "%VENV_NAME%\Scripts\activate".
endlocal
