@echo off
REM Root dispatcher for DSO helper scripts (Windows)
set SCRIPT_DIR=%~dp0
auto
set CLI_PATH=%SCRIPT_DIR%tools\bat\cli\cli_legacy.bat
if not exist "%CLI_PATH%" (
  echo Error: CLI script not found at %CLI_PATH%
  exit /b 1
)
call "%CLI_PATH%" %*
