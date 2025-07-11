@echo off
:main
cls
echo.
echo Deep Symbolic Optimization - Windows CLI
echo.
echo 1^) Setup
echo 2^) Run tests
echo 3^) Run benchmark (Nguyen-5)
echo 4^) Quit
set /p choice=Enter choice [1-4]:
if "%choice%"=="1" goto setup_menu
if "%choice%"=="2" goto tests
if "%choice%"=="3" goto bench
if "%choice%"=="4" goto eof
echo Invalid choice.
pause
goto main

:setup_menu
cls
echo Setup options:
echo 1^) Legacy stack (Python 3.6 + TF1)
echo 2^) Modern stack (placeholder)
echo 3^) Back
set /p setup_choice=Enter choice [1-3]:
if "%setup_choice%"=="1" goto setup_legacy_profile
if "%setup_choice%"=="2" echo Modern setup not implemented yet.& pause & goto setup_menu
if "%setup_choice%"=="3" goto main
echo Invalid choice.& pause
goto setup_menu

:setup_legacy_profile
cls
echo Choose installation profile:
echo 1^) Regression-only (light)
echo 2^) Full legacy (all extras)
set /p prof_choice=Profile [1/2]:
if "%prof_choice%"=="2" (
  call "%~dp0..\setup\setup_legacy.bat" full
) else (
  call "%~dp0..\setup\setup_legacy.bat" regression
)
 pause
 goto main

:tests
if exist .venv_36\Scripts\activate.bat (
  call .venv_36\Scripts\activate.bat
) else (
  echo No virtualenv found. Run setup first.& pause& goto main
)
python -m pytest -q || echo Tests failed.
pause
goto main

:bench
if exist .venv_36\Scripts\activate.bat (
  call .venv_36\Scripts\activate.bat
) else (
  echo No virtualenv found. Run setup first.& pause& goto main
)
python -m dso.run dso\dso\config\examples\regression\Nguyen-2.json --benchmark Nguyen-5
pause
goto main

eof
exit /b
