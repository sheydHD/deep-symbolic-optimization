@echo off
:: Central dispatcher for legacy Batch scripts.

set SCRIPT_DIR=%~dp0

:main_loop
    echo.
    echo === Deep Symbolic Optimization - Legacy CLI ===
    echo Select an option:
    echo   1) Setup
    echo   2) Run tests
    echo   3) Run benchmark (Nguyen-5)
    echo   4) Quit
    set /p CHOICE="Enter choice [1-4]: "

    if "%CHOICE%"=="1" goto :run_setup
    if "%CHOICE%"=="2" goto :run_tests
    if "%CHOICE%"=="3" goto :run_benchmark
    if "%CHOICE%"=="4" goto :quit
    echo Invalid choice. Please try again.
    goto :main_loop

:run_setup
    call "%SCRIPT_DIR%\setup\setup_legacy.bat" %*
    goto :main_loop

:run_tests
    call "%SCRIPT_DIR%\test\test_legacy.bat" %*
    goto :main_loop

:run_benchmark
    call "%SCRIPT_DIR%\benchmark\benchmark_legacy.bat" %*
    goto :main_loop

:quit
    echo Bye!
    exit /b 0
