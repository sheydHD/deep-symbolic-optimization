@echo off
REM Root dispatcher for DSO helper scripts (Windows)

set SCRIPT_DIR=%~dp0

:: Function to display the main interactive menu
call :show_main_menu
goto :eof

:show_main_menu
    :menu_loop
    echo.
    echo === Deep Symbolic Optimization ===
    echo Choose your environment:
    echo   1) Legacy (Python 3.6, TF1)
    echo   2) Modern (Python 3.11, TF2)
    echo   3) Quit
    set /p CHOICE="Enter choice [1-3]: "

    if "%CHOICE%"=="1" (
        call "%SCRIPT_DIR%tools\bat\run.bat"
        goto :menu_loop
    )
    if "%CHOICE%"=="2" (
        call python "%SCRIPT_DIR%tools\python\run.py"
        goto :menu_loop
    )
    if "%CHOICE%"=="3" (
        echo Bye!
        exit /b 0
    )
    echo Invalid choice. Please try again.
    goto :menu_loop

:: Process command-line arguments
if "%1"=="" (
    :: No arguments, show interactive menu
    call :show_main_menu
) else (
    set ENV_CHOICE=%1
    shift

    if /i "%ENV_CHOICE%"=="legacy" (
        call "%SCRIPT_DIR%tools\bat\run.bat" %*
    ) else if /i "%ENV_CHOICE%"=="modern" (
        call python "%SCRIPT_DIR%tools\python\run.py" %*
    ) else (
        echo Error: Invalid environment specified: "%ENV_CHOICE%". Use 'legacy' or 'modern'.
        exit /b 1
    )
)
