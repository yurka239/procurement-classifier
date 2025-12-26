@echo off
echo ========================================
echo  PA Text Classification Tool
echo  Installation Script
echo ========================================
echo.

cd /d "%~dp0"

REM Check if Python is installed
echo [1/4] Checking Python installation...

REM Try standard Python first
python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PYTHON_CMD=python
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
    echo [OK] Python %PYTHON_VERSION% found
    goto :python_found
)

REM Try Anaconda in common locations
echo Standard Python not found, checking for Anaconda...

if exist "%USERPROFILE%\anaconda3\python.exe" (
    set PYTHON_CMD="%USERPROFILE%\anaconda3\python.exe"
    goto :test_anaconda
)

if exist "C:\ProgramData\anaconda3\python.exe" (
    set PYTHON_CMD="C:\ProgramData\anaconda3\python.exe"
    goto :test_anaconda
)

if exist "C:\Anaconda3\python.exe" (
    set PYTHON_CMD="C:\Anaconda3\python.exe"
    goto :test_anaconda
)

if exist "%USERPROFILE%\Anaconda3\python.exe" (
    set PYTHON_CMD="%USERPROFILE%\Anaconda3\python.exe"
    goto :test_anaconda
)

if exist "%LOCALAPPDATA%\Continuum\anaconda3\python.exe" (
    set PYTHON_CMD="%LOCALAPPDATA%\Continuum\anaconda3\python.exe"
    goto :test_anaconda
)

REM Anaconda not found
echo [ERROR] Python/Anaconda is not found!
echo.
echo Please either:
echo 1. Install Python from https://www.python.org/downloads/
echo    (Make sure to check "Add Python to PATH")
echo.
echo 2. Or install Anaconda from https://www.anaconda.com/
echo.
pause
exit /b 1

:test_anaconda
for /f "tokens=2" %%i in ('%PYTHON_CMD% --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Anaconda Python %PYTHON_VERSION% found
echo.

:python_found

REM Check if pip is available
echo [2/4] Checking pip...
%PYTHON_CMD% -m pip --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] pip is not available!
    echo.
    echo Installing pip...
    %PYTHON_CMD% -m ensurepip --default-pip
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to install pip
        pause
        exit /b 1
    )
)
echo [OK] pip is available
echo.

REM Upgrade pip
echo [3/4] Upgrading pip...
%PYTHON_CMD% -m pip install --upgrade pip --quiet
echo [OK] pip upgraded
echo.

REM Install dependencies
echo [4/4] Installing required packages...
echo This may take a few minutes...
echo.
%PYTHON_CMD% -m pip install -r requirements.txt

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo  Installation Complete!
    echo ========================================
    echo.
    echo Next steps:
    echo 1. Configure your API keys in _Config\config.ini
    echo 2. Double-click RUN_APP.bat to start the application
    echo.
) else (
    echo.
    echo ========================================
    echo  Installation Failed!
    echo ========================================
    echo.
    echo Please check the error messages above.
    echo You may need to run this as Administrator.
    echo.
)

pause