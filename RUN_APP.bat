@echo off
echo ========================================
echo  Procurement Classifier v2.1
echo  AI-Powered Classification with
echo  Attribute Extraction & Normalization
echo ========================================
echo.

cd /d "%~dp0"

REM Disable Python bytecode caching
set PYTHONDONTWRITEBYTECODE=1
set PYTHONPYCACHEPREFIX=

REM Detect Python installation
REM Try standard Python first
python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PYTHON_CMD=python
    goto :python_found
)

REM Try Anaconda in common locations
if exist "%USERPROFILE%\anaconda3\python.exe" (
    set PYTHON_CMD="%USERPROFILE%\anaconda3\python.exe"
    goto :python_found
)

if exist "C:\ProgramData\anaconda3\python.exe" (
    set PYTHON_CMD="C:\ProgramData\anaconda3\python.exe"
    goto :python_found
)

if exist "C:\Anaconda3\python.exe" (
    set PYTHON_CMD="C:\Anaconda3\python.exe"
    goto :python_found
)

if exist "%USERPROFILE%\Anaconda3\python.exe" (
    set PYTHON_CMD="%USERPROFILE%\Anaconda3\python.exe"
    goto :python_found
)

if exist "%LOCALAPPDATA%\Continuum\anaconda3\python.exe" (
    set PYTHON_CMD="%LOCALAPPDATA%\Continuum\anaconda3\python.exe"
    goto :python_found
)

REM Python not found
echo [ERROR] Python/Anaconda not found!
echo.
echo Please run INSTALL.bat first to set up the environment.
echo.
pause
exit /b 1

:python_found

REM Check if streamlit is installed
%PYTHON_CMD% -c "import streamlit" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Streamlit found
    echo Starting enhanced UI...
    echo Browser will open automatically at http://localhost:8501
    echo.
    echo Features:
    echo   - Structured attribute extraction (14 attributes)
    echo   - Fingerprinting normalization engine
    echo   - Benchmarking and analytics
    echo.
    echo Press Ctrl+C to stop the server.
    echo.
    %PYTHON_CMD% -m streamlit run app.py
) else (
    REM Try Anaconda path
    if exist "C:\ProgramData\anaconda3\Scripts\streamlit.exe" (
        echo [OK] Using Anaconda Streamlit
        echo Starting enhanced UI...
        echo Browser will open automatically at http://localhost:8501
        echo.
        echo Features:
        echo   - Structured attribute extraction (14 attributes)
        echo   - Fingerprinting normalization engine
        echo   - Benchmarking and analytics
        echo.
        echo Press Ctrl+C to stop the server.
        echo.
        C:\ProgramData\anaconda3\Scripts\streamlit.exe run app.py
    ) else (
        echo [ERROR] Streamlit is not installed!
        echo.
        echo Please install dependencies:
        echo   pip install -r requirements.txt
        echo.
        echo Or if using Anaconda:
        echo   conda install -c conda-forge streamlit
        echo.
        pause
        exit /b 1
    )
)

pause