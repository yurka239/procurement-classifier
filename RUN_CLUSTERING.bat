@echo off
REM Interactive Clustering Tool Launcher
REM Similar to OpenRefine - Fast clustering with user review

echo ========================================
echo  Interactive Clustering Tool
echo ========================================
echo.
echo Starting clustering tool...
echo.
echo The tool will open in your browser automatically.
echo If it doesn't open, go to: http://localhost:8502
echo.
echo To stop the tool: Close this window or press Ctrl+C
echo.

REM Try to find Python and run streamlit
REM Option 1: Try Anaconda Python
if exist "C:\ProgramData\anaconda3\python.exe" (
    echo Using Anaconda Python...
    start http://localhost:8502
    "C:\ProgramData\anaconda3\python.exe" -m streamlit run clustering_app_v4_complete.py --server.port 8502 --server.headless true
    goto :end
)

REM Option 2: Try system Python
start http://localhost:8502
python -m streamlit run clustering_app_v4_complete.py --server.port 8502 --server.headless true
if %ERRORLEVEL% EQU 0 goto :end

REM Option 3: Try streamlit directly
start http://localhost:8502
streamlit run clustering_app_v4_complete.py --server.port 8502 --server.headless true

:end
pause
