@echo off
title FlexFlowSim - Manufacturing Routing Benchmark

:: Always run from the folder where this .bat lives
cd /d "%~dp0"

echo.
echo   ============================================
echo     FlexFlowSim v0.3.0
echo     DES + RL Benchmark for Manufacturing Routing
echo   ============================================
echo.

:: Find Python
python --version >nul 2>&1
if %errorlevel%==0 (
    set PYTHON=python
    goto :found
)
py --version >nul 2>&1
if %errorlevel%==0 (
    set PYTHON=py
    goto :found
)
echo ERROR: Python not found. Install from https://www.python.org
echo Make sure to check "Add Python to PATH" during installation.
pause
exit /b

:found
echo [1/3] Checking dependencies...
%PYTHON% -m pip install -q simpy gymnasium stable-baselines3 scipy pandas matplotlib streamlit plotly openpyxl "kaleido<1.0" 2>nul

:: Kill any existing Streamlit on port 8501
echo [2/3] Clearing port 8501...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8501 ^| findstr LISTENING 2^>nul') do (
    taskkill /PID %%a /F >nul 2>&1
)
timeout /t 1 /nobreak >nul

echo [3/3] Starting FlexFlowSim...
echo.
echo   Dashboard will open in your browser at:
echo   http://localhost:8501
echo.
echo   Press Ctrl+C in this window to stop.
echo.

%PYTHON% -m streamlit run app.py --server.headless false --browser.gatherUsageStats false
pause
