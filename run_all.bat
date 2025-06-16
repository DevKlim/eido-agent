@echo off
ECHO =======================================================
ECHO == EIDO Sentinel - Local Development Startup (Windows) ==
ECHO =======================================================
ECHO.
ECHO This script will start the FastAPI backend and the Streamlit UI.
ECHO Two new command prompt windows will open.
ECHO Please close this window and the two new windows to stop the servers.
ECHO.

REM Check for python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: 'python' command not found. Please ensure Python is installed and in your PATH.
    pause
    exit /b
)

REM Check for uvicorn
uvicorn --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: 'uvicorn' not found. Have you installed requirements? (pip install -r requirements.txt)
    pause
    exit /b
)

REM Check for streamlit
streamlit --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: 'streamlit' not found. Have you installed requirements? (pip install -r requirements.txt)
    pause
    exit /b
)

REM Load variables from .env file if it exists
if exist .env (
    for /f "tokens=1,* delims==" %%a in ('findstr /R /V "^#" .env') do (
        set "%%a=%%b"
    )
)

REM Use defaults if not set in .env
if not defined API_PORT set API_PORT=8000
if not defined STREAMLIT_SERVER_PORT set STREAMLIT_SERVER_PORT=8501
if not defined API_HOST set API_HOST=127.0.0.1

ECHO Starting FastAPI backend on http://%API_HOST%:%API_PORT%...
start "FastAPI Backend" cmd /c "uvicorn api.main:app --host %API_HOST% --port %API_PORT% --reload"

ECHO.
ECHO Starting Streamlit UI on http://localhost:%STREAMLIT_SERVER_PORT%...
timeout /t 3 /nobreak >nul
start "Streamlit UI" cmd /c "streamlit run ui/app.py --server.port %STREAMLIT_SERVER_PORT%"

ECHO.
ECHO Both servers are starting in new windows.
ECHO =======================================================