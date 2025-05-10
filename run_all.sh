#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
VENV_DIR="${SCRIPT_DIR}/venv"

# --- Configuration (Can be overridden by .env file if your apps read it) ---
# Default values if not found in .env or if .env is not sourced here
DEFAULT_API_HOST="0.0.0.0" # Listen on all interfaces
DEFAULT_API_PORT="8000"
DEFAULT_STREAMLIT_PORT="8501"
# Note: Your Python apps (api/main.py, ui/app.py) should ideally load these from .env via config.settings

echo "🚀 Launching EIDO Sentinel (FastAPI Backend & Streamlit UI)..."

# --- Activate Virtual Environment ---
if [ -d "${VENV_DIR}" ]; then
    echo "🐍 Activating virtual environment: ${VENV_DIR}"
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
        source "${VENV_DIR}/Scripts/activate"
    else
        source "${VENV_DIR}/bin/activate"
    fi
else
    echo "⚠️ WARNING: Virtual environment '${VENV_DIR}' not found. Running with system Python."
    echo "Please ensure dependencies are installed in your active Python environment."
fi

# --- Check for uvicorn and streamlit ---
if ! command -v uvicorn &> /dev/null; then
    echo "❌ ERROR: uvicorn command not found. Please install dependencies (pip install -r requirements.txt)."
    exit 1
fi
if ! command -v streamlit &> /dev/null; then
    echo "❌ ERROR: streamlit command not found. Please install dependencies (pip install -r requirements.txt)."
    exit 1
fi

# --- Function to clean up background processes ---
cleanup() {
    echo -e "\n🧹 Cleaning up background processes..."
    if [ -n "$FASTAPI_PID" ]; then
        echo "🔪 Stopping FastAPI server (PID: $FASTAPI_PID)..."
        kill "$FASTAPI_PID" 2>/dev/null
        wait "$FASTAPI_PID" 2>/dev/null # Wait for it to actually terminate
    fi
    echo "🧼 Cleanup complete."
    exit 0
}

# Trap SIGINT (Ctrl+C) and SIGTERM to run cleanup
trap cleanup SIGINT SIGTERM

# --- Start FastAPI Backend in the background ---
# The api/main.py should read host/port from config.settings (which reads .env)
echo "⚙️ Starting FastAPI backend (uvicorn)..."
# Use environment variables for host/port if set, otherwise defaults
# These are passed to uvicorn command line, which might override settings in api.main.py if it doesn't prioritize its own config loading.
# It's generally better if api.main.py's uvicorn.run() call is the source of truth for host/port from .env.
# For this script, we'll assume api.main.py handles its own config for host/port.
uvicorn api.main:app --reload --log-level info &
FASTAPI_PID=$!
echo "🔗 FastAPI backend started (PID: $FASTAPI_PID). Access API docs at http://localhost:${DEFAULT_API_PORT}/docs (adjust port if changed in .env)"
sleep 2 # Give uvicorn a moment to start

# --- Start Streamlit UI in the foreground ---
# ui/app.py should use settings.streamlit_server_port
echo "🎨 Starting Streamlit UI..."
streamlit run ui/app.py
# Streamlit will run in the foreground. When it's stopped (e.g., Ctrl+C in terminal),
# the trap will trigger the cleanup function.

# Fallback cleanup if Streamlit exits normally without trap
cleanup