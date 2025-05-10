#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
VENV_DIR="${SCRIPT_DIR}/venv"
API_MAIN_MODULE="api.main:app"

echo "Attempting to run FastAPI API..."

# Check if virtual environment exists and activate it
if [ -d "${VENV_DIR}" ]; then
    echo "Activating virtual environment: ${VENV_DIR}"
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
        source "${VENV_DIR}/Scripts/activate"
    else
        source "${VENV_DIR}/bin/activate"
    fi
else
    echo "WARNING: Virtual environment '${VENV_DIR}' not found. Running with system Python."
    echo "Please ensure dependencies are installed in your active Python environment."
fi

# Check if uvicorn is installed
if ! command -v uvicorn &> /dev/null; then
    echo "ERROR: uvicorn command not found."
    echo "Please ensure dependencies are installed (e.g., pip install -r requirements.txt)"
    echo "If using a virtual environment, make sure it's activated."
    exit 1
fi

# Run Uvicorn
# Host/Port are typically read from config.settings by api.main.py when uvicorn.run is called there.
# If running directly like this, these args override.
# The api.main.py itself also calls uvicorn.run if run as __main__, so this script is an alternative way.
echo "Launching Uvicorn server for '${API_MAIN_MODULE}'..."
echo "API docs will be accessible at http://<configured_host>:<configured_port>/docs"
echo "API root will be at http://<configured_host>:<configured_port>/"
echo "To use settings from .env for host/port, ensure api.main.py reads them for its uvicorn.run call if you execute that directly."

# Use 0.0.0.0 to be accessible on the network. Use 127.0.0.1 for local only.
# The port 8000 is a common default. These will be overridden by what's in your .env if api.main.py's uvicorn.run is used.
uvicorn ${API_MAIN_MODULE} --reload --host 0.0.0.0 --port 8000 --log-level info

# Deactivate virtual environment (optional, as script ends here or uvicorn takes over)
# if [ -d "${VENV_DIR}" ]; then
#     deactivate
# fi