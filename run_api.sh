#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
VENV_DIR="${SCRIPT_DIR}/venv"
API_MAIN_PATH="${SCRIPT_DIR}/api/main.py" # Relative path within repo assumed

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
fi

# Check if the API main file exists (though uvicorn checks module path)
if [ ! -f "${API_MAIN_PATH}" ]; then
    echo "Warning: API main file not directly found at ${API_MAIN_PATH}, ensure module path 'api.main:app' is correct."
fi

# Check if uvicorn is installed in the environment
if ! command -v uvicorn &> /dev/null; then
     # Check specifically in the venv if active
     if [ -d "${VENV_DIR}" ] && ! "${VENV_DIR}/bin/uvicorn" --version &> /dev/null; then
          echo "ERROR: uvicorn command not found, even in virtual environment."
          echo "Please ensure dependencies are installed (run ./install_dependencies.sh)"
          exit 1
     elif ! [ -d "${VENV_DIR}" ]; then
          echo "ERROR: uvicorn command not found."
           echo "Please ensure dependencies are installed (run ./install_dependencies.sh and activate the venv)."
          exit 1
     fi
fi


# Run Uvicorn using the module path
# Uses host/port from .env via config/settings.py if available, else defaults
echo "Launching Uvicorn server for 'api.main:app'..."
echo "(API will be accessible at http://<host>:<port>/api/v1, Docs at http://<host>:<port>/docs)"
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# Note: Host/Port here override settings unless uvicorn is configured to read them.
# For production, remove --reload and set host/port explicitly or via config read in api/main.py

# deactivate