#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
VENV_DIR="${SCRIPT_DIR}/venv"
UI_APP_PATH="${SCRIPT_DIR}/ui/app.py"

echo "Attempting to run Streamlit UI..."

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

# Check if the UI app file exists
if [ ! -f "${UI_APP_PATH}" ]; then
    echo "ERROR: Streamlit app file not found at ${UI_APP_PATH}"
    exit 1
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "ERROR: streamlit command not found."
    echo "Please ensure dependencies are installed (e.g., pip install -r requirements.txt)"
    echo "If using a virtual environment, make sure it's activated."
    exit 1
fi

# Run Streamlit
echo "Launching Streamlit app: ${UI_APP_PATH}"
echo "(Access at http://localhost:8501 or the URL provided by Streamlit)"
# You can pass server.port from .env here if needed, but Streamlit usually defaults to 8501
# Example: streamlit run "${UI_APP_PATH}" --server.port ${STREAMLIT_SERVER_PORT:-8501}
streamlit run "${UI_APP_PATH}"

# Deactivate virtual environment (optional, as script ends or is backgrounded by Streamlit)
# if [ -d "${VENV_DIR}" ]; then
#     deactivate
# fi