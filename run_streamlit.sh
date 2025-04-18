#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
VENV_DIR="${SCRIPT_DIR}/venv"
UI_APP_PATH="${SCRIPT_DIR}/ui/app.py"

echo "Attempting to run Streamlit UI..."

# Check if virtual environment exists and activate it
if [ -d "${VENV_DIR}" ]; then
    echo "Activating virtual environment: ${VENV_DIR}"
    # Use appropriate activation command based on OS
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
        source "${VENV_DIR}/Scripts/activate"
    else
        source "${VENV_DIR}/bin/activate"
    fi
else
    echo "WARNING: Virtual environment '${VENV_DIR}' not found. Running with system Python."
fi

# Check if the UI app file exists
if [ ! -f "${UI_APP_PATH}" ]; then
    echo "ERROR: Streamlit app file not found at ${UI_APP_PATH}"
    exit 1
fi

# Run Streamlit
echo "Launching Streamlit app: ${UI_APP_PATH}"
echo "(Access at http://localhost:8501 or the URL provided by Streamlit)"
streamlit run "${UI_APP_PATH}"

# Deactivate might not run if streamlit run keeps the shell busy,
# but good practice to include if the command exits cleanly.
# deactivate