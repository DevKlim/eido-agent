#!/bin/sh
# file: entrypoint.sh

# Exit immediately if a command exits with a non-zero status.
set -e

# The first argument to this script is the command to run (e.g., "api" or "ui")
COMMAND=$1

echo "Entrypoint received command: '$COMMAND'"

if [ "$COMMAND" = "api" ]; then
    echo "--- Starting FastAPI server ---"
    # Fly.io sets the PORT environment variable to the internal_port from fly.toml.
    # Uvicorn needs to listen on this port.
    log_level_lower=$(echo "${LOG_LEVEL:-info}" | tr '[:upper:]' '[:lower:]')
    
    UVICORN_CMD="uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --log-level $log_level_lower"
    echo "Will execute: $UVICORN_CMD"
    exec $UVICORN_CMD

elif [ "$COMMAND" = "ui" ]; then
    echo "--- Starting Streamlit UI ---"
    # Fly.io also sets the PORT env var for the 'ui' process group.
    STREAMLIT_CMD="streamlit run ui/app.py --server.port ${PORT:-8501} --server.address 0.0.0.0"
    echo "Will execute: $STREAMLIT_CMD"
    exec $STREAMLIT_CMD

else
    echo "Error: Unknown command '$COMMAND'"
    echo "Available commands: api, ui"
    exit 1
fi