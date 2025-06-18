#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

# The first argument to this script is the command to run (e.g., "api" or "ui")
COMMAND=$1

echo "Entrypoint received command: '$COMMAND'"

if [ "$COMMAND" = "api" ]; then
    echo "Starting FastAPI server..."
    # Bind to 0.0.0.0 to be accessible from outside the container.
    # Use the PORT environment variable if set by the platform, otherwise default to 8000.
    log_level_lower=$(echo "${LOG_LEVEL:-info}" | tr '[:upper:]' '[:lower:]')
    exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}" --log-level "$log_level_lower"

elif [ "$COMMAND" = "ui" ]; then
    echo "Starting Streamlit UI on port 8501..."
    # The API_BASE_URL environment variable is set in docker-compose.yml to
    # point the UI to the API container.
    exec streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0

else
    echo "Error: Unknown command '$COMMAND'"
    echo "Available commands: api, ui"
    exit 1
fi