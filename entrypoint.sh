# entrypoint.sh
#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

# The first argument to this script is the command to run (e.g., "api" or "ui")
COMMAND=$1

echo "Entrypoint received command: '$COMMAND'"

if [ "$COMMAND" = "api" ]; then
    echo "Starting FastAPI server on port 8000..."
    # Bind to 0.0.0.0 to make the server accessible from outside the container.
    # Convert LOG_LEVEL to lowercase for uvicorn compatibility.
    log_level_lower=$(echo "${LOG_LEVEL:-info}" | tr '[:upper:]' '[:lower:]')
    exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level "$log_level_lower"

elif [ "$COMMAND" = "ui" ]; then
    # The port is set to 8000 to match the expected internal_port for Fly.io deployment, based on error logs.
    # The local docker-compose.yml has been updated to map external port 8501 to this internal port 8000.
    echo "Starting Streamlit UI on port 8000..."
    exec streamlit run ui/app.py --server.port 8000 --server.address 0.0.0.0

else
    echo "Error: Unknown command '$COMMAND'"
    echo "Available commands: api, ui"
    exit 1
fi