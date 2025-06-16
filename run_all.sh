#!/bin/bash

# =======================================================
# == EIDO Sentinel - Local Development Startup (Linux/macOS) ==
# =======================================================
echo "This script will start the FastAPI backend and the Streamlit UI."

# Function to clean up background processes on exit
cleanup() {
    echo -e "\nShutting down servers..."
    # Kill all background jobs of this script
    if jobs -p | grep . > /dev/null; then
        kill $(jobs -p)
    fi
    echo "Shutdown complete."
    exit
}

# Trap CTRL+C (SIGINT) and call cleanup
trap cleanup SIGINT

# Source environment variables from .env file if it exists
if [ -f .env ]; then
    echo "Sourcing environment variables from .env file..."
    set -a # automatically export all variables
    source .env
    set +a
else
    echo "WARNING: .env file not found. Please copy .env.example to .env and fill it out."
    # Optional: exit if .env is critical
    # exit 1
fi

# Set default values if not defined in .env
API_HOST=${API_HOST:-127.0.0.1}
API_PORT=${API_PORT:-8000}
STREAMLIT_SERVER_PORT=${STREAMLIT_SERVER_PORT:-8501}
LOG_LEVEL=${LOG_LEVEL:-INFO}

# Construct API URL for health check
API_URL="http://${API_HOST}:${API_PORT}"

# Start FastAPI backend in the background
echo "Starting FastAPI backend on ${API_URL}..."
uvicorn api.main:app --host "${API_HOST}" --port "${API_PORT}" --log-level "${LOG_LEVEL,,}" --reload &
FASTAPI_PID=$!
echo "FastAPI backend started with PID: $FASTAPI_PID"

# Health check loop to wait for backend to become available
echo "Waiting for backend to become available..."
max_retries=15 # Max seconds to wait
count=0
while ! curl -s --fail "${API_URL}/" > /dev/null; do
    count=$((count+1))
    if [ "$count" -ge "$max_retries" ]; then
        echo -e "\nError: Backend did not become available after ${max_retries} seconds. Exiting."
        cleanup # Call cleanup to kill FastAPI process
    fi
    printf "." # Print a dot for each retry
    sleep 1
done

echo -e "\nBackend is up and running!"

# Start Streamlit UI in the foreground
echo "Starting Streamlit UI on http://localhost:${STREAMLIT_SERVER_PORT}..."
streamlit run ui/app.py --server.port "${STREAMLIT_SERVER_PORT}"

# The script will wait here until Streamlit is closed.
# When the user presses Ctrl+C, the trap will execute the cleanup function.
wait $FASTAPI_PID