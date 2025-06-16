#!/bin/bash
echo "--- Starting FastAPI Backend Server ---"

# Source environment variables from .env file if it exists
if [ -f .env ]; then
    echo "Sourcing environment variables from .env file..."
    set -a # automatically export all variables
    source .env
    set +a
else
    echo "WARNING: .env file not found. Using default values."
fi

# Set default values if not defined in .env
API_HOST=${API_HOST:-127.0.0.1}
API_PORT=${API_PORT:-8000}
LOG_LEVEL=${LOG_LEVEL:-INFO}

uvicorn api.main:app --host "${API_HOST}" --port "${API_PORT}" --log-level "${LOG_LEVEL,,}" --reload