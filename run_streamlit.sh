#!/bin/bash
echo "--- Starting Streamlit Frontend UI ---"

# Source environment variables from .env file if it exists
if [ -f .env ]; then
    echo "Sourcing environment variables from .env file..."
    set -a # automatically export all variables
    source .env
    set +a
else
    echo "WARNING: .env file not found. Using default values."
fi

STREAMLIT_SERVER_PORT=${STREAMLIT_SERVER_PORT:-8501}

streamlit run ui/app.py --server.port "${STREAMLIT_SERVER_PORT}"