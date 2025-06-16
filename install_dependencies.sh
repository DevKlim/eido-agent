#!/bin/bash
echo "--- Installing Python dependencies from requirements.txt ---"

# Ensure pip is up to date
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

echo ""
echo "--- Dependencies installed. ---"
echo "Next steps:"
echo "1. If you haven't already, copy .env.example to .env and fill in your API keys."
echo "2. Run the RAG indexer: python utils/rag_indexer.py"
echo "3. Run the application: ./run_all.sh"