# Dockerfile
# ./Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install Tesseract OCR which is a dependency for pytesseract
RUN apt-get update && \
    apt-get install -y --no-install-recommends tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run the RAG indexer during the build
RUN python utils/rag_indexer.py

# Create a non-root user and give it permissions
RUN addgroup --system app && adduser --system --group app
RUN chown -R app:app /app
USER app

# Expose the port the app will listen on
EXPOSE 8000

# DIRECTLY DEFINE THE COMMAND TO RUN
# This avoids any issues with shell scripts (like entrypoint.sh)
# This CMD specifies the default command, which will be overridden by fly.toml for deployment.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]