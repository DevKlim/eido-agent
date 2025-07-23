# file: Dockerfile
FROM python:3.11-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies required by the application
RUN apt-get update && \
    apt-get install -y --no-install-recommends dos2unix tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# The --extra-index-url from requirements.txt will be used here
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run the RAG indexer during the build to pre-process schema data
RUN python utils/rag_indexer.py

# Copy the entrypoint script and ensure it has correct line endings and permissions
COPY entrypoint.sh /usr/local/bin/
RUN dos2unix /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

# Create a non-root user for security best practices
RUN addgroup --system app && adduser --system --group app
# Give ownership of the app directory to the new user
RUN chown -R app:app /app
# Switch to the non-root user
USER app

# Expose the ports for the API and UI services
EXPOSE 8000
EXPOSE 8501

# Set the entrypoint script to be executed when the container starts
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Set the default command, which will be passed as the first argument to the entrypoint
CMD ["api"]
