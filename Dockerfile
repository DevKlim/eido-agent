FROM python:3.10-slim-buster

# Set environment variables to prevent generating .pyc files and to run python in unbuffered mode
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies required for the project
# - tesseract-ocr: for the OCR functionality in utils/ocr_processor.py
# - sed: to fix Windows line endings in scripts
RUN apt-get update && \
    apt-get install -y --no-install-recommends tesseract-ocr sed && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source code into the container
COPY . .

# Run the RAG indexer script using the local schema file.
RUN python utils/rag_indexer.py

# Copy the entrypoint script, fix its line endings, and make it executable
# This sequence is critical for cross-platform compatibility (Windows -> Linux)
COPY entrypoint.sh /usr/local/bin/
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# --- All privileged operations are done. Now, switch to a non-root user. ---

# Create a non-root user and group for security purposes
RUN addgroup --system app && adduser --system --group app

# Change ownership of the app directory to the new user
RUN chown -R app:app /app

# Switch to the non-root user for runtime
USER app

# Expose the ports the API and UI will run on
EXPOSE 8000
EXPOSE 8501

# Set the entrypoint for the container.
# Call the interpreter explicitly to avoid shebang-related "exec format errors".
ENTRYPOINT ["/bin/sh", "/usr/local/bin/entrypoint.sh"]

# Set the default command to run the API service
CMD ["api"]