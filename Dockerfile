FROM python:3.10-slim-buster

# Set environment variables to prevent generating .pyc files and to run python in unbuffered mode
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# - tesseract-ocr: for the OCR functionality in utils/ocr_processor.py
# - sed: to fix Windows line endings in scripts
RUN apt-get update && \
    apt-get install -y --no-install-recommends tesseract-ocr sed && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Run the RAG indexer script using the local schema file.
RUN python utils/rag_indexer.py
COPY entrypoint.sh /usr/local/bin/
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Create a non-root user and group for security purposes
RUN addgroup --system app && adduser --system --group app
RUN chown -R app:app /app

# Switch to the non-root user for runtime
USER app

EXPOSE 8000
EXPOSE 8501

ENTRYPOINT ["/bin/sh", "/usr/local/bin/entrypoint.sh"]
CMD ["api"]