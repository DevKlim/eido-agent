FROM python:3.10-slim-buster

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install dos2unix and tesseract-ocr
RUN apt-get update && \
    apt-get install -y --no-install-recommends dos2unix tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Run the RAG indexer during the build
RUN python utils/rag_indexer.py

# Copy the entrypoint script and ensure it's in the correct format
COPY entrypoint.sh /usr/local/bin/
RUN dos2unix /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Create a non-root user and give it permissions
RUN addgroup --system app && adduser --system --group app
RUN chown -R app:app /app
USER app

# Expose the ports the app will listen on
EXPOSE 8000
EXPOSE 8501

# Set the entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Set the default command, which will be passed to the entrypoint script
CMD ["api"]