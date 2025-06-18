# ./Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1


RUN apt-get update && \
    apt-get install -y --no-install-recommends tesseract-ocr sed && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

RUN python utils/rag_indexer.py

COPY entrypoint.sh /usr/local/bin/
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

RUN addgroup --system app && adduser --system --group app
RUN chown -R app:app /app
USER app
EXPOSE 8000
EXPOSE 8501
ENTRYPOINT ["/bin/sh", "/usr/local/bin/entrypoint.sh"]

# Set the default command to run the UI (Streamlit) service for deployment
CMD ["ui"]