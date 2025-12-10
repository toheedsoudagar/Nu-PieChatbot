# Dockerfile — Streamlit RAG (Chroma offline + Ollama compatible)
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off

WORKDIR /app

# Install system packages for PDF -> image, OCR, and common build deps
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
       build-essential \
       git \
       curl \
       poppler-utils \     # for pdf2image
       tesseract-ocr \     # for pytesseract
       libgl1 \            # sometimes needed by image libs
       libglib2.0-0 \
       libsm6 \
       libxrender1 \
       libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# Copy application code
COPY . /app

# Create a non-root user (optional but recommended)
RUN useradd -m -s /bin/bash appuser || true
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
