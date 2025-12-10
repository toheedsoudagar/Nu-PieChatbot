FROM python:3.12-slim

WORKDIR /app

# Optional system deps (adjust if you need tesseract/onnx etc)
RUN apt-get update && apt-get install -y build-essential git curl --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# copy project
COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
