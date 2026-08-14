# TontumaBot V1 — image de production
FROM python:3.11-slim

# Système : ffmpeg (whisper/audio), tesseract (OCR fra), libgomp (onnxruntime/chromadb)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        tesseract-ocr \
        tesseract-ocr-fra \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/models/hf \
    SENTENCE_TRANSFORMERS_HOME=/app/models/sentence-transformers \
    TRANSFORMERS_CACHE=/app/models/hf \
    STT_MODEL_PATH=/app/models/wolof-whisper-small-lora

WORKDIR /app

# Torch CPU d'abord (évite l'énorme wheel CUDA, ~10x plus petit)
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu

# Dépendances applicatives (contient python-multipart, datasets, scipy…)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Code applicatif (modèles/data persistent via volumes)
COPY app.py demo.py pytest.ini ./
COPY src ./src
COPY data ./data
COPY static ./static

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "app.py"]