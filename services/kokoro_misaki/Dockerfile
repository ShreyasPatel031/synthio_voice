FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf \
    KOKORO_VOICE=af_heart \
    KOKORO_SPEED=1.0 \
    KOKORO_SPEED_LOCK=1 \
    PORT=8080 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
      espeak-ng \
      libsndfile1 \
      ffmpeg \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
      torch==2.5.1 torchaudio==2.5.1 \
      --index-url https://download.pytorch.org/whl/cu124 \
 && pip install --no-cache-dir -r requirements.txt

COPY app.py prefetch_weights.py kokoro_misaki_lexicon.json ./
RUN python prefetch_weights.py && rm -f prefetch_weights.py

EXPOSE 8080
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1
