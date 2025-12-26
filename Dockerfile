FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY app.py .
COPY static/ static/
COPY tiny_clip/ tiny_clip/
COPY data/cleaned_emojis.parquet data/
COPY data/text_embeddings.pt data/
COPY data/image_embeddings.pt data/

ENV PORT=7860
EXPOSE 7860

CMD ["gunicorn", "-w", "1", "--threads", "4", "-b", "0.0.0.0:7860", "app:app"]
