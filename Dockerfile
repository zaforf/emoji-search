FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY embeddings.py .
COPY data/emoji.json data/
RUN mkdir -p data && python embeddings.py

COPY app.py .
COPY static/ static/

ENV PORT=7860
EXPOSE 7860

CMD ["gunicorn", "-w", "1", "--threads", "2", "-b", "0.0.0.0:7860", "app:app"]
