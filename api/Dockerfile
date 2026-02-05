FROM python:3.11-slim

# claude-agent-sdk bundles Claude Code CLI which needs node
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY tests/ ./tests/

ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/data
ENV DB_PATH=/data/tasks.db

RUN groupadd -g 1000 appuser && useradd -u 1000 -g 1000 -m appuser \
    && mkdir -p /data/uploads /data/cache \
    && chown -R 1000:1000 /data /app

USER 1000:1000

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
