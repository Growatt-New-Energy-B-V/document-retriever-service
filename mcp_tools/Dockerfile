FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server/ ./server/

ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8001

RUN groupadd -g 1000 appuser && useradd -u 1000 -g 1000 -m appuser \
    && chown -R 1000:1000 /app

USER 1000:1000

EXPOSE 8001

CMD ["python", "-m", "server.main"]
