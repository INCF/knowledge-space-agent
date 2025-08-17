FROM python:3.12-slim
WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock .env ./

COPY backend/service-account.json ./service-account.json

RUN UV_HTTP_TIMEOUT=300 uv sync

COPY backend/ ./

EXPOSE 8000

ENV GOOGLE_APPLICATION_CREDENTIALS=/app/service-account.json

CMD ["uv", "run", "main.py", "--host", "0.0.0.0", "--port", "8000"]
