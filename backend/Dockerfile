FROM python:3.12-slim
WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock .env ./

RUN uv sync

COPY backend/ ./

EXPOSE 8000

CMD ["uv", "run", "main.py", "--host", "0.0.0.0", "--port", "8000"]
