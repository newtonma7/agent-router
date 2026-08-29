FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data

RUN pip install --no-cache-dir .

ENV ADAPTIVE_ROUTER_MOCK_MODE=true
ENV ADAPTIVE_ROUTER_PERSISTENCE_PATH=/app/runs.jsonl
EXPOSE 8000

CMD ["uvicorn", "adaptive_router.main:app", "--host", "0.0.0.0", "--port", "8000"]
