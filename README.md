# Adaptive Agent Router

A small contextual-bandit research service that routes tasks to `direct`, `strong`, or bounded `tool` strategies. The local service defaults to deterministic mock mode and records each inference attempt as JSONL.

## Run locally

```bash
python3 -m pip install -e '.[dev]'
uvicorn adaptive_router.main:app --reload
```

The API is available at <http://127.0.0.1:8000/docs>.

```bash
curl http://127.0.0.1:8000/health
curl -X POST 'http://127.0.0.1:8000/infer?evaluate=true' \
  -H 'content-type: application/json' \
  -d '{"id":"A1","prompt":"What is 2 + 2?","category":"arithmetic","evaluation_type":"numeric","expected_answer":4}'
```

The request body is a validated `Task`; `evaluate` is optional and defaults to false. A nested form (`{"task": {...}, "evaluate": true}`) is also accepted. Set `ADAPTIVE_ROUTER_MOCK_MODE=false` and `OPENAI_API_KEY` to use the OpenAI-compatible provider. The API never persists the key. `ADAPTIVE_ROUTER_PERSISTENCE_PATH` controls the JSONL path and defaults to `runs.jsonl`.

## Docker

```bash
docker build -t adaptive-agent-router .
docker run --rm -p 8000:8000 adaptive-agent-router
```

Mock mode needs no credentials. The container exposes `/health`, `/infer`, and interactive `/docs`.

## Checks

```bash
python3 -m pytest -q
python3 -m compileall src
```
