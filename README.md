# Adaptive Agent Router

A small contextual-bandit research service that routes tasks to `direct`, `strong`, or bounded `tool` strategies. The local service defaults to deterministic mock mode and records each inference attempt as JSONL.

## Run locally

```bash
python3 -m pip install -e '.[dev]'
cp .env.example .env
uvicorn adaptive_router.main:app --reload
```

Edit `.env` with your provider key and model settings first. The application loads it automatically; process environment values override it. `.env` is ignored by Git.

The API is available at <http://127.0.0.1:8000/docs>.

```bash
curl http://127.0.0.1:8000/health
curl -X POST 'http://127.0.0.1:8000/infer?evaluate=true' \
  -H 'content-type: application/json' \
  -d '{"id":"A1","prompt":"What is 2 + 2?","category":"arithmetic","evaluation_type":"numeric","expected_answer":4}'
```

The request body is a validated `Task`; `evaluate` is optional and defaults to false. Set `ADAPTIVE_ROUTER_MOCK_MODE=false` and `OPENAI_API_KEY` to use the OpenAI-compatible provider. The API never persists the key. `ADAPTIVE_ROUTER_PERSISTENCE_PATH` controls the JSONL path and defaults to `runs.jsonl`.

## Docker

```bash
docker build -t adaptive-agent-router .
# Default deterministic mock mode:
docker run --rm -p 8000:8000 adaptive-agent-router
# To pass your local .env (for live/custom settings):
docker run --rm --env-file .env -p 8000:8000 adaptive-agent-router
```

Mock mode needs no credentials. The container exposes `/health`, `/infer`, and interactive `/docs`.

## Experiments

Run experiments directly in Python; FastAPI and Podman are not required:

```bash
# Full-information comparison: 3 strategies × 12 seed tasks
python scripts/run_experiment.py

# Online replay for one routing policy
python scripts/run_experiment.py --mode replay --policy linucb --seed 7
```

The script loads `.env`, writes each run to `experiments/runs/`, and writes a unique report to `experiments/reports/` using a timestamp and hash. Use `--records` or `--output` to change those paths. Live runs require provider credits; mock mode only checks the experiment plumbing.

## Rubric judge calibration

Before using live rubric results as experiment evidence, score a small balanced sample of explanation responses by humans, compare those scores with the blind judge, inspect disagreements, and adjust rubric guidance or thresholds. Human scores calibrate and audit the judge; they do not replace it for every run.

## Checks

```bash
python3 -m pytest -q
python3 -m compileall src
./scripts/docker_smoke.sh  # requires Docker and curl
```
