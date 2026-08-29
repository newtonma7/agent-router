#!/usr/bin/env bash
set -euo pipefail

image="adaptive-agent-router:smoke"
container="adaptive-agent-router-smoke-$$"
cleanup() {
  docker rm -f "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker build -t "$image" .
docker run -d --name "$container" -p 18000:8000 "$image" >/dev/null
for _ in $(seq 1 30); do
  if curl --fail --silent http://127.0.0.1:18000/health >/dev/null \
    && curl --fail --silent -X POST 'http://127.0.0.1:18000/infer?evaluate=true' \
      -H 'content-type: application/json' \
      -d '{"id":"A1","prompt":"What is 2 + 2?","category":"arithmetic","evaluation_type":"numeric","expected_answer":4}' \
      >/dev/null; then
    echo "Docker smoke test passed"
    exit 0
  fi
  sleep 1
done

echo "Docker service did not become healthy" >&2
exit 1
