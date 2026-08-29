from fastapi.testclient import TestClient

from adaptive_router.api import create_app
from adaptive_router.application import ApplicationService
from adaptive_router.models import AgentResult
from adaptive_router.persistence import JSONLRecorder
from adaptive_router.routing import StaticPolicy


def task_payload():
    return {
        "id": "A1",
        "prompt": "What is 2 + 2?",
        "category": "arithmetic",
        "evaluation_type": "numeric",
        "expected_answer": 4,
    }


def test_health_and_inference_are_public_synchronous_endpoints(tmp_path):
    service = ApplicationService(
        policy=StaticPolicy("direct"),
        strategies={"direct": lambda task: AgentResult(
            task_id=task.id, strategy="direct", answer="4", latency_seconds=0.01,
            input_tokens=2, output_tokens=1, estimated_cost_usd=0.02,
        )},
        recorder=JSONLRecorder(tmp_path / "runs.jsonl"),
    )
    client = TestClient(create_app(service))

    health = client.get("/health")
    response = client.post("/infer?evaluate=true", json=task_payload())

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "direct"
    assert body["answer"] == "4"
    assert body["evaluation"]["passed"] is True
    assert body["latency_seconds"] == 0.01


def test_inference_rejects_invalid_task_before_service_execution(tmp_path):
    service = ApplicationService(
        policy=StaticPolicy("direct"),
        strategies={},
        recorder=JSONLRecorder(tmp_path / "runs.jsonl"),
    )
    client = TestClient(create_app(service))

    response = client.post("/infer", json={**task_payload(), "expected_answer": None})

    assert response.status_code == 422
    assert "expected_answer" in response.text
