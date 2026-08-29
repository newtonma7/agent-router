from adaptive_router.application import ApplicationService, build_service
from adaptive_router.config import Settings
from adaptive_router.models import AgentResult, EvaluationType, Task, TaskCategory
from adaptive_router.persistence import JSONLRecorder, load_records
from adaptive_router.routing import StaticPolicy
from adaptive_router.evaluation.numeric import NumericEvaluator


def numeric_task() -> Task:
    return Task(
        id="A1",
        prompt="What is 2 + 2?",
        category=TaskCategory.ARITHMETIC,
        evaluation_type=EvaluationType.NUMERIC,
        expected_answer=4,
    )


class RecordingPolicy(StaticPolicy):
    def __init__(self) -> None:
        super().__init__("direct")
        self.contexts: list[tuple[float, ...]] = []

    def select(self, context=None):
        self.contexts.append(tuple(context or ()))
        return super().select(context)


def test_service_routes_evaluates_rewards_and_persists_one_record(tmp_path):
    policy = RecordingPolicy()
    service = ApplicationService(
        policy=policy,
        strategies={"direct": lambda task: AgentResult(
            task_id=task.id,
            strategy="direct",
            answer="4",
            latency_seconds=0.01,
            input_tokens=2,
            output_tokens=1,
            estimated_cost_usd=0.02,
        )},
        evaluators={"numeric": NumericEvaluator()},
        recorder=JSONLRecorder(tmp_path / "runs.jsonl"),
        policy_name="test-policy",
    )

    response = service.infer(numeric_task(), evaluate=True)

    assert response.strategy == "direct"
    assert response.answer == "4"
    assert response.evaluation is not None and response.evaluation.passed is True
    assert response.reward is not None
    assert len(policy.contexts) == 1
    records = load_records(tmp_path / "runs.jsonl")
    assert len(records) == 1
    assert records[0]["task_id"] == "A1"
    assert records[0]["action"] == "direct"
    assert records[0]["evaluation"]["passed"] is True


def test_service_returns_typed_execution_failure_and_still_persists(tmp_path):
    service = ApplicationService(
        policy=StaticPolicy("strong"),
        strategies={"strong": lambda task: (_ for _ in ()).throw(RuntimeError("offline"))},
        recorder=JSONLRecorder(tmp_path / "runs.jsonl"),
    )

    response = service.infer(numeric_task())

    assert response.strategy == "strong"
    assert response.answer is None
    assert response.error is not None and "offline" in response.error
    assert load_records(tmp_path / "runs.jsonl")[0]["error"] is not None


def test_service_validates_mapping_tasks_and_skips_unrequested_evaluation(tmp_path):
    service = ApplicationService(
        policy=StaticPolicy("direct"),
        strategies={"direct": lambda task: AgentResult(
            task_id=task.id, strategy="direct", answer="4", latency_seconds=0,
            input_tokens=0, output_tokens=0, estimated_cost_usd=0,
        )},
        evaluators={"numeric": NumericEvaluator()},
        recorder=JSONLRecorder(tmp_path / "runs.jsonl"),
    )

    response = service.infer(numeric_task().model_dump(), evaluate=False)

    assert response.evaluation is None
    assert response.reward is None


def test_build_service_defaults_to_no_network_mock_mode(tmp_path):
    settings = Settings(persistence_path=str(tmp_path / "runs.jsonl"))
    service = build_service(settings)

    response = service.infer(numeric_task(), evaluate=True)

    assert settings.mock_mode is True
    assert response.strategy == "tool"
    assert response.error is None
    assert response.reward is not None
