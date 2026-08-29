"""The shared route, execute, evaluate, reward, and record service."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from adaptive_router.evaluation import ExactEvaluator, NumericEvaluator, StructuredEvaluator
from adaptive_router.features import extract_features
from adaptive_router.models import AgentResult, AgentStrategy, EvaluationResult, EvaluationType, Task
from adaptive_router.persistence import JSONLRecorder, RunRecord
from adaptive_router.routing import CategoryPolicy
from adaptive_router.routing.base import action_name
from adaptive_router.routing.reward import Reward, calculate_reward


class ServiceError(RuntimeError):
    """A configured application dependency failed."""


class InferenceResponse(BaseModel):
    """Stable response returned by the synchronous inference service."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    policy: str = Field(min_length=1)
    context: tuple[float, ...]
    action: str = Field(min_length=1)
    strategy: str = Field(min_length=1)
    answer: Any = None
    result: AgentResult
    evaluation: EvaluationResult | None = None
    quality: float | None = Field(default=None, ge=0.0, le=1.0)
    passed: bool | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0.0)
    latency_seconds: float = Field(ge=0.0)
    tool_calls: int = Field(ge=0)
    normalized_cost: float | None = None
    normalized_latency: float | None = None
    reward: float | None = None
    error: str | None = None


def _lookup(mapping: Mapping[Any, Any], key: Any) -> Any:
    wanted = action_name(key)
    for candidate, value in mapping.items():
        if action_name(candidate) == wanted:
            return value
    raise KeyError(wanted)


def _evaluation_lookup(mapping: Mapping[Any, Any], key: EvaluationType) -> Any:
    for candidate, value in mapping.items():
        if action_name(candidate) == key.value:
            return value
    return None


def _call_strategy(strategy: Any, task: Task) -> Any:
    method = getattr(strategy, "execute", None) or getattr(strategy, "run", None)
    if method is None and callable(strategy):
        method = strategy
    if method is None:
        raise TypeError("strategy must be callable or expose execute/run")
    return method(task)


def _typed_failure(task: Task, action: str, started: float, error: Exception) -> AgentResult:
    return AgentResult(
        task_id=task.id,
        strategy=AgentStrategy(action),
        answer=None,
        latency_seconds=max(0.0, time.perf_counter() - started),
        input_tokens=None,
        output_tokens=None,
        estimated_cost_usd=None,
        tool_calls=int(getattr(error, "tool_calls", 0)),
        error=f"{type(error).__name__}: {error}",
    )


def _object_data(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    return vars(value)


def _as_result(value: Any, task: Task, action: str) -> AgentResult:
    if isinstance(value, AgentResult):
        return value
    data = dict(_object_data(value))
    data.setdefault("task_id", task.id)
    data.setdefault("strategy", action)
    return AgentResult.model_validate(data)


def _as_evaluation(value: Any) -> EvaluationResult:
    if isinstance(value, EvaluationResult):
        return value
    return EvaluationResult.model_validate(dict(_object_data(value)))


def _public_configuration(settings: Any | None) -> dict[str, Any]:
    if settings is None:
        return {}
    if is_dataclass(settings):
        names = (field.name for field in fields(settings))
        values = {name: getattr(settings, name) for name in names}
    else:
        values = dict(vars(settings)) if hasattr(settings, "__dict__") else {}
    # Never put credentials in a persisted record.
    values.pop("openai_api_key", None)
    return values


class ApplicationService:
    """One synchronous application path shared by the API and experiments."""

    def __init__(
        self,
        *,
        policy: Any,
        strategies: Mapping[Any, Any],
        evaluators: Mapping[Any, Any] | None = None,
        recorder: JSONLRecorder | None = None,
        feature_extractor: Callable[[Any], Sequence[float]] = extract_features,
        policy_name: str | None = None,
        reference_cost_usd: float = 1.0,
        reference_latency_seconds: float = 1.0,
        cost_penalty: float = 0.0,
        latency_penalty: float = 0.0,
        settings: Any | None = None,
    ) -> None:
        self.policy = policy
        self.strategies = strategies
        self.evaluators = (
            {
                EvaluationType.NUMERIC.value: NumericEvaluator(),
                EvaluationType.EXACT.value: ExactEvaluator(),
                EvaluationType.STRUCTURED.value: StructuredEvaluator(),
            }
            if evaluators is None
            else evaluators
        )
        self.recorder = recorder or JSONLRecorder("runs.jsonl")
        self.feature_extractor = feature_extractor
        self.policy_name = policy_name or getattr(policy, "name", policy.__class__.__name__)
        self.reference_cost_usd = reference_cost_usd
        self.reference_latency_seconds = reference_latency_seconds
        self.cost_penalty = cost_penalty
        self.latency_penalty = latency_penalty
        self.settings = settings
        self.configuration = _public_configuration(settings)

    def infer(self, task: Task | Mapping[str, Any], *, evaluate: bool = False) -> InferenceResponse:
        task = Task.model_validate(task)
        context = tuple(float(value) for value in self.feature_extractor(task))
        action = action_name(self.policy.select(context))
        try:
            strategy = _lookup(self.strategies, action)
        except KeyError as exc:
            raise ServiceError(f"no strategy configured for action {action!r}") from exc

        started = time.perf_counter()
        try:
            result = _as_result(_call_strategy(strategy, task), task, action)
        except Exception as exc:
            result = _typed_failure(task, action, started, exc)
        if result.strategy.value != action:
            # The selected action is authoritative in the shared contract.
            result = result.model_copy(update={"strategy": action})

        evaluation: EvaluationResult | None = None
        evaluation_error: str | None = None
        if evaluate:
            evaluator = _evaluation_lookup(self.evaluators, task.evaluation_type)
            if evaluator is not None:
                try:
                    method = getattr(evaluator, "evaluate", evaluator)
                    evaluation = _as_evaluation(method(task, result.answer))
                except Exception as exc:
                    evaluation_error = f"{type(exc).__name__}: {exc}"

        reward: Reward | None = None
        if (
            evaluation is not None
            and result.estimated_cost_usd is not None
            and result.latency_seconds is not None
        ):
            reward = calculate_reward(
                evaluation.quality,
                result.estimated_cost_usd,
                result.latency_seconds,
                reference_cost_usd=self.reference_cost_usd,
                reference_latency_seconds=self.reference_latency_seconds,
                cost_penalty=self.cost_penalty,
                latency_penalty=self.latency_penalty,
            )

        run_id = str(uuid.uuid4())
        error = result.error or evaluation_error
        record = RunRecord(
            task_id=task.id,
            policy=str(self.policy_name),
            context=context,
            action=action,
            strategy=action,
            category=task.category.value,
            answer=result.answer,
            evaluation=evaluation,
            quality=evaluation.quality if evaluation is not None else None,
            passed=evaluation.passed if evaluation is not None else None,
            cost_usd=result.estimated_cost_usd,
            latency_seconds=result.latency_seconds,
            normalized_cost=reward.normalized_cost if reward else None,
            normalized_latency=reward.normalized_latency if reward else None,
            reward=reward.reward if reward else None,
            error=error,
            timestamp=datetime.now(timezone.utc).isoformat(),
            run_id=run_id,
            configuration=self.configuration,
            config=self.configuration,
            result=result,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            tool_calls=result.tool_calls,
        )
        try:
            self.recorder.append(record)
        except Exception as exc:
            raise ServiceError(f"could not persist run record: {exc}") from exc
        if reward is not None:
            self.policy.update(context, action, reward.reward)

        return InferenceResponse(
            run_id=run_id,
            task_id=task.id,
            policy=str(self.policy_name),
            context=context,
            action=action,
            strategy=action,
            answer=result.answer,
            result=result,
            evaluation=evaluation,
            quality=evaluation.quality if evaluation is not None else None,
            passed=evaluation.passed if evaluation is not None else None,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            estimated_cost_usd=result.estimated_cost_usd,
            latency_seconds=result.latency_seconds,
            tool_calls=result.tool_calls,
            normalized_cost=reward.normalized_cost if reward else None,
            normalized_latency=reward.normalized_latency if reward else None,
            reward=reward.reward if reward else None,
            error=error,
        )

    # Names used by callers that describe this boundary as a route operation.
    execute = infer
    route = infer


def build_service(settings: Any | None = None) -> ApplicationService:
    """Build the local service with mock mode as the safe default."""
    from adaptive_router.config import Settings
    from adaptive_router.agents import DirectStrategy, StrongStrategy, ToolStrategy
    from adaptive_router.providers import MockProvider, OpenAICompatibleProvider
    from adaptive_router.models import MockRubricJudge
    from adaptive_router.evaluation import RubricEvaluator

    settings = settings or Settings.from_env()
    provider = (
        MockProvider(input_tokens=1, output_tokens=1)
        if settings.mock_mode
        else OpenAICompatibleProvider(settings=settings)
    )
    strategies = {
        "direct": DirectStrategy(provider, settings=settings),
        "strong": StrongStrategy(provider, settings=settings),
        "tool": ToolStrategy(provider, settings=settings),
    }
    evaluators: dict[str, Any] = {
        EvaluationType.NUMERIC.value: NumericEvaluator(),
        EvaluationType.EXACT.value: ExactEvaluator(),
        EvaluationType.STRUCTURED.value: StructuredEvaluator(),
    }
    if settings.mock_mode:
        evaluators[EvaluationType.RUBRIC.value] = RubricEvaluator(MockRubricJudge())
    return ApplicationService(
        policy=CategoryPolicy(),
        strategies=strategies,
        evaluators=evaluators,
        recorder=JSONLRecorder(settings.persistence_path),
        policy_name="category",
        cost_penalty=settings.cost_penalty,
        latency_penalty=settings.latency_penalty,
        settings=settings,
    )


create_service = build_service

__all__ = [
    "ApplicationService",
    "InferenceResponse",
    "ServiceError",
    "build_service",
    "create_service",
]
