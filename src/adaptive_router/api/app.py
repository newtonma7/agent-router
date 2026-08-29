from __future__ import annotations

from typing import Annotated

from fastapi import Body, FastAPI, HTTPException, Query

from adaptive_router.application import (
    ApplicationService,
    InferenceResponse,
    ServiceError,
    build_service,
)
from adaptive_router.models import Task


def create_app(service: ApplicationService | None = None) -> FastAPI:
    """Create the API app, optionally with an injected service for tests."""
    if service is None:
        service = build_service()
    app = FastAPI(title="Adaptive Agent Router", version="0.1.0")
    app.state.service = service

    @app.get("/health")
    def health() -> dict[str, object]:
        settings = getattr(service, "settings", None)
        return {
            "status": "ok",
            "mock_mode": getattr(settings, "mock_mode", True),
        }

    @app.post("/infer", response_model=InferenceResponse)
    def infer(
        task: Annotated[Task, Body()],
        evaluate: Annotated[bool, Query()] = False,
    ) -> InferenceResponse:
        try:
            return service.infer(task, evaluate=evaluate)
        except ServiceError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app
