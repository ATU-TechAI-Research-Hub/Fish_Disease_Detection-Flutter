"""Local-only FastAPI contract, including NDJSON token streaming."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from ..runtime import AssistantRuntime

SESSION_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
GENERATION_BUSY_MESSAGE = (
    "The local assistant is already answering another prompt. "
    "Wait for it to finish before sending a new request."
)


class ChatRequest(BaseModel):
    session_id: str
    question: str = Field(default="", max_length=4000)
    model: str | None = None
    regenerate: bool = False

    @field_validator("session_id")
    @classmethod
    def validate_session(cls, value: str) -> str:
        if not SESSION_PATTERN.fullmatch(value):
            raise ValueError(
                "session_id must contain 8-128 letters, numbers, _ or -."
            )
        return value


class PredictionContextRequest(BaseModel):
    session_id: str
    prediction: dict[str, Any]

    @field_validator("session_id")
    @classmethod
    def validate_session(cls, value: str) -> str:
        if not SESSION_PATTERN.fullmatch(value):
            raise ValueError("Invalid session_id.")
        return value


assistant_runtime = AssistantRuntime()
assistant_router = APIRouter(prefix="/assistant", tags=["assistant"])


@assistant_router.get("/health")
def assistant_health() -> dict[str, Any]:
    return assistant_runtime.service.status()


@assistant_router.get("/models")
def assistant_models() -> dict[str, Any]:
    status = assistant_runtime.service.status()
    return {
        "active_model": status["active_model"],
        "default_model": status["default_model"],
        "models": status["models"],
    }


@assistant_router.get("/history/{session_id}")
def assistant_history(session_id: str, limit: int = 50) -> dict[str, Any]:
    if not SESSION_PATTERN.fullmatch(session_id):
        raise HTTPException(400, "Invalid session_id.")
    return {
        "session_id": session_id,
        "messages": assistant_runtime.service.history(session_id, limit),
    }


@assistant_router.delete("/history/{session_id}")
def clear_assistant_history(session_id: str) -> dict[str, Any]:
    if not SESSION_PATTERN.fullmatch(session_id):
        raise HTTPException(400, "Invalid session_id.")
    assistant_runtime.service.clear_history(session_id)
    return {"cleared": True, "session_id": session_id}


@assistant_router.delete("/session/{session_id}")
def delete_assistant_session(session_id: str) -> dict[str, Any]:
    if not SESSION_PATTERN.fullmatch(session_id):
        raise HTTPException(400, "Invalid session_id.")
    assistant_runtime.service.delete_session(session_id)
    return {"deleted": True, "session_id": session_id}


@assistant_router.post("/prediction-context")
def set_prediction_context(request: PredictionContextRequest) -> dict[str, Any]:
    context = assistant_runtime.service.publish_prediction(
        request.session_id, request.prediction
    )
    return {"stored": True, "prediction": context.to_dict()}


@assistant_router.post("/reindex")
def rebuild_assistant_index(request: Request) -> dict[str, Any]:
    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        raise HTTPException(403, "Reindexing is restricted to the backend host.")
    try:
        return assistant_runtime.service.rebuild_knowledge(force=True)
    except Exception as exc:
        raise HTTPException(503, str(exc)) from exc


@assistant_router.post("/chat")
def assistant_chat(request: ChatRequest) -> dict[str, Any]:
    if not assistant_runtime.service.try_begin_generation():
        raise HTTPException(409, GENERATION_BUSY_MESSAGE)
    try:
        try:
            return assistant_runtime.service.chat(
                session_id=request.session_id,
                question=request.question,
                model=request.model,
                regenerate=request.regenerate,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(503, str(exc)) from exc
    finally:
        assistant_runtime.service.end_generation()


def _ndjson_events(request: ChatRequest) -> Iterator[str]:
    if not assistant_runtime.service.try_begin_generation():
        yield json.dumps(
            {"type": "error", "message": GENERATION_BUSY_MESSAGE},
            ensure_ascii=False,
        ) + "\n"
        return
    try:
        try:
            for event in assistant_runtime.service.stream_chat(
                session_id=request.session_id,
                question=request.question,
                model=request.model,
                regenerate=request.regenerate,
            ):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as exc:
            yield json.dumps(
                {"type": "error", "message": str(exc)}, ensure_ascii=False
            ) + "\n"
    finally:
        assistant_runtime.service.end_generation()


@assistant_router.post("/chat/stream")
def assistant_chat_stream(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _ndjson_events(request),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
