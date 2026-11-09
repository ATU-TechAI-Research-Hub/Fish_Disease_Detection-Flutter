"""Framework-neutral domain models used across RAG, API, and history."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RetrievedDocument:
    text: str
    source: str
    title: str
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionContext:
    disease_name: str
    confidence: float
    confidence_tier: str
    filename: str
    timestamp: str
    disease_type: str = ""
    cause: str = ""
    symptoms: str = ""
    treatment: str = ""
    prevention: str = ""
    fish_species: str | None = None
    model_source: str = ""
    inference_ms: float = 0.0
    warning: str | None = None
    recommendation: str | None = None
    top_predictions: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["top_predictions"] = list(self.top_predictions)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PredictionContext":
        return cls(
            disease_name=str(payload.get("disease_name", "Unknown")),
            confidence=float(payload.get("confidence", 0.0)),
            confidence_tier=str(payload.get("confidence_tier", "low")),
            filename=str(payload.get("filename", "")),
            timestamp=str(payload.get("timestamp") or utc_now_iso()),
            disease_type=str(payload.get("disease_type", "")),
            cause=str(payload.get("cause", "")),
            symptoms=str(payload.get("symptoms", "")),
            treatment=str(payload.get("treatment", "")),
            prevention=str(payload.get("prevention", "")),
            fish_species=payload.get("fish_species"),
            model_source=str(payload.get("model_source", "")),
            inference_ms=float(payload.get("inference_ms", 0.0)),
            warning=payload.get("warning"),
            recommendation=payload.get("recommendation"),
            top_predictions=tuple(payload.get("top_predictions") or ()),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class ChatMessage:
    id: str
    session_id: str
    role: str
    content: str
    created_at: str
    model: str | None = None
    sources: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sources"] = list(self.sources)
        return payload
