"""Convert prediction API payloads into persistent assistant context."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..models import PredictionContext


class PredictionContextService:
    def __init__(self, history_store) -> None:
        self.history_store = history_store

    def publish(
        self,
        session_id: str,
        prediction_payload: dict[str, Any],
    ) -> PredictionContext:
        disease = dict(prediction_payload.get("prediction") or {})
        context = PredictionContext(
            disease_name=str(disease.get("name", "Unknown")),
            disease_type=str(disease.get("type", "")),
            confidence=float(prediction_payload.get("confidence", 0.0)),
            confidence_tier=str(
                prediction_payload.get("confidence_tier", "low")
            ),
            filename=str(prediction_payload.get("filename", "")),
            timestamp=datetime.now(timezone.utc).isoformat(),
            cause=str(disease.get("cause", "")),
            symptoms=str(disease.get("symptoms", "")),
            treatment=str(disease.get("treatment", "")),
            prevention=str(disease.get("prevention", "")),
            fish_species=prediction_payload.get("fish_species"),
            model_source=str(prediction_payload.get("source", "")),
            inference_ms=float(prediction_payload.get("inference_ms", 0.0)),
            warning=prediction_payload.get("warning"),
            recommendation=prediction_payload.get("recommendation"),
            top_predictions=tuple(
                prediction_payload.get("top_predictions") or ()
            ),
            metadata={
                key: value
                for key, value in prediction_payload.items()
                if key
                not in {
                    "prediction",
                    "confidence",
                    "confidence_tier",
                    "filename",
                    "source",
                    "inference_ms",
                    "warning",
                    "recommendation",
                    "top_predictions",
                    "fish_species",
                }
            },
        )
        self.history_store.set_prediction(session_id, context)
        return context

    def get(self, session_id: str) -> PredictionContext | None:
        return self.history_store.get_prediction(session_id)
