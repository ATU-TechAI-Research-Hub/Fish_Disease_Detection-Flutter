"""High-level prediction service used by FastAPI routes.

Wraps:
  - the unified `BaseModel` (Keras .h5 → ONNX fallback)
  - the centralised `preprocess_image` pipeline (matches the paper)
  - rich `diseases.json` metadata (cause, symptoms, treatment, prevention)

The service is responsible for:
  * loading every disease record
  * deciding when a prediction is too uncertain ("No Fish Detected")
  * categorising confidence as High / Medium / Low for the UI
  * returning a fully populated `PredictionResponse` ready for serialisation
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from app.core import (
    BaseModel,
    LabelMap,
    PreprocessingError,
    detect_fish,
    is_low_quality,
    load_label_map,
    load_model,
    preprocess_image,
    warmup_fish_gate,
)
from app.models import (
    ClassProbability,
    ConfidenceTier,
    Disease,
    ModelStatus,
    PredictionResponse,
)

logger = logging.getLogger(__name__)

NO_FISH_DISEASE_ID = 0

DEFAULT_HIGH_CONFIDENCE = 0.70
DEFAULT_MEDIUM_CONFIDENCE = 0.45

# "No Fish Detected" is only meant to catch genuine non-fish / garbage inputs
# where the model has no signal at all (a near-uniform softmax). Real-world
# phone photos often classify at 30-45% confidence, which is still a valid
# diagnosis for a 7-class problem (chance = 1/7 ≈ 14%). We therefore only
# reject when the output is *both* very low-confidence *and* near-uniform.
# Max entropy for 7 classes is ln(7) ≈ 1.9459.
DEFAULT_NO_FISH_THRESHOLD = 0.20
DEFAULT_ENTROPY_THRESHOLD = 1.90
DEFAULT_FISH_GATE_THRESHOLD = 0.05
TOP_K_PREDICTIONS = 3


class PredictionService:
    """Loads the model + labels + disease metadata and serves predictions."""

    def __init__(
        self,
        diseases_file: Path,
        labels_file: Path,
        h5_model_path: Optional[Path] = None,
        onnx_model_path: Optional[Path] = None,
        no_fish_threshold: float = DEFAULT_NO_FISH_THRESHOLD,
        entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
        high_confidence: float = DEFAULT_HIGH_CONFIDENCE,
        medium_confidence: float = DEFAULT_MEDIUM_CONFIDENCE,
        enable_fish_gate: bool = True,
        fish_gate_threshold: float = DEFAULT_FISH_GATE_THRESHOLD,
    ) -> None:
        self._diseases = self._load_diseases(diseases_file)
        self._disease_by_id = {d.id: d for d in self._diseases}

        self._label_map: LabelMap = load_label_map(labels_file)

        self._no_fish_threshold = no_fish_threshold
        self._entropy_threshold = entropy_threshold
        self._high_confidence = high_confidence
        self._medium_confidence = medium_confidence
        self._enable_fish_gate = enable_fish_gate
        self._fish_gate_threshold = fish_gate_threshold

        if enable_fish_gate:
            # Warm the ImageNet gate model so the first /predict isn't slow.
            try:
                warmup_fish_gate()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Fish gate warmup failed: %s", exc)

        self._h5_path = h5_model_path
        self._onnx_path = onnx_model_path
        self._model: Optional[BaseModel] = None
        self._load_error: Optional[str] = None

        try:
            self._model = load_model(
                h5_path=h5_model_path,
                onnx_path=onnx_model_path,
                num_classes=self._label_map.num_classes,
                image_size=self._label_map.image_size,
            )
            logger.info(
                "Inference model loaded: %s (path=%s, device=%s)",
                self._model.info().backend,
                self._model.info().path,
                self._model.info().device,
            )
        except Exception as exc:
            self._load_error = str(exc)
            logger.error("Failed to load any inference backend: %s", exc)

    @staticmethod
    def _load_diseases(diseases_file: Path) -> List[Disease]:
        if not diseases_file.exists():
            raise FileNotFoundError(f"Disease metadata file not found: {diseases_file}")

        payload = json.loads(diseases_file.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload:
            raise ValueError("Disease metadata must be a non-empty list.")
        return [Disease.model_validate(item) for item in payload]

    @property
    def model_ready(self) -> bool:
        return self._model is not None

    @property
    def label_map(self) -> LabelMap:
        return self._label_map

    def status(self) -> ModelStatus:
        """Status payload returned by `/health`."""
        if self._model is None:
            return ModelStatus(
                ready=False,
                backend="none",
                model_path="",
                device="n/a",
                num_classes=self._label_map.num_classes,
                image_size=self._label_map.image_size,
                error=self._load_error,
            )
        info = self._model.info()
        return ModelStatus(
            ready=True,
            backend=info.backend,
            model_path=info.path,
            device=info.device,
            num_classes=info.num_classes,
            image_size=info.image_size,
        )

    def get_all_diseases(self) -> List[Disease]:
        """All disease entries except the synthetic "No Fish Detected" record."""
        return [d for d in self._diseases if d.id != NO_FISH_DISEASE_ID]

    @staticmethod
    def _entropy(probabilities: np.ndarray) -> float:
        clipped = np.clip(probabilities, 1e-10, 1.0)
        return float(-np.sum(clipped * np.log(clipped)))

    def _confidence_tier(self, confidence: float) -> ConfidenceTier:
        if confidence >= self._high_confidence:
            return ConfidenceTier.HIGH
        if confidence >= self._medium_confidence:
            return ConfidenceTier.MEDIUM
        return ConfidenceTier.LOW

    def _build_top_predictions(
        self, probabilities: np.ndarray
    ) -> List[ClassProbability]:
        top_indices = np.argsort(probabilities)[::-1][:TOP_K_PREDICTIONS]
        items: List[ClassProbability] = []
        for idx in top_indices:
            entry = self._label_map.by_index(int(idx))
            items.append(
                ClassProbability(
                    disease_id=entry.disease_id,
                    disease_name=entry.disease_name,
                    confidence=round(float(probabilities[idx]), 4),
                )
            )
        return items

    def _no_fish_response(
        self,
        filename: str,
        confidence: float,
        inference_ms: float,
        top_predictions: List[ClassProbability],
        warning: str,
    ) -> PredictionResponse:
        no_fish = self._disease_by_id.get(NO_FISH_DISEASE_ID) or Disease(
            id=0,
            name="No Fish Detected",
            type="Unknown",
            cause="Image not recognized.",
            symptoms="N/A",
            treatment="Try a clearer fish photo.",
            prevention="N/A",
        )
        return PredictionResponse(
            prediction=no_fish,
            confidence=round(confidence, 4),
            confidence_tier=ConfidenceTier.LOW,
            source=self._model.info().backend if self._model else "unavailable",
            filename=filename,
            inference_ms=round(inference_ms, 1),
            top_predictions=top_predictions,
            warning=warning,
            recommendation=(
                "Try again with a well-lit photo where the fish fills most of "
                "the frame, against a plain background."
            ),
        )

    async def predict(self, image_bytes: bytes, filename: str) -> PredictionResponse:
        """Run a single-image prediction.

        Raises:
            RuntimeError: if no inference backend is loaded.
            ValueError: for invalid / undecodable images.
        """
        if self._model is None:
            raise RuntimeError(
                "Inference model not loaded. "
                "Place `model.h5` in /model or train one with `python -m train.train`."
            )

        low_quality, reason = is_low_quality(image_bytes)
        warning = reason if low_quality else None

        # Stage 1: is there a fish in the image at all? The disease CNN only
        # knows 7 fish classes, so non-fish inputs must be rejected here.
        if self._enable_fish_gate:
            t_gate = time.perf_counter()
            detection = detect_fish(
                image_bytes, threshold=self._fish_gate_threshold
            )
            gate_ms = (time.perf_counter() - t_gate) * 1000
            if detection.available and not detection.is_fish:
                logger.info(
                    "Fish gate rejected %s (top=%s %.2f, fish_score=%.3f, %.0f ms)",
                    filename,
                    detection.top_label,
                    detection.top_prob,
                    detection.fish_score,
                    gate_ms,
                )
                return self._no_fish_response(
                    filename=filename,
                    confidence=detection.fish_score,
                    inference_ms=gate_ms,
                    top_predictions=[],
                    warning=(
                        "No fish detected in this image. The closest match was "
                        f"\u201c{detection.top_label}\u201d, which is not a fish."
                    ),
                )

        try:
            tensor = preprocess_image(
                image_bytes, image_size=self._label_map.image_size
            )
        except PreprocessingError as exc:
            raise ValueError(str(exc)) from exc

        t0 = time.perf_counter()
        probabilities = self._model.predict(tensor)[0]
        inference_ms = (time.perf_counter() - t0) * 1000

        top_predictions = self._build_top_predictions(probabilities)
        class_index = int(np.argmax(probabilities))
        confidence = float(probabilities[class_index])
        entropy = self._entropy(probabilities)

        # Only treat as "No Fish Detected" when the softmax is genuinely
        # near-uniform: the top class is barely above chance AND the
        # distribution is almost flat. A clear top guess (even at ~30-45%)
        # is still a real diagnosis and should be surfaced to the user.
        uncertain = (
            confidence < self._no_fish_threshold
            and entropy > self._entropy_threshold
        )
        if uncertain:
            return self._no_fish_response(
                filename=filename,
                confidence=confidence,
                inference_ms=inference_ms,
                top_predictions=top_predictions,
                warning=(
                    warning
                    or "Prediction confidence was too low to identify a disease."
                ),
            )

        class_entry = self._label_map.by_index(class_index)
        disease = self._disease_by_id[class_entry.disease_id]
        tier = self._confidence_tier(confidence)

        recommendation = self._build_recommendation(disease, tier)
        return PredictionResponse(
            prediction=disease,
            confidence=round(confidence, 4),
            confidence_tier=tier,
            source=self._model.info().backend,
            filename=filename,
            inference_ms=round(inference_ms, 1),
            top_predictions=top_predictions,
            warning=warning,
            recommendation=recommendation,
        )

    @staticmethod
    def _build_recommendation(disease: Disease, tier: ConfidenceTier) -> str:
        if tier == ConfidenceTier.LOW:
            return (
                "Confidence is low — re-photograph the fish under better light "
                "and re-run the scan. Treat this result as a hint, not a diagnosis."
            )
        if tier == ConfidenceTier.MEDIUM:
            return (
                "Confidence is moderate. Consider a second photo from a "
                "different angle to confirm before starting treatment. "
                f"Suggested treatment: {disease.treatment}"
            )
        return f"Recommended action: {disease.treatment}"
